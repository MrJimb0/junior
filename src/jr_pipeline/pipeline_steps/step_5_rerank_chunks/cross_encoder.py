"""A model-based reranker: a cross-encoder reads the question and one chunk
together and outputs a single relevance number per pair. The model used here is
the gte ModernBERT reranker.

The model is loaded only when first needed, and only from local files on disk
(``local_files_only=True``), so a machine holding protected health information
never reaches out to download model weights.

The settings below make the model give the exact same answer every run (no
compilation/optimization shortcuts, full-precision math, relevance numbers
rounded to 6 decimal places before sorting). Loads via transformers'
``AutoModelForSequenceClassification``, NOT the sentence-transformers package.
"""
from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_5_rerank_chunks.shared_reranking_contract import (
    RerankedCandidate,
    RerankerInfo,
    RerankInput,
    chunk_text_or_empty,
    rank_and_trim,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import PatientChunkStore

_VERSION = "v1"
_DEFAULT_BATCH_SIZE = 16
_DEFAULT_MAX_LENGTH = 8192  # longest input (question + chunk) the gte ModernBERT model accepts, in tokens
_SCORE_ROUNDING = 6


class CrossEncoderReranker:
    """Re-order candidates by the relevance score a local cross-encoder model gives each (question, chunk) pair."""

    info: RerankerInfo

    def __init__(
        self,
        *,
        model_path: str | Path,
        max_length: int | None = None,
        batch_size: int = _DEFAULT_BATCH_SIZE,
        device_preference: str = "auto",
    ) -> None:
        self._model_path = str(model_path)
        self._max_length = int(max_length) if max_length else None
        self._batch_size = int(batch_size)
        self._device_preference = device_preference
        # None until something is actually reranked — loading a cross-encoder costs
        # seconds and most calls never need one. Reached through _loaded_model(), so
        # "used before it was loaded" is one refusal rather than a None turning up
        # partway down a batch loop.
        self._tokenizer: Any | None = None
        self._model: Any | None = None
        self._device: Any | None = None
        self.info = RerankerInfo(
            kind="cross_encoder",
            version=_VERSION,
            config={
                "model_path": self._model_path,
                "max_length": self._max_length,
                "batch_size": self._batch_size,
            },
        )

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo
            torch._dynamo.config.disable = True
        except Exception:  # noqa: BLE001 — best-effort on older torches
            pass
        try:
            import torch as _torch  # torch = PyTorch, the numeric/deep-learning library
            from transformers import (
                AutoConfig,
                AutoModelForSequenceClassification,
                AutoTokenizer,
            )
        except ImportError as e:
            # never silently fall back to the model-free reranker — a recipe that asks
            # for the cross_encoder reranker must get it or get a clear, loud failure.
            raise RuntimeError(
                "the cross_encoder reranker requires torch + transformers; install with "
                "`pip install -e '.[torch]'`. It never silently degrades to a "
                "model-free reranker."
            ) from e

        from jr_pipeline.runtime_infrastructure.torch_device_selection import select_device

        device, dtype = select_device(self._device_preference)
        tok = AutoTokenizer.from_pretrained(self._model_path, local_files_only=True)
        model_config = AutoConfig.from_pretrained(self._model_path, local_files_only=True)
        # ModernBERT's reference_compile path causes the same run-to-run variation as
        # the dynamo compiler above; force plain run mode so the relevance scores are
        # exactly reproducible.
        if hasattr(model_config, "reference_compile"):
            model_config.reference_compile = False
        mdl = AutoModelForSequenceClassification.from_pretrained(
            self._model_path, config=model_config, local_files_only=True, torch_dtype=dtype
        ).to(device)
        mdl.eval()

        _torch.manual_seed(0)
        if hasattr(_torch, "use_deterministic_algorithms"):
            try:
                _torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # noqa: BLE001 — best-effort on older torches
                pass
        # CUBLAS_WORKSPACE_CONFIG, the other half of deterministic cuBLAS, cannot be set
        # here: CUDA has already initialized by this line. select_device sets it.
        if hasattr(_torch.backends, "cuda") and hasattr(_torch.backends.cuda, "matmul"):
            _torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(_torch.backends, "cudnn"):
            _torch.backends.cudnn.allow_tf32 = False
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False

        if self._max_length is None:
            limit = int(getattr(tok, "model_max_length", _DEFAULT_MAX_LENGTH) or _DEFAULT_MAX_LENGTH)
            # some tokenizers report a huge placeholder number when no max length is set;
            # in that case fall back to the model's real maximum.
            self._max_length = limit if limit < 1_000_000 else _DEFAULT_MAX_LENGTH
        self._tokenizer = tok
        self._model = mdl
        self._device = device

    def _loaded_model(self) -> tuple[Any, Any]:
        """The tokenizer and model, after ``_ensure_loaded`` has put them there.

        Both start as None, so every use has to establish they are not — otherwise the
        first sign of a load that did not happen is a None being called halfway through
        a batch of chart passages, which says nothing about what went wrong."""
        if self._tokenizer is None or self._model is None:
            raise RuntimeError(
                "The reranking model was asked to score passages before it was loaded. "
                "This is a bug in the reranker, not something to fix in a config."
            )
        return self._tokenizer, self._model

    def rerank(
        self,
        corpus: PatientChunkStore,
        inp: RerankInput,
        *,
        top_n: int,
    ) -> list[RerankedCandidate]:
        if not inp.candidates or top_n <= 0:
            return []  # nothing to rank, or top_n <= 0 — never bother loading the model
        self._ensure_loaded()  # raises a clear error if the model libraries are absent
        tokenizer, model = self._loaded_model()
        import torch as _torch
        chunk_texts = [chunk_text_or_empty(corpus, c.chunk_id) for c in inp.candidates]
        scores: list[float] = []
        for start in range(0, len(inp.candidates), self._batch_size):
            batch = chunk_texts[start : start + self._batch_size]
            enc = tokenizer(
                [inp.query_text] * len(batch),
                batch,
                padding=True,
                truncation=True,
                max_length=self._max_length,
                return_tensors="pt",
            ).to(self._device)
            with _torch.inference_mode():
                logits = model(**enc).logits.to(_torch.float32)
            scores.extend(self._flatten_logits(logits, len(batch)))

        scored = [
            RerankedCandidate(
                chunk_id=c.chunk_id,
                rank=0,
                score=s,
                prior_rank=c.rank,
                features={"cross_encoder_logit": round(s, _SCORE_ROUNDING)},
            )
            for c, s in zip(inp.candidates, scores, strict=True)
        ]
        # Rounded comparison: model logits carry float noise past the 6th decimal that
        # must not reorder two otherwise-tied chunks between runs.
        return rank_and_trim(scored, top_n, rounding=_SCORE_ROUNDING)

    @staticmethod
    def _flatten_logits(logits, n: int) -> list[float]:
        """Turn the model's raw relevance output (a ``(n,)``, ``(n, 1)``, or single-value
        tensor) into a plain list of ``n`` finite numbers, one per input pair; reject any
        other shape or a non-finite (NaN/infinite) score with a loud error."""
        flat = logits.reshape(-1).tolist()
        if len(flat) != n:
            raise ValueError(
                f"cross-encoder returned {len(flat)} logits for {n} inputs "
                f"(shape {tuple(logits.shape)}); expected one score per (query, chunk) pair"
            )
        for v in flat:
            if not math.isfinite(v):
                raise ValueError(f"cross-encoder returned a non-finite logit ({v!r})")
        return [float(v) for v in flat]

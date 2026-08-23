"""Default model: BioClinical-ModernBERT-base, but you can change it to whatever works
for you.

HFEncoder is the production implementation. It loads the model only when first needed ("lazy") and
picks the fastest available hardware (CUDA>MPS>CPU)

fingerprint() records which model and settings were used. The embed step stores
this alongside the vectors so a re-run can detect if the model was swapped out and
re-embed only when it changed (instead of re-embedding everything every time).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
    hash_json,
)
from jr_pipeline.runtime_infrastructure.torch_device_selection import _DTYPE_NAMES, select_device

# build_encoder spawns a fresh HFEncoder per patient; a 30k-patient run would
# otherwise re-compute the content hash (fingerprint) of the same 600 MB weights
# file 30k times. Cache it keyed on (path, size, last-modified time) so that
# overwriting the model file in place changes the key and forces a re-hash.
_WEIGHTS_HASH_CACHE: dict[tuple[str, int, float], str] = {}


def _cached_weights_hash(weights: Path) -> str:
    st = weights.stat()
    key = (str(weights.resolve()), st.st_size, st.st_mtime)
    if key not in _WEIGHTS_HASH_CACHE:
        _WEIGHTS_HASH_CACHE[key] = hash_file(weights)
    return _WEIGHTS_HASH_CACHE[key]


# Tokenizer files whose contents change how text is split — and thus chunk
# boundaries, token counts, and num_special_tokens (the default chunk window).
# A tokenizer swap touches none of the model id, config, or weights, so a hash of
# these files is folded into the fingerprint to bust the embed cache. Covers
# WordPiece (vocab.txt), BPE (vocab.json + merges.txt), fast (tokenizer.json),
# and SentencePiece families, plus config/special-token files.
_TOKENIZER_IDENTITY_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    # SentencePiece model files, named differently across families:
    # tokenizer.model (Llama/Qwen), spiece.model (T5/ALBERT/XLNet),
    # sentencepiece.bpe.model (XLM-R/NLLB).
    "tokenizer.model",
    "spiece.model",
    "sentencepiece.bpe.model",
)


# tokenizer.json alone is multi-MB, and fingerprint() runs per patient in embed and per
# variable per patient via EmbeddingRetriever.__init__ (a fresh encoder each time). Cache the
# combined tokenizer hash the same way _WEIGHTS_HASH_CACHE caches the weights — keyed on each
# present file's (path, size, mtime), so an in-place tokenizer swap re-hashes but a repeat
# fingerprint() on unchanged files (even from a different encoder instance) is free.
_TOKENIZER_HASH_CACHE: dict[tuple[str, tuple[tuple[str, int, float], ...]], str] = {}


def _cached_tokenizer_hash(root: Path) -> str | None:
    present = [(name, root / name) for name in _TOKENIZER_IDENTITY_FILES if (root / name).is_file()]
    if not present:
        return None
    key = (
        str(root.resolve()),
        tuple((name, p.stat().st_size, p.stat().st_mtime) for name, p in present),
    )
    cached = _TOKENIZER_HASH_CACHE.get(key)
    if cached is None:
        cached = hash_json({name: hash_file(p) for name, p in present})
        _TOKENIZER_HASH_CACHE[key] = cached
    return cached


# The weights are loaded and hashed from safetensors only — one format, one hash,
# no chance of hashing one file while transformers loads another.
_WEIGHTS_FILENAME = "model.safetensors"


# The fingerprint fields that determine the output vectors. Compared by the embed
# cache (embed.py) and the retrieval-time alignment check (embedding_v1.py, which
# drops model_id — a load path, so the same weights at a new path still match).
# Excluded: dim (derived from the weights), device (only affects output via the
# default dtype, keyed directly).
# Included: tokenizer_hash — a tokenizer swap changes chunk boundaries and token
# counts without touching the model path or weights.
VECTOR_AFFECTING_FINGERPRINT_FIELDS = (
    "model_id", "model_sha256", "tokenizer_hash", "pooling", "max_tokens", "normalize", "dtype",
)


class Encoder(Protocol):
    """The interface every encoder must provide: length-1 float32 vectors."""

    dim: int
    max_tokens: int
    model_id: str

    @property
    def num_special_tokens(self) -> int:
        """How many start/end marker tokens this encoder adds to a chunk at embed time.
        Read-only, because the encoder reads it off its tokenizer rather than being told
        it — declaring it as a plain attribute would demand that every encoder let a
        caller assign one."""
        ...

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Turn N text chunks into an (N, dim) array of length-1 float32 vectors."""
        ...

    def tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into tokens and report each token's (start, end) character span."""
        ...

    def fingerprint(self) -> dict[str, Any]:
        """A short summary identifying this encoder's settings — never the weights themselves."""
        ...


class _LoadedTokenizer(Protocol):
    """The tokenizer members this encoder uses, named here because the real class ships
    with `transformers` — an optional extra imported only when the model loads, so there
    is nothing to annotate against at import time."""

    def __call__(self, text: Any = None, **kwargs: Any) -> Any: ...

    def num_special_tokens_to_add(self, pair: bool = False) -> int: ...

    def convert_ids_to_tokens(self, ids: Any) -> Any: ...


# Vectors with a length (norm) below this are treated as zero-length. Real
# embeddings never get this small, but half-precision (fp16) rounding or a chunk
# that tokenizes to nothing but special tokens can produce them.
_NORM_EPSILON = 1e-12


def _unit_normalize(x: np.ndarray) -> np.ndarray:
    """Scale each row to length 1 so that a dot product equals cosine similarity."""
    if x.size == 0:
        return x.astype(np.float32, copy=False)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    # replace near-zero lengths with 1.0 so dividing returns the original near-zero vector instead of NaN.
    norm = np.where(norm < _NORM_EPSILON, 1.0, norm)
    return (x / norm).astype(np.float32, copy=False)


# ── HFEncoder defaults ───────────────────────────────────────────────────────
# Used in HFEncoder.__init__ and build_encoder so both share one source of truth.
#
#   pooling          how the model's per-token vectors are combined into one
#                    vector for the whole chunk; "mean" averages all real
#                    (non-padding) tokens, "cls" takes only the first token —
#                    mean works better for retrieval
#   max_tokens       input is cut off at this many tokens; must be <= the model's
#                    own limit (512 for ModernBERT)
#   device           "auto" tries a CUDA GPU, then Apple's MPS GPU, then CPU; set
#                    "cuda" to fail loudly if no GPU is found instead of quietly
#                    falling back to the (much slower) CPU
#   normalize        scale every output vector to length 1; must stay True for
#                    cosine similarity to work correctly in the search index
#   dtype            numeric precision of the weights. None → half precision (fp16)
#                    on CUDA, full precision (fp32) elsewhere; set explicitly when
#                    one cohort runs across mixed hardware
_DEFAULT_POOLING = "mean"
_DEFAULT_MAX_TOKENS = 512
_DEFAULT_DEVICE = "auto"
_DEFAULT_NORMALIZE = True
_DEFAULT_DTYPE: str | None = None
_DEFAULT_LOCAL_FILES_ONLY = True


class HFEncoder:
    """Encoder backed by a HuggingFace model: loads only when first used, and
    verifies the weight files' content hashes against the pinned expected values
    before putting anything on the GPU."""

    def __init__(
        self,
        *,
        model_id: str,
        pooling: str = _DEFAULT_POOLING,
        max_tokens: int = _DEFAULT_MAX_TOKENS,
        device_preference: str = _DEFAULT_DEVICE,
        normalize: bool = _DEFAULT_NORMALIZE,
        expected_file_sha256: dict[str, str] | None = None,
        dtype: str | None = _DEFAULT_DTYPE,
        local_files_only: bool = _DEFAULT_LOCAL_FILES_ONLY,
    ):
        if pooling not in {"mean", "cls"}:
            raise ValueError("pooling must be 'mean' or 'cls'")
        if dtype is not None and dtype not in _DTYPE_NAMES:
            raise ValueError(
                f"dtype must be one of {list(_DTYPE_NAMES)}; got {dtype!r}"
            )

        self.model_id = model_id
        self.pooling = pooling
        self.max_tokens = int(max_tokens)
        self.device_preference = device_preference
        # None defers the precision choice to load time (CUDA -> fp16, else -> fp32).
        self.dtype_preference = dtype

        self._normalize = normalize
        # local_files_only=True on a machine holding PHI (protected health
        # information): never reach out to the HuggingFace Hub to download weights
        # over the network — only use files already on disk. Dev/non-PHI runs may
        # set this False to allow auto-download.
        self._local_files_only = bool(local_files_only)
        self._expected_file_sha256: dict[str, str] = dict(expected_file_sha256 or {})

        # Both come from `transformers`, an optional extra that is imported inside
        # _ensure_loaded so the package installs without it. The tokenizer gets the
        # small interface below rather than Any, so a misspelled tokenizer method is
        # still caught; the model is called straight through to torch, which has no
        # useful shape to name here.
        self._tokenizer: _LoadedTokenizer | None = None
        self._model: Any = None
        self._device: str | None = None
        # the real vector length, filled in by _ensure_loaded from the model's
        # hidden_size once the model is loaded.
        self.dim = 0
        self._model_weight_sha256: str | None = None

    def _verify_local_files(self) -> None:
        """Content-hash each pinned weight file and raise if it doesn't match the
        expected value — done before loading onto the GPU."""
        if not self._expected_file_sha256:
            return

        root = Path(self.model_id)
        if not root.is_dir():
            raise RuntimeError(
                f"expected_file_sha256 requires model_id to be a local directory; "
                f"got {self.model_id!r}"
            )
        for fname, expected in sorted(self._expected_file_sha256.items()):
            target = root / fname
            if not target.is_file():
                raise RuntimeError(f"Pinned model file missing: {target}")
            actual = hash_file(target).split(":", 1)[1]
            # config entries may be prefixed ("sha256:abc...") or bare hex
            expected_hex = expected.split(":", 1)[1] if ":" in expected else expected
            if actual.lower() != expected_hex.lower():
                raise RuntimeError(
                    f"SHA-256 mismatch for {target}: expected {expected_hex}, "
                    f"got {actual}"
                )
            if fname == _WEIGHTS_FILENAME:
                self._model_weight_sha256 = f"sha256:{actual.lower()}"

    def _ensure_loaded(self) -> None:
        """Load the tokenizer + model on the first call; verify the pinned weight
        hashes before putting anything on the GPU."""
        if self._model is not None:
            return
        # Turn off torch.compile / dynamo (PyTorch's just-in-time kernel compiler)
        # BEFORE any model module is imported. ModernBERT wraps methods in
        # @torch.compile at import time, but compiled kernels are not bit-for-bit
        # identical across torch versions or machines — which would break our
        # promise that the same code produces the same embeddings — the compiler's
        # warmup spawns process pools that locked-down environments forbid, and at
        # our small per-patient batch sizes compiling costs more time than it saves.
        os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
        try:
            import torch._dynamo
            torch._dynamo.config.disable = True
        except Exception:  # noqa: BLE001 — best-effort on older torches
            pass
        from transformers import AutoModel, AutoTokenizer

        self._verify_local_files()

        device, dtype = select_device(
            self.device_preference, dtype_override=self.dtype_preference
        )
        tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=self._local_files_only)

        model_kwargs: dict[str, Any] = {"torch_dtype": dtype, "use_safetensors": True}
        # ModernBERT's reference_compile path has the same reproducibility/warmup
        # problems as the compiler above — force the plain step-by-step ("eager")
        # path instead.
        from transformers import AutoConfig
        model_config = AutoConfig.from_pretrained(self.model_id, local_files_only=self._local_files_only)
        if hasattr(model_config, "reference_compile"):
            model_config.reference_compile = False
        mdl = AutoModel.from_pretrained(
            self.model_id, config=model_config, local_files_only=self._local_files_only, **model_kwargs
        ).to(device)
        mdl.eval()  # switch off dropout (training-only randomness) — required so embeddings are repeatable

        # Fix the random seed and turn off kernels that give slightly different
        # results each run, so the same config + input always produces identical
        # embeddings (the reproducibility promise we audit against).
        import torch as _torch
        _torch.manual_seed(0)
        if hasattr(_torch, "use_deterministic_algorithms"):
            try:
                _torch.use_deterministic_algorithms(True, warn_only=True)
            except Exception:  # noqa: BLE001 — best-effort on older torches
                pass
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        if hasattr(_torch.backends, "cuda") and hasattr(_torch.backends.cuda, "matmul"):
            _torch.backends.cuda.matmul.allow_tf32 = False
        if hasattr(_torch.backends, "cudnn"):
            _torch.backends.cudnn.allow_tf32 = False
            _torch.backends.cudnn.deterministic = True
            _torch.backends.cudnn.benchmark = False

        # Batch padding needs a pad token; borrow EOS or CLS if the tokenizer
        # didn't ship one.
        if tok.pad_token_id is None:
            if tok.eos_token is not None:
                tok.pad_token = tok.eos_token
            elif tok.cls_token is not None:
                tok.pad_token = tok.cls_token

        self._tokenizer = tok
        self._model = mdl
        self._device = device
        self.dim = int(mdl.config.hidden_size)

    def _loaded_tokenizer(self) -> _LoadedTokenizer:
        """The tokenizer, loading the model first if it isn't loaded yet. Every caller
        needs one that is actually there; ``_ensure_loaded`` either sets it or raises,
        which is what this return type states and what the assertion holds it to."""
        self._ensure_loaded()
        assert self._tokenizer is not None
        return self._tokenizer

    @property
    def num_special_tokens(self) -> int:
        """How many special tokens ([CLS]/[SEP]) the tokenizer adds to a single
        chunk at embed time — ``embed_batch`` tokenizes with the default
        ``add_special_tokens=True``. The chunker subtracts this from its window so
        a full-window chunk isn't silently truncated once the markers are added.
        Triggers loading the tokenizer."""
        return int(self._loaded_tokenizer().num_special_tokens_to_add(pair=False))

    def tokenize_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        """Split text into tokens and report each token's character span.

        add_special_tokens=False — the [CLS]/[SEP] start/end markers don't point at
        any real text, so they have no character span; the e > s filter drops these
        zero-width marker tokens.
        """
        tokenizer = self._loaded_tokenizer()
        enc = tokenizer(
            text,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=False,
        )
        toks = tokenizer.convert_ids_to_tokens(enc["input_ids"])
        return [
            (tok, int(s), int(e))
            for tok, (s, e) in zip(toks, enc["offset_mapping"], strict=True)
            if e > s
        ]

    def embed_batch(self, texts: list[str]) -> np.ndarray:
        """Full path for one batch: split into tokens, run the model, combine the
        per-token vectors into one ("pool"), and scale to length 1. Returns an
        (N, dim) float32 array.

        The model's per-token outputs are converted to full precision (float32)
        before pooling — adding up hundreds of half-precision (fp16) numbers
        accumulates enough rounding error to reshuffle which chunks rank as most
        similar.
        """
        tok = self._loaded_tokenizer()
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        import torch

        enc = tok(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
        ).to(self._device)

        with torch.inference_mode():
            out = self._model(**enc)
            hidden_fp32 = out.last_hidden_state.to(torch.float32)
            if self.pooling == "cls":
                pooled = hidden_fp32[:, 0, :]
            else:
                # average over real tokens only — the padding added to short chunks
                # must not be counted, or it would drag their vectors toward zero.
                mask = enc["attention_mask"].unsqueeze(-1).to(torch.float32)
                summed = (hidden_fp32 * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                pooled = summed / denom

        arr = pooled.detach().to("cpu").numpy()
        if self._normalize:
            arr = _unit_normalize(arr)
        return arr

    def _tokenizer_identity_hash(self) -> str | None:
        """Combined content hash of the tokenizer files on disk (computed without
        loading the model). Swapping the tokenizer while keeping the same model path
        changes how text is split — chunk boundaries, token counts, and
        num_special_tokens (the default chunk window) — but touches none of the
        model path, config, or weights, so without this the embed cache would
        silently reuse stale chunks. Returns None when the model is referenced by a
        HuggingFace Hub id rather than a local directory (no local files to hash,
        same as local_config_hash / model_sha256)."""
        root = Path(self.model_id)
        if not root.is_dir():
            return None
        return _cached_tokenizer_hash(root)

    def fingerprint(self) -> dict[str, Any]:
        """A summary dict stored alongside the embeddings, used to notice when the
        encoder (model or its settings) has changed since they were written. A new
        field that changes the output vectors also belongs in
        VECTOR_AFFECTING_FINGERPRINT_FIELDS."""
        local = Path(self.model_id)
        local_hash = None
        model_sha256 = self._model_weight_sha256
        if local.is_dir():
            cfg = local / "config.json"
            if cfg.is_file():
                local_hash = hash_file(cfg)
            if model_sha256 is None:
                weights = local / _WEIGHTS_FILENAME
                if weights.is_file():
                    model_sha256 = _cached_weights_hash(weights)
                    self._model_weight_sha256 = model_sha256
        return {
            "model_id": self.model_id,
            "pooling": self.pooling,
            "max_tokens": self.max_tokens,
            "normalize": self._normalize,
            "dtype": self.dtype_preference,
            "dim": self.dim,
            "device_preference": self.device_preference,
            "local_config_hash": local_hash,
            "model_sha256": model_sha256,
            "tokenizer_hash": self._tokenizer_identity_hash(),
            "expected_file_sha256_verified": sorted(self._expected_file_sha256.keys()),
        }

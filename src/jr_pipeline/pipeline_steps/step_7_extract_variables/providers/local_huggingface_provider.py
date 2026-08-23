"""LLM provider that runs a model in-process for local dev + smoke testing.

Instead of calling a remote service, this loads a chat-tuned HuggingFace
text-generation model (HF = the HuggingFace model hub/library; a "causal
LM" is a standard generate-the-next-word language model, e.g.
qwen2.5-0.5b-instruct) into this same process on first use — same pattern
as the encoder. No HTTP, no allowlist URL, no external server: nothing
leaves the machine.

OFFLINE BY DEFAULT (``local_files_only=True``, like the encoder and the
reranker): a run on real patient data (PHI) never silently pulls model
weights from the HuggingFace Hub. A mistyped/missing local model directory
fails LOUDLY instead of being reinterpreted as a Hub repo name and reaching
out to the network. Dev/demo can set ``allow_hub_download=True``
(``allow_download: true`` on the endpoint) to permit a one-time download into
~/.cache/huggingface/ — the only code path here that touches the network.

This is the local-only alternative to the openai_compat / gemini_compat
providers. Production still goes through those (institution-approved
endpoints, covered by a Business Associate Agreement, over HTTPS). This one
exists so a developer can clone the repo, pip install, and run the full
extraction pipeline against a synthetic patient with zero server setup, and
so the Shiny demo can run end-to-end with one auto-downloaded model.

When temperature == 0 it uses greedy decoding (always pick the single most
likely next word), which makes output deterministic so the LLM result cache
hits on reruns.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMRequest,
    LLMResponse,
    ResolvedModel,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
    hash_file,
    hash_json,
)
from jr_pipeline.runtime_infrastructure.torch_device_selection import (
    release_cached_device_memory,
)

# Weight files are multi-GB, and provider_config() runs once per (patient, recipe) to form
# the LLM cache key even on a full cache-hit rerun where the model never loads. Cache each
# weight file's hash by (path, size, mtime) across ALL provider instances (mirrors
# step_2_embed_chunks.encoder._WEIGHTS_HASH_CACHE), so a run hashes each weight file once
# rather than once per recipe per patient. An in-place weight swap changes size or mtime and
# re-hashes; the hash value itself is unchanged from a direct hash_file.
_WEIGHT_SHA_CACHE: dict[tuple[str, int, float], str] = {}


def _cached_file_sha(path: Path) -> str:
    st = path.stat()
    key = (str(path.resolve()), st.st_size, st.st_mtime)
    if key not in _WEIGHT_SHA_CACHE:
        _WEIGHT_SHA_CACHE[key] = hash_file(path)
    return _WEIGHT_SHA_CACHE[key]


def _resolve_local_weight_sha(model_id: str) -> str | None:
    """Content hash of the on-disk weights for a local model directory, so the
    receipt records exactly which weights produced this extraction — mirroring the
    encoder's model_sha256. Single-file weights (model.safetensors, else
    pytorch_model.bin) hash directly; sharded safetensors (model-XXXXX-of-YYYYY,
    e.g. the 3B extractor) hash as a combined hash over every shard. None when
    model_id is a Hub name (no local weight file to hash)."""
    root = Path(model_id)
    if not root.is_dir():
        return None
    for fname in ("model.safetensors", "pytorch_model.bin"):
        weight = root / fname
        if weight.is_file():
            return _cached_file_sha(weight)
    shards = sorted(root.glob("model-*-of-*.safetensors"))
    if shards:
        return hash_json({p.name: _cached_file_sha(p) for p in shards})
    return None


@dataclass
class LocalHFProvider:
    """LLM provider that loads a HuggingFace text-generation model into this process."""

    endpoint_name: str
    model_id: str
    device_preference: str = "auto"
    dtype: str | None = None
    max_new_tokens_cap: int = 16000
    max_prompt_tokens_cap: int | None = None
    # What hardware that ceiling was measured on, for the screen that shows it.
    measured_on: str | None = None
    # offline by default — a run on real patient data never silently downloads
    # weights. dev/demo must opt in explicitly.
    allow_hub_download: bool = False

    # loaded only on the first chat() call (not at import) so importing this module
    # stays fast and cheap in test environments that never call the model.
    _tokenizer: Any = field(default=None, init=False, repr=False)
    _model: Any = field(default=None, init=False, repr=False)
    _device: str | None = field(default=None, init=False, repr=False)
    _model_weight_sha256: str | None = field(default=None, init=False, repr=False)
    _weight_sha_resolved: bool = field(default=False, init=False, repr=False)

    def _weight_sha256(self) -> str | None:
        """Content hash of the on-disk weights, memoized (the weight files are fixed for
        the life of this provider). Read directly from disk with no model load, so it is
        available for the cache key and fingerprint even on a cache hit where the model
        is never loaded. None for a Hub-name load (no local weight file to hash)."""
        if not self._weight_sha_resolved:
            self._model_weight_sha256 = _resolve_local_weight_sha(self.model_id)
            self._weight_sha_resolved = True
        return self._model_weight_sha256

    def _usable_context_tokens(self) -> int | None:
        """The loaded model's usable context length in tokens, or None if it can't be
        determined. Prefer the model config's max_position_embeddings (authoritative for
        the weights actually loaded); fall back to the tokenizer's model_max_length,
        ignoring the huge sentinel some tokenizers report when no real limit is set."""
        config = getattr(self._model, "config", None)
        max_positions = getattr(config, "max_position_embeddings", None) if config else None
        if isinstance(max_positions, int) and 0 < max_positions < 1_000_000:
            return max_positions
        tokenizer_max = getattr(self._tokenizer, "model_max_length", None)
        if isinstance(tokenizer_max, int) and 0 < tokenizer_max < 1_000_000:
            return tokenizer_max
        return None

    @property
    def _local_files_only(self) -> bool:
        """Offline unless the endpoint explicitly opted into a Hub download."""
        return not self.allow_hub_download

    @staticmethod
    def _looks_like_local_path(model_id: str) -> bool:
        """True if model_id points at a local filesystem path (in which case a missing
        directory must be a loud error), as opposed to a bare HuggingFace Hub name like
        ``Qwen/Qwen2.5-0.5B-Instruct``."""
        return (
            model_id.startswith((".", "/", "~"))
            or Path(model_id).is_absolute()
            or Path(model_id).is_dir()
        )

    def provider_config(self) -> dict[str, Any]:
        """The provider identity used in the result-cache key (a config change here
        forces re-extraction). Includes the on-disk weight hash, so swapping the weights
        at the same path changes the key and busts stale cache entries for every run —
        pinned or not. The make_key layer excludes the model fingerprint, so this is the
        only place a weight change can invalidate a cache hit."""
        return {
            "kind": "local_hf",
            "endpoint_name": self.endpoint_name,
            "model_id": self.model_id,
            "device_preference": self.device_preference,
            "dtype": self.dtype,
            "max_new_tokens_cap": self.max_new_tokens_cap,
            "max_prompt_tokens_cap": self.max_prompt_tokens_cap,
            "model_weight_sha256": self._weight_sha256(),
        }

    def _ensure_loaded(self) -> None:
        """load the model on first call. Offline by default (``local_files_only``)."""
        if self._model is not None:
            return
        local_only = self._local_files_only
        # Fail loudly and early when a local model directory is mistyped/missing,
        # instead of letting the transformers library reinterpret the path as a Hub
        # repo name and reach out to the network. A bare Hub name (no path markers)
        # is handed to from_pretrained, which with local_files_only=True loads from
        # the local HuggingFace cache or fails loudly — it never downloads.
        if (
            local_only
            and self._looks_like_local_path(self.model_id)
            and not Path(self.model_id).is_dir()
        ):
            raise RuntimeError(
                f"local extraction model dir not found: {self.model_id!r}. Place the "
                "model there, or (dev/demo only) set 'allow_download: true' on the "
                "endpoint to permit a HuggingFace Hub download."
            )
        # import these heavy ML libraries here, not at module top, so this file can be
        # imported (e.g. in tests) on a machine without torch+transformers installed.
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from jr_pipeline.runtime_infrastructure.torch_device_selection import select_device

        device, torch_dtype = select_device(
            self.device_preference, dtype_override=self.dtype
        )
        tok = AutoTokenizer.from_pretrained(self.model_id, local_files_only=local_only)
        mdl = AutoModelForCausalLM.from_pretrained(
            self.model_id, torch_dtype=torch_dtype, local_files_only=local_only
        ).to(device)
        # eval() disables dropout — required for deterministic outputs.
        mdl.eval()

        # record exactly which weights produced this patient's extraction (a content
        # hash for provenance), mirroring the encoder's model_sha256. Memoized; None for
        # a Hub-name load (there is no local file to hash).
        self._weight_sha256()

        # some chat models leave the padding token unset; reuse the end-of-sequence
        # token for padding so generation doesn't error.
        if tok.pad_token_id is None and tok.eos_token_id is not None:
            tok.pad_token = tok.eos_token

        self._tokenizer = tok
        self._model = mdl
        self._device = device
        _ = torch

    def chat(self, req: LLMRequest) -> LLMResponse:
        """generate a chat completion for req, wrapped in the shared retry layer.

        Like the openai/gemini providers, delegate through chat_with_retries so a
        truncated local generation (finish_reason='length') gets the same bigger-budget
        content retry and truncation quarantine as a remote call, instead of returning a
        cut-off answer unguarded. The bigger-budget retry runs only for recipes that
        opted in via llm.retry_cut_off_answers (carried on the request)."""
        from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_call_retry_logic import (
            ResilienceContext,
            chat_with_retries,
        )

        ctx = ResilienceContext(
            endpoint_name=self.endpoint_name,
            task_name=req.task_name,
            quarantine_path=req.quarantine_path,
        )
        return chat_with_retries(inner=self._raw_chat, req=req, ctx=ctx)

    def _raw_chat(self, req: LLMRequest) -> LLMResponse:
        """generate a chat completion for req — one call, no retry layer."""
        self._ensure_loaded()
        import torch

        t0 = time.perf_counter()
        # Keep every device tensor in explicit locals so the finally block can release
        # the last references before asking torch to empty its allocator cache. Merely
        # calling empty_cache while these are live does not reclaim their memory.
        encoded = prompt_ids = attention_mask = out_ids = new_tokens = None
        try:
            # return_dict=True pins the return to a BatchEncoding across
            # transformers versions (4.x returned a bare tensor; 5.x raises
            # on tensor-style attribute access).
            encoded = self._tokenizer.apply_chat_template(
                req.messages,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
            prompt_ids = encoded["input_ids"]
            prompt_len = int(prompt_ids.shape[-1])
            max_new = min(int(req.max_tokens), self.max_new_tokens_cap)

            # MPS's attention implementation needs a large temporary whose size grows
            # quadratically with the prompt. A model's advertised context window can
            # therefore be much larger than this machine can execute safely. Endpoint
            # config may impose a measured hardware ceiling; reject above it before even
            # copying the prompt to MPS, rather than pressure-killing the whole CLI.
            if (
                self.max_prompt_tokens_cap is not None
                and prompt_len > self.max_prompt_tokens_cap
            ):
                raise ValueError(
                    f"local model {self.model_id!r} for recipe {req.task_name!r}: the "
                    f"prompt is {prompt_len} tokens, over this endpoint's safe prompt "
                    f"ceiling of {self.max_prompt_tokens_cap}. Reduce retrieval k/top_n "
                    "or evidence size; the model was not run and evidence was not "
                    "silently truncated."
                )

            # A remote API 400s loudly when the prompt overflows the context window; a local
            # model would instead generate silently degraded output that still parses as JSON.
            # Guard loudly here rather than let that happen — never silently truncate evidence.
            usable_context = self._usable_context_tokens()
            if usable_context is not None and prompt_len + max_new > usable_context:
                raise ValueError(
                    f"local model {self.model_id!r} for recipe {req.task_name!r}: the prompt "
                    f"is {prompt_len} tokens and max_new_tokens is {max_new}, a total of "
                    f"{prompt_len + max_new} that exceeds the model's usable context of "
                    f"{usable_context} tokens. Tighten the step-5 selection (filters / top_n) "
                    "or evidence.max_context_tokens, or use a model with a larger context "
                    "window; the evidence is not silently truncated."
                )

            encoded = encoded.to(self._device)
            prompt_ids = encoded["input_ids"]
            attention_mask = encoded.get("attention_mask")

            gen_kwargs: dict[str, Any] = {
                "max_new_tokens": max_new,
                "pad_token_id": self._tokenizer.pad_token_id,
            }
            if req.temperature <= 0.0:
                gen_kwargs["do_sample"] = False
                gen_kwargs["num_beams"] = 1
            else:
                gen_kwargs["do_sample"] = True
                gen_kwargs["temperature"] = float(req.temperature)
            if req.seed is not None:
                torch.manual_seed(int(req.seed))

            with torch.inference_mode():
                out_ids = self._model.generate(
                    prompt_ids, attention_mask=attention_mask, **gen_kwargs
                )
            new_tokens = out_ids[0, prompt_len:]
            content = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
            n_new = int(new_tokens.shape[-1])
            # generating right up to the token cap means the output was cut off, not
            # finished naturally — report "length" so the retry layer's bigger-budget
            # retry for cut-off answers can fire.
            finish_reason = "length" if n_new >= max_new else "stop"

            # The weight hash is part of the fingerprint so a pinned run
            # (llm.expected_fingerprint) verifies against weight identity, not just the
            # path/device/dtype.
            resolved = ResolvedModel(
                endpoint_name=self.endpoint_name,
                api_model_id=self.model_id,
                api_version=None,
                deployment_name="local",
                fingerprint=(
                    f"local_hf|{self.model_id}|{self.device_preference}|"
                    f"{self.dtype or 'auto'}|new_cap={self.max_new_tokens_cap}|"
                    f"prompt_cap={self.max_prompt_tokens_cap or 'none'}|"
                    f"{self._weight_sha256() or 'nohash'}"
                ),
            )
            response_raw = {
                "object": "chat.completion",
                "model": self.model_id,
                "choices": [{"index": 0, "finish_reason": finish_reason,
                             "message": {"role": "assistant", "content": content}}],
                "usage": {
                    "prompt_tokens": prompt_len,
                    "completion_tokens": n_new,
                    "total_tokens": prompt_len + n_new,
                },
                "_local_hf": True,
                "_model_weight_sha256": self._weight_sha256(),
            }
            return LLMResponse(
                content=content,
                response_raw=response_raw,
                resolved_model=resolved,
                usage=response_raw["usage"],
                latency_s=round(time.perf_counter() - t0, 6),
            )
        finally:
            # This also runs when generate() raises OOM. Drop all tensor aliases first;
            # new_tokens is a view of out_ids and encoded still owns prompt_ids.
            new_tokens = out_ids = attention_mask = prompt_ids = encoded = None
            release_cached_device_memory(str(self._device))

"""Device + dtype selection for torch models.

A step-independent leaf module so step 7's local HF provider can pick a device
without importing the step-2 encoder package — that would invert the step
layering (step 7 must not depend on step 2). torch is imported lazily so
`import jr_pipeline` and `--help` work even when torch is absent.
"""
from __future__ import annotations

import os
from typing import Any

# _DTYPE_NAMES is the validation allowlist; _torch_dtype maps names to torch constants.
# Always validate against _DTYPE_NAMES before calling _torch_dtype.
_DTYPE_NAMES = ("float16", "bfloat16", "float32")


def _torch_dtype(name: str) -> Any:
    import torch
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def select_device(
    preferred: str = "auto",
    *,
    dtype_override: str | None = None,
) -> tuple[str, Any]:
    """pick (device, dtype); auto cascades CUDA -> MPS -> CPU, explicit names fail loudly."""
    # CUDA reads CUBLAS_WORKSPACE_CONFIG once, when its runtime initializes, and ignores
    # every later change (short of a new process). Deterministic cuBLAS therefore has to
    # be asked for BEFORE anything touches a GPU — and in a cohort run the first thing
    # that does is the embed encoder, long before the cross-encoder that wants it. This
    # is the one place every torch model passes through on its way to a device, so it is
    # the last point that is still early enough. setdefault, so an operator can override.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError as e:
        raise RuntimeError(
            "HFEncoder requires torch. Install with `pip install -e '.[torch]'`."
        ) from e

    if dtype_override is not None and dtype_override not in _DTYPE_NAMES:
        raise ValueError(
            f"dtype must be one of {list(_DTYPE_NAMES)}; got {dtype_override!r}"
        )

    if preferred == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif (
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        ):
            device = "mps"
        else:
            device = "cpu"
    elif preferred == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "device=cuda requested but torch.cuda.is_available() is False"
            )
        device = "cuda"
    elif preferred == "mps":
        if not (
            getattr(getattr(torch, "backends", None), "mps", None)
            and torch.backends.mps.is_available()
        ):
            raise RuntimeError("device=mps requested but MPS is unavailable")
        device = "mps"
    elif preferred == "cpu":
        device = "cpu"
    else:
        raise ValueError(f"Unknown device preference: {preferred!r}")

    if dtype_override is not None:
        dtype = _torch_dtype(dtype_override)
    elif device == "cuda":
        dtype = torch.float16
    else:
        dtype = torch.float32

    return device, dtype


def release_cached_device_memory(device: str) -> None:
    """Hand freed GPU blocks back to the operating system.

    Torch keeps a caching allocator: freeing a tensor returns its block to a pool rather
    than to the OS, so the next allocation is fast. Measured on this pipeline, one
    extraction left 11.8 GiB of live tensors and 47 GiB held by the allocator — the
    difference being transient attention and generation buffers that were freed but never
    returned. A few patients of that reaches the MPS high-watermark and every subsequent
    allocation raises "MPS backend out of memory", including the small ones the query
    encoder needs, so retrieval starts failing for reasons that have nothing to do with
    the chart being read.

    CONTRACT: only call this immediately after using ``device`` in this process. Calling
    `torch.mps.empty_cache()` where MPS was never initialized SEGFAULTS — the process
    dies, no exception is raised, and no try/except can save it. There is no reliable way
    to ask torch whether a backend is initialized without risking the same crash, so the
    safety here is the caller's knowledge, not a runtime check: the local provider calls
    it having just run a generation on that exact device. Do not add a speculative call
    somewhere that merely hopes a GPU is in play — that is how this crashed the test
    suite the first time.

    Called after each local generation, where the cost — the next allocation asking the
    driver again — is nothing beside a generation that takes seconds to minutes. A no-op
    on CPU, and on any torch too old to expose it."""
    if device not in ("mps", "cuda"):
        return
    try:
        import torch
    except ImportError:
        return
    backend = getattr(torch, device, None)
    empty = getattr(backend, "empty_cache", None)
    if empty is None:
        return
    # Generations can enqueue asynchronous GPU work. Wait for it to finish so buffers
    # released by the caller are genuinely unoccupied before empty_cache examines the
    # allocator. Still attempt the reclaim if synchronize itself is unavailable/fails.
    synchronize = getattr(backend, "synchronize", None)
    if synchronize is not None:
        try:
            synchronize()
        except Exception:  # noqa: BLE001 — best-effort cleanup only
            pass
    try:
        empty()
    except Exception:  # noqa: BLE001 — reclaiming memory must never fail a run
        return

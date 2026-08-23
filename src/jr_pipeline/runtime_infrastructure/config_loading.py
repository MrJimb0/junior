"""YAML config loader: OmegaConf handles ``${VAR}`` interpolation; the per-step
shape checks below validate the resulting dict."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from omegaconf import OmegaConf


def load_config(path: str | Path) -> dict:
    """Load one self-contained YAML, interpolate env vars, return a plain dict."""
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"Config not found: {p}")
    try:
        with p.open("r", encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        # A hand-edited settings file with a typo in it is an ordinary mistake, and
        # PyYAML's own exception reaches the terminal as a forty-line traceback
        # through click and site-packages. Keep the part that says where, drop the
        # rest — the operator needs the line number, not the parser's call stack.
        where = ""
        mark = getattr(e, "problem_mark", None)
        if mark is not None:
            where = f" (line {mark.line + 1}, column {mark.column + 1})"
        problem = getattr(e, "problem", None) or "could not be parsed"
        raise ValueError(f"{p.name} is not valid YAML{where}: {problem}.") from e
    if not isinstance(raw, dict):
        raise ValueError(f"Config root must be a mapping: {p}")
    cfg = OmegaConf.create(raw)
    OmegaConf.resolve(cfg)
    resolved = OmegaConf.to_container(cfg, resolve=True)
    if isinstance(resolved, dict) and "extends" in resolved:
        raise ValueError(
            f"{p}: 'extends' was removed — configs are self-contained now. Inline the base "
            "file's keys into this config; a leftover 'extends' key is otherwise silently "
            "ignored, so a child config would lose the inherited keys (e.g. weight pins)."
        )
    return resolved


# Step-specific shape checks (candidates to migrate to Pydantic models per step).

def require(cfg: dict, key: str, kind: type | tuple[type, ...] = object) -> Any:
    if key not in cfg:
        raise ValueError(f"Missing required config key: {key!r}")
    v = cfg[key]
    if not isinstance(v, kind) and kind is not object:
        raise ValueError(f"Config key {key!r} must be of type {kind}; got {type(v).__name__}")
    return v


def validate_ingest_config(cfg: dict) -> None:
    require_run_id(cfg)
    if cfg.get("source_root") is not None:
        require(cfg, "source_root", str)
    if cfg.get("project") is not None:
        require(cfg, "project", str)
    # files: omitted or "auto" -> discover; else list of {stem, optional?}.
    # Step 1 always preserves whole source files as structured parquets.
    # Column selection belongs to Step 2 embed configs.
    files = cfg.get("files")
    if files is None or files == "auto":
        return
    if not isinstance(files, list):
        raise ValueError(
            f"Config key 'files' must be a list of file specs or the string 'auto'; "
            f"got {type(files).__name__}"
        )
    for i, entry in enumerate(files):
        if not isinstance(entry, dict):
            raise ValueError(
                f"files[{i}] must be a mapping with a 'stem' key; got {type(entry).__name__}"
            )
        if "stem" not in entry or not isinstance(entry["stem"], str):
            raise ValueError(f"files[{i}] is missing a string 'stem' key")
        if "optional" in entry and not isinstance(entry["optional"], bool):
            raise ValueError(
                f"files[{i}].optional must be bool; got {type(entry['optional']).__name__}"
            )
        cols = entry.get("columns", "auto")
        if cols not in ("auto", None):
            raise ValueError(
                "Step 1 ingest preserves whole source files. Remove "
                f"files[{i}].columns and choose embed text columns in Step 2 instead."
            )


def require_run_id(cfg: dict) -> None:
    if not cfg.get("run_id"):
        raise ValueError(
            "run_id is not set. Pass run_id in your settings (interface) "
            "or export JR_RUN_ID (SLURM)."
        )
    require(cfg, "run_id", str)


def validate_embed_config(cfg: dict) -> None:
    require_run_id(cfg)
    # encoder is required — embed never falls back to a downloaded model
    # (silent network egress on a PHI machine; fails on an air-gapped cluster anyway).
    require(cfg, "encoder", dict)
    if not cfg["encoder"].get("model_id"):
        raise ValueError(
            "encoder.model_id is not set. Point embedding_model_path (interface) "
            "or encoder.model_id (config) at a local model folder, "
            "e.g. ./models/embedding/<model>."
        )
    if "use_safetensors" in cfg["encoder"]:
        raise ValueError(
            "encoder.use_safetensors was removed — encoder weights are safetensors-only "
            f"now. Convert {cfg['encoder'].get('model_id', '<model dir>')} with "
            "safetensors.torch.save_file and drop the key (a .bin-only value is otherwise "
            "ignored and later fails with a generic transformers error)."
        )
    if cfg.get("chunker") is not None:
        require(cfg, "chunker", dict)
    # chunker.window must fit encoder.max_tokens, else embeddings only cover
    # the first max_tokens of each chunk while retrieval shows the full window.
    enc = cfg.get("encoder") or {}
    chunker = cfg.get("chunker") or {}
    enc_max = int(enc.get("max_tokens", 512))
    win = int(chunker.get("window", 0)) if chunker.get("kind") == "token_window" else 0
    if win and win > enc_max:
        raise ValueError(
            f"chunker.window ({win}) exceeds encoder.max_tokens ({enc_max}). "
            f"Embeddings would only represent the first {enc_max} tokens of each chunk "
            "while retrieval text shows the full window. Lower the chunker window "
            "or pick a longer-context encoder."
        )
    files = cfg.get("files")
    if files is not None and files != "auto" and not isinstance(files, list):
        raise ValueError(
            f"Config key 'files' must be a list of embed specs or 'auto'; "
            f"got {type(files).__name__}"
        )
    if cfg.get("text_column") is not None:
        require(cfg, "text_column", str)
    if isinstance(files, list):
        for i, entry in enumerate(files):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"files[{i}] must be a mapping with a 'stem' key; got {type(entry).__name__}"
                )
            if "stem" not in entry or not isinstance(entry["stem"], str):
                raise ValueError(f"files[{i}] is missing a string 'stem' key")
            if "embed" in entry and not isinstance(entry["embed"], bool):
                raise ValueError(
                    f"files[{i}].embed must be bool; got {type(entry['embed']).__name__}"
                )
            cols = entry.get("text_columns")
            if cols is not None and (
                not isinstance(cols, list) or not all(isinstance(c, str) for c in cols)
            ):
                raise ValueError(
                    f"files[{i}].text_columns must be a list of column-name strings; "
                    f"got {type(cols).__name__}"
                )
            # chunk_id is {patient}:{stem}:{row}:{chunk_idx} and omits the source
            # column, so two text columns on the same row would collide. Reject multi-
            # column specs loudly; add source_column to the id only when a real
            # multi-column need appears.
            if isinstance(cols, list) and len(cols) > 1:
                raise ValueError(
                    f"files[{i}].text_columns has {len(cols)} columns ({cols}); embed "
                    "supports exactly one text column per file spec today. The chunk_id "
                    "omits the source column, so two columns on the same row would "
                    "collide. Use one column per spec (add a separate files[] entry "
                    "per column)."
                )


def validate_index_config(cfg: dict) -> None:
    require_run_id(cfg)
    if cfg.get("index") is not None:
        require(cfg, "index", dict)

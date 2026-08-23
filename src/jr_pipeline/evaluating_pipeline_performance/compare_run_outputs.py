"""Run-vs-run diffs over the sealed artifact layer.

``compare_run_outputs`` diffs patient retrieval/source artifacts, ``code_diff`` diffs
hashes.json, and ``compare_outputs_with_code_changes`` pairs a per-variable result diff
with the scoped code-hash changes that could explain it.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jr_pipeline.evaluating_pipeline_performance.run_output_reader import (
    RunLayout,
    hashes,
    load_json,
)


def _deep_diff(a: Any, b: Any, prefix: str = "") -> list[dict[str, Any]]:
    """Yield ``{path, a, b}`` entries for every differing leaf."""
    if type(a) is not type(b):
        return [{"path": prefix or "$", "a": a, "b": b}]
    out: list[dict[str, Any]]
    if isinstance(a, dict):
        keys = sorted(set(a) | set(b))
        out = []
        for k in keys:
            out.extend(_deep_diff(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k))
        return out
    if isinstance(a, list):
        if len(a) != len(b):
            return [{"path": prefix or "$", "a": a, "b": b, "list_len_diff": True}]
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out.extend(_deep_diff(x, y, f"{prefix}[{i}]"))
        return out
    if a != b:
        return [{"path": prefix or "$", "a": a, "b": b}]
    return []


def diff_artifacts(a: dict, b: dict) -> list[dict[str, Any]]:
    """Diff two artifact envelopes' payloads. Ignores envelope metadata."""
    return _deep_diff(a.get("payload", {}), b.get("payload", {}), prefix="payload")


def compare_run_outputs(run_a: Path, run_b: Path, patient_id: str) -> dict:
    """Diff two runs for one patient: source snapshot + retrieval-layer artifacts."""
    la = RunLayout(run_a)
    lb = RunLayout(run_b)
    out: dict[str, Any] = {"run_a": la.run_root.name, "run_b": lb.run_root.name, "patient_id": patient_id}

    pa = la.patient_dir(patient_id) / "source_snapshot.json"
    pb = lb.patient_dir(patient_id) / "source_snapshot.json"
    if pa.is_file() and pb.is_file():
        out["source_snapshot"] = diff_artifacts(load_json(pa), load_json(pb))

    ra = {p.name: load_json(p) for p in la.retrieval_artifacts(patient_id)}
    rb = {p.name: load_json(p) for p in lb.retrieval_artifacts(patient_id)}
    retrieval_diffs: dict[str, Any] = {}
    for name in sorted(set(ra) | set(rb)):
        if name in ra and name in rb:
            retrieval_diffs[name] = diff_artifacts(ra[name], rb[name])
        else:
            retrieval_diffs[name] = {"only_in": "a" if name in ra else "b"}
    out["retrieval"] = retrieval_diffs
    return out


def code_diff(
    run_a: Path, run_b: Path, *, scope: str | None = None
) -> dict[str, Any]:
    """Diff code/hashes.json between two runs; ``scope`` narrows to one sub-hash family."""
    ha = hashes(run_a)["payload"]
    hb = hashes(run_b)["payload"]

    if scope is None:
        return {
            "run_a": run_a.name,
            "run_b": run_b.name,
            "diffs": _deep_diff(ha, hb),
        }

    scope_map = {
        "recipe": "per_recipe",
        "prompt": "per_recipe",
        "schema": "per_recipe",
        "python_helpers": "per_recipe",
        "provider": "provider_config_hash",
        "retrieval": "retrieval_config_hash",
        "config": "config_hash",
        "env": "env_hash",
    }
    if scope not in scope_map:
        raise ValueError(f"Unknown scope {scope!r}; supported: {sorted(scope_map)}")
    key = scope_map[scope]

    if key == "per_recipe":
        subfield = {
            "recipe": "recipe_hash",
            "schema": "schema_hash",
            "prompt": "prompt_hashes",
            "python_helpers": "python_helpers_hash",
        }[scope]
        a_sub = {name: rec.get(subfield) for name, rec in (ha.get("per_recipe") or {}).items()}
        b_sub = {name: rec.get(subfield) for name, rec in (hb.get("per_recipe") or {}).items()}
        return {"run_a": run_a.name, "run_b": run_b.name, "scope": scope, "diffs": _deep_diff(a_sub, b_sub)}

    return {
        "run_a": run_a.name,
        "run_b": run_b.name,
        "scope": scope,
        "diffs": _deep_diff(ha.get(key), hb.get(key)),
    }


def _load_result(run_root: Path, patient_id: str, variable: str) -> dict:
    # writers use run_root/patients/<pid>/...
    path = run_root / "patients" / patient_id / "extract" / variable / "result.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8")).get("payload") or {}


def _per_recipe_for(hashes_payload: dict, variable: str) -> dict:
    """The scoped per-recipe sub-hashes for ``variable``. per_recipe is keyed
    ``"<variable>/<version>"`` by the producer, so resolve the versioned key
    (highest if several), with a bare-key fallback."""
    per_recipe = hashes_payload.get("per_recipe") or {}
    if variable in per_recipe:
        return per_recipe[variable]
    matches = sorted(k for k in per_recipe if k.split("/", 1)[0] == variable)
    return per_recipe[matches[-1]] if matches else {}


def compare_outputs_with_code_changes(
    run_a: Path,
    run_b: Path,
    patient_id: str,
    variable: str,
) -> dict[str, Any]:
    """Pair a patient-level result diff with the scoped code-hash changes that could explain it."""
    a = _load_result(run_a, patient_id, variable)
    b = _load_result(run_b, patient_id, variable)
    data_diff = _deep_diff(a.get("data"), b.get("data"), prefix="data")

    ha = hashes(run_a)["payload"]
    hb = hashes(run_b)["payload"]

    per_a = _per_recipe_for(ha, variable)
    per_b = _per_recipe_for(hb, variable)
    recipe_diff = _deep_diff(per_a, per_b, prefix=f"per_recipe.{variable}")

    global_scopes = {
        "provider_config_hash": (ha.get("provider_config_hash"), hb.get("provider_config_hash")),
        "retrieval_config_hash": (ha.get("retrieval_config_hash"), hb.get("retrieval_config_hash")),
        "config_hash": (ha.get("config_hash"), hb.get("config_hash")),
        "env_hash": (ha.get("env_hash"), hb.get("env_hash")),
    }
    global_diffs = {k: {"a": v[0], "b": v[1], "changed": v[0] != v[1]} for k, v in global_scopes.items()}

    return {
        "run_a": run_a.name,
        "run_b": run_b.name,
        "patient_id": patient_id,
        "variable": variable,
        "data_diff": data_diff,
        "code_diff": {
            "recipe_scope": recipe_diff,
            "global_scopes": global_diffs,
        },
    }



"""Cross-run retrieval-quality comparison.

Takes two or more sealed run roots, each of which
already has ``data/eval/candidate_metrics.json`` and
``data/eval/bundle_metrics.json`` written by :mod:`jr_pipeline.evaluating_pipeline_performance.run`,
and produces a flat comparison table: one row per run, columns for
recall@k, MRR, and bundle sufficiency.

The intent is to sanity-check a chunk-size or encoder swap by running
``jr_pipeline eval`` once per config and then ``jr_pipeline compare-runs``
to see whether the new run's retrieval quality cleared or regressed
against the incumbent.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    eval_metrics_dir,
)


def _read_candidate(run_root: Path) -> dict[str, Any]:
    p = eval_metrics_dir(run_root) / "candidate_metrics.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"No candidate_metrics.json under {run_root}. "
            f"Run `jr_pipeline eval` against this run first."
        )
    return json.loads(p.read_text(encoding="utf-8"))["payload"]


def _read_bundle(run_root: Path) -> dict[str, Any]:
    p = eval_metrics_dir(run_root) / "bundle_metrics.json"
    if not p.is_file():
        raise FileNotFoundError(
            f"No bundle_metrics.json under {run_root}. "
            f"Run `jr_pipeline eval` against this run first."
        )
    return json.loads(p.read_text(encoding="utf-8"))["payload"]


def compare_runs(
    run_roots: list[Path],
    *,
    k_values: tuple[int, ...] = (1, 3, 5, 10, 20),
) -> dict[str, Any]:
    """Return a structured comparison of candidate + bundle metrics across runs."""
    if len(run_roots) < 2:
        raise ValueError("compare_runs requires at least two run roots")

    rows: list[dict[str, Any]] = []
    for rr in run_roots:
        rr = Path(rr)
        cand = _read_candidate(rr)
        bun = _read_bundle(rr)
        row = {
            "run_id": rr.name,
            "n_queries": int(cand.get("n_queries", 0)),
            "recall_at_k": {int(k): float(v) for k, v in cand.get("mean_recall_at_k", {}).items()},
            "mrr": float(cand.get("mean_mrr", 0.0)),
            "bundle_sufficient_rate": float(bun.get("bundle_sufficient_rate", 0.0)),
            "elapsed_s": cand.get("elapsed_s"),
        }
        rows.append(row)

    # Ties broken by first run (max is stable on equal keys).
    def _best_by(key_fn) -> str:
        return max(rows, key=key_fn)["run_id"]

    best = {
        "mrr": _best_by(lambda r: r["mrr"]),
        "bundle_sufficient_rate": _best_by(lambda r: r["bundle_sufficient_rate"]),
    }
    for k in k_values:
        if any(k in r["recall_at_k"] for r in rows):
            best[f"recall_at_{k}"] = _best_by(lambda r, k=k: r["recall_at_k"].get(k, 0.0))

    baseline = rows[0]
    deltas: list[dict[str, Any]] = []
    for r in rows[1:]:
        d: dict[str, Any] = {"run_id": r["run_id"], "baseline": baseline["run_id"]}
        d["mrr_delta"] = round(r["mrr"] - baseline["mrr"], 4)
        d["bundle_sufficient_rate_delta"] = round(
            r["bundle_sufficient_rate"] - baseline["bundle_sufficient_rate"], 4
        )
        d["recall_at_k_delta"] = {
            k: round(r["recall_at_k"].get(k, 0.0) - baseline["recall_at_k"].get(k, 0.0), 4)
            for k in k_values
            if k in baseline["recall_at_k"] or k in r["recall_at_k"]
        }
        deltas.append(d)

    return {"runs": rows, "best": best, "deltas": deltas}


def format_table(report: dict[str, Any]) -> str:
    """Render the comparison as a monospace text table for CLI output."""
    rows = report["runs"]
    headers = ["run_id", "n", "MRR", "R@1", "R@5", "R@10", "bundle"]
    lines = []
    widths = [max(len(h), 12) for h in headers]
    for r in rows:
        widths[0] = max(widths[0], len(r["run_id"]))

    def _fmt_row(cells: list[str]) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths, strict=True))

    lines.append(_fmt_row(headers))
    lines.append(_fmt_row(["-" * w for w in widths]))
    for r in rows:
        rcall = r["recall_at_k"]
        lines.append(
            _fmt_row([
                r["run_id"],
                str(r["n_queries"]),
                f"{r['mrr']:.4f}",
                f"{rcall.get(1, 0.0):.4f}",
                f"{rcall.get(5, 0.0):.4f}",
                f"{rcall.get(10, 0.0):.4f}",
                f"{r['bundle_sufficient_rate']:.4f}",
            ])
        )
    lines.append("")
    lines.append("best:")
    for k, v in sorted(report["best"].items()):
        lines.append(f"  {k:26s} {v}")
    if report.get("deltas"):
        lines.append("")
        lines.append(f"deltas vs baseline ({rows[0]['run_id']}):")
        for d in report["deltas"]:
            lines.append(
                f"  {d['run_id']:26s} MRR {d['mrr_delta']:+.4f}  "
                f"bundle {d['bundle_sufficient_rate_delta']:+.4f}"
            )
    return "\n".join(lines)

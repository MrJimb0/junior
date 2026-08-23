"""A project's runs should be findable by date, with the answers pointed at.

A project accumulates folders named 20260820_105033_b74d inside one called
pipeline_run_receipts, each holding fifteen entries of which two are the point — the
rest being a response cache, an hmac secret, state fragments and two jsonl logs. Every
piece of that is defensible and the whole is unreadable.
"""
from __future__ import annotations

from pathlib import Path

from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    recipe_table_name,
    render_project_run_index,
    run_output_dir,
    write_project_run_index,
)


def _receipts(tmp_path: Path) -> Path:
    """The receipts root inside a real project tree.

    The answers now sit one level out from it, under the project's CONTAINS_PHI folder,
    so a fixture that treats any old directory as the receipts root stops describing
    where the files actually go."""
    receipts = tmp_path / "CONTAINS_PHI" / "pipeline_run_receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    return receipts


def _a_run(receipts: Path, run_id: str, *, values=False, rows_table=None, extracted=True):
    run = receipts / run_id
    (run / "patients").mkdir(parents=True)
    if extracted:
        (run / "summary.json").write_text("{}", encoding="utf-8")
    if values or rows_table:
        output = run_output_dir(run)
        output.mkdir(parents=True, exist_ok=True)
        # Named through the layout module, not spelled here: a fixture that invents its
        # own filename keeps passing after the real one changes shape.
        for variable in ([("date_of_birth")] if values else []) + (
                [rows_table] if rows_table else []):
            (output / recipe_table_name(run_id, variable)).write_bytes(b"")
    # The machinery a reader does not want listed.
    (run / "llm_cache.db").write_text("", encoding="utf-8")
    (run / "exhaust_hmac_secret").write_text("", encoding="utf-8")
    (run / "state_fragments").mkdir()
    return run


def test_a_run_id_is_shown_as_a_date(tmp_path):
    """20260820_105033_b74d is 20 Aug 2026 at 10:50 and reads as one to nobody."""
    _a_run(_receipts(tmp_path), "20260820_105033_b74d", values=True)

    index = render_project_run_index(_receipts(tmp_path))

    assert "20 Aug 2026  10:50" in index
    assert "20260820_105033_b74d" in index, "the real folder name still has to be there"


def test_the_answers_come_first(tmp_path):
    """What somebody opens a run for, before the material behind it. The answers are a
    folder rather than loose files, so they are not listed one by one — a run with
    twenty recipes would bury everything else."""
    _a_run(_receipts(tmp_path), "20260820_105033_b74d", values=True, rows_table="a_table_recipe")

    lines = [line.strip() for line in render_project_run_index(_receipts(tmp_path)).splitlines() if line.strip()]
    order = [i for i, line in enumerate(lines)
             if line.startswith(("../answers/", "patients", "summary.json"))]

    assert lines[order[0]].startswith("../answers/")
    assert "2 recipe table(s)" in lines[order[0]]
    assert lines[order[1]].startswith("patients")


def test_the_machinery_is_not_listed(tmp_path):
    """Listing everything rebuilds the haystack this exists to remove."""
    _a_run(_receipts(tmp_path), "20260820_105033_b74d", values=True)

    index = render_project_run_index(_receipts(tmp_path))

    for machinery in ("llm_cache", "exhaust_hmac_secret", "state_fragments",
                      "run_log.jsonl"):
        assert machinery not in index, machinery


def test_newest_first(tmp_path):
    _a_run(_receipts(tmp_path), "20260820_101842_953c", values=True)
    _a_run(_receipts(tmp_path), "20260820_105033_b74d", values=True)

    index = render_project_run_index(_receipts(tmp_path))

    assert index.index("105033") < index.index("101842")


def test_a_run_with_nothing_extracted_says_so(tmp_path):
    """Rather than an empty entry, which reads as the index being broken."""
    _a_run(_receipts(tmp_path), "20260820_101842_953c", extracted=False)

    assert "has not been through extract" in render_project_run_index(_receipts(tmp_path))


def test_a_project_with_no_runs_says_what_to_do(tmp_path):
    assert "No runs yet" in render_project_run_index(_receipts(tmp_path))


def test_it_is_rewritten_in_place(tmp_path):
    """Refreshed whenever a command touches the project, so it cannot go stale — the
    same reason the per-run guide is rendered fresh rather than written once."""
    _a_run(_receipts(tmp_path), "20260820_101842_953c", values=True)
    first = write_project_run_index(_receipts(tmp_path)).read_text(encoding="utf-8")

    _a_run(_receipts(tmp_path), "20260820_105033_b74d", values=True)
    second = write_project_run_index(_receipts(tmp_path)).read_text(encoding="utf-8")

    assert "105033" not in first
    assert "105033" in second and "101842" in second


def test_it_never_creates_the_folder_it_indexes(tmp_path):
    """It ran on every ensure_layout and did mkdir(parents=True), which made an empty
    project look started — and `new-project` refuses to overwrite a project that already
    exists, so the second creation of the same name silently succeeded. An index of runs
    where there are no runs is nothing."""
    never_made = tmp_path / "not_a_project" / "CONTAINS_PHI" / "pipeline_run_receipts"

    write_project_run_index(never_made)

    assert not never_made.exists()
    assert not never_made.parent.exists()


def test_a_finished_run_is_not_still_listed_as_having_no_values(tmp_path, monkeypatch):
    """The index is rendered by ensure_layout, which runs DURING the stages — before
    the answer tables exist, since those are written at close-out. So every finished run
    was listed as "no values yet, this run has not been through extract" until some
    later command happened to touch the project.

    Seen on a real finished run: four tables on disk, RUNS.txt saying it had never
    extracted. Close-out re-renders the index after writing them."""
    import json

    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (  # noqa: E501
        build_manifest,
        write_manifest,
    )
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.run_summary import (
        write_summary,
    )

    run_id = "20260820_105033_b74d"
    receipts = tmp_path / "CONTAINS_PHI" / "pipeline_run_receipts"
    run = receipts / run_id
    extract = run / "patients" / "P1" / "extract" / "date_of_birth"
    extract.mkdir(parents=True)
    (extract / "result.json").write_text(json.dumps(
        {"payload": {"ok": True, "variable": "date_of_birth",
                     "data": {"date_of_birth": "1973-01-02"}}}), encoding="utf-8")
    write_manifest(run, build_manifest(
        run_id=run_id, code_lock_hash="sha256:" + "0" * 64,
        entry_point_name="test", config_alias="test", target_patients=["P1"],
    ))
    # The index as it stands mid-run, before close-out has written anything.
    write_project_run_index(receipts)
    assert "has not been through extract" in (receipts / "RUNS.txt").read_text()

    write_summary(run)

    index = (receipts / "RUNS.txt").read_text()
    assert "has not been through extract" not in index
    assert f"../answers/{run_id}/" in index and "1 recipe table(s)" in index
    # Rendered after the summary is on disk, not between the tables and the summary.
    # The index lists what a run holds worth opening, and the line an operator looks
    # for after a run they were not watching is the one naming summary.json. Seen
    # missing on a real finished run that had one.
    assert "summary.json" in index


def test_touching_a_run_that_already_exists_does_not_rebuild_the_index(tmp_path, monkeypatch):
    """ensure_layout is called per patient per stage, and per QUERY in retrieve, and it
    rebuilt the whole project index every time — a full scan of every run in the
    project, measured at 10ms against 200 runs. That is half a minute of pure metadata
    I/O on a 500-patient cohort and far more during a retrieval eval, to re-render a
    file whose content cannot have changed: the index gains a line when a run appears
    (here) and gains its tables at close-out (write_summary re-renders it there)."""
    from jr_pipeline.runtime_infrastructure import data_directory_layout_and_safe_writes as layout

    renders: list[Path] = []
    real = layout.write_project_run_index
    monkeypatch.setattr(
        layout, "write_project_run_index",
        lambda receipts_root: (renders.append(Path(receipts_root)), real(receipts_root))[1],
    )

    layout.ensure_layout("20260820_105033_b74d", tmp_path)
    assert len(renders) == 1, "a run appearing must put a line in the index"

    for _ in range(50):
        layout.ensure_layout("20260820_105033_b74d", tmp_path)

    assert len(renders) == 1, "touching an existing run cannot change the index"


def test_closing_out_says_where_the_answers_are(tmp_path, capsys):
    """A run ends on three ticks and then explains how to iterate — which is the second
    question. The first is "where is the thing I just waited forty minutes for", and
    nothing printed it: not the close-out, not either interface. An operator had to
    know the layout, and the layout moved."""
    import json

    from apps_and_interfaces.command_line_interface import _what_to_do_next
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        run_output_dir,
    )
    from jr_pipeline.runtime_infrastructure.values_table import write_values_tables

    run = tmp_path / "CONTAINS_PHI" / "pipeline_run_receipts" / "20260820_105033_b74d"
    extract = run / "patients" / "P1" / "extract" / "date_of_birth"
    extract.mkdir(parents=True)
    (extract / "result.json").write_text(json.dumps(
        {"payload": {"ok": True, "variable": "date_of_birth",
                     "data": {"date_of_birth": "1973-01-02"}}}), encoding="utf-8")
    write_values_tables(run)

    _what_to_do_next(run)

    printed = capsys.readouterr().out
    assert str(run_output_dir(run)) in printed, "the close-out never names the answers folder"
    assert "1 recipe table" in printed

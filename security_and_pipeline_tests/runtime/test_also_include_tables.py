"""The `also_include_tables` knob merges builder-table rows into the bundle.

Two layers:
  * ``_gather_builder_table_evidence`` resolves each recipe entry to pre-resolved
    evidence items (normalize the builder CSV, filter/project/cap, render rows) and
    a receipt record — including a missing CSV (skipped, not raised) and a pinned
    cross-run ``source_run_id``; and
  * the real ``RetrieveAndPromptHandler.execute`` actually hands those items to step
    6, so a builder ``:TABLE`` row lands in the bundle and is recorded in the receipt.
"""
from __future__ import annotations

import types
from pathlib import Path

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMResponse,
    ResolvedModel,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import load_recipe
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps import (
    retrieve_and_prompt_step as rp,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
)
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    builder_intermediates_dir,
)

REPO = Path(__file__).resolve().parents[2]
RECIPE = next((REPO / "var_extraction_recipes").rglob("genetics_germline_v1_recipe.yaml"))

_CAP_CSV = (
    "specimen,result_type,diagnosis\n"
    "left breast,Surgical Path,invasive ductal carcinoma\n"
    "blood,Lab,CBC normal\n"
    "right breast,Surgical Path,DCIS\n"
)


def _seed_builder_csv(run_id, patient_id, name, content):
    _seed_builder_file(run_id, patient_id, f"{name}.csv", content)


def _seed_builder_file(run_id, patient_id, filename, content):
    d = builder_intermediates_dir(run_id, patient_id)
    d.mkdir(parents=True, exist_ok=True)
    (d / filename).write_text(content)


def _step_context(tmp_path, *, run_id, patient_id, specs, code_lock_hash=None):
    return types.SimpleNamespace(
        step=types.SimpleNamespace(id="gather", config={"also_include_tables": specs}),
        run_id=run_id,
        patient_id=patient_id,
        corpus=types.SimpleNamespace(patient_root=tmp_path / "pt"),
        code_lock_hash=code_lock_hash,
    )


# --- _gather_builder_table_evidence -----------------------------------------

def test_gather_renders_table_rows_as_evidence(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    _seed_builder_csv("R1", "P1", "path_cap", _CAP_CSV)

    items, records = rp._gather_builder_table_evidence(
        _step_context(tmp_path, run_id="R1", patient_id="P1", specs=[{"name": "path_cap"}])
    )

    assert len(items) == 3
    assert all(i["chunk_id"].startswith("P1:path_cap:") and i["chunk_id"].endswith(":TABLE") for i in items)
    assert "invasive ductal carcinoma" in items[0]["text"]
    assert items[0]["doc_type"] == "builder_table"
    [rec] = records
    assert rec["rows_included"] == 3 and rec["source_run_id"] == "R1"
    assert rec["parquet_content_hash"].startswith("sha256:")


def test_gather_applies_filter_and_columns(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    _seed_builder_csv("R1", "P1", "path_cap", _CAP_CSV)

    items, _ = rp._gather_builder_table_evidence(
        _step_context(tmp_path, run_id="R1", patient_id="P1", specs=[{
            "name": "path_cap",
            "filter": "result_type == 'Surgical Path'",
            "columns": ["diagnosis"],
        }])
    )
    assert len(items) == 2  # the Lab row is filtered out
    # only the projected column is rendered
    assert items[0]["text"].startswith("diagnosis: ")
    assert "specimen" not in items[0]["text"]


def test_gather_respects_max_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    _seed_builder_csv("R1", "P1", "path_cap", _CAP_CSV)

    items, _ = rp._gather_builder_table_evidence(
        _step_context(tmp_path, run_id="R1", patient_id="P1", specs=[{"name": "path_cap", "max_rows": 1}])
    )
    assert len(items) == 1


def test_gather_missing_csv_is_skipped_not_raised(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    # no CSV seeded for "absent"
    items, records = rp._gather_builder_table_evidence(
        _step_context(tmp_path, run_id="R1", patient_id="P1", specs=[{"name": "absent"}])
    )
    assert items == []
    assert records == [{"name": "absent", "source_run_id": "R1", "missing": True}]


def test_gather_reuses_a_pinned_prior_run(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    # the builder table was produced under a PRIOR run; the current run pins it.
    _seed_builder_csv("PRIOR", "P1", "path_cap", _CAP_CSV)

    items, records = rp._gather_builder_table_evidence(
        _step_context(tmp_path, run_id="CURRENT", patient_id="P1",
             specs=[{"name": "path_cap", "source_run_id": "PRIOR"}])
    )
    assert len(items) == 3
    assert records[0]["source_run_id"] == "PRIOR"


def test_gather_discovers_non_csv_formats(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    # A builder may emit a .tsv (or .xlsx/.json/...); discovery + reading route
    # through the same code ingest uses, so it works without special-casing CSV.
    _seed_builder_file(
        "R1", "P1", "path_cap.tsv",
        "specimen\tdiagnosis\nleft breast\tinvasive ductal carcinoma\n",
    )
    items, records = rp._gather_builder_table_evidence(
        _step_context(tmp_path, run_id="R1", patient_id="P1", specs=[{"name": "path_cap"}])
    )
    assert len(items) == 1
    assert "invasive ductal carcinoma" in items[0]["text"]
    assert records[0]["rows_included"] == 1


# --- end-to-end through the real handler ------------------------------------

class _StubRetriever:
    info = types.SimpleNamespace(kind="bm25", version="stub", config={})
    score_normalization = "raw"

    def query(self, corpus, *, text, k):
        return []  # no free-text candidates; the bundle is the builder rows only


class _StubProvider:
    def chat(self, req):
        return LLMResponse(
            content='{"genetic_testing_done": false, "genes": []}',
            response_raw={},
            resolved_model=ResolvedModel(
                endpoint_name="stub", api_model_id="stub", api_version=None,
                deployment_name=None, fingerprint="stub",
            ),
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_s=0.0,
        )

    def provider_config(self):
        return {"endpoint_name": "stub"}


class _RecordingProvider:
    """A provider that must NOT be called: it flags any call and refuses."""

    def __init__(self):
        self.called = False

    def chat(self, req):
        self.called = True
        raise AssertionError("the model must not be called when there is no evidence")

    def provider_config(self):
        return {"endpoint_name": "stub"}


def test_empty_evidence_skips_model_and_records_error(tmp_path, monkeypatch):
    # Clinical-safety guard: when retrieval returns nothing and there are no builder
    # rows, the step must NOT call the model (an empty prompt invites a hallucinated
    # answer) and must record an empty_evidence error. This replaces the end-to-end
    # guard lost when the budget-reserve test was deleted.
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(rp, "build_retriever", lambda *a, **k: _StubRetriever())  # returns []

    recipe = load_recipe(RECIPE)
    recipe.steps[0].config.pop("also_include_tables", None)  # no builder rows either
    provider = _RecordingProvider()
    step_context = StepContext(
        recipe=recipe,
        step=recipe.steps[0],
        patient_id="Test_Patient",
        corpus=types.SimpleNamespace(patient_root=tmp_path / "pt", row_for=lambda _cid: None),
        provider=provider,
        provider_config_hash=None,
        llm_cache=None,
        run_id="20990101_000000",
        code_lock_hash=None,
    )

    result = rp.RetrieveAndPromptHandler().execute(step_context)

    packet = result.receipt_payload["evidence_packet"]
    assert packet["empty_evidence"] is True
    assert packet["block_count"] == 0
    assert provider.called is False                       # model was NOT called
    assert "empty_evidence" in (result.error or "")      # failure surfaced


def test_builder_rows_reach_the_bundle_through_execute(tmp_path, monkeypatch):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(rp, "build_retriever", lambda *a, **k: _StubRetriever())
    _seed_builder_csv("20990101_000000", "Test_Patient", "path_cap", _CAP_CSV)

    recipe = load_recipe(RECIPE)
    recipe.steps[0].config["also_include_tables"] = [{"name": "path_cap", "max_rows": 2}]
    step_context = StepContext(
        recipe=recipe,
        step=recipe.steps[0],
        patient_id="Test_Patient",
        corpus=types.SimpleNamespace(patient_root=tmp_path / "pt", row_for=lambda _cid: None),
        provider=_StubProvider(),
        provider_config_hash=None,
        llm_cache=None,
        run_id="20990101_000000",
        code_lock_hash=None,
    )

    result = rp.RetrieveAndPromptHandler().execute(step_context)

    packet = result.receipt_payload["evidence_packet"]
    assert packet["empty_evidence"] is False                     # builder rows are evidence
    assert any(cid.endswith(":TABLE") for cid in packet["included_chunk_ids"])
    assert packet["builder_tables"][0]["rows_included"] == 2     # max_rows honored end-to-end

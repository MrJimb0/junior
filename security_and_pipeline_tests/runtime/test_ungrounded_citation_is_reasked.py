"""A step that gets a citation to a passage it never showed asks again.

In one real run, the clinical_tnm step showed 16 chunks and the model
answered citing ``clinical_note:368:0``, which was not one of them. Nothing asked again.
The pointer was nulled at the end of the recipe and the value became an unsupported
claim -- correct, but a whole variable lost to one bad line when the step still had the
right ids in hand.

Re-asking only works because the prompt CHANGES. Decoding is greedy (the engine pins
temperature at 0.0) and the response cache is keyed on the prompt, so re-sending the
same request returns the same answer, out of cache, instantly. These tests pin that the
correction reaches the model and that the retry is bounded and recorded.
"""
from __future__ import annotations

import json
import re
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
RUN = "20990101_000000"
CSV = "specimen,result_type,diagnosis\nleft breast,Surgical Path,invasive ductal carcinoma\n"

INVENTED = "Test_Patient:clinical_note:368:0"


class _StubRetriever:
    """No vector hits; the builder table supplies the evidence (and the chunk ids)."""

    info = types.SimpleNamespace(kind="bm25", version="stub", config={})
    score_normalization = "raw"

    def query(self, corpus, *, text, k):
        return []


class _ModelThatCitesAPassageItWasNeverShown:
    """Fabricates on the first ask; on a correction, cites an id the correction offered.

    Reading the ids back out of the correction prompt is the point: it proves the
    correction actually carries them, rather than just scolding the model."""

    def __init__(self, *, comply_on_retry=True):
        self.prompts_seen: list[str] = []
        self.comply_on_retry = comply_on_retry

    def chat(self, req):
        last_user = [m for m in req.messages if m["role"] == "user"][-1]["content"]
        self.prompts_seen.append(last_user)
        cited = INVENTED
        if len(self.prompts_seen) > 1 and self.comply_on_retry:
            offered = re.findall(r"^\s{2}(\S+)$", last_user, re.MULTILINE)
            assert offered, f"the correction offered no ids to choose from:\n{last_user}"
            cited = offered[0]
        return LLMResponse(
            content=json.dumps({
                "genetic_testing_done": True, "genes": [],
                "evidence_chunk_id": cited,
            }),
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


def _run(tmp_path, monkeypatch, provider, *, retries):
    monkeypatch.setenv("JR_DATA_ROOT", str(tmp_path / "data"))
    monkeypatch.setattr(rp, "build_retriever", lambda *a, **k: _StubRetriever())
    d = builder_intermediates_dir(RUN, "Test_Patient")
    d.mkdir(parents=True, exist_ok=True)
    (d / "path_cap.csv").write_text(CSV)

    recipe = load_recipe(RECIPE)
    recipe.steps[0].config["also_include_tables"] = [{"name": "path_cap", "max_rows": 2}]
    return rp.RetrieveAndPromptHandler().execute(StepContext(
        recipe=recipe,
        step=recipe.steps[0],
        patient_id="Test_Patient",
        corpus=types.SimpleNamespace(patient_root=tmp_path / "pt", row_for=lambda _cid: None),
        provider=provider,
        provider_config_hash=None,
        llm_cache=None,
        run_id=RUN,
        code_lock_hash=None,
        max_provenance_retries=retries,
    ))


def test_an_invented_citation_is_corrected_and_the_step_asks_again(tmp_path, monkeypatch):
    provider = _ModelThatCitesAPassageItWasNeverShown()

    result = _run(tmp_path, monkeypatch, provider, retries=1)

    assert len(provider.prompts_seen) == 2, "the step accepted an invented citation"
    correction = provider.prompts_seen[1]
    assert INVENTED in correction, "the correction must name what was invented"
    assert "return null" in correction, (
        "without an explicit null option a model will pick a plausible id to satisfy "
        "the constraint, which hides the failure instead of fixing it"
    )
    shown = result.receipt_payload["evidence_packet"]["included_chunk_ids"]
    assert result.data["evidence_chunk_id"] in shown, "the retry did not land on a real passage"


def test_the_correction_is_kept_on_the_receipt(tmp_path, monkeypatch):
    """An answer corrected into citing its source is weaker than one that cited it
    first time. Only this record can tell a reviewer which they are looking at."""
    provider = _ModelThatCitesAPassageItWasNeverShown()

    result = _run(tmp_path, monkeypatch, provider, retries=1)

    retries = result.receipt_payload["provenance_retries"]
    assert len(retries) == 1
    assert retries[0]["attempt"] == 1
    assert retries[0]["ungrounded"][0]["chunk_id"] == INVENTED


def test_retries_are_bounded_and_the_bad_answer_still_stands(tmp_path, monkeypatch):
    """A model that keeps inventing must not loop. After the budget, the answer stands
    as-is; extract nulls the pointer and the value fails as an unsupported claim."""
    provider = _ModelThatCitesAPassageItWasNeverShown(comply_on_retry=False)

    result = _run(tmp_path, monkeypatch, provider, retries=2)

    assert len(provider.prompts_seen) == 3, "asked a different number of times than allowed"
    assert result.data["evidence_chunk_id"] == INVENTED
    assert len(result.receipt_payload["provenance_retries"]) == 2


def test_zero_retries_asks_once(tmp_path, monkeypatch):
    provider = _ModelThatCitesAPassageItWasNeverShown()

    result = _run(tmp_path, monkeypatch, provider, retries=0)

    assert len(provider.prompts_seen) == 1
    assert result.receipt_payload["provenance_retries"] == []


def test_a_grounded_answer_is_not_re_asked(tmp_path, monkeypatch):
    """The gate must cost nothing when the model behaves."""
    class _Compliant(_ModelThatCitesAPassageItWasNeverShown):
        def chat(self, req):
            self.prompts_seen.append("")
            return LLMResponse(
                content=json.dumps({"genetic_testing_done": False, "genes": []}),
                response_raw={},
                resolved_model=ResolvedModel(
                    endpoint_name="stub", api_model_id="stub", api_version=None,
                    deployment_name=None, fingerprint="stub",
                ),
                usage={}, latency_s=0.0,
            )

    provider = _Compliant()
    result = _run(tmp_path, monkeypatch, provider, retries=1)

    assert len(provider.prompts_seen) == 1
    assert result.receipt_payload["provenance_retries"] == []

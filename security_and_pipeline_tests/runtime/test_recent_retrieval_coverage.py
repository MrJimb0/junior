from types import SimpleNamespace

import polars as pl
import pytest

from jr_pipeline.pipeline_steps.step_7_extract_variables.recent_retrieval_coverage import (
    append_recent_candidates,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
    load_recipe,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import Candidate


def test_recent_coverage_unions_newest_source_chunks_without_duplicates() -> None:
    corpus = SimpleNamespace(
        chunk_index=pl.DataFrame(
            [
                {"chunk_id": "old", "source_file": "clinical_note.csv", "document_date": "2020-01-01"},
                {"chunk_id": "new", "source_file": "clinical_note.csv", "document_date": "2024-03-01"},
                {"chunk_id": "newer", "source_file": "clinical_note.csv", "document_date": "2024-04-01"},
                {"chunk_id": "path", "source_file": "pathology_report.csv", "document_date": "2025-01-01"},
                {"chunk_id": "undated", "source_file": "clinical_note.csv", "document_date": None},
            ]
        )
    )
    original = [Candidate(chunk_id="newer", rank=1, score=2.0, retriever="bm25")]

    combined, added = append_recent_candidates(
        original,
        corpus,
        {"source_file": "clinical_note.csv", "include_recent": 2},
    )

    assert [candidate.chunk_id for candidate in combined] == ["newer", "new", "old"]
    assert added == ["new", "old"]
    assert combined[1].retriever == "recent_coverage"


def test_recent_coverage_uses_undated_chunks_only_as_a_coverage_fallback() -> None:
    corpus = SimpleNamespace(
        chunk_index=pl.DataFrame(
            [
                {"chunk_id": "dated", "source_file": "clinical_note.csv", "document_date": "2024-01-01"},
                {"chunk_id": "undated", "source_file": "clinical_note.csv", "document_date": None},
            ]
        )
    )

    combined, added = append_recent_candidates(
        [],
        corpus,
        {"source_file": "clinical_note.csv", "include_recent": 2},
    )

    assert added == ["dated", "undated"]
    assert combined[-1].retriever == "recent_coverage_undated_fallback"


@pytest.mark.parametrize("value", [0, -1, True, "10"])
def test_recipe_rejects_invalid_include_recent(tmp_path, value) -> None:
    folder = tmp_path / "recent_test" / "v1"
    folder.mkdir(parents=True)
    (folder / "recent_test_v1_output_schema.json").write_text(
        '{"type":"object"}', encoding="utf-8"
    )
    (folder / "prompt.md").write_text(
        "# SYSTEM\nx\n# USER\n{{ evidence_text }}", encoding="utf-8"
    )
    (folder / "recent_test_v1_recipe.yaml").write_text(
        "\n".join(
            [
                "name: recent_test",
                "version: v1",
                "output_schema: recent_test_v1_output_schema.json",
                "llm: {model: local_qwen}",
                "steps:",
                "  - id: extract",
                "    kind: retrieve_and_prompt",
                "    retrieval:",
                "      kind: bm25",
                "      query: cancer",
                f"      include_recent: {value!r}",
                "    prompt: prompt.md",
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="include_recent"):
        load_recipe(folder / "recent_test_v1_recipe.yaml")

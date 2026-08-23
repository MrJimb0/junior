"""The shared building blocks every recipe step uses: StepContext, StepResult, StepHandler.

StepContext is what a step is given to work with, StepResult is what it returns,
and StepHandler is the common interface that every step type follows.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_interface import (
    LLMProvider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import (
    RecipeSpec,
    StepSpec,
)
from jr_pipeline.runtime_infrastructure.patient_chunk_store import PatientChunkStore


@dataclass
class StepContext:
    """Everything a single recipe step is handed when it runs.

    This bundles the recipe and the specific step being run, which patient it is
    for, the patient's searchable record (``corpus``), the model connector and
    response cache, and a few cross-step extras (outputs from earlier steps, the
    encoder/chunker settings, where to log problem cases).
    """

    recipe: RecipeSpec
    step: StepSpec
    patient_id: str
    corpus: PatientChunkStore
    provider: LLMProvider | None
    provider_config_hash: str | None
    # Holds the LLM response cache. Typed as Any only to avoid an import loop
    # with the cache module (importing its real type here would be circular).
    llm_cache: Any | None
    run_id: str
    code_lock_hash: str | None

    upstream_vars: Mapping[str, Any] = field(default_factory=dict)
    step_outputs: dict[str, Any] = field(default_factory=dict)
    encoder_cfg: dict | None = None
    chunker_cfg: dict | None = None
    quarantine_dir: Path | None = None
    # How many chunks may reach one prompt on THIS machine, capping whatever a recipe
    # asks for. A recipe's top_n is a research judgement about how much evidence the
    # question needs; this is a fact about the hardware, so it travels with the run
    # rather than being edited into shared recipes. None means the recipe decides.
    max_chunks_per_prompt: int | None = None
    # How many times a prompt step may re-ask after the model cites a passage it was
    # never shown. 0 disables it: the answer stands, and the ungrounded pointer is
    # nulled downstream so the value fails as an unsupported claim.
    max_provenance_retries: int | None = None


@dataclass
class StepResult:
    """What one step produces: its data, an audit receipt, and an optional error.

    ``data`` is the step's structured result; ``receipt_payload`` is the saved
    record of how it was produced; ``error`` records a failure.
    """

    data: dict[str, Any] | None
    receipt_payload: dict[str, Any]
    error: str | None = None
    # Extra data this step passes forward to step 8 (organize_output) — for
    # example the evidence packet a retrieve_and_prompt step assembled. Left as
    # None for steps that produce nothing step 8 needs.
    step8_payload: dict[str, Any] | None = None


class StepHandler(Protocol):
    """The shape every step type must implement: one ``execute(ctx) -> StepResult`` method."""

    kind: str

    def execute(self, ctx: StepContext) -> StepResult: ...

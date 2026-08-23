"""recipe runner — extracts the requested clinical variables for one patient.

This is the top of the extract step: given one patient and a list of variables
to pull from their chart, it runs each variable's recipe and writes the answers
to disk. Files written per patient under .../patients/<pid>/extract/<variable>/:
  result.json                  the final answer for this variable, with metadata
  invariants.json              results of the clinical consistency checks
  transcript.json              the full record of this recipe run (contains PHI)
  steps/<step_id>/receipt.json an audit record for each step (contains PHI)

Recipes run in dependency order (recipe_execution_order.py), so an upstream
variable is always extracted before the recipes that depend on it read it. Each
recipe's steps run in order (any step may end the recipe early via a `stop_if`
condition); the answer is checked against the recipe's output schema, then the
clinical consistency checks run. A failed consistency check is written to
invariants.json but does not stop the run: a clinically implausible value
usually means one bad model answer for one patient — something the operator
reviews after the batch, not a reason to halt it (ADR 0025).
"""
from __future__ import annotations

import json
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from jinja2 import Environment, StrictUndefined

from jr_pipeline.pipeline_steps.step_7_extract_variables.llm_response_cache import (
    LLMCache,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_allowlist import (
    Allowlist,
    load_allowlist,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_provider_lookup import (
    build_provider,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_execution_order import (
    plan as plan_recipes,
)
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_specification import RecipeSpec
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps import build_step
from jr_pipeline.pipeline_steps.step_7_extract_variables.recipe_steps.recipe_step_base_types import (
    StepContext,
    StepResult,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.cross_variable_invariants import (
    organize_clinical_invariants,
)
from jr_pipeline.pipeline_steps.step_8_organize_output.organize_output import (
    organize_per_recipe_outputs,
    record_extract_completion,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.pipeline_progress_log import (
    Entity,
    record_transition,
)
from jr_pipeline.runtime_enforcing_safety_and_reproducibility.schemas.output_validation_schemas import (
    envelope_for,
)
from jr_pipeline.runtime_infrastructure.artifact_store import write_artifact
from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
    atomic_write_json,
    ensure_layout,
    extract_output_dir,
    phi_intermediate_run_dir,
    phi_patient_run_dir,
)
from jr_pipeline.runtime_infrastructure.json_event_logging import get_logger
from jr_pipeline.runtime_infrastructure.patient_chunk_store import (
    PatientChunkStore,
    parse_structured_evidence_pointer,
)

_log = get_logger("extract")


def _reject_ungrounded_evidence_pointers(
    data: Any, shown_chunk_ids: set[str], step_id: str | None
) -> list[dict[str, Any]]:
    """Null every evidence pointer in ``data`` that names a chunk the model was never
    shown and is not a well-formed structured evidence pointer, and return the rejections.

    Step-8 provenance validation only checks a pointer is non-empty, so a model that
    fabricates an ``evidence_chunk_id`` would ship an ungrounded value as "grounded". This
    is the central, recipe-agnostic guard: an evidence-pointer field is any key containing
    ``evidence_chunk_id`` (the exact key, a prefixed key like ``dob_evidence_chunk_id``, or
    a per-item list ``evidence_chunk_ids`` — the same key test step 8 uses). A cited id is
    grounded when it is in ``shown_chunk_ids`` (the union of every step's shown-chunk list)
    or is a well-formed structured evidence pointer (``patient:...:structured:...``, which is
    pipeline-authored and carries its own file+row hash). A fabricated id is nulled — never shipped —
    and the pointer-less value then flows through step 8's ordinary unprovenanced-value
    handling (it appears in ``unprovenanced_value_paths`` and drives ``ok``). List fields
    keep their grounded items and drop only the fabricated ones. Each rejection records the
    path, the rejected id, and the step, for the receipt.
    """
    rejections: list[dict[str, Any]] = []

    def _is_grounded(chunk_id: str) -> bool:
        return chunk_id in shown_chunk_ids or parse_structured_evidence_pointer(chunk_id) is not None

    def _walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if "evidence_chunk_id" in key.lower():
                    if isinstance(value, str) and value and not _is_grounded(value):
                        rejections.append({"path": f"{path}.{key}", "chunk_id": value, "step": step_id})
                        node[key] = None
                    elif isinstance(value, list):
                        kept = []
                        for item in value:
                            if isinstance(item, str) and item and not _is_grounded(item):
                                rejections.append(
                                    {"path": f"{path}.{key}", "chunk_id": item, "step": step_id}
                                )
                            else:
                                kept.append(item)
                        node[key] = kept
                elif isinstance(value, (dict, list)):
                    _walk(value, f"{path}.{key}")
        elif isinstance(node, list):
            for index, item in enumerate(node):
                _walk(item, f"{path}[{index}]")

    _walk(data, "data")
    return rejections
_expr_env = Environment(undefined=StrictUndefined)


def _eval_expr(expr: str, context: dict[str, Any]) -> bool:
    """evaluate a step's `stop_if` early-exit condition; returns False on any error
    (a broken condition should never silently stop the recipe)."""
    if not expr or not expr.strip():
        return False
    try:
        compiled = _expr_env.compile_expression(expr)
        result = compiled(**context)
        return bool(result)
    except Exception as e:
        # warning, not info: a condition that cannot evaluate is a recipe bug the
        # operator should see — otherwise the recipe just runs on with no visible sign.
        _log.warning("stop_if_eval_error", extra_={"expr": expr, "error": str(e)})
        return False


def _run_one_stage(
    *,
    recipe: RecipeSpec,
    ctx: StepContext,
    extract_dir: Path,
    scoped_hashes: dict[str, Any] | None = None,
) -> tuple[StepResult, dict[str, Any]]:
    step = ctx.step
    handler = build_step(step.kind)
    t0 = time.perf_counter()
    try:
        result = handler.execute(ctx)
        status = "completed"
        err = None
    except Exception as e:
        import traceback
        err = f"{type(e).__name__}: {e}"
        # some errors have an empty text form — keep the full traceback so the
        # step's audit record still says what actually went wrong
        result = StepResult(
            data=None,
            receipt_payload={"error": err, "traceback": traceback.format_exc()},
            error=err,
        )
        status = "failed"
    elapsed = time.perf_counter() - t0

    receipt_dir = extract_dir / "steps" / step.id
    receipt_dir.mkdir(parents=True, exist_ok=True)
    payload = dict(result.receipt_payload)
    payload.setdefault("kind", step.kind)
    payload.setdefault("timings", {}).update({"total_s": round(elapsed, 6)})

    env = envelope_for(
        artifact_type="step_receipt",
        sensitivity="high",
        stream="data",
        run_id=ctx.run_id,
        step="extract",
        patient_id=ctx.patient_id,
        variable=recipe.name,
        step_id=step.id,
        payload=payload,
        code_lock_hash=ctx.code_lock_hash,
        provider_config_hash=ctx.provider_config_hash,
        **(scoped_hashes or {}),
    )
    write_artifact(env, path=receipt_dir / "receipt.json")

    stage_summary = {
        "step_id": step.id,
        "kind": step.kind,
        "status": status,
        "receipt_path": str(Path("steps") / step.id / "receipt.json"),
        "elapsed_s": round(elapsed, 6),
    }
    return result, stage_summary


def _named_relative_to(path: Path, root: Path) -> str:
    """``path`` as the run should record it: relative to ``root`` where it sits inside.

    A run is copied between a cluster and a laptop, and a path recorded absolutely stops
    being true the moment it moves. Falls back to the absolute form when the path is not
    under the root at all, because a wrong relative path would be worse than a stale
    absolute one — it would resolve to something."""
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)


def resolve_llm_cache_path(cfg: dict) -> Path:
    """Where the language-model response cache lives — this cache holds PHI.

    The cache stores the rendered prompts and the model's responses, which contain
    patient clinical text (PHI = protected health information). So by default it lives
    INSIDE the run's PHI directory tree (``phi_intermediate_run_dir(run_id)/
    llm_cache.db``): there it inherits the same "contains PHI" access controls, and it
    never lands in ``~/.cache`` outside the PHI boundary, where a home-directory backup
    / iCloud / Time Machine could silently copy patient data off this machine. An
    explicit ``llm_cache_path`` override still wins for an operator-approved location
    (for example a per-compute-node scratch directory on the cluster). The path is tied
    to the run id, so re-running the same pinned run still reuses its cached answers."""
    override = cfg.get("llm_cache_path")
    if override:
        return Path(override)
    return phi_intermediate_run_dir(cfg["run_id"]) / "llm_cache.db"


def run_extract_one(
    *,
    cfg: dict,
    patient_id: str,
    code_lock_hash: str | None = None,
    force: bool = False,
    on_variable: Callable[[str, str, float | None], None] | None = None,
) -> dict:
    """Run the recipe(s) named in ``cfg`` for one patient. This wrapper guarantees that,
    even if something crashes hard, the patient is recorded in a final state (completed
    or failed) and the model-response cache file is closed cleanly.

    A failure inside an individual step is already caught lower down
    (``_run_one_stage``); this wrapper additionally catches a crash *outside* the step
    loop — for example the chart store raising because embeddings were never built
    (running extract before the embed step) — so the patient never stays stuck in the
    ``running`` state with the cache file left open.

    A variable that already succeeded under these exact settings is not extracted
    again; ``force`` re-extracts everything regardless. ``on_variable(name, state,
    seconds)`` is called as each variable is decided and finished, for progress
    display -- states are "already complete", "running", "retrying", "done", "failed"."""
    run_id = cfg["run_id"]
    cache_path = resolve_llm_cache_path(cfg)
    llm_cache = LLMCache(path=cache_path)
    try:
        return _run_extract_one_body(
            cfg=cfg,
            patient_id=patient_id,
            force=force,
            on_variable=on_variable,
            llm_cache=llm_cache,
            code_lock_hash=code_lock_hash,
        )
    except Exception as exc:
        # A crash before the run could mark itself completed left this patient stuck in
        # the 'running' state; record the honest 'failed' state instead. Wrap the
        # bookkeeping in its own try so a failure here can never hide the original `exc`.
        try:
            record_transition(
                phi_intermediate_run_dir(run_id),
                entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="extract"),
                from_state="running",
                to_state="failed",
                reason=f"{type(exc).__name__}: {exc}",
                step_context="extract",
                code_lock_hash=code_lock_hash,
            )
        except Exception:
            pass
        raise
    finally:
        llm_cache.close()


def _why_it_failed_last_time(patient_out: Path, variable: str) -> list[str] | None:
    """The errors this variable failed with before, or None if it did not fail.

    Read before it is retried, so the retry can be compared against it. Extraction is
    deterministic — greedy decoding, and a response cache keyed on the prompt — so a
    variable that fails on unchanged evidence will fail the same way every time. Saying
    "retry" without noticing that is telling somebody to do the same thing again."""
    result_file = patient_out / "extract" / variable / "result.json"
    if not result_file.is_file():
        return None
    try:
        payload = json.loads(result_file.read_text(encoding="utf-8")).get("payload") or {}
    except (OSError, json.JSONDecodeError):
        return None
    if payload.get("ok"):
        return None
    return [str(e) for e in (payload.get("errors") or [])]


def _how_much_a_step_found(data: Any) -> int:
    """How much a step's answer actually carries, counted the way a reader would.

    Records when the step answers with a table of them, since "two treatment lines" is
    what a reader means; otherwise the number of value-bearing leaves. Judged by the
    same rule step 8 uses to decide what needs provenance, so a rationale, a certainty
    score, an evidence pointer and a null placeholder are not findings — which is what
    makes an answer of {"lines": []} count zero while a real one does not."""
    from jr_pipeline.pipeline_steps.step_8_organize_output.provenance_validation import (
        _is_value_bearing_leaf,
    )

    if not isinstance(data, dict):
        return 0
    record_lists = [v for v in data.values()
                    if isinstance(v, list) and v and isinstance(v[0], dict)]
    if len(record_lists) == 1:
        return len(record_lists[0])

    def leaves(node: Any) -> int:
        if isinstance(node, dict):
            return sum(
                leaves(v) if isinstance(v, (dict, list))
                else int(_is_value_bearing_leaf(k, v))
                for k, v in node.items()
            )
        if isinstance(node, list):
            return sum(leaves(item) for item in node)
        return 0

    return leaves(data)


def _settings_a_result_was_made_under(
    *, code_lock_hash: str | None, per_recipe: dict
) -> dict[str, str | None]:
    """The identity of the settings a variable is extracted under.

    Every field here is known WITHOUT building a provider, which is the constraint that
    shapes it: deciding not to run a variable must not cost a model load, or resuming a
    finished patient would be as expensive as extracting it. That rules out
    provider_config_hash, the one value that would also pin which model answered.

    It is not the hole it looks like. The endpoint set is fixed by the run's sealed
    allowlist, allowlist_path is run-invariant, and result.json still records
    provider_config_hash for every value, so a model swap remains visible to an audit
    afterwards even though it is not what decides the skip."""
    return {
        "code_lock_hash": code_lock_hash,
        "recipe_hash": per_recipe.get("recipe_hash"),
        "schema_hash": per_recipe.get("schema_hash"),
        "python_helpers_hash": per_recipe.get("python_helpers_hash"),
    }


def _the_corpus_changed_after(result_file: Path, patient_out: Path) -> bool:
    """Whether anything extraction reads for this patient was rebuilt since that answer.

    None of the hashes the skip compares can see this. They all identify the CODE that
    ran — recipe, schema, helpers, bundle — and all of them come from the run's sealed
    index, so re-ingesting and re-embedding a corrected chart under the same run id
    moves none of them. Without this check, `junior ingest --force && junior embed
    --force && junior extract` skips every ok variable and never reads the new document,
    and the run reports completed.

    The watched set is the corpus ENTRY SET the layout guide declares for ingest,
    embed and index — not a hand-picked list. The hand-picked list was
    (chunk_index, embeddings, hnsw), which missed structured/: `direct_parquet`
    retrieval reads those tables straight off disk, and for a patient with a
    structured date it decides the whole variable — so a re-ingested demographics
    table left "already complete" answers that had never read the correction.
    Deriving the set from RUN_ARTIFACT_GUIDE also means a corpus artifact added to
    one of those stages later is covered without anyone remembering to come back.

    Two mechanics matter. Directory entries are judged by the newest FILE beneath
    them, never the directory's own mtime: replacing a file inside structured/ does
    not always move the parent, and a fresh directory made while hard-linking an
    inherited corpus must not read as change (the linked files keep the source
    inode's mtime; copy2, the cross-device fallback, preserves it too — the same
    corpus is the same corpus). And source_snapshot.json is excluded: ingest
    rewrites it even when nothing changed, so including it would turn every
    no-op re-ingest into a full cohort redo.
    """
    from jr_pipeline.runtime_infrastructure.corpus_inheritance import corpus_entry_names

    try:
        answered_at = result_file.stat().st_mtime
    except OSError:
        return True
    for entry_name in sorted(corpus_entry_names() - {"source_snapshot.json"}):
        entry = patient_out / entry_name
        try:
            if entry.is_file():
                if entry.stat().st_mtime > answered_at:
                    return True
            elif entry.is_dir():
                for inner in entry.rglob("*"):
                    if inner.is_file() and inner.stat().st_mtime > answered_at:
                        return True
        except OSError:
            return True
    return False


def _work_already_done(
    patient_out: Path, variable: str, made_under: dict[str, str | None]
) -> dict | None:
    """This variable's prior answer, if it succeeded under exactly these settings.

    Returns None -- meaning "do the work" -- when the variable was never extracted, when
    it failed, when the corpus was rebuilt after it was answered, or when anything
    identifying how it was produced is unknown or differs.

    The hash comparison is NOT a second line of defence against a recipe edit, which an
    earlier version of this docstring claimed. Within a run both sides of it read the
    same sealed index, so it cannot fire; a recipe edit is caught by
    _recipes_that_changed_since_sealing at the CLI, and that is the only thing catching
    it. What this comparison does catch is a result that cannot say what produced it,
    which is why an unknown expected value means redo.

    Conservative on absence: a result whose receipt does not record what made it cannot
    be shown to match, so it is redone. Runs sealed before those hashes were recorded
    re-extract once, which is the right way round -- the alternative is trusting an
    answer nothing can vouch for."""
    result_file = patient_out / "extract" / variable / "result.json"
    if not result_file.is_file():
        return None
    try:
        envelope = json.loads(result_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    payload = envelope.get("payload") or {}
    if not payload.get("ok"):
        return None                      # failed last time: that is work to retry
    if _the_corpus_changed_after(result_file, patient_out):
        return None
    produced_by = envelope.get("produced_by") or {}
    for field, expected in made_under.items():
        # An unknown expected value is a reason to REDO, not a reason to match. It was
        # the other way round, which made the skip fire hardest exactly where least was
        # known: an unsealed run (compare-models builds one), or a recipe whose version
        # was bumped so the sealed per-recipe entry is missing under both its versioned
        # and bare keys. In both, every field is None, nothing is compared, and any
        # ok result is kept.
        if expected is None or produced_by.get(field) != expected:
            return None
    return payload


def _run_extract_one_body(
    *,
    cfg: dict,
    patient_id: str,
    llm_cache: LLMCache,
    code_lock_hash: str | None = None,
    force: bool = False,
    on_variable: Callable[[str, str, float | None], None] | None = None,
) -> dict:
    """The actual work of running the recipe(s) for one patient. The caller
    ``run_extract_one`` owns opening/closing the model-response cache and recording the
    failed state if this body raises."""
    run_id = cfg["run_id"]
    ensure_layout(run_id)
    run_root = phi_intermediate_run_dir(run_id)

    patient_out = phi_patient_run_dir(run_id, patient_id)
    if not patient_out.is_dir():
        raise FileNotFoundError(
            f"No patient dir for {patient_id!r}; run ingest + embed first."
        )

    recipes_root = Path(cfg.get("recipes_root", "recipes")).resolve()
    names = cfg.get("recipes") or []
    if not names:
        raise ValueError(
            "extract config must set 'recipes' to a non-empty list"
        )
    # a missing recipes_root must fail loudly rather than quietly proceed — otherwise
    # the recipe folder would be left out of the run's provenance fingerprint (the
    # content hash that records exactly which code and recipes produced this run).
    if not recipes_root.is_dir():
        raise FileNotFoundError(
            f"recipes_root {recipes_root} does not exist — check the path / cwd."
        )

    plan = plan_recipes(recipes_root=recipes_root, requested=names)
    # A requested variable that is misspelled or has no recipe shows up in
    # plan.missing; fail loudly — the run cannot honestly claim to have extracted a
    # variable whose recipe was never found.
    requested_missing = [n for n in names if n in plan.missing]
    if requested_missing:
        raise FileNotFoundError(
            f"requested variable(s) {requested_missing} have no recipe under "
            f"{recipes_root} — check the spelling and recipes_root."
        )
    if plan.missing:  # any remaining entries are dependencies-of-dependencies we
                      # could not resolve (not directly requested)
        _log.info("dag_missing_dependencies", extra_={"missing": plan.missing})
    ordered_names = [r.name for r in plan.recipes]

    allowlist_path = cfg.get("allowlist_path")
    allowlist: Allowlist = load_allowlist(Path(allowlist_path)) if allowlist_path else Allowlist([])

    log = _log.bind(run_id=run_id, patient_id=patient_id)
    record_transition(
        run_root,
        entity=Entity(kind="step", run_id=run_id, patient_id=patient_id, step="extract"),
        from_state=None,
        to_state="running",
        reason=f"extract recipes={names}",
        step_context="extract",
        code_lock_hash=code_lock_hash,
    )

    results: dict[str, dict[str, Any]] = {}

    # One chunk store for the whole patient, shared by every recipe. Construction
    # eagerly reads chunk_index.parquet, and the embeddings matrix + hnsw index are
    # lazily loaded and cached on the instance — so building it per recipe (as this
    # once did) reloads all three for every variable of every patient.
    corpus = PatientChunkStore(patient_root=patient_out)

    # Content hashes computed when the run was "sealed" (its code + recipes frozen).
    # We stamp these onto every step's audit record so a later diff can tell exactly
    # which recipe/prompt/schema version produced a given result (ADR 0018).
    hashes_index = _load_hashes_index(run_root)
    retrieval_config_hash = hashes_index.get("retrieval_config_hash")

    model_override = cfg.get("model_override")

    for recipe in plan.recipes:
        recipe_name = recipe.name
        endpoint_name = model_override or recipe.llm.model

        # The per-recipe hash table is keyed "<variable>/<version>"; look it up by that
        # versioned key so the audit record's recipe/prompt/schema hashes resolve,
        # falling back to the bare variable name for older runs.
        per_recipe_index = hashes_index.get("per_recipe") or {}
        per_recipe = (per_recipe_index.get(f"{recipe_name}/{recipe.version}")
                      or per_recipe_index.get(recipe_name) or {})
        recipe_hash = per_recipe.get("recipe_hash")
        schema_hash_val = per_recipe.get("schema_hash")
        hooks_hash_val = per_recipe.get("python_helpers_hash")
        prompt_hashes = per_recipe.get("prompt_hashes") or {}

        # Resume: a variable that already succeeded under exactly these settings is not
        # extracted again. Decided BEFORE the provider is built, so a patient with
        # nothing to do costs no model load; and before any step runs, so it costs no
        # model call. The prior answer is still put into `results`, because a variable
        # that depends on this one reads it from there and would otherwise see a
        # variable that "ran" with no data.
        made_under = _settings_a_result_was_made_under(
            code_lock_hash=code_lock_hash, per_recipe=per_recipe,
        )
        finished_earlier = None if force else _work_already_done(
            patient_out, recipe_name, made_under
        )
        if finished_earlier is not None:
            # The whole recorded result, not just its data. A variable that depends on
            # this one reads `data`; the cross-variable clinical invariants below read
            # the same shape organize produces for a variable that ran, and handing them
            # a thinner one for a resumed variable would make the checks depend on
            # whether the value happened to be computed this time.
            results[recipe_name] = finished_earlier
            log.info(
                "extract_variable_already_done",
                extra_={"recipe": recipe_name, "variable": recipe_name},
            )
            if on_variable is not None:
                on_variable(recipe_name, "already complete", None)
            continue

        failed_with_before = _why_it_failed_last_time(patient_out, recipe_name)
        if on_variable is not None:
            on_variable(
                recipe_name, "retrying" if failed_with_before is not None else "running", None
            )
        variable_started = time.perf_counter()

        try:
            endpoint = allowlist.get(endpoint_name)
            provider = build_provider(endpoint)
            provider_config_hash = _hash_provider_config(provider.provider_config())
        except Exception as e:
            # A recipe made only of python / table-lookup steps needs no model, so an
            # unusable one is not its problem. A recipe that DOES ask a model cannot be
            # answered at all — and the failure must not be allowed to look like an
            # answer. Left as a warning, every step failed identically, the recipe's
            # final helper emitted its "nothing found" default, schema validation
            # passed, and the run reported success with a null value: "we never
            # looked" was indistinguishable from "it is not in the chart".
            if _recipe_needs_a_model(recipe):
                raise RuntimeError(
                    f"variable {recipe_name!r} needs a language model and none is "
                    f"available: {e}"
                ) from e
            log.info("provider_unavailable", extra_={"recipe": recipe_name, "error": str(e)})
            provider = None
            provider_config_hash = None

        variable_dir = extract_output_dir(patient_out) / recipe_name
        variable_dir.mkdir(parents=True, exist_ok=True)

        step_outputs: dict[str, Any] = {}
        data_final: dict[str, Any] | None = None
        data_final_step_id: str | None = None
        content_found_by_step: dict[str, int] = {}
        stage_summaries: list[dict[str, Any]] = []
        recipe_step_evidence: list[dict[str, Any]] = []
        t0_recipe = time.perf_counter()
        errors: list[str] = []
        upstream_vars = {
            name: {"data": results.get(name, {}).get("data")}
            for name in recipe.depends_on
        }

        for i, step in enumerate(recipe.steps):
            ctx = StepContext(
                recipe=recipe,
                step=step,
                patient_id=patient_id,
                corpus=corpus,
                provider=provider,
                provider_config_hash=provider_config_hash,
                llm_cache=llm_cache,
                run_id=run_id,
                code_lock_hash=code_lock_hash,
                upstream_vars=upstream_vars,
                step_outputs=step_outputs,
                encoder_cfg=cfg.get("encoder"),
                max_chunks_per_prompt=cfg.get("max_chunks_per_prompt"),
                max_provenance_retries=cfg.get("max_provenance_retries"),
                chunker_cfg=cfg.get("chunker"),
                quarantine_dir=variable_dir,
            )
            scoped = {
                "recipe_hash": recipe_hash,
                "schema_hash": schema_hash_val,
                "python_helpers_hash": hooks_hash_val,
                "retrieval_config_hash": retrieval_config_hash,
                "prompt_hash": _prompt_hash_for_stage(step, prompt_hashes),
            }
            result, summary = _run_one_stage(
                recipe=recipe,
                ctx=ctx,
                extract_dir=variable_dir,
                scoped_hashes=scoped,
            )
            stage_summaries.append(summary)
            step_outputs[step.id] = {
                "data": result.data,
                "raw": (result.receipt_payload or {}).get("response_raw"),
                # The ids of the chart passages this step actually put in front of the
                # model (empty for python / non-retrieval steps). A later helper uses
                # this to anchor an answer to "the passage the model actually read" even
                # when the model forgets to cite its source — and to reject a made-up
                # citation that points at a passage the model was never shown. Kept in
                # memory and only on the PHI side; never written to the de-identified,
                # shareable (NO_PHI) output tree.
                "evidence_chunk_ids": (
                    ((result.step8_payload or {}).get("evidence_packet") or {}).get("included_chunk_ids") or []
                ),
            }
            if result.step8_payload is not None:
                recipe_step_evidence.append(result.step8_payload)
            if result.error:
                # Named, because a variable's warnings are read after the fact by
                # somebody holding one row of a table. "parsed response was not a JSON
                # object" was true of one of nine steps and said which of them nowhere.
                errors.append(f"{step.id}: {result.error}")

            # the most recent step that produced a non-empty answer wins as the
            # variable's output
            if result.data is not None:
                data_final = result.data
                data_final_step_id = step.id
            # What each step actually found, whatever became of it. A step can COMPLETE
            # and answer with nothing — a small model handed an empty output example
            # returns it verbatim — and that answer then overwrites a step that did find
            # something. The recipe's answer is still the last one, which is deliberate;
            # but a row reading n_lines=0, ok=TRUE over two lines a previous step found
            # says the chart is silent when the chart was not. This is what lets the
            # table say which it was.
            content_found_by_step[step.id] = _how_much_a_step_found(result.data)

            # A step that raised produced no answer, so its stop_if has nothing to judge:
            # it would be evaluated against an empty dict, where a condition like
            # "no answer yet" reads as "good enough to stop". That both ends the recipe
            # early and rewrites this step's recorded status from failed to stopped —
            # and "failed" is what step 8 scans for to decide the variable is not ok, so
            # the variable would come back ok with a null answer and "we never looked"
            # would read as "the chart is silent". A failed step stays failed.
            if step.stop_if and summary["status"] != "failed":
                ctx_expr = {"data": result.data or {}, "steps": step_outputs, "vars": upstream_vars}
                if _eval_expr(step.stop_if, ctx_expr):
                    summary["status"] = "stopped"
                    for s in recipe.steps[i + 1:]:
                        stage_summaries.append({
                            "step_id": s.id,
                            "kind": s.kind,
                            "status": "skipped",
                            "receipt_path": None,
                            "elapsed_s": None,
                        })
                    break

        # Central grounding guard: reject any evidence pointer in the final data that
        # cites a passage the model was never shown (a fabricated citation) and is not a
        # self-verifying structured evidence pointer. The union of every step's
        # shown-chunk list is the set of passages actually put in front of the model. A
        # rejected pointer is nulled here so step 8 handles the ungrounded value with
        # its ordinary unprovenanced-value machinery (it drives `ok`); the rejection is
        # noted so the reviewer can tell "cited nothing" from "fabricated a citation".
        shown_chunk_ids: set[str] = set()
        for step_output in step_outputs.values():
            shown_chunk_ids.update(step_output.get("evidence_chunk_ids") or [])
        provenance_rejected: list[dict[str, Any]] = []
        if data_final is not None:
            provenance_rejected = _reject_ungrounded_evidence_pointers(
                data_final, shown_chunk_ids, data_final_step_id
            )
            if provenance_rejected:
                log.warning(
                    "provenance_rejected_fabricated_citation",
                    extra_={"recipe": recipe.name, "rejected": provenance_rejected},
                )

        elapsed = time.perf_counter() - t0_recipe
        # The extract step (step 7) hands this per-recipe transcript to the organize
        # step (step 8), which checks the answer against the schema, writes result.json
        # + invariants.json + the evidence files, and derives the final variable result.
        # This transcript stays on the PHI side (it carries the answer plus chart text)
        # and is never allowed to leave the machine.
        atomic_write_json(variable_dir / "transcript.json", {
            "sensitivity": "high",
            "run_id": run_id,
            "patient_id": patient_id,
            "variable": recipe.name,
            "recipe": {"name": recipe.name, "version": recipe.version},
            "recipe_path": str(recipe.path),
            # Named relative to the recipe library rather than absolutely, for the same
            # reason the receipts below are: this is read again when the run is
            # organized, and an absolute path is true only on the machine that
            # extracted. Below the library the layout is what the library ships, so it
            # is the same everywhere. Absolute when the recipe somehow sits outside the
            # library, which is a shape nothing in the tree produces.
            "output_schema_path": _named_relative_to(recipe.output_schema_path, recipes_root),
            "validation_rules": list(recipe.validation.rules),
            "depends_on": list(recipe.depends_on),
            "data_final": data_final,
            # Which step's answer became the variable's answer, and what every step
            # found on the way. Both were computed already and thrown away, so nothing
            # downstream could tell an empty chart from an answer a later step replaced
            # with nothing.
            "answer_came_from_step": data_final_step_id,
            "content_found_by_step": content_found_by_step,
            "errors": errors,
            # Every stage summary names its receipt relative to this folder, and nothing
            # in the summary says where this folder is. A run gets copied between a
            # cluster and a laptop, so a location recorded here would stop being true the
            # moment it moved; the receipts are anchored instead by whoever reads this
            # transcript, to the folder they read it from.
            "steps": stage_summaries,
            "total_elapsed_s": round(elapsed, 6),
            "provider_config_hash": provider_config_hash,
            "code_lock_hash": code_lock_hash,
            # Per-variable and well defined, unlike prompt_hash which differs per step
            # and is recorded on each step receipt. Carried here so result.json can
            # assert them too: two_stream_provenance and ADR 0018 both state that every
            # data-side artifact's produced_by carries the relevant sub-hashes, and on
            # disk the result envelope was shipping them all null while the step
            # receipts beside it had them.
            "recipe_hash": recipe_hash,
            "schema_hash": schema_hash_val,
            "python_helpers_hash": hooks_hash_val,
            "step_evidence": recipe_step_evidence,
            "provenance_rejected": provenance_rejected,
        })

        # While this loop is still running, a recipe that depends on this variable reads
        # its value here; the organize step (step 8), after the loop, fills in whether
        # it passed and any errors/warnings.
        # Organized here rather than after every variable, so this one's outcome is
        # known before the next one starts. Whether it succeeded is decided by step 8,
        # against its schema and its provenance — a recipe whose steps all ran cleanly
        # can still fail there — so this is the earliest moment the answer exists.
        # Doing it at the end instead meant the display printed every start and then
        # every finish, which is not what the work looked like.
        results.update(organize_per_recipe_outputs(
            run_id=run_id,
            patient_id=patient_id,
            patient_out=patient_out,
            recipes_root=recipes_root,
            ordered_names=[recipe.name],
            code_lock_hash=code_lock_hash,
            results_so_far=results,
        ))
        if on_variable is not None:
            organized_now = results.get(recipe.name) or {}
            if organized_now.get("ok"):
                outcome = "done"
            elif failed_with_before is not None and [
                str(e) for e in (organized_now.get("errors") or [])
            ] == failed_with_before:
                # Retried and landed in exactly the same place. Worth its own word: the
                # thing to do about it is not to run it again.
                outcome = "failed again"
            else:
                outcome = "failed"
            on_variable(recipe.name, outcome, time.perf_counter() - variable_started)

    # Organize step (step 8): for every recipe that ran this time, in dependency order,
    # check it against its schema and write result.json / invariants.json / evidence,
    # then derive the final per-variable results. This step does no model calls and is
    # safe to re-run (it just rewrites the same files).
    # No post-loop organize: every variable in ordered_names either ran and was organized
    # as it finished, or was resumed and carries its recorded result. The filtered call
    # that used to be here could only ever receive an empty list, and its comment
    # described a rebuild that never happened.

    # The cross-variable clinical invariants are consistency checks across several
    # variables at once (e.g. a death date can't precede the diagnosis date; therapy
    # lines must be in a sensible order). They are the one check that does not rely on
    # the model at all, so they act as a free clinical sanity backstop. Step 8 runs them
    # over ALL of this patient's results, writes the detailed PHI record, and emits a
    # de-identified (NO_PHI) pass/fail signal that carries no chart text.
    n_invariant_violations = organize_clinical_invariants(
        run_id=run_id,
        patient_id=patient_id,
        patient_out=patient_out,
        recipes_root=recipes_root,
        results=results,
    )

    # Final bookkeeping: step 8 counts how many variables did not pass and records the
    # one running->completed state change for this patient (any failed variable will
    # keep later stages from treating the run as cleanly completed). The outer
    # run_extract_one still owns recording the 'failed' state and closing the cache.
    n_failed = record_extract_completion(
        run_root=run_root,
        run_id=run_id,
        patient_id=patient_id,
        requested_recipes=names,
        results=results,
        code_lock_hash=code_lock_hash,
    )
    log.info("extract_done", extra_={"variables": ordered_names, "missing_deps": plan.missing, "n_failed": n_failed})
    return {
        "patient_id": patient_id,
        "variables": results,
        "order": ordered_names,
        "missing_deps": plan.missing,
        "n_failed": n_failed,
        "n_invariant_violations": n_invariant_violations,
    }


# Step kinds that put a prompt in front of a language model. Kept beside the check
# that uses it so adding a prompting step kind without listing it here is a visible
# omission rather than a silent one.
_STEP_KINDS_NEEDING_A_MODEL = frozenset(
    {"retrieve_and_prompt", "llm_only", "map_table_rows_and_prompt"}
)


def _recipe_needs_a_model(recipe: RecipeSpec) -> bool:
    """Whether any of this variable's steps asks a language model anything."""
    return any(step.kind in _STEP_KINDS_NEEDING_A_MODEL for step in recipe.steps)


def _hash_provider_config(cfg: dict[str, Any]) -> str:
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.content_fingerprinting import (
        hash_json,
    )
    return hash_json(cfg)


def _load_hashes_index(run_root: Path) -> dict[str, Any]:
    """read code/hashes.json (the content hashes recorded when the run was sealed);
    return an empty dict if it is missing."""
    path = Path(run_root) / "code" / "hashes.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data.get("payload") or {}

def _prompt_hash_for_stage(step, prompt_hashes: dict[str, str]) -> str | None:
    """look up the hash of the prompt file this step uses, if any."""
    if step.prompt is None:
        return None
    stem = Path(step.prompt).stem
    return prompt_hashes.get(stem)


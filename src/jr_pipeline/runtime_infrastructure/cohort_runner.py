"""End-to-end cohort runner.

One entry point: ``run_cohort(settings)`` walks seal → preflight → ingest →
embed → index → extract in a single process. quickstart.py, the app's Start
tab, and the ``junior run`` command all call it, so the runner is the single
source of truth for what a whole-cohort run does.

Sealing is step 0, not an optional extra: the per-step CLI commands
(``junior ingest --patient ...``) refuse to run without a sealed bundle, so
sealing here is what makes them a decomposition of this same path rather than
a second one with its own rules.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

# What extraction reads out of the config that is not a recipe, a path or an endpoint.
# Kept here rather than spelled out at each call site: extract_cfg below is built from a
# hand-written set of keys, so anything missing from it is dropped in silence — which is
# how max_chunks_per_prompt came to apply on `junior extract` and not on `junior run`,
# giving one run 7-chunk prompts and 16-chunk prompts for the same variables.
EXTRACT_EXECUTION_CFG_KEYS = (
    "max_chunks_per_prompt",
    "max_provenance_retries",
    # Where model responses are cached. Honoured by `junior extract` and dropped
    # by `junior run`, which then silently fell back to the run folder — the same
    # divergence as the two above, found by asking what extract actually reads
    # rather than by hitting it.
    "llm_cache_path",
)


@dataclass
class CohortSettings:
    """Every cohort-specific knob lives here. Per-step defaults (chunk size,
    retriever weights, retry policy, ...) stay inside the step modules."""
    project: str = "your_cohort_name"
    input_folder: Path = field(default_factory=lambda: Path("./examples"))
    data_root: Path = field(default_factory=lambda: Path("./data"))

    # "auto" → every subfolder of input_folder; or an explicit list of MRNs.
    patients: list[str] | str = "auto"

    # "auto" → every supported file, all columns; or explicit {stem, optional?} dicts.
    # Ingest preserves whole source files as structured parquets.
    files_to_ingest: list[dict] | str = "auto"
    # "auto" → the `text` column from every ingested parquet that has one.
    # Embed controls the retrieval corpus; Step 3 indexes those embeddings in HNSW.
    files_to_embed: list[dict] | str = "auto"

    # This site's column map: chart_columns_file points at an institution's map under
    # deployment/, chunk_metadata_columns is the same mapping written inline, and
    # text_column renames the default free-text column. Ingest and embed both read
    # these to interpret source files, so they have to ride the cohort path — the
    # per-step CLI hands each stage the whole config file and would otherwise read
    # the same charts differently from `run`.
    chart_columns_file: str | None = None
    chunk_metadata_columns: dict = field(default_factory=dict)
    text_column: str | None = None

    # Variable names — must match recipe folder names under recipes_root.
    variables: list[str] = field(default_factory=lambda: ["date_of_birth"])
    recipes_root: Path = field(default_factory=lambda: Path("./var_extraction_recipes"))

    # Local folder with the encoder weights. Required — embed never downloads models.
    embedding_model_path: Path = field(
        default_factory=lambda: Path(
            "./models/embedding/thomas-sounack:BioClinical-ModernBERT-base23APR2026"
        )
    )
    # Encoder keys layered over the defaults below — e.g. dtype: float16 to match a
    # cluster build. embedding_model_path supplies model_id. Anything a YAML config
    # names under `encoder:` arrives here, so a config that round-trips through
    # CohortSettings seals the same retrieval fingerprint it started with.
    encoder_options: dict = field(default_factory=dict)
    # Chunk windowing for embed. Its fingerprint is part of the run's method identity.
    chunker_options: dict = field(
        default_factory=lambda: {"kind": "token_window", "overlap": 128}
    )
    # HNSW build parameters (M, ef_construction, space, random_seed). Empty leaves the
    # index step on its own defaults.
    index_options: dict = field(default_factory=dict)
    # Settings that decide how extraction runs rather than what it extracts. They are
    # run-invariant, so a run is fixed to them, and they reach extract through this
    # object on the `junior run` path and through the config directly on the per-stage
    # path. Both have to carry them or the same run means two different things
    # depending on which command drove it.
    extract_execution_settings: dict = field(default_factory=dict)
    # When these settings came from a YAML config, that config's own keys, recorded
    # verbatim in the sealed bundle. The per-step CLI compares the config it is handed
    # against the sealed one, so the bundle has to hold the file's values as written —
    # a path this runner resolved to absolute would read as drift against the same file.
    sealed_config_base: dict = field(default_factory=dict)

    # "local" → local Qwen, no network. "openai_compatible" → an allowlisted
    # institutional endpoint. Set llm_endpoint_name to force one specific endpoint.
    llm_mode: str = "local"
    llm_local_model_path: Path | None = field(
        default_factory=lambda: Path(
            "./models/extraction/3B_param_local_qwen_extractor_for_example"
        )
    )
    llm_allowlist: Path = field(
        default_factory=lambda: Path("./deployment/local/llm_allowlist.yaml")
    )
    llm_endpoint_name: str | None = None


@dataclass
class CohortResult:
    """run_cohort return value. Use for downstream rollups / dashboards."""
    run_id: str
    patients: list[str]
    ingested: list[str]
    failed_ingest: list[tuple[str, str]]
    blocked_by_preflight: list[str]
    embed_status: dict[str, str]
    index_status: dict[str, str]
    extract_status: dict[str, str]


def _new_run_id() -> str:
    """Timestamp + short entropy suffix so two runs in the same second don't
    interleave one dir. Pinning a run_id to re-enter a run re-seals it; the
    sealed bundle is what rejects re-entry under a drifted config."""
    import secrets

    return datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + secrets.token_hex(2)


# The canonical run-id shape _new_run_id produces: date_time_<hex entropy>. The single
# owner of this pattern, so a consumer that filters run dirs (e.g. the review app picking
# the newest run) shares it instead of copying the regex. Scratch dirs from model
# comparison are named <base>__compare__<name> and never match it. Used with fullmatch.
RUN_ID_PATTERN = re.compile(r"\d{8}_\d{6}_[0-9a-f]+")


def cohort_settings_from_config(cfg: dict, *, variables: tuple[str, ...] = ()) -> CohortSettings:
    """Map a pipeline YAML config onto CohortSettings.

    The YAML and the ``CohortSettings`` dataclass are the same settings under two
    names, so this is the one place the two vocabularies meet — every interface
    (the CLI, the review app's Start tab, quickstart) builds its settings here so
    none of them grows a dialect of its own. Relative paths resolve against the
    repo root rather than the caller's cwd: the config names ``./models/...`` and
    ``./var_extraction_recipes``, which are repo locations, and resolving them
    against cwd would silently write a run's outputs wherever the operator
    happened to be standing."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
        resolve_repo_root,
    )

    repo_root = resolve_repo_root()

    def _path(value: str | None, default: Path) -> Path:
        """A setting that always resolves to somewhere, so the caller gets a Path.

        Split from _path_or_none so the type says which settings can be absent. Most of
        these cannot: CohortSettings gives every one of them a default, and returning a
        maybe-Path for them pushed a None check onto every caller for a case that
        cannot arise, which is how a real one gets lost among them."""
        # Defaults go through the same resolution as configured values — a default
        # of "./data" left relative would put outputs under the caller's cwd.
        candidate = Path(value if value is not None else default).expanduser()
        return candidate if candidate.is_absolute() else (repo_root / candidate).resolve()

    def _path_or_none(value: str | None, default: Path | None) -> Path | None:
        """The same, for a setting that may legitimately be unset."""
        raw = value if value is not None else default
        return None if raw is None else _path(str(raw), Path(raw))

    defaults = CohortSettings()
    encoder = cfg.get("encoder") or {}
    # `recipes` is the key every config writer and every other reader uses. Reading
    # only `variables` meant no config in the repo ever matched, so a cohort created
    # with a chosen variable list silently extracted the dataclass default instead —
    # and the sealed bundle, built from the file, then recorded the list that was asked
    # for rather than the one that ran.
    chosen_variables = (
        list(variables) or cfg.get("recipes") or cfg.get("variables") or defaults.variables
    )
    column_map_file = _path_or_none(cfg.get("chart_columns_file"), None)

    return CohortSettings(
        project=cfg.get("project") or Path(cfg.get("run_id") or "cohort").name,
        input_folder=_path(cfg.get("source_root"), defaults.input_folder),
        data_root=_path(cfg.get("output_root"), defaults.data_root),
        patients=cfg.get("patients") or "auto",
        files_to_ingest=cfg.get("files") or "auto",
        files_to_embed=cfg.get("files_to_embed") or "auto",
        chart_columns_file=str(column_map_file) if column_map_file else None,
        chunk_metadata_columns=cfg.get("chunk_metadata_columns") or {},
        text_column=cfg.get("text_column"),
        variables=chosen_variables,
        recipes_root=_path(cfg.get("recipes_root"), defaults.recipes_root),
        embedding_model_path=_path(encoder.get("model_id"), defaults.embedding_model_path),
        # Everything the config says about the encoder except the path, which
        # embedding_model_path owns. Passing these through is what lets the sealed
        # retrieval fingerprint match the config that produced it.
        encoder_options={k: v for k, v in encoder.items() if k != "model_id"},
        chunker_options=cfg.get("chunker") or defaults.chunker_options,
        index_options=cfg.get("index") or {},
        extract_execution_settings={
            key: cfg[key] for key in EXTRACT_EXECUTION_CFG_KEYS if cfg.get(key) is not None
        },
        sealed_config_base={k: v for k, v in cfg.items() if v is not None},
        llm_mode=cfg.get("llm_mode") or defaults.llm_mode,
        llm_local_model_path=_path_or_none(
            cfg.get("llm_local_model_path"), defaults.llm_local_model_path
        ),
        llm_allowlist=_path(cfg.get("allowlist_path"), defaults.llm_allowlist),
        llm_endpoint_name=cfg.get("model_override"),
    )


def _sealed_config(settings: CohortSettings, run_id: str, cfgs: dict[str, dict]) -> dict:
    """The single config the sealed bundle records for a cohort run.

    run_cohort drives four per-step configs, but a run seals once. The bundle
    records their union so the per-step CLI's drift gate compares like against
    like: it checks the run-invariant keys (run_id, output_root, allowlist_path,
    recipes_root) plus the scoped provider/retrieval sub-hashes, so all of those
    must be present here for someone re-entering this run with
    ``jr-pipeline ingest --patient ...`` to pass rather than trip a false drift."""
    extract_cfg = cfgs["extract"]
    sealed = {
        "run_id": run_id,
        "project": settings.project,
        "output_root": str(Path(settings.data_root).expanduser().resolve()),
        "source_root": cfgs["ingest"]["source_root"],
        "recipes_root": extract_cfg["recipes_root"],
        "recipes": extract_cfg["recipes"],
        "allowlist_path": extract_cfg["allowlist_path"],
        "encoder": extract_cfg["encoder"],
        "chunker": extract_cfg["chunker"],
    }
    # `index` is part of the retrieval fingerprint, so it belongs in the sealed cfg
    # whenever the run configures one — omitting it would make a per-step CLI call
    # that passes the same config read as drifted.
    if settings.index_options:
        sealed["index"] = dict(settings.index_options)
    if "model_override" in extract_cfg:
        sealed["model_override"] = extract_cfg["model_override"]
    # The originating config file's keys win: the bundle must record what that file
    # says, so handing the same file to `jr-pipeline ingest` compares equal.
    sealed.update(settings.sealed_config_base)
    # That restore can put the file's allowlist_path back over the one extraction
    # actually loads (a local-mode run loads the generated run_config/ file). The
    # gate needs the file's spelling; provenance needs the truth — so both are kept.
    if sealed.get("allowlist_path") != extract_cfg["allowlist_path"]:
        sealed["allowlist_path_used"] = extract_cfg["allowlist_path"]
    return sealed


def _continue_the_sealed_run(run_root: Path, sealed_config_candidate: dict) -> str:
    """Re-enter a run that is already sealed: verify the bundle, check for drift, and
    return the existing code_lock_hash. Never re-seals — see the caller's comment."""
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.code_to_data_cross_link import (  # noqa: E501
        verify_code_bundle,
    )
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (  # noqa: E501
        sealed_code_lock_hash,
        why_a_sealed_run_cannot_continue,
    )

    changed = why_a_sealed_run_cannot_continue(sealed_config_candidate, run_root)
    if changed:
        raise RuntimeError(
            f"Run '{run_root.name}' was started with different settings than the ones "
            f"you are using now.\n"
            f"  what changed: {', '.join(changed)}\n"
            "A run is locked to the settings it began with, so its results stay "
            "reproducible. Either:\n"
            "  • put those settings back, and carry on with this run; or\n"
            "  • changed a recipe only?  junior extract --new-run   reuses this run's "
            "corpus under a fresh run id; or\n"
            "  • start over under the new settings:  junior ingest --new-run   then  junior embed, index, extract"
        )
    ok, detail = verify_code_bundle(run_root)
    if not ok:
        raise RuntimeError(
            "The code changed after this run started.\n"
            f"  what changed: {detail}\n"
            "A run records the exact code that produced it, so it will not continue "
            "under different code. Either put the change back, or start over:\n"
            "  junior ingest --new-run   then  junior embed, index, extract"
        )
    code_version = sealed_code_lock_hash(run_root)
    if not code_version:
        raise RuntimeError(
            f"This run's saved copy of the code is incomplete under {run_root / 'code'} "
            "— its code.lock.json cannot be read. Start over: "
            "junior ingest --new-run   then  junior embed, index, extract"
        )
    return code_version


def run_cohort(
    settings: CohortSettings,
    *,
    run_id: str | None = None,
    security_check: bool = True,
) -> CohortResult:
    """Walk the full pipeline for this cohort. ``run_id`` defaults to a
    fresh ``YYYYmmdd_HHMMSS``; pin it to re-enter an existing run."""
    os.environ["JR_DATA_ROOT"] = str(Path(settings.data_root).resolve())
    if security_check:
        _print_security_check(settings)

    # Lazy import: avoids dragging torch / polars / hnswlib into the
    # dataclass module just so a caller can construct CohortSettings.
    from jr_pipeline.pipeline_steps.step_1_ingest_raw_files.ingest import (
        preflight_patients,
        run_ingest_one,
    )
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.frozen_code_snapshot import (
        build_and_seal,
        resolve_repo_root,
    )
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.run_manifest_builder import (
        build_manifest,
        build_roster,
        model_sha256_from_cfg,
        write_manifest,
        write_roster,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        ensure_layout,
        no_phi_run_dir,
        phi_intermediate_run_dir,
        phi_patient_run_dir,
    )
    from jr_pipeline.runtime_infrastructure.json_event_logging import (
        clear_run_log_file,
        set_run_log_file,
    )

    run_id = run_id or _new_run_id()
    ensure_layout(run_id)
    run_root = phi_intermediate_run_dir(run_id)
    # Durable per-run log sink: every structured event also lands in run_log.jsonl
    # so a crashed or SLURM-array run keeps a record beyond the lost stdout.
    set_run_log_file(run_root / "run_log.jsonl")
    patients = _resolve_patients(settings)
    # Mutable accumulator persisted as cohort_result.json in the finally below, so a
    # half-built cohort (e.g. embed skipped: no torch) leaves a durable, honest record
    # instead of vanishing into an unpersisted return value.
    run_record: dict[str, Any] = {
        "run_id": run_id,
        "project": settings.project,
        "patients": patients,
        "ingested": [],
        "failed_ingest": [],
        "blocked_by_preflight": [],
        "embed_status": {},
        "index_status": {},
        "extract_status": {},
    }
    print(f"Run ID:  {run_id}")
    print(f"Project: {settings.project} — {len(patients)} patient(s)")
    print(f"PHI outputs:     {phi_patient_run_dir(run_id, patients[0]) if patients else '(no patients)'}")
    print(f"Non-PHI outputs: {no_phi_run_dir(run_id)}")

    # Build the step configs before sealing: the bundle records the resolved config,
    # and for a local-LLM run _resolve_extract_allowlist materializes the allowlist
    # file that the bundle's allowlist_names.json is derived from.
    cfgs = _build_step_configs(settings, run_id)

    # Seal the code bundle — the same call `jr-pipeline seal` makes. Every run is
    # sealed, so a laptop run and a cluster run carry identical provenance and both
    # answer to `jr-pipeline verify`. This is also what lets someone re-enter this
    # run_id with the per-step CLI: those commands require a sealed bundle.
    #
    # Sealed ONCE. Re-entering a run that already holds a bundle (a bare
    # `junior run` continues the newest run; the app's Run button spawns exactly
    # that) must never re-seal: build_and_seal deletes and recopies code/ and
    # rewrites hashes.json — the provenance every existing receipt claims — and,
    # because resume compares recorded hashes against the sealed index, a re-seal
    # under edited code silently turns finished variables back into pending ones.
    # An existing bundle is therefore verified and drift-checked, exactly like the
    # per-stage commands check it, and the run refuses on drift instead of
    # papering over it.
    print("\n=== Step 0: Seal code bundle ===")
    sealed_config_candidate = _sealed_config(settings, run_id, cfgs)
    if (run_root / "code" / "code.lock.json").is_file():
        code_version = _continue_the_sealed_run(run_root, sealed_config_candidate)
        print(f"  ✓ {code_version} (sealed earlier — verified, not re-sealed)")
    else:
        sealed = build_and_seal(
            run_id=run_id,
            run_root=run_root,
            repo_root=resolve_repo_root(),
            cfg=sealed_config_candidate,
            entry_point={
                "argv_program": "python",
                "argv": sys.argv[:],
                "step": "seal",
                "config_alias": settings.project,
            },
            recipes_root=Path(settings.recipes_root).expanduser().resolve(),
            allowlist_path=Path(cfgs["extract"]["allowlist_path"]),
        )
        code_version = sealed["code_lock_hash"]
        print(f"  ✓ {code_version}")

    write_manifest(run_root, build_manifest(
        run_id=run_id,
        code_lock_hash=code_version,
        entry_point_name="jr_pipeline.runtime_infrastructure.cohort_runner.run_cohort",
        config_alias=settings.project,
        target_patients=patients,
        model_sha256=model_sha256_from_cfg(cfgs["embed"]),
    ))
    # The raw roster lives PHI-side (never exported); the manifest holds only a count.
    write_roster(run_root, build_roster(run_id=run_id, target_patients=patients))

    # Run every stage inside a try whose finally ALWAYS writes the summary
    # (the one place the manifest flips off "running"). A mid-run crash then leaves
    # a `failed` summary instead of a manifest stuck "running" that a later
    # `summarize` would wrongly flip to "completed".
    try:
        print("\n=== Step 1a: Preflight ===")
        preflight = preflight_patients(cfg=cfgs["ingest"], patients=patients)
        print(preflight.summary())

        print("\n=== Step 1b: Ingest ===")
        ingested, failed_ingest = [], []
        for pid in preflight.ok:
            try:
                summary = run_ingest_one(cfg=cfgs["ingest"], patient_id=pid, code_lock_hash=code_version)
            except Exception as e:
                failed_ingest.append((pid, f"{type(e).__name__}: {e}"))
                print(f"  ✗ {pid}: {type(e).__name__}: {e}")
                continue
            ingested.append(pid)
            tag = " (cached)" if summary["cached"] else ""
            print(f"  ✓ {pid}: {len(summary['files_written'])} files{tag}")

        print(
            f"\nIngest tally: {len(ingested)} ok, {len(failed_ingest)} failed, "
            f"{len(preflight.blocked)} blocked by preflight (of {len(patients)} total)"
        )
        run_record["ingested"] = ingested
        run_record["failed_ingest"] = failed_ingest
        run_record["blocked_by_preflight"] = preflight.blocked

        embed_status = _run_embed(ingested, cfgs["embed"], code_version)
        run_record["embed_status"] = embed_status
        index_status = _run_index(ingested, cfgs["index"], code_version)
        run_record["index_status"] = index_status
        # Steps 4–6 (retrieve, rerank, prepare_evidence) run inside extract per recipe.
        extract_status = _run_extract(ingested, settings, cfgs["extract"], code_version)
        run_record["extract_status"] = extract_status

        return CohortResult(
            run_id=run_id, patients=patients,
            ingested=ingested, failed_ingest=failed_ingest,
            blocked_by_preflight=preflight.blocked,
            embed_status=embed_status, index_status=index_status, extract_status=extract_status,
        )
    finally:
        # Durable per-patient per-stage record (incl. skips) written FIRST, so the
        # summary below can read it for status honesty. Best-effort — never sink the run.
        try:
            from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
                atomic_write_json,
            )
            atomic_write_json(run_root / "cohort_result.json", run_record)
        except Exception as e:
            print(f"  (cohort_result write skipped: {type(e).__name__}: {e})")
        # Merge state fragments -> state.jsonl, write summary.json, flip manifest
        # status. Best-effort — a summary failure must not sink the return or mask a crash.
        try:
            from jr_pipeline.runtime_enforcing_safety_and_reproducibility.run_summary import (
                write_summary,
            )
            write_summary(run_root)
        except Exception as e:
            print(f"  (summary write skipped: {type(e).__name__}: {e})")
        # The run's single method_provenance record (the RunHeader join key), emitted
        # BEFORE finalize compacts the shards. Run-level method identity, no patient
        # data. Best-effort -- telemetry must not sink the run.
        try:
            from jr_pipeline.runtime_enforcing_safety_and_reproducibility.evidence_selection_trace import (
                encoder_fingerprint,
                fingerprint,
                record_method_provenance,
            )
            embed_cfg = cfgs.get("embed", {})
            record_method_provenance(
                run_id,
                encoder_fingerprint=encoder_fingerprint(embed_cfg.get("encoder") or {}),
                chunking_config_id=fingerprint(embed_cfg.get("chunker") or {}),
                code_lock_hash=code_version,
            )
        except Exception as e:
            print(f"  (method provenance skipped: {type(e).__name__}: {e})")
        # Compact this run's NO_PHI exhaust shards into <record_type>.parquet + the
        # shareable manifest. Best-effort — exhaust is telemetry; a finalize failure
        # must not sink the run or mask a crash.
        try:
            from jr_pipeline.runtime_infrastructure.exhaust.finalize import finalize_exhaust
            finalize_exhaust(run_id)
        except Exception as e:
            print(f"  (exhaust finalize skipped: {type(e).__name__}: {e})")
        # Stop teeing logs to this run's file so the next run in the same process
        # (quickstart, tests) doesn't append to a stale run_log.jsonl.
        clear_run_log_file()


def _resolve_patients(settings: CohortSettings) -> list[str]:
    if isinstance(settings.patients, str):
        if settings.patients == "auto":
            ip = Path(settings.input_folder)
            if not ip.is_dir():
                return []
            return sorted(
                p.name for p in ip.iterdir()
                if p.is_dir() and not p.name.startswith(".")
            )
        return [settings.patients]
    if settings.patients != "auto":
        return list(settings.patients)
    return []


def _build_step_configs(settings: CohortSettings, run_id: str) -> dict[str, dict]:
    if settings.embedding_model_path is None:
        raise ValueError(
            "embedding_model_path is required — point it at a local encoder "
            "folder, e.g. ./models/embedding/<model>. Embed never downloads models."
        )
    encoder: dict[str, Any] = {
        "model_id": str(settings.embedding_model_path),
        "pooling": "mean", "normalize": True,
        "max_tokens": 512, "device": "auto",
    }
    # Config-supplied keys win over the defaults. Merging rather than replacing keeps
    # model_id under embedding_model_path's control while letting a config add the
    # keys it cares about (dtype, expected_file_sha256, ...) — dropping them here
    # would seal a retrieval fingerprint the originating config can never match.
    encoder.update(settings.encoder_options)
    # Present only when configured: a null key in a stage config would read as "this
    # run explicitly set no map", which is not what an absent setting means.
    site_column_map: dict[str, Any] = {}
    if settings.chart_columns_file:
        site_column_map["chart_columns_file"] = str(settings.chart_columns_file)
    if settings.chunk_metadata_columns:
        site_column_map["chunk_metadata_columns"] = dict(settings.chunk_metadata_columns)
    embed_cfg: dict[str, Any] = {
        "run_id": run_id,
        "files": settings.files_to_embed,
        "encoder": encoder,
        # Explicit chunker so its fingerprint is a real config identity in the
        # exhaust header, not fingerprint({}).
        "chunker": dict(settings.chunker_options),
        **site_column_map,
    }
    if settings.text_column:
        embed_cfg["text_column"] = settings.text_column
    extract_cfg = {
        "run_id": run_id,
        "recipes_root": str(settings.recipes_root),
        "recipes": settings.variables,
        "allowlist_path": str(_resolve_extract_allowlist(settings, run_id)),
        # Hybrid retrieval embeds the query at extract time, so the extract
        # step needs the same encoder the embed step used.
        "encoder": embed_cfg["encoder"],
        # Feed the real chunker so the trace's chunking_config_id is a
        # genuine fingerprint, not a non-distinguishing constant.
        "chunker": embed_cfg["chunker"],
        **settings.extract_execution_settings,
    }
    if settings.llm_endpoint_name is not None:
        extract_cfg["model_override"] = settings.llm_endpoint_name
    return {
        "ingest": {
            "run_id": run_id,
            "source_root": str(settings.input_folder),
            "project": settings.project,
            "files": settings.files_to_ingest,
            **site_column_map,
        },
        "embed": embed_cfg,
        "index": {"run_id": run_id, **({"index": dict(settings.index_options)} if settings.index_options else {})},
        "extract": extract_cfg,
    }


# The keys that IDENTIFY an endpoint, as opposed to the ones that make it safe to
# run. When a generated local endpoint borrows settings from the configured
# allowlist, identity stays with the generator — the model the operator picked is
# the model, wherever the template pointed.
_ENDPOINT_IDENTITY_KEYS = frozenset({
    "name", "url", "provider", "attestation", "default_model", "allow_download", "auth",
})


def _local_endpoint_safety_settings(configured_allowlist: Path) -> dict:
    """The non-identity settings of the configured allowlist's first ``local_hf``
    endpoint — the prompt-token ceiling above all, plus dtype/device and any
    measurement note — to carry into a generated run-local allowlist."""
    try:
        loaded = yaml.safe_load(
            Path(configured_allowlist).expanduser().read_text(encoding="utf-8")
        ) or {}
    except (OSError, yaml.YAMLError):
        return {}
    for entry in loaded.get("allowed_endpoints") or []:
        if isinstance(entry, dict) and entry.get("provider") == "local_hf":
            return {k: v for k, v in entry.items() if k not in _ENDPOINT_IDENTITY_KEYS}
    return {}


def _resolve_extract_allowlist(settings: CohortSettings, run_id: str) -> Path:
    """Return the allowlist path for extract, generating one for local runs.

    A generated endpoint carries the safety settings of the configured allowlist's
    local_hf entry. Generating a bare endpoint quietly discarded that file:
    `junior run` extracted with no prompt ceiling and no dtype while `junior extract`
    under the same config honoured both — and an oversized prompt on an in-process
    model does not fail, it takes the machine down with nothing in the log."""
    if settings.llm_mode != "local":
        return Path(settings.llm_allowlist)
    if settings.llm_local_model_path is None:
        raise ValueError("llm_local_model_path is required when llm_mode='local'")

    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        run_config_dir,
    )

    model_path = Path(settings.llm_local_model_path).expanduser().resolve()
    endpoint = {
        "name": "local_qwen",
        "url": str(model_path),
        "provider": "local_hf",
        "attestation": "self_hosted",
        "default_model": str(model_path),
    }
    endpoint.update(_local_endpoint_safety_settings(Path(settings.llm_allowlist)))
    if endpoint.get("max_prompt_tokens_cap") is None:
        print(
            "  ⚠ no prompt-token ceiling is set for the in-process model — an "
            "oversized prompt can take this machine down rather than fail. Add a "
            "local_hf entry with max_prompt_tokens_cap to the allowlist named by "
            "allowlist_path (deployment/local/llm_allowlist_local3b.yaml is the "
            "measured example)."
        )
    out = (
        run_config_dir(run_id, Path(settings.data_root).expanduser().resolve())
        / "local_llm_allowlist.yaml"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        yaml.safe_dump({"allowed_endpoints": [endpoint]}, sort_keys=False),
        encoding="utf-8",
    )
    return out


def _run_embed(ingested: list[str], cfg: dict, code_version: str | None = None) -> dict[str, str]:
    print("\n=== Step 2: Embed ===")
    try:
        from jr_pipeline.pipeline_steps.step_2_embed_chunks.embed import run_embed_one
    except ImportError as e:
        print(f"  needs torch extra: pip install -e \".[torch]\"  ({e})")
        return {pid: "skipped: no torch" for pid in ingested}
    status: dict[str, str] = {}
    for pid in ingested:
        try:
            s = run_embed_one(cfg=cfg, patient_id=pid, code_lock_hash=code_version)
        except Exception as e:
            status[pid] = f"failed: {type(e).__name__}: {e}"
            print(f"  ✗ {pid}: {type(e).__name__}: {e}")
            continue
        if s.get("cached"):
            status[pid] = "cached"
            print(f"  ✓ {pid}: embeddings up to date (cached)")
        else:
            status[pid] = f"ok: {s.get('chunks', '?')} chunks"
            print(f"  ✓ {pid}: {s.get('chunks', '?')} chunks embedded")
    return status


def _run_index(ingested: list[str], cfg: dict, code_version: str | None = None) -> dict[str, str]:
    print("\n=== Step 3: Build vector index ===")
    from jr_pipeline.pipeline_steps.step_3_build_vector_index.build_index import run_index_one
    status: dict[str, str] = {}
    for pid in ingested:
        try:
            s = run_index_one(cfg=cfg, patient_id=pid, code_lock_hash=code_version)
        except FileNotFoundError:
            status[pid] = "skipped: no embeddings"
            print(f"  – {pid}: no embeddings yet (Step 2 failed?)")
            continue
        except Exception as e:
            status[pid] = f"failed: {type(e).__name__}: {e}"
            print(f"  ✗ {pid}: {type(e).__name__}: {e}")
            continue
        if s.get("cached"):
            status[pid] = "cached"
            print(f"  ✓ {pid}: index up to date (cached)")
        else:
            status[pid] = f"ok: {s.get('size', '?')} vectors"
            print(f"  ✓ {pid}: index built ({s.get('size', '?')} vectors)")
    return status


def _stdout_values_allowed() -> bool:
    """Whether patient-derived extracted values may be printed to stdout.

    Extracted values are PHI and stdout persists outside the PHI tree (terminal
    scrollback, SLURM logfiles, screen-shares), so the default is to REDACT. Set
    ``JR_SHOW_STDOUT_VALUES=1`` to print them (a trusted local terminal, or synthetic
    demo data). Setting ``JR_REDACT_STDOUT`` forces redaction regardless."""
    if os.environ.get("JR_REDACT_STDOUT"):
        return False
    return os.environ.get("JR_SHOW_STDOUT_VALUES") == "1"


# A leading "<field name>:" on an error message. Schema keys are code, not chart
# content, so the field name may survive redaction and tell the operator WHERE the
# failure was without saying what the model extracted.
_ERROR_FIELD_PREFIX = re.compile(r"^[A-Za-z0-9_.\-]{1,80}(?=:)")


def stdout_safe_error(err: str) -> str:
    """One extraction error, safe for stdout.

    Validation messages embed the offending extracted VALUE (jsonschema's message
    quotes the instance), so an error line is as value-bearing as the value line the
    gate above it redacts — printing it verbatim defeated the redaction one line up.
    The full message stays in the run's receipts, which live inside the PHI tree."""
    if _stdout_values_allowed():
        return err
    named = _ERROR_FIELD_PREFIX.match(err or "")
    where = f" in {named.group(0)}" if named else ""
    return (f"<redacted{where} — full message in the run's receipts; "
            "set JR_SHOW_STDOUT_VALUES=1 to display it>")


# Summary fields whose values are (or quote) chart-derived content. Everything else in
# a stage summary — counts, paths, timings, ok flags — is operational and may print.
_VALUE_BEARING_SUMMARY_KEYS = frozenset({
    "data", "raw", "text", "content", "answer", "evidence", "messages", "response",
})


def redact_summary_for_stdout(summary):
    """A stage summary that is safe to dump to stdout.

    Stage summaries can carry patient-derived content — an extract summary holds every
    variable's extracted ``data``, and its error strings quote the offending values —
    while stdout persists outside the PHI tree (scrollback, SLURM logs, the review
    app's log panel). Values print only on explicit opt-in; otherwise value-bearing
    fields become a marker that keeps the structure's size, and each error string
    keeps only its leading field name."""
    if _stdout_values_allowed():
        return summary

    def walk(node):
        if isinstance(node, dict):
            scrubbed = {}
            for key, value in node.items():
                if key == "errors" and isinstance(value, list):
                    scrubbed[key] = [stdout_safe_error(str(item)) for item in value]
                elif key in _VALUE_BEARING_SUMMARY_KEYS:
                    if isinstance(value, (dict, list)):
                        scrubbed[key] = f"<redacted: {len(value)} item(s)>"
                    else:
                        scrubbed[key] = None if value is None else "<redacted>"
                else:
                    scrubbed[key] = walk(value)
            return scrubbed
        if isinstance(node, list):
            return [walk(item) for item in node]
        return node

    return walk(summary)


def _run_extract(ingested: list[str], settings: CohortSettings, cfg: dict, code_version: str | None = None) -> dict[str, str]:
    print("\n=== Step 7: Extract (steps 4–6 run inside per recipe) ===")
    allowlist_path = Path(cfg["allowlist_path"])
    if not allowlist_path.exists():
        print(f"  allowlist not found at {allowlist_path} — set llm_allowlist in settings")
        return {pid: "skipped: no allowlist" for pid in ingested}
    from jr_pipeline.pipeline_steps.step_7_extract_variables.extract import run_extract_one
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        phi_intermediate_run_dir,
    )
    from jr_pipeline.runtime_infrastructure.extraction_progress import (
        report_nothing_to_do,
        report_patient,
        report_variables,
    )

    # Per-variable lines exist because this stage was silent for twenty minutes and a
    # healthy run read as a hang. On a TERMINAL the CLI's ticker now answers that — one
    # self-updating line carrying the patient and the estimate — and the two displays
    # cannot share a screen: the ticker owns its line with a carriage return, so every
    # stdout line printed underneath it was overwritten or stranded a copy of the
    # ticker mid-list. Piped to a log there is no ticker and no clobbering, and a log
    # is where the detail is wanted, so it is kept there.
    a_ticker_owns_the_screen = sys.stderr.isatty()

    status: dict[str, str] = {}
    print(f"  {len(ingested)} patient(s) to extract")
    from jr_pipeline.runtime_enforcing_safety_and_reproducibility.reproducibility.sealed_run_continuity import (  # noqa: E501
        recipes_that_changed_since_sealing,
    )

    # No run id (an embedded caller extracting outside a sealed run) means no sealed
    # tree to compare against; the guard below then never fires.
    sealed_recipes_dir = (
        phi_intermediate_run_dir(cfg["run_id"]) / "code" / "recipes"
        if cfg.get("run_id") else None
    )
    for position, pid in enumerate(ingested, start=1):
        # Re-checked per patient, not once at stage entry: extraction reads recipes
        # LIVE, and a cohort takes long enough that a recipe edited mid-run is an
        # ordinary event. Without this, every patient after the edit gets receipts
        # stamped with the sealed hash of a recipe that did not run — the exact
        # dishonesty the seal exists to prevent. Outside the per-patient try:
        # a drifted tree is a run-level stop, not one patient's failure.
        drifted = (
            recipes_that_changed_since_sealing(cfg.get("recipes_root"), sealed_recipes_dir)
            if sealed_recipes_dir is not None else []
        )
        if drifted:
            print(f"  ✋ stopping before {pid}: a recipe changed while this run was "
                  f"extracting — {', '.join(drifted)}")
            print("     Finished patients above stand. Put the recipe back to continue "
                  "this run, or take the change to a fresh run: junior extract --new-run")
            for remaining in ingested[position - 1:]:
                status[remaining] = "stopped: recipes changed mid-run"
            break
        try:
            # Said before the work starts, not after it finishes. This stage is the only
            # one measured in minutes and was the only one silent until a patient was
            # done, which is why a healthy run read as a hang.
            if not a_ticker_owns_the_screen:
                report_patient(position, len(ingested), pid)
            show = None if a_ticker_owns_the_screen else report_variables()
            anything_ran = {"yes": False}

            def _on_variable(name, state, seconds, _ran=anything_ran, _show=show):
                # "already complete" is the one state that means no work happened, so it
                # is what decides whether this patient needed anything at all. Tracked
                # whether or not anything is displayed.
                if state != "already complete":
                    _ran["yes"] = True
                if _show is not None:
                    _show(name, state, seconds)

            s = run_extract_one(
                cfg=cfg, patient_id=pid, code_lock_hash=code_version,
                on_variable=_on_variable,
            )
            if not anything_ran["yes"] and not a_ticker_owns_the_screen:
                report_nothing_to_do()
            for var, v in s.get("variables", {}).items():
                mark = "✓" if v.get("ok") else "✗"
                # The extracted data is patient-derived PHI; stdout persists outside the
                # PHI tree (terminal scrollback, SLURM logfiles, screen-shares). REDACT by
                # default; print values only on explicit opt-in (see _stdout_values_allowed).
                if _stdout_values_allowed():
                    print(f"  {mark} {pid}/{var}: {json.dumps(v.get('data'))}"[:200])
                else:
                    print(f"  {mark} {pid}/{var}: <redacted>")
                for err in v.get("errors") or []:
                    # The same gate as the value line above it: schema messages quote
                    # the extracted value, so an unredacted error IS the value.
                    print(f"      error: {stdout_safe_error(err)}")
            n_failed = s.get("n_failed", 0)
            status[pid] = "ok" if not n_failed else f"completed_with_errors: {n_failed} variable(s)"
        except FileNotFoundError as e:
            status[pid] = f"skipped: missing upstream: {e}"
            print(f"  – {pid}: missing upstream artifact (Step 2 or 3 failed?): {e}")
        except ImportError as e:
            status[pid] = "skipped: no torch"
            print(f"  – {pid}: local LLM needs torch: pip install -e \".[torch]\"  ({e})")
        except Exception as e:
            status[pid] = f"failed: {type(e).__name__}: {e}"
            print(f"  ✗ {pid}: {type(e).__name__}: {e}")
    return status


def _print_security_check(settings: CohortSettings) -> None:
    from jr_pipeline.pipeline_steps.step_7_extract_variables.providers.llm_endpoint_denylist import (
        FORBIDDEN_HOSTS,
    )
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import data_root
    print("SECURITY CHECK")
    print("=" * 50)
    print(f"  LLM mode:        {settings.llm_mode}")
    if settings.llm_mode == "local":
        print(f"  Local model:     {settings.llm_local_model_path}")
        print("  Network calls:   NONE — all processing is local")
    else:
        print(f"  Endpoint name:   {settings.llm_endpoint_name}")
        print("  ⚠ Verify this is a BAA-attested institutional endpoint")
    print(f"  Data root:       {data_root()}")
    print(f"  PHI writes to:   {data_root()}/CONTAINS_PHI/")
    print(f"  Shareable to:    {data_root()}/NO_PHI__shareable/")
    print(f"  Public API denylist: ACTIVE ({len(FORBIDDEN_HOSTS)} hosts blocked)")
    print("=" * 50)


# Display helpers — optional rollups after run_cohort. Quickstart calls all three.

def view_results(result: CohortResult, settings: CohortSettings) -> None:
    """Print every extracted variable's final value for every patient."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import (
        extract_output_dir,
        phi_patient_run_dir,
    )
    print("RESULTS")
    print("=" * 50)
    show = _stdout_values_allowed()
    for pid in result.ingested:
        result_dir = extract_output_dir(phi_patient_run_dir(result.run_id, pid))
        for var in settings.variables:
            rf = result_dir / var / "result.json"
            if not rf.exists():
                print(f"  {pid} / {var}: not yet extracted")
                continue
            print(f"  {pid} / {var}:")
            if show:
                print(f"    {json.dumps(json.loads(rf.read_text()), indent=4)}")
            else:
                # the result envelope is patient-derived PHI; print only on opt-in.
                print("    <redacted — set JR_SHOW_STDOUT_VALUES=1 to display PHI values>")
    print("=" * 50)


def check_phi_containment(settings: CohortSettings) -> None:
    """List every file in NO_PHI__shareable so a human can eyeball that
    nothing patient-identifiable leaked."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import data_root
    phi_dir, no_phi_dir = data_root() / "CONTAINS_PHI", data_root() / "NO_PHI__shareable"
    phi_files = list(phi_dir.rglob("*")) if phi_dir.exists() else []
    no_phi_files = list(no_phi_dir.rglob("*")) if no_phi_dir.exists() else []
    print("PHI CONTAINMENT CHECK")
    print("=" * 50)
    print(f"  CONTAINS_PHI:       {sum(1 for f in phi_files if f.is_file())} files")
    print(f"  NO_PHI__shareable:  {sum(1 for f in no_phi_files if f.is_file())} files")
    print("\n  NO_PHI files (verify none contain patient text):")
    for f in sorted(no_phi_files):
        if f.is_file():
            print(f"    {f.relative_to(no_phi_dir)}")
    print("=" * 50)


def print_slurm_commands(settings: CohortSettings, run_id: str) -> None:
    """sbatch commands to scale this cohort on a SLURM cluster. Settings here do NOT
    transfer automatically — cluster runs use the per-environment configs
    under deployment/."""
    from jr_pipeline.runtime_infrastructure.data_directory_layout_and_safe_writes import data_root
    ip = Path(settings.input_folder)
    all_patients = sorted([
        p.name for p in ip.iterdir() if p.is_dir() and not p.name.startswith(".")
    ]) if ip.is_dir() else _resolve_patients(settings)

    patient_list_path = data_root() / "CONTAINS_PHI" / f"patient_list_{run_id}.txt"
    patient_list_path.parent.mkdir(parents=True, exist_ok=True)
    patient_list_path.write_text("\n".join(all_patients) + "\n", encoding="utf-8")
    print(f"Patient list written to: {patient_list_path}")

    slurm = "deployment/Local_SLURM_Cluster/slurm"
    n = len(all_patients)
    print(f"\n{'=' * 60}")
    print("SLURM SUBMISSION COMMANDS (run on the cluster, from the repo root)")
    print(f"{'=' * 60}")
    print(f"""# Adjust --partition/--time/--mem and cluster paths first.

export JR_RUN_ID={run_id}
export JR_CODE_ROOT=/shared/<lab>/<project>/code   # the repo checkout on the cluster
export JR_DATA_ROOT=/shared/<lab>/<project>/data
export JR_OUT_ROOT=$JR_DATA_ROOT
export JR_SOURCE_ROOT=/shared/<lab>/<project>/raw
export PATIENT_LIST=/shared/<lab>/<project>/patient_list_{run_id}.txt

# Each stage cd's to $JR_CODE_ROOT and ensures logs/ exists first: the scripts'
# #SBATCH --output=logs/ paths are relative to the submit directory, so submitting
# from anywhere without logs/ kills every array task with no captured stderr.

# Seal once on the login node, then fan out:
cd "$JR_CODE_ROOT" && jr-pipeline seal --config {slurm}/ingest_override.yaml

INGEST_JOB=$(cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch --parsable --array=1-{n}%20 \\
  --export=ALL,CFG={slurm}/ingest_override.yaml,PATIENT_LIST=$PATIENT_LIST \\
  {slurm}/ingest.sh)

EMBED_JOB=$(cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch --parsable --array=1-{n}%20 --dependency=afterok:$INGEST_JOB \\
  --export=ALL,CFG={slurm}/embed_override.yaml,PATIENT_LIST=$PATIENT_LIST \\
  {slurm}/embed.sh)

INDEX_JOB=$(cd "$JR_CODE_ROOT" && mkdir -p logs && sbatch --parsable --array=1-{n}%20 --dependency=afterok:$EMBED_JOB \\
  --export=ALL,CFG={slurm}/index_override.yaml,PATIENT_LIST=$PATIENT_LIST \\
  {slurm}/index.sh)

# Extract (steps 4–7 per patient): extract.sh not written yet — run via CLI per patient.""")
    print(f"{'=' * 60}")

"""CPU-only contract checks for the versioned phase-two controller."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

import scripts.autoresearch_full_training_phase2_controller as phase2


ROOT = Path(__file__).resolve().parents[1]


def _session_for_resources(profile_name: str) -> dict:
    profile = phase2.PROFILES[profile_name]
    return {
        "pilot_max_analysis_trials": 2,
        "pilot_execution_wall_seconds": phase2.PHASE_SECONDS,
        "pilot_wall_seconds_per_job": phase2.ARM_SECONDS,
        "weekly_usage_stop_used_percent": 50,
        "weekly_snapshot_max_age_seconds": 120,
        "max_parallel_gpu_jobs": 1,
        "started_at": "2026-10-23T00:00:00+00:00",
        "resource_limits": {
            "phase_wall_seconds": phase2.PHASE_SECONDS,
            "per_arm_wall_seconds": phase2.ARM_SECONDS,
            "host_rss_limit_mb": phase2.RSS_MB,
            "cuda_allocator_fraction": phase2.GPU_FRACTION,
            "main_cpu_threads": 4,
            "dataloader_workers": profile["workers"],
            "worker_cpu_threads": 1,
            "total_cpu_execution_slots": profile["cpu_slots"],
            "terminate_grace_seconds": phase2.GRACE_SECONDS,
            "phase_output_bytes": phase2.PHASE_BYTES,
            "per_arm_output_bytes": phase2.ARM_BYTES,
        },
    }


@pytest.mark.parametrize("profile_name", tuple(phase2.PROFILES))
def test_resource_profiles_accept_exact_cpu_budget_and_reject_drift(profile_name):
    profile = phase2.PROFILES[profile_name]
    session = _session_for_resources(profile_name)
    accepted = phase2._validate_resources(session, profile, allow_preparation=True)
    assert accepted["phase_seconds_remaining"] > 0

    for field, bad_value in (
        ("dataloader_workers", profile["workers"] + 1),
        ("total_cpu_execution_slots", profile["cpu_slots"] + 1),
        ("per_arm_wall_seconds", phase2.ARM_SECONDS - 1),
        ("host_rss_limit_mb", phase2.RSS_MB + 1),
        ("cuda_allocator_fraction", 1.0),
    ):
        changed = deepcopy(session)
        changed["resource_limits"][field] = bad_value
        with pytest.raises(phase2.PhaseControllerError, match="resource limit"):
            phase2._validate_resources(changed, profile, allow_preparation=True)


def test_quota_and_phase_bounds_are_not_relaxed():
    session = _session_for_resources("prefetch-fresh-v1")
    for field, value in (
        ("weekly_usage_stop_used_percent", 51),
        ("weekly_snapshot_max_age_seconds", 121),
        ("pilot_execution_wall_seconds", phase2.PHASE_SECONDS + 1),
    ):
        changed = deepcopy(session)
        changed[field] = value
        with pytest.raises(phase2.PhaseControllerError):
            phase2._validate_resources(changed, phase2.PROFILES["prefetch-fresh-v1"], allow_preparation=True)


def test_prefetch_backend_metadata_is_exact_and_resource_backend_stays_workers_zero():
    import autoresearch_full_training_prefetch as prefetch

    metadata = prefetch.execution_backend_metadata()
    phase2.validate_backend_metadata("prefetch-fresh-v1", metadata)
    for key, value in (("workers_per_objective_loader", 0), ("loader_worker_processes", 6),
                       ("declared_cpu_execution_slots", 4), ("persistent_workers", True)):
        changed = deepcopy(metadata)
        changed[key] = value
        with pytest.raises(phase2.PhaseControllerError, match="prefetch backend metadata"):
            phase2.validate_backend_metadata("prefetch-fresh-v1", changed)

    phase2.validate_backend_metadata("resource-only-resume-v1", {"id": "full-pretraining-workers0-v1", "workers": 0})
    with pytest.raises(phase2.PhaseControllerError, match="workers=0"):
        phase2.validate_backend_metadata("resource-only-resume-v1", {"id": "objective-loader-prefetch-v1", "workers": 1})


def test_prefetch_command_requires_root_resolved_absolute_anchor():
    session, jobs, _job_paths = _load_preparation_contract()
    anchor_value = session["input_identities"]["initialization_anchor"]["path"]
    anchor_path = Path(anchor_value)
    if not anchor_path.is_absolute():
        anchor_path = ROOT / anchor_path
    expected = str(anchor_path.resolve(strict=True))
    profile = phase2.PROFILES["prefetch-fresh-v1"]

    job = deepcopy(jobs[0])
    worker_idx = job["command"].index("--initialization-anchor")
    job["command"][worker_idx + 1] = expected
    phase2._validate_command(ROOT, job, profile, "df2", anchor_value)

    job["command"][worker_idx + 1] = Path(anchor_value).as_posix()
    if Path(anchor_value).is_absolute():
        job["command"][worker_idx + 1] = Path(anchor_value).relative_to(ROOT).as_posix()
    with pytest.raises(phase2.PhaseControllerError, match="--initialization-anchor"):
        phase2._validate_command(ROOT, job, profile, "df2", anchor_value)


def test_workspace_writable_output_rejects_symlinks_but_hf_input_symlinks_resolve(tmp_path):
    root = tmp_path / "worktree"
    root.mkdir()
    target = root / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    output_link = root / "output-link.json"
    output_link.symlink_to(target)
    with pytest.raises(phase2.PhaseControllerError, match="may not be a symlink"):
        phase2._workspace_file(root, output_link, "writable output")

    # Exercise the controller's actual immutable-input validation with HF-style
    # symlinks into the local blob cache, while the output helper rejects links.
    old_session = json.loads((ROOT / "autoresearch/full_training_devicefix_continuation1_session.json").read_text())
    session = {"input_identities": deepcopy(old_session["input_identities"])}
    anchor_path = ROOT / "autoresearch/full-training/initialization-seed1804-prefetch-v1.json"
    anchor = json.loads(anchor_path.read_text())
    session["input_identities"]["initialization_anchor"] = {
        "path": str(anchor_path),
        "sha256": phase2.sha256_file(anchor_path),
    }

    cache_links = tmp_path / "hf-snapshots"
    cache_links.mkdir()
    assets = anchor["offline_backbones"]
    for path_key, hash_key, name in (
        ("bert_config_path", "bert_config_sha256", "config.json"),
        ("bert_weights_path", "bert_weights_sha256", "bert.safetensors"),
        ("vision_weights_path", "vision_weights_sha256", "vit.safetensors"),
    ):
        link = cache_links / name
        link.symlink_to(Path(assets[path_key]))
        assets[path_key] = str(link)
    tokenizer_link = tmp_path / "tokenizer-snapshot"
    tokenizer_link.symlink_to(Path(assets["tokenizer_path"]), target_is_directory=True)
    assets["tokenizer_path"] = str(tokenizer_link)

    driver = phase2._workspace_file(ROOT, phase2.PREFETCH_DRIVER, "prefetch driver")
    closure = phase2._source_paths(driver)
    source_map = {name: phase2.sha256_file(ROOT / name) for name in closure}
    source_map["full_training_driver"] = source_map[phase2.PREFETCH_DRIVER]
    anchor["source_sha256"] = source_map
    session["frozen_files_sha256"] = dict(source_map)

    phase2._validate_anchor_inputs(ROOT, session, anchor, phase2.PROFILES["prefetch-fresh-v1"])

    anchor["offline_backbones"]["bert_config_sha256"] = "0" * 64
    with pytest.raises(phase2.PhaseControllerError, match="offline input hash mismatch"):
        phase2._validate_anchor_inputs(ROOT, session, anchor, phase2.PROFILES["prefetch-fresh-v1"])


def _load_preparation_contract() -> tuple[dict, list[dict], list[Path]]:
    session_path = ROOT / "autoresearch/full_training_prefetch_phase2_session.json"
    job_paths = [
        ROOT / "autoresearch/jobs/full-training-df2-seed1804-prefetch-phase2.json",
        ROOT / "autoresearch/jobs/full-training-rev-seed1804-prefetch-phase2.json",
    ]
    missing = [path for path in [session_path, *job_paths] if not path.is_file()]
    if missing:
        pytest.skip("phase2 preparation contracts are not published yet")
    session = json.loads(session_path.read_text(encoding="utf-8"))
    jobs = [json.loads(path.read_text(encoding="utf-8")) for path in job_paths]
    return session, jobs, job_paths


def test_frozen_prefetch_preparation_contract_and_dry_run_accept_wrapper_alias():
    session, jobs, job_paths = _load_preparation_contract()
    frozen, _budget = phase2.validate_contract(
        ROOT, session, jobs, job_paths, profile_name="prefetch-fresh-v1", allow_preparation=True
    )
    assert frozen[phase2.PREFETCH_DRIVER] == phase2.sha256_file(ROOT / phase2.PREFETCH_DRIVER)
    anchor = json.loads((ROOT / session["input_identities"]["initialization_anchor"]["path"]).read_text())
    assert anchor["source_sha256"]["full_training_driver"] == frozen[phase2.PREFETCH_DRIVER]

    # Dry run uses the actual guarded integration path, but does not write state
    # or launch a runner/GPU job.
    result = phase2.run_phase_controller(
        workspace_root=ROOT,
        session_path="autoresearch/full_training_prefetch_phase2_session.json",
        job_paths=[
            "autoresearch/jobs/full-training-df2-seed1804-prefetch-phase2.json",
            "autoresearch/jobs/full-training-rev-seed1804-prefetch-phase2.json",
        ],
        state_path="autoresearch/full-training/controller-state-prefetch-phase2.json",
        profile_name="prefetch-fresh-v1",
        execute=False,
        resume=False,
    )
    assert result["status"] == "ready"
    assert list(result["job_order"]) == [job["trial_id"] for job in jobs]


@pytest.mark.parametrize(
    "mutation, message",
    [
        ("recipe", "scientific recipe"),
        ("workers", "dataloader_workers"),
        ("quota", "quota"),
        ("backend", "backend"),
        ("command_worker", "--workers"),
        ("review_status", "acceptance"),
    ],
)
def test_frozen_prefetch_contract_rejects_changed_execution_gates(mutation, message):
    session, jobs, job_paths = _load_preparation_contract()
    session = deepcopy(session)
    jobs = deepcopy(jobs)
    if mutation == "recipe":
        session["scientific_recipe"]["batch_size"] += 1
    elif mutation == "workers":
        session["resource_limits"]["dataloader_workers"] = 0
    elif mutation == "quota":
        session["weekly_usage_stop_used_percent"] = 51
    elif mutation == "backend":
        jobs[0]["execution_backend"]["id"] = "full-pretraining-workers0-v1"
    elif mutation == "command_worker":
        command = jobs[0]["command"]
        worker_idx = command.index("--workers")
        command[worker_idx + 1] = "0"
    elif mutation == "review_status":
        session["prefetch_input_review"]["status"] = "pending"

    with pytest.raises(phase2.PhaseControllerError, match=message):
        phase2.validate_contract(
            ROOT, session, jobs, job_paths, profile_name="prefetch-fresh-v1", allow_preparation=True
        )

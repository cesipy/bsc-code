import datetime as dt
import json
from pathlib import Path

import pytest

import scripts.autoresearch_full_training_controller_repair1 as controller


def _job(root, trial_id, arm, *, device="cuda:0"):
    output_root = root / "autoresearch/full-training"
    run_dir = output_root / trial_id
    command = [
        "python", "src/autoresearch_full_training.py", "train",
        "--arm", arm, "--seed", "1804", "--run-id", trial_id,
        "--output-root", str(output_root), "--initialization-anchor",
        str(root / controller.ANCHOR_PATH), "--device", device, "--workers", "0",
    ]
    return {
        "trial_id": trial_id,
        "run_id": trial_id,
        "seed": 1804,
        "mode": "training",
        "uses_gpu": True,
        "timeout_seconds": 259200,
        "gpu_wall_limit_seconds": 259200,
        "memory_limit_mb": 16384,
        "disk_limit_mb": 40960,
        "run_directory_cap_mb": 20480,
        "terminate_grace_seconds": 180,
        "min_free_gpu_mb": 16000,
        "command": command,
        "expected_result_path": str(run_dir / "result.json"),
        "metrics_path": str(run_dir / "result.json"),
        "initialization_anchor": str(root / controller.ANCHOR_PATH),
        "expected_checkpoint_path": str(run_dir / "resume_latest.pt"),
        "env": {name: "4" for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
    }


def _session(session_id="repair-session"):
    now = dt.datetime.now(dt.timezone.utc).isoformat()
    return {
        "session_id": session_id,
        "status": "active",
        "execution_started_at": now,
        "weekly_usage_stop_used_percent": 50,
        "weekly_snapshot_max_age_seconds": 120,
        "max_parallel_gpu_jobs": 1,
        "pilot_max_analysis_trials": 48,
        "pilot_execution_wall_seconds": 604800,
        "global_trial_count_at_preparation": 46,
        "trial_order": list(controller.EXPECTED_JOBS),
        "repair_status": "approved_by_root",
        "preupdate_repair_reserve": 1,
        "repair_reserve_used": 1,
        "repair_reserve_total": 0,
        "repair_authorization": {
            "approved_by": "root",
            "max_attempts": 1,
            "repair_of": controller.PREVIOUS_FAILED_DF2,
            "scope": controller.REPAIR_SCOPE,
        },
        "resource_limits": {
            "phase_wall_seconds": 604800,
            "per_arm_wall_seconds": 259200,
            "host_rss_limit_mb": 16384,
            "cuda_allocator_fraction": 0.90,
            "cpu_threads": 4,
            "dataloader_workers": 0,
            "terminate_grace_seconds": 180,
            "phase_output_bytes": 40 * 1024**3,
            "per_arm_output_bytes": 20 * 1024**3,
            "minimum_free_gpu_mb": 16000,
        },
        "protocol_sha256": "protocol-hash",
        "input_identities": {"initialization_anchor": {"sha256": "anchor-hash"}},
    }


def _contract_fixture(tmp_path, *, device="cuda:0"):
    (tmp_path / controller.ANCHOR_PATH).parent.mkdir(parents=True)
    (tmp_path / controller.ANCHOR_PATH).write_text("anchor")
    repair = _job(tmp_path, controller.EXPECTED_JOBS[0], "df2", device=device)
    repair.update({
        "repair_of": controller.PREVIOUS_FAILED_DF2,
        "repair_authorization": {
            "approved_by": "root",
            "max_attempts": 1,
            "scope": controller.REPAIR_SCOPE,
        },
    })
    rev = _job(tmp_path, controller.EXPECTED_JOBS[1], "rev")
    rev["repair_session_id"] = "repair-session"
    paths = []
    for index, job in enumerate((repair, rev)):
        path = tmp_path / "autoresearch/jobs" / f"job{index}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(job))
        paths.append(path)
    return _session(), [repair, rev], paths


def test_repair_contract_accepts_cuda_zero_and_rejects_original_trial_id(tmp_path, monkeypatch):
    session, jobs, paths = _contract_fixture(tmp_path)
    monkeypatch.setattr(
        controller,
        "_validate_frozen_files",
        lambda *_args: {controller.PROTOCOL_PATH: "protocol-hash", controller.ANCHOR_PATH: "anchor-hash"},
    )
    frozen, _ = controller._check_session_and_jobs(tmp_path, session, jobs, paths)
    assert frozen[controller.PROTOCOL_PATH] == "protocol-hash"

    jobs[0]["trial_id"] = controller.PREVIOUS_FAILED_DF2
    with pytest.raises(controller.ControllerError, match="ordered repair1 DF2"):
        controller._check_session_and_jobs(tmp_path, session, jobs, paths)


def test_repair_contract_rejects_unindexed_device(tmp_path, monkeypatch):
    session, jobs, paths = _contract_fixture(tmp_path, device="cuda")
    monkeypatch.setattr(
        controller,
        "_validate_frozen_files",
        lambda *_args: {controller.PROTOCOL_PATH: "protocol-hash", controller.ANCHOR_PATH: "anchor-hash"},
    )
    with pytest.raises(controller.ControllerError, match="runtime identity"):
        controller._check_session_and_jobs(tmp_path, session, jobs, paths)


def test_consumed_repair_budget_allows_only_two_remaining_trial_ids():
    session = _session()
    used = {f"previous-{index}" for index in range(45)} | {controller.PREVIOUS_FAILED_DF2}
    controller._check_repair_trial_budget(session, used)
    controller._check_repair_trial_budget(session, used | {controller.EXPECTED_JOBS[0]})
    with pytest.raises(controller.ControllerError, match=r"46\+2=48"):
        controller._check_repair_trial_budget(session, used | {"unauthorized-third-trial"})


def test_original_failed_trial_must_be_exactly_one_preupdate_cuda_error(tmp_path):
    failure = "ValueError: Expected a torch.device with a specified index or an integer, but got:cuda"
    result_path = tmp_path / "autoresearch/full-training/original/result.json"
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({
        "status": "failed",
        "run_id": controller.PREVIOUS_FAILED_DF2,
        "failure_reason": failure,
        "latest_checkpoint_path": None,
    }))
    launch = {
        "event": "run_launching",
        "trial_id": controller.PREVIOUS_FAILED_DF2,
        "run_id": "old-run-1",
        "command": ["python", "driver", "--device", "cuda"],
    }
    finish = {
        "event": "run_finished",
        "trial_id": controller.PREVIOUS_FAILED_DF2,
        "run_id": "old-run-1",
        "session_id": "old-session",
        "status": "crash",
        "reason": "nonzero_exit",
        "metrics": {
            "status": "failed",
            "run_id": controller.PREVIOUS_FAILED_DF2,
            "arm": "df2",
            "seed": 1804,
            "failure_reason": failure,
            "latest_checkpoint_path": None,
        },
    }
    ledger = tmp_path / "autoresearch/results.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("\n".join(json.dumps(row) for row in (launch, finish)) + "\n")
    session = {
        "failed_attempt_reference": {
            "trial_id": controller.PREVIOUS_FAILED_DF2,
            "status": "failed_before_model_initialization",
            "failure": failure,
            "optimizer_updates": 0,
            "checkpoint_written": False,
            "result_path": str(result_path.relative_to(tmp_path)),
            "session_id": "old-session",
        }
    }
    controller._validate_preupdate_repair_evidence(tmp_path, session)

    ledger.write_text(ledger.read_text() + json.dumps(launch) + "\n")
    with pytest.raises(controller.ControllerError, match="exactly one preserved launch"):
        controller._validate_preupdate_repair_evidence(tmp_path, session)

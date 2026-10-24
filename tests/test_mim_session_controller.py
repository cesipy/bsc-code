from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from scripts.mim_session_controller import ControllerError, validate_next_job


TRIALS = [
    "mim-seed1801-legacy",
    "mim-seed1801-unit_scale",
    "mim-seed1802-unit_scale",
    "mim-seed1802-legacy",
    "mim-seed1803-legacy",
    "mim-seed1803-unit_scale",
]


def _session(now: datetime) -> dict:
    return {
        "session_id": "mim-test",
        "status": "active",
        "execution_started_at": now.isoformat(),
        "pilot_max_analysis_trials": 38,
        "pilot_execution_wall_seconds": 5400,
        "mim_session_control": {
            "predeclared_trial_order": TRIALS,
            "cumulative_gpu_wall_limit_seconds": 3600,
            "minimum_gpu_wall_remaining_seconds": 600,
            "max_artifact_bytes": 10 * 1024**3,
            "max_job_artifact_bytes": 1536 * 1024**2,
            "shared_artifact_paths": [
                "autoresearch/training-study/seed-initializations",
                "autoresearch/verification/study-input-preflight-v2.json",
            ],
        },
    }


def _job(trial_id: str, *, repair_of: str | None = None, disk_mb: int = 1536) -> dict:
    output_root = f"autoresearch/trials/{trial_id}"
    job = {
        "trial_id": trial_id,
        "mode": "training",
        "command": [
            "python",
            "src/autoresearch_training_study.py",
            "--output-root",
            output_root,
            "--run-id",
            "payload",
            "--initialization-anchor-root",
            "autoresearch/training-study/seed-initializations",
            "--paired-input-artifact",
            "autoresearch/verification/study-input-preflight-v2.json",
        ],
        "cwd": ".",
        "timeout_seconds": 600,
        "memory_limit_mb": 8192,
        "disk_limit_mb": disk_mb,
        "artifact_dir": output_root,
        "stdout_log": f"{output_root}/stdout.log",
        "metrics_path": None,
        "uses_gpu": True,
        "gpu_wall_limit_seconds": 600,
        "min_free_gpu_mb": 16000,
    }
    if repair_of:
        job["repair_of"] = repair_of
    return job


def _events_for_run(
    session_id: str,
    trial_id: str,
    run_id: str,
    *,
    wall: float,
    status: str = "completed",
    artifact_bytes: int = 0,
    repair_of: str | None = None,
) -> list[dict]:
    return [
        {
            "event": "run_launching",
            "session_id": session_id,
            "trial_id": trial_id,
            "run_id": run_id,
            "uses_gpu": True,
            "command": [
                "python",
                "--output-root",
                f"autoresearch/trials/{trial_id}",
                "--run-id",
                "payload",
            ],
        },
        {
            "event": "quota_check",
            "session_id": session_id,
            "trial_id": trial_id,
            "run_id": run_id,
            "repair_of": repair_of,
        },
        {
            "event": "run_finished",
            "session_id": session_id,
            "trial_id": trial_id,
            "run_id": run_id,
            "status": status,
            "observed_resources": {
                "gpu_wall_seconds": wall,
                "artifact_bytes": artifact_bytes,
            },
        },
    ]


def _write_result(root: Path, trial_id: str) -> None:
    path = root / "autoresearch" / "trials" / trial_id / "payload" / "result.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('{"status": "completed"}', encoding="utf-8")


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "autoresearch").mkdir(parents=True)
    preflight = root / "autoresearch" / "verification" / "study-input-preflight-v2.json"
    preflight.parent.mkdir()
    preflight.touch()
    return root


def test_accepts_exact_next_trial_and_returns_bounded_spec(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    root = _workspace(tmp_path)
    session = _session(now)
    job = _job(TRIALS[0])

    bounded, summary = validate_next_job(
        session, job, [], workspace_root=root, now=now, executing=True
    )

    assert bounded["disk_limit_mb"] == 1536
    assert summary["trial_id"] == TRIALS[0]
    assert summary["session_gpu_wall_seconds_remaining"] == 3600


def test_rejects_out_of_order_or_skipped_planned_job(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    with pytest.raises(ControllerError, match="out-of-order"):
        validate_next_job(
            _session(now), _job(TRIALS[1]), [], workspace_root=_workspace(tmp_path), now=now
        )


def test_includes_failed_original_and_repair_wall_before_next_job(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    root = _workspace(tmp_path)
    evidence = root / "autoresearch" / "repair-evidence.json"
    payload = json.dumps({"diagnosis": "import error before update"}, sort_keys=True).encode()
    evidence.write_bytes(payload)
    session = _session(now)
    session["repair_authorizations"] = {
        TRIALS[0]: {
            "repair_trial_id": f"{TRIALS[0]}-repair1",
            "authorized_by": "/root",
            "failure_class": "import",
            "optimizer_updates_completed": 0,
            "evidence_path": "autoresearch/repair-evidence.json",
            "evidence_sha256": hashlib.sha256(payload).hexdigest(),
        }
    }
    events = _events_for_run(
        "mim-test", TRIALS[0], "failed", wall=1500, status="crash", artifact_bytes=20
    )
    events += _events_for_run(
        "mim-test",
        f"{TRIALS[0]}-repair1",
        "repair",
        wall=1500,
        status="completed",
        artifact_bytes=30,
        repair_of=TRIALS[0],
    )
    _write_result(root, f"{TRIALS[0]}-repair1")

    bounded, summary = validate_next_job(
        session, _job(TRIALS[1]), events, workspace_root=root, now=now, executing=True
    )

    assert bounded["trial_id"] == TRIALS[1]
    assert summary["session_gpu_wall_seconds_used"] == 3000
    assert summary["session_gpu_wall_seconds_remaining"] == 600
    assert summary["session_artifact_bytes_used"] == 50


def test_failed_run_blocks_next_trial_without_parent_repair_authorization(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    events = _events_for_run(
        "mim-test", TRIALS[0], "failed", wall=12, status="crash"
    )
    with pytest.raises(ControllerError, match="failed run blocks"):
        validate_next_job(
            _session(now), _job(TRIALS[1]), events, workspace_root=_workspace(tmp_path), now=now
        )


def test_exactly_600_cumulative_seconds_allows_launch_but_less_does_not(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    root = _workspace(tmp_path)
    events = _events_for_run(
        "mim-test", TRIALS[0], "run0", wall=3000, status="completed"
    )
    _write_result(root, TRIALS[0])
    # Progress is seeded through the first planned trial; its wall counts.
    _, summary = validate_next_job(
        _session(now), _job(TRIALS[1]), events, workspace_root=root, now=now, executing=True
    )
    assert summary["session_gpu_wall_seconds_remaining"] == 600

    events[-1]["observed_resources"]["gpu_wall_seconds"] = 3000.001
    with pytest.raises(ControllerError, match="less than 600 cumulative"):
        validate_next_job(
            _session(now), _job(TRIALS[1]), events, workspace_root=root, now=now, executing=True
        )


def test_unfinished_previous_run_fails_closed(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    event = {
        "event": "run_launching",
        "session_id": "mim-test",
        "trial_id": TRIALS[0],
        "run_id": "unclosed",
    }
    with pytest.raises(ControllerError, match="unfinished or uncertain"):
        validate_next_job(
            _session(now), _job(TRIALS[1]), [event], workspace_root=_workspace(tmp_path), now=now
        )


def test_phase_wall_requires_at_least_one_job_allowance(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    session = _session(now - timedelta(seconds=4801))
    with pytest.raises(ControllerError, match="phase-wall"):
        validate_next_job(
            session,
            _job(TRIALS[0]),
            [],
            workspace_root=_workspace(tmp_path),
            now=now,
            executing=True,
        )


def test_storage_budget_clamps_job_and_preserves_prior_failed_artifacts(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    root = _workspace(tmp_path)
    session = _session(now)
    used = 10 * 1024**3 - 2 * 1024**2
    events = _events_for_run(
        "mim-test", TRIALS[0], "run0", wall=1, status="crash", artifact_bytes=used
    )
    # Failed outputs remain counted and require adjudication; authorize a repair
    # so the next proposed spec can be checked against the remaining bytes.
    evidence = root / "autoresearch" / "repair-evidence.json"
    evidence.write_text('{"optimizer_updates_completed": 0}', encoding="utf-8")
    evidence_bytes = evidence.read_bytes()
    session["repair_authorizations"] = {
        TRIALS[0]: {
            "repair_trial_id": f"{TRIALS[0]}-repair1",
            "authorized_by": "/root",
            "failure_class": "configuration",
            "optimizer_updates_completed": 0,
            "evidence_path": "autoresearch/repair-evidence.json",
            "evidence_sha256": hashlib.sha256(evidence_bytes).hexdigest(),
        }
    }
    anchor = root / "autoresearch" / "training-study" / "seed-initializations" / "seed-1801.json"
    anchor.parent.mkdir(parents=True)
    anchor.write_bytes(b"a" * 100)
    events += _events_for_run(
        "mim-test",
        f"{TRIALS[0]}-repair1",
        "repair",
        wall=1,
        status="completed",
        repair_of=TRIALS[0],
    )
    _write_result(root, f"{TRIALS[0]}-repair1")
    bounded, summary = validate_next_job(
        session, _job(TRIALS[1]), events, workspace_root=root, now=now, executing=True
    )

    assert summary["session_artifact_bytes_used"] == used + 100
    assert summary["shared_artifact_bytes_used"] == 100
    assert bounded["disk_limit_mb"] == 1
    assert summary["effective_job_disk_limit_mb"] == 1


def test_bad_repair_evidence_hash_is_rejected(tmp_path: Path) -> None:
    now = datetime.now(timezone.utc)
    root = _workspace(tmp_path)
    evidence = root / "autoresearch" / "evidence.json"
    evidence.write_text("{}", encoding="utf-8")
    session = _session(now)
    session["repair_authorizations"] = {
        TRIALS[0]: {
            "repair_trial_id": f"{TRIALS[0]}-repair1",
            "authorized_by": "/root",
            "failure_class": "launch",
            "optimizer_updates_completed": 0,
            "evidence_path": "autoresearch/evidence.json",
            "evidence_sha256": "0" * 64,
        }
    }
    events = _events_for_run(
        "mim-test", TRIALS[0], "failed", wall=1, status="crash"
    )
    with pytest.raises(ControllerError, match="hash does not match"):
        validate_next_job(
            session,
            _job(f"{TRIALS[0]}-repair1", repair_of=TRIALS[0]),
            events,
            workspace_root=root,
            now=now,
        )

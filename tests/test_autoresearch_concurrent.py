from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

from scripts.autoresearch_concurrent import (
    ConcurrentRunnerError,
    run_concurrent_job,
)
from scripts.autoresearch_coordination import Registry


UTC = dt.timezone.utc


@pytest.fixture(autouse=True)
def one_cpu_affinity():
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        yield
        return
    original = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(original)})
    try:
        yield
    finally:
        os.sched_setaffinity(0, original)


def _workspace(tmp_path: Path) -> Path:
    root = tmp_path / "workspace"
    (root / "autoresearch").mkdir(parents=True)
    policy = {
        "weekly_usage_stop_used_percent": 100,
        "snapshot_max_age_seconds": None,
        "stop_on_unavailable": False,
        "usage_readings_required": False,
        "enforce_usage_freshness": False,
    }
    (root / "autoresearch" / "active_usage_policy.json").write_text(
        json.dumps(policy), encoding="utf-8"
    )
    return root


def _session(session_id: str = "test-session", *, now: dt.datetime | None = None) -> dict[str, Any]:
    current = now or dt.datetime.now(UTC)
    return {
        "session_id": session_id,
        "status": "active",
        "host": None,
        "boot_id": None,
        "execution_started_at": (current - dt.timedelta(seconds=1)).isoformat(),
        "execution_allowance_seconds": 90,
        "pilot_execution_wall_seconds": 90,
        "pilot_max_analysis_trials": 4,
        "weekly_usage_stop_used_percent": 100,
        "weekly_snapshot_max_age_seconds": 120,
        "max_parallel_gpu_jobs": 1,
        "pilot_training_enabled": True,
        "read_only_roots": [],
        "phase_expires_at": (current + dt.timedelta(minutes=5)).isoformat(),
    }


def _job(
    root: Path,
    trial_id: str = "trial-1",
    *,
    command: list[str] | None = None,
    expected_metrics: list[str] | None = None,
) -> dict[str, Any]:
    artifact = root / "autoresearch" / "concurrency" / "runs" / trial_id
    return {
        "trial_id": trial_id,
        "mode": "analysis",
        "command": command
        or [
            sys.executable,
            "-c",
            "import json,os,pathlib; p=pathlib.Path(os.environ['AUTORESEARCH_ARTIFACT_DIR'])/'metrics.json'; p.write_text(json.dumps({'score': 1, 'visible': os.environ['CUDA_VISIBLE_DEVICES'], 'threads': os.environ['OMP_NUM_THREADS'], 'affinity': len(os.sched_getaffinity(0))}))",
        ],
        "cwd": str(root),
        "timeout_seconds": 10,
        "memory_limit_mb": 1024,
        "disk_limit_mb": 32,
        "artifact_dir": str(artifact),
        "stdout_log": str(artifact / "stdout.log"),
        "metrics_path": str(artifact / "metrics.json"),
        "expected_metrics": expected_metrics if expected_metrics is not None else ["score"],
        "expected_metric_values": {},
        "uses_gpu": False,
        "cpu_threads": 1,
    }


def _last_run_finish(ledger: Path) -> dict[str, Any]:
    records = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    finishes = [item for item in records if item.get("event") == "run_finished"]
    return finishes[-1]


def test_cpu_child_runs_in_private_session_with_strict_metrics(tmp_path: Path):
    root = _workspace(tmp_path)
    session = _session()
    job = _job(root)
    job["expected_metric_values"] = {"score": 1}

    result = run_concurrent_job(session, job, workspace_root=root, dry_run=False)

    assert result["status"] == "completed"
    assert result["selected_device"] == "cpu"
    metrics = json.loads(Path(job["metrics_path"]).read_text(encoding="utf-8"))
    assert metrics["visible"] == ""
    assert metrics["threads"] == "1"
    assert metrics["affinity"] == 1
    session_dir = root / "autoresearch" / "concurrency" / "sessions" / session["session_id"]
    assert (session_dir / "results.jsonl").is_file()
    assert (session_dir / "handoff.json").is_file()
    assert not (root / "autoresearch" / "results.jsonl").exists()
    assert not (root / "autoresearch" / "handoff.json").exists()
    assert Registry(root).snapshot()["claims"][0]["status"] == "completed"


def test_missing_required_metric_is_invalid_and_cannot_complete_claim(tmp_path: Path):
    root = _workspace(tmp_path)
    session = _session()
    job = _job(
        root,
        command=[
            sys.executable,
            "-c",
            "import json,os,pathlib; (pathlib.Path(os.environ['AUTORESEARCH_ARTIFACT_DIR'])/'metrics.json').write_text(json.dumps({'other': 1}))",
        ],
    )

    result = run_concurrent_job(session, job, workspace_root=root, dry_run=False)

    assert result["status"] == "invalid"
    finish = _last_run_finish(
        root / "autoresearch" / "concurrency" / "sessions" / session["session_id"] / "results.jsonl"
    )
    assert finish["status"] == "invalid"
    claim = Registry(root).snapshot()["claims"][0]
    assert claim["status"] == "failed"


def test_timeout_cleans_owned_cpu_child_and_records_failure(tmp_path: Path):
    root = _workspace(tmp_path)
    session = _session()
    job = _job(
        root,
        command=[sys.executable, "-c", "import time; time.sleep(4)"],
        expected_metrics=[],
    )
    job["timeout_seconds"] = 0.25

    result = run_concurrent_job(session, job, workspace_root=root, dry_run=False)

    assert result["status"] == "timeout"
    assert result["observed_resources"]["exit_code"] is not None
    assert Registry(root).snapshot()["resources"] == []
    assert Registry(root).snapshot()["claims"][0]["status"] == "failed"


def test_dry_run_has_no_registry_claim_or_session_output(tmp_path: Path):
    root = _workspace(tmp_path)
    session = _session()
    job = _job(root)

    result = run_concurrent_job(session, job, workspace_root=root, dry_run=True)

    assert result["status"] == "dry_run"
    assert result["will_launch"] is True
    assert result["registry_claimed"] is False
    assert Registry(root).snapshot()["claims"] == []
    assert not (root / "autoresearch" / "results.jsonl").exists()
    assert not (root / "autoresearch" / "handoff.json").exists()
    assert not (root / "autoresearch" / "concurrency" / "sessions" / session["session_id"]).exists()


def test_device_visibility_selects_only_visible_uuid_and_ignores_other_gpu(tmp_path: Path, monkeypatch):
    root = _workspace(tmp_path)
    session = _session()
    job = _job(root)
    job.update(
        {
            "uses_gpu": True,
            "min_free_gpu_mb": 100,
            "gpu_wall_limit_seconds": 10,
        }
    )
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-B")
    probes: list[str | None] = []

    def probe(device_uuid: str | None):
        probes.append(device_uuid)
        if device_uuid is None:
            return {
                "status": "inventory",
                "devices": [
                    {"index": 0, "uuid": "GPU-A", "name": "A"},
                    {"index": 1, "uuid": "GPU-B", "name": "B"},
                ],
            }
        return {
            "status": "idle" if device_uuid == "GPU-B" else "busy",
            "uuid": device_uuid,
            "memory": [{"free_mb": 1000, "total_mb": 2000}],
        }

    result = run_concurrent_job(
        session, job, workspace_root=root, dry_run=True, gpu_probe_fn=probe
    )

    assert result["selected_device"] == "GPU-B"
    assert result["will_launch"] is True
    assert "GPU-A" not in probes
    assert probes.count("GPU-B") >= 2
    assert Registry(root).snapshot()["claims"] == []


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("host", "some-other-node", "host does not match"),
        ("phase_expires_at", "2000-01-01T00:00:00+00:00", "phase has expired"),
        ("weekly_usage_stop_used_percent", 50, "100% usage policy"),
    ],
)
def test_contract_guards_block_without_claims(tmp_path: Path, field, value, message):
    root = _workspace(tmp_path)
    session = _session()
    session[field] = value

    with pytest.raises(ConcurrentRunnerError, match=message):
        run_concurrent_job(session, _job(root), workspace_root=root, dry_run=False)

    assert Registry(root).snapshot()["claims"] == []


def test_source_pin_is_checked_before_and_after_run(tmp_path: Path):
    root = _workspace(tmp_path)
    session = _session()
    pinned = root / "frozen-source.json"
    pinned.write_text('{"version": 1}', encoding="utf-8")
    digest = hashlib.sha256(pinned.read_bytes()).hexdigest()
    session["frozen_files_sha256"] = {"frozen-source.json": digest}
    job = _job(
        root,
        command=[
            sys.executable,
            "-c",
            "import json,os,pathlib; root=pathlib.Path(os.environ['AUTORESEARCH_ARTIFACT_DIR']).parent.parent.parent.parent; (pathlib.Path(os.environ['AUTORESEARCH_ARTIFACT_DIR'])/'metrics.json').write_text(json.dumps({'score': 1})); (root/'frozen-source.json').write_text('changed')",
        ],
    )

    result = run_concurrent_job(session, job, workspace_root=root, dry_run=False)

    assert result["status"] == "invalid"
    assert "source_pin_changed_after_run" in result["reason"]
    assert Registry(root).snapshot()["claims"][0]["status"] == "failed"


def test_same_trial_is_blocked_globally_even_in_another_session(tmp_path: Path):
    root = _workspace(tmp_path)
    first_session = _session("session-one")
    first_job = _job(root, "globally-unique")
    first = run_concurrent_job(first_session, first_job, workspace_root=root, dry_run=False)
    assert first["status"] == "completed"

    second_session = _session("session-two")
    second_job = _job(root, "globally-unique")
    second_job["artifact_dir"] = str(root / "autoresearch" / "concurrency" / "runs" / "second-output")
    second_job["stdout_log"] = str(Path(second_job["artifact_dir"]) / "stdout.log")
    second_job["metrics_path"] = str(Path(second_job["artifact_dir"]) / "metrics.json")
    with pytest.raises(ConcurrentRunnerError, match="reservation failed"):
        run_concurrent_job(second_session, second_job, workspace_root=root, dry_run=False)

    assert len(Registry(root).snapshot()["claims"]) == 1


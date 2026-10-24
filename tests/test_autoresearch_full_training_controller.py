import datetime as dt
import hashlib
import json
from pathlib import Path
import threading

import pytest

import scripts.autoresearch_full_training_controller as controller


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n", encoding="utf-8")


def test_input_identity_scan_keeps_training_csv_when_training_feasibility_follows(tmp_path, monkeypatch):
    train = tmp_path / "train.csv"
    validation = tmp_path / "validation.csv"
    feasibility = tmp_path / "autoresearch/training_feasibility.json"
    train.write_text("train", encoding="utf-8")
    validation.write_text("validation", encoding="utf-8")
    feasibility.parent.mkdir(parents=True, exist_ok=True)
    feasibility.write_text("feasibility", encoding="utf-8")
    monkeypatch.setattr(controller, "EXPECTED_TRAIN_SHA256", _sha(train))
    monkeypatch.setattr(controller, "EXPECTED_VALIDATION_SHA256", _sha(validation))

    assets_dir = tmp_path / "offline"
    assets_dir.mkdir()
    asset_paths = {}
    for name in ("bert-config", "bert-weights", "vision-weights"):
        path = assets_dir / name
        path.write_text(name, encoding="utf-8")
        asset_paths[name] = path
    tokenizer_dir = assets_dir / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_hashes = {}
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt"):
        path = tokenizer_dir / name
        path.write_text(name, encoding="utf-8")
        tokenizer_hashes[name] = _sha(path)

    anchor = {
        "seed": 1804,
        "status": "passed",
        "initial_model_state_sha256_by_arm": {"df2": "same", "rev": "same"},
        "configs": {
            "df2": {"text_cross_attention_layers_zero_based": [3, 4, 5], "vision_cross_attention_layers_zero_based": [9, 10, 11]},
            "rev": {"text_cross_attention_layers_zero_based": [9, 10, 11], "vision_cross_attention_layers_zero_based": [3, 4, 5]},
        },
        "offline_backbones": {
            "bert_config_path": str(asset_paths["bert-config"]),
            "bert_config_sha256": _sha(asset_paths["bert-config"]),
            "bert_weights_path": str(asset_paths["bert-weights"]),
            "bert_weights_sha256": _sha(asset_paths["bert-weights"]),
            "vision_weights_path": str(asset_paths["vision-weights"]),
            "vision_weights_sha256": _sha(asset_paths["vision-weights"]),
            "tokenizer_path": str(tokenizer_dir),
            "tokenizer_files_sha256": tokenizer_hashes,
        },
    }
    anchor_path = tmp_path / controller.ANCHOR_PATH
    _json(anchor_path, anchor)
    session = {
        "input_identities": {
            "training_csv": {"path": str(train), "sha256": _sha(train)},
            "validation_csv": {"path": str(validation), "sha256": _sha(validation)},
            # This later entry contains "training" in its name but must not
            # overwrite the successful training_csv identity.
            "training_feasibility": {"path": "autoresearch/training_feasibility.json", "sha256": _sha(feasibility)},
            "initialization_anchor": {"path": controller.ANCHOR_PATH, "sha256": _sha(anchor_path)},
            "image_root": {"path": "/readonly/images", "content_hash_scope": "not globally hashed"},
        }
    }

    controller._check_input_files(tmp_path, session, anchor)


def _session_fixture(root):
    now = dt.datetime.now(dt.timezone.utc)
    session = {
        "session_id": "test-full-training",
        "status": "active",
        "execution_started_at": now.isoformat(),
        "pilot_max_analysis_trials": 48,
        "pilot_execution_wall_seconds": 604800,
        "weekly_usage_stop_used_percent": 50,
        "weekly_snapshot_max_age_seconds": 120,
        "max_parallel_gpu_jobs": 1,
        "global_trial_count_at_preparation": 45,
        "trial_order": list(controller.EXPECTED_JOBS),
        "resource_limits": {"per_arm_output_bytes": 20 * 1024**3},
    }
    session_path = root / "autoresearch/full_training_session.json"
    _json(session_path, session)
    jobs = []
    job_paths = []
    for trial, arm in zip(controller.EXPECTED_JOBS, ("df2", "rev")):
        job = {
            "trial_id": trial,
            "run_id": trial,
            "seed": 1804,
            "metrics_path": str(root / "autoresearch/full-training" / trial / "result.json"),
            "timeout_seconds": 259200,
            "run_directory_cap_mb": 20480,
            "command": ["python", "driver.py", "--output-root", "autoresearch/full-training"],
        }
        path = root / "autoresearch/jobs" / f"{trial}.json"
        _json(path, job)
        jobs.append(job)
        job_paths.append(path)
    ledger = root / "autoresearch/results.jsonl"
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text("".join(json.dumps({"event": "run_launching", "trial_id": f"old-{i}"}) + "\n" for i in range(45)))
    anchor_path = root / controller.ANCHOR_PATH
    _json(anchor_path, {})
    return session_path, jobs, job_paths


class _Quota:
    def check(self):
        return {"status": "continue", "max_age_seconds": 120}


def test_controller_runs_df2_then_rev_only_after_verified_gate(tmp_path, monkeypatch):
    session_path, jobs, job_paths = _session_fixture(tmp_path)
    monkeypatch.setattr(controller, "_check_input_files", lambda *_args: None)
    contract = lambda *_args: ({"frozen": "sha"}, {"phase_seconds_remaining": 604800})
    calls = []

    def runner(session, job, **_kwargs):
        calls.append(job["trial_id"])
        return {
            "status": "completed",
            "trial_id": job["trial_id"],
            "exit_code": 0,
            "observed_resources": {"peak_rss_mb": 1, "gpu_wall_seconds": 1, "artifact_bytes": 1},
            "quota": {"status": "continue"},
            "gpu_preflight": {"status": "idle"},
        }

    def validate(_root, job, _result, **_kwargs):
        return {"trial_id": job["trial_id"], "arm": controller.EXPECTED_ARMS[job["trial_id"]], "checkpoint_sha256": "hash"}

    result = controller.run_controller(
        workspace_root=tmp_path,
        session_path=session_path,
        job_paths=job_paths,
        execute=True,
        runner_fn=runner,
        quota_factory=_Quota,
        completion_validator=validate,
        contract_validator=contract,
        poll_seconds=1,
    )
    assert result["status"] == "completed"
    assert calls == list(controller.EXPECTED_JOBS)
    saved = json.loads((tmp_path / controller.DEFAULT_STATE).read_text())
    assert saved["completed_jobs"][0]["arm"] == "df2"
    assert saved["completed_jobs"][1]["arm"] == "rev"


def test_controller_stops_after_df2_failure_without_launching_rev(tmp_path, monkeypatch):
    session_path, _jobs, job_paths = _session_fixture(tmp_path)
    monkeypatch.setattr(controller, "_check_input_files", lambda *_args: None)
    contract = lambda *_args: ({"frozen": "sha"}, {"phase_seconds_remaining": 604800})
    calls = []

    def runner(_session, job, **_kwargs):
        calls.append(job["trial_id"])
        return {"status": "resource_stop", "reason": "memory_limit_exceeded", "trial_id": job["trial_id"]}

    with pytest.raises(controller.ControllerError, match="runner stopped"):
        controller.run_controller(
            workspace_root=tmp_path,
            session_path=session_path,
            job_paths=job_paths,
            execute=True,
            runner_fn=runner,
            quota_factory=_Quota,
            contract_validator=contract,
            poll_seconds=1,
        )
    assert calls == [controller.EXPECTED_JOBS[0]]
    saved = json.loads((tmp_path / controller.DEFAULT_STATE).read_text())
    assert saved["status"] == "stopped"


def test_active_run_directory_cap_requests_owned_safe_stop(tmp_path, monkeypatch):
    trial = controller.EXPECTED_JOBS[0]
    child_stopped = threading.Event()
    monkeypatch.setattr(
        controller,
        "_ledger_trial_events",
        lambda *_args: [{
            "event": "run_started",
            "trial_id": trial,
            "run_id": trial,
            "pid": 123,
            "pgid": 123,
            "process_start_ticks": 42,
        }],
    )
    monkeypatch.setattr(
        controller.session_runner,
        "_proc_identity",
        lambda _pid: {"pid": 123, "pgrp": 123, "start_ticks": 42},
    )
    monkeypatch.setattr(controller, "_tree_size", lambda _path: 1024 * 1024 + 1)
    monkeypatch.setattr(
        controller,
        "_signal_owned_child",
        lambda _child: (child_stopped.set() or True),
    )

    def runner(_session, _job, **_kwargs):
        assert child_stopped.wait(timeout=2)
        return {"status": "resource_stop", "reason": "output_cap"}

    job = {
        "trial_id": trial,
        "run_directory_cap_mb": 1,
        "command": ["python", "driver.py", "--output-root", "autoresearch/full-training"],
    }
    session = {"resource_limits": {"per_arm_output_bytes": 1024 * 1024}}
    state = {"status": "running", "stop_reason": None}
    state_path = tmp_path / "controller-state.json"
    stop_requested = threading.Event()
    state_result = controller._run_monitored(
        root=tmp_path,
        session=session,
        job=job,
        state=state,
        state_path=state_path,
        runner_fn=runner,
        quota_monitor=object(),
        poll_seconds=0.01,
        stop_requested=stop_requested,
        stop_signal=[None],
    )
    assert stop_requested.is_set()
    assert state_result["status"] == "resource_stop"
    assert state["stop_reason"] == "per_run_output_cap_exceeded"


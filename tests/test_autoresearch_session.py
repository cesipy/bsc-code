import datetime as dt
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

import pytest

import scripts.check_weekly_usage as usage_checker
import scripts.autoresearch_session as session_runner
from scripts.autoresearch_session import (
    QuotaMonitor,
    RunnerError,
    _pid_start_ticks,
    _normalize_terminate_grace_seconds,
    evaluate_quota_snapshot,
    read_cached_weekly_snapshot,
    run_one_job,
    snapshot_from_app_server_result,
)


UTC = dt.timezone.utc


def test_terminate_grace_keeps_default_and_caps_at_180_seconds():
    assert _normalize_terminate_grace_seconds(session_runner.DEFAULT_TERMINATE_GRACE_SECONDS) == 2.0
    assert _normalize_terminate_grace_seconds(45) == 45.0
    assert _normalize_terminate_grace_seconds(240) == 180.0
    assert _normalize_terminate_grace_seconds(-1) == 0.0
    for invalid in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(RunnerError, match="must be finite"):
            _normalize_terminate_grace_seconds(invalid)


class SequenceQuota:
    """In-process test dependency; production CLI exposes no synthetic quota mode."""

    def __init__(self, results):
        self.results = list(results)
        self.calls = 0

    def check(self):
        index = min(self.calls, len(self.results) - 1)
        self.calls += 1
        return self.results[index]


def _quota_result(status="continue", used=20.0, *, snapshot_at=None):
    stamp = snapshot_at or dt.datetime.now(UTC).isoformat()
    return {
        "status": status,
        "checked_at": dt.datetime.now(UTC).isoformat(),
        "max_age_seconds": 120,
        "stop_used_percent": 50,
        "snapshot": {
            "source": "test_dependency",
            "snapshot_at": stamp,
            "observed_at": dt.datetime.now(UTC).isoformat(),
            "limit_id": "codex",
            "buckets": [
                {
                    "bucket": "primary",
                    "window_minutes": 10080,
                    "used_percent": used,
                    "resets_at": 123,
                }
            ],
        },
    }


def _session(*, wall_seconds=20):
    return {
        "session_id": "unit-test-session",
        "status": "active",
        "started_at": (dt.datetime.now(UTC) - dt.timedelta(seconds=1)).isoformat(),
        "execution_started_at": (dt.datetime.now(UTC) - dt.timedelta(seconds=1)).isoformat(),
        "pilot_max_analysis_trials": 3,
        "pilot_execution_wall_seconds": wall_seconds,
        "max_parallel_gpu_jobs": 1,
        "weekly_usage_stop_used_percent": 50,
        "weekly_snapshot_max_age_seconds": 120,
        "read_only_roots": [],
    }


def _job(tmp_path, trial="trial-01", **updates):
    job = {
        "trial_id": trial,
        "mode": "analysis",
        "command": [sys.executable, "-c", "print('runner smoke')"],
        "cwd": ".",
        "timeout_seconds": 3,
        "memory_limit_mb": 1024,
        "disk_limit_mb": 4,
        "artifact_dir": f"autoresearch/runs/{trial}",
        "stdout_log": f"autoresearch/runs/{trial}/stdout.log",
        "metrics_path": f"autoresearch/runs/{trial}/metrics.json",
        "expected_metrics": ["accuracy", "cycle_knn"],
        "uses_gpu": False,
        "quota_poll_seconds": 0.02,
        "terminate_grace_seconds": 0.1,
    }
    job.update(updates)
    return job


def test_app_server_snapshot_uses_only_codex_weekly_bucket_and_default100_threshold():
    now = dt.datetime.now(UTC)
    result = {
        "rateLimitsByLimitId": {
            "codex": {
                "limitId": "codex",
                "primary": {"usedPercent": 99.9, "windowDurationMins": 10080, "resetsAt": 200},
                "secondary": {"usedPercent": 99, "windowDurationMins": 60, "resetsAt": 100},
            },
            "codex_other": {
                "limitId": "codex_other",
                "primary": {"usedPercent": 100, "windowDurationMins": 10080},
            },
        }
    }
    snapshot = snapshot_from_app_server_result(result, observed_at=now)
    assert snapshot["limit_id"] == "codex"
    assert snapshot["buckets"] == [
        {"bucket": "primary", "window_minutes": 10080, "used_percent": 99.9, "resets_at": 200}
    ]
    assert evaluate_quota_snapshot(snapshot, now=now)["status"] == "continue"
    snapshot["buckets"][0]["used_percent"] = 100
    assert evaluate_quota_snapshot(snapshot, now=now)["status"] == "stop_usage"
    snapshot["buckets"][0]["used_percent"] = 50
    assert evaluate_quota_snapshot(snapshot, now=now, stop_percent=50)["status"] == "stop_usage"
    assert session_runner._session_budgets(_session())[2] == 50
    assert snapshot_from_app_server_result(
        {"rateLimits": {"limitId": "codex_other", "primary": {"usedPercent": 3, "windowDurationMins": 10080}}},
        observed_at=now,
    ) is None


def test_cached_log_mtime_does_not_freshen_stale_snapshot(tmp_path):
    now = dt.datetime.now(UTC)
    timestamp = (now - dt.timedelta(seconds=180)).isoformat()
    day = tmp_path / (now.strftime("%Y/%m/%d"))
    day.mkdir(parents=True)
    log = day / "session.jsonl"
    log.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "payload": {
                    "rate_limits": {
                        "limit_id": "codex",
                        "primary": {"used_percent": 21, "window_minutes": 10080, "resets_at": 300},
                        "secondary": {"used_percent": 90, "window_minutes": 60},
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    os.utime(log, (now.timestamp(), now.timestamp()))

    snapshot = read_cached_weekly_snapshot(tmp_path, now=now)
    assert snapshot is not None
    assert snapshot["snapshot_at"] == timestamp
    assert snapshot["age_seconds"] == pytest.approx(180, abs=0.1)
    assert evaluate_quota_snapshot(snapshot, now=now)["status"] == "stop_stale"


def test_quota_monitor_prefers_live_snapshot_and_never_estimates_usage():
    now = dt.datetime.now(UTC)
    live = {
        "source": "app_server",
        "snapshot_at": now.isoformat(),
        "limit_id": "codex",
        "buckets": [{"bucket": "primary", "window_minutes": 10080, "used_percent": 100}],
    }

    def cached_reader(_now):
        pytest.fail("cached snapshot must not override a valid live account snapshot")

    monitor = QuotaMonitor(
        live_reader=lambda: live,
        cached_reader=cached_reader,
        now_fn=lambda: now,
    )
    checked = monitor.check()
    assert checked["status"] == "stop_usage"
    assert checked["snapshot"]["source"] == "app_server"


def test_live_app_server_uses_documented_stdio_rpc_and_cleans_owned_group(monkeypatch):
    import subprocess

    received = []
    process_ref = {}

    class FakeProcess:
        def __init__(self, command, **kwargs):
            assert command == ["/fake/codex", "app-server"]
            assert kwargs["start_new_session"] is True
            assert kwargs["stderr"] == subprocess.DEVNULL
            self.pid = 424242
            self.returncode = None
            server_input, client_input = os.pipe()
            client_output, server_output = os.pipe()
            self.stdin = os.fdopen(client_input, "wb", buffering=0)
            self.stdout = os.fdopen(client_output, "rb", buffering=0)

            def serve():
                with os.fdopen(server_input, "rb") as requests, os.fdopen(server_output, "wb", buffering=0) as responses:
                    for raw in requests:
                        message = json.loads(raw)
                        received.append(message)
                        if message.get("id") == 1:
                            result = {}
                        elif message.get("id") == 2:
                            result = {
                                "rateLimitsByLimitId": {
                                    "codex": {
                                        "limitId": "codex",
                                        "primary": {
                                            "usedPercent": 37,
                                            "windowDurationMins": 10080,
                                            "resetsAt": 1793091754,
                                        },
                                    }
                                }
                            }
                        else:
                            continue
                        responses.write(json.dumps({"id": message["id"], "result": result}).encode() + b"\n")
                self.returncode = 0

            self.thread = threading.Thread(target=serve, daemon=True)
            self.thread.start()
            process_ref["process"] = self

        def poll(self):
            return self.returncode

        def wait(self, timeout=None):
            self.thread.join(timeout)
            if self.thread.is_alive():
                raise subprocess.TimeoutExpired("fake-codex", timeout)
            self.returncode = 0
            return self.returncode

    kill_calls = []
    monkeypatch.setattr(session_runner.shutil, "which", lambda _name: "/fake/codex")
    monkeypatch.setattr(session_runner.subprocess, "Popen", FakeProcess)
    monkeypatch.setattr(session_runner.os, "killpg", lambda pid, sig: kill_calls.append((pid, sig)))

    snapshot = session_runner.read_live_app_server_snapshot(timeout_seconds=3)

    assert snapshot is not None
    assert snapshot["source"] == "app_server"
    assert snapshot["buckets"][0]["used_percent"] == 37
    assert [message.get("method") for message in received] == [
        "initialize",
        "initialized",
        "account/rateLimits/read",
    ]
    assert (424242, signal.SIGTERM) in kill_calls
    assert (424242, signal.SIGKILL) in kill_calls
    process_ref["process"].thread.join(timeout=0.5)
    assert not process_ref["process"].thread.is_alive()


def test_quota_monitor_uses_time_after_live_failure_for_cached_freshness():
    started = dt.datetime(2026, 10, 22, 11, 0, tzinfo=UTC)
    after_live_wait = started + dt.timedelta(seconds=3)
    cached_stamp = started - dt.timedelta(seconds=119)
    times = iter((after_live_wait,))
    seen = {}

    def cached_reader(now):
        seen["now"] = now
        return {
            "source": "cached_jsonl",
            "snapshot_at": cached_stamp.isoformat(),
            "limit_id": "codex",
            "buckets": [{"bucket": "primary", "window_minutes": 10080, "used_percent": 30}],
        }

    monitor = QuotaMonitor(
        live_reader=lambda: None,
        cached_reader=cached_reader,
        now_fn=lambda: next(times),
    )
    checked = monitor.check()

    assert seen["now"] == after_live_wait
    assert checked["status"] == "stop_stale"
    assert checked["snapshot"]["age_seconds"] == pytest.approx(122)


def _quota_monitor_snapshot(stamp, *, used=30, source="app_server", observed_at=None):
    return {
        "source": source,
        "snapshot_at": stamp.isoformat(),
        "observed_at": (observed_at or stamp).isoformat(),
        "limit_id": "codex",
        "buckets": [
            {
                "bucket": "primary",
                "window_minutes": 10080,
                "used_percent": used,
                "resets_at": 123,
            }
        ],
    }


class FakeClock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_quota_monitor_remembers_live_snapshot_without_refreshing_its_timestamps():
    started = dt.datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
    observed = started - dt.timedelta(seconds=1)
    live = _quota_monitor_snapshot(started, observed_at=observed)
    older_cache = _quota_monitor_snapshot(
        started - dt.timedelta(seconds=90), source="cached_jsonl"
    )
    clock = FakeClock(started)
    live_results = iter((live, None, None))
    monitor = QuotaMonitor(
        live_reader=lambda: next(live_results),
        cached_reader=lambda _now: older_cache,
        now_fn=clock,
    )

    assert monitor.check()["status"] == "continue"
    clock.now = started + dt.timedelta(seconds=10)
    transient = monitor.check()
    assert transient["status"] == "continue"
    assert transient["snapshot"]["source"] == "app_server"
    assert transient["snapshot"]["snapshot_at"] == started.isoformat()
    assert transient["snapshot"]["observed_at"] == observed.isoformat()
    assert transient["snapshot"]["age_seconds"] == 10

    clock.now = started + dt.timedelta(seconds=121)
    stale = monitor.check()
    assert stale["status"] == "stop_stale"
    assert stale["snapshot"]["snapshot_at"] == started.isoformat()
    assert stale["snapshot"]["observed_at"] == observed.isoformat()
    assert stale["snapshot"]["age_seconds"] == 121


def test_quota_monitor_without_any_snapshot_is_unavailable():
    clock = FakeClock(dt.datetime(2026, 10, 23, 8, 0, tzinfo=UTC))
    monitor = QuotaMonitor(
        live_reader=lambda: None,
        cached_reader=lambda _now: None,
        now_fn=clock,
    )

    checked = monitor.check()
    assert checked["status"] == "stop_unavailable"
    assert checked["snapshot"] is None


def test_quota_monitor_stops_on_fresh_usage_at_threshold():
    now = dt.datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
    monitor = QuotaMonitor(
        live_reader=lambda: _quota_monitor_snapshot(now, used=50),
        cached_reader=lambda _current: pytest.fail("fresh live snapshot must win"),
        now_fn=FakeClock(now),
        stop_percent=50,
    )

    assert monitor.check()["status"] == "stop_usage"


def test_quota_monitor_prefers_and_remembers_newer_valid_cache_snapshot():
    started = dt.datetime(2026, 10, 23, 8, 0, tzinfo=UTC)
    live = _quota_monitor_snapshot(started)
    newer_stamp = started + dt.timedelta(seconds=5)
    newer_cache = _quota_monitor_snapshot(
        newer_stamp,
        source="cached_jsonl",
        observed_at=newer_stamp + dt.timedelta(seconds=1),
    )
    older_cache = _quota_monitor_snapshot(
        started - dt.timedelta(seconds=5), source="cached_jsonl"
    )
    clock = FakeClock(started)
    live_results = iter((live, None, None))
    cache_results = iter((newer_cache, older_cache))
    monitor = QuotaMonitor(
        live_reader=lambda: next(live_results),
        cached_reader=lambda _now: next(cache_results),
        now_fn=clock,
    )

    assert monitor.check()["snapshot"]["source"] == "app_server"
    clock.now = started + dt.timedelta(seconds=10)
    checked = monitor.check()
    assert checked["status"] == "continue"
    assert checked["snapshot"]["source"] == "cached_jsonl"
    assert checked["snapshot"]["snapshot_at"] == newer_stamp.isoformat()
    assert checked["snapshot"]["observed_at"] == (
        newer_stamp + dt.timedelta(seconds=1)
    ).isoformat()

    clock.now = started + dt.timedelta(seconds=11)
    after_cache_regression = monitor.check()
    assert after_cache_regression["status"] == "continue"
    assert after_cache_regression["snapshot"]["snapshot_at"] == newer_stamp.isoformat()


def test_standalone_usage_helper_defaults_to100_and_records_only_check(tmp_path, monkeypatch):
    calls = []

    class FakeMonitor:
        def __init__(self, *, max_age_seconds, stop_percent):
            calls.append({"max_age_seconds": max_age_seconds, "stop_percent": stop_percent})

        def check(self):
            return {
                "status": "continue",
                "checked_at": "2026-10-22T11:00:00+00:00",
                "snapshot": {
                    "source": "app_server",
                    "snapshot_at": "2026-10-22T11:00:00+00:00",
                    "limit_id": "codex",
                    "buckets": [{"bucket": "primary", "window_minutes": 10080, "used_percent": 37}],
                },
            }

    ledger = tmp_path / "autoresearch" / "usage_checks.jsonl"
    monkeypatch.setattr(usage_checker, "QuotaMonitor", FakeMonitor)
    monkeypatch.setattr(usage_checker, "LEDGER_PATH", ledger)

    assert usage_checker.main([]) == 0
    assert calls[-1]["stop_percent"] == 100
    recorded = json.loads(ledger.read_text(encoding="utf-8").splitlines()[0])
    assert recorded["snapshot"]["buckets"][0]["used_percent"] == 37
    assert "token" not in json.dumps(recorded).lower()

    assert usage_checker.main(["--stop-percent", "50"]) == 0
    assert calls[-1]["stop_percent"] == 50


def test_missing_or_stale_quota_snapshot_fails_closed():
    now = dt.datetime.now(UTC)
    assert evaluate_quota_snapshot(None, now=now)["status"] == "stop_unavailable"
    stale = _quota_result("continue", 20, snapshot_at=(now - dt.timedelta(seconds=121)).isoformat())[
        "snapshot"
    ]
    assert evaluate_quota_snapshot(stale, now=now)["status"] == "stop_stale"
    high_but_stale = dict(stale)
    high_but_stale["buckets"] = [dict(stale["buckets"][0], used_percent=90)]
    assert evaluate_quota_snapshot(high_but_stale, now=now)["status"] == "stop_stale"


def test_one_job_records_missing_metrics_as_null_and_writes_handoff(tmp_path):
    quota = SequenceQuota([_quota_result(), _quota_result()])
    result = run_one_job(
        _session(),
        _job(tmp_path),
        workspace_root=tmp_path,
        quota_monitor=quota,
    )

    assert result["status"] == "completed"
    assert result["metrics"] == {"accuracy": None, "cycle_knn": None}
    events = [json.loads(line) for line in (tmp_path / "autoresearch/results.jsonl").read_text().splitlines()]
    finished = [event for event in events if event["event"] == "run_finished"][-1]
    started = [event for event in events if event["event"] == "run_started"][-1]
    assert finished["metrics"] == {"accuracy": None, "cycle_knn": None}
    assert finished["observed_resources"]["exit_code"] == 0
    assert started["pgid"] == started["pid"]
    assert (tmp_path / "autoresearch/handoff.json").is_file()
    assert "only the process group" in json.loads((tmp_path / "autoresearch/handoff.json").read_text())[
        "process_ownership_scope"
    ]
    assert "runner smoke" in (tmp_path / "autoresearch/runs/trial-01/stdout.log").read_text()


def test_exit_between_poll_and_rss_read_is_completed(tmp_path, monkeypatch):
    reads = []

    def rss_after_exit(pid):
        reads.append(pid)
        # Observe completion without reaping: the runner must still poll it.
        os.waitid(os.P_PID, pid, os.WEXITED | os.WNOWAIT)
        return None

    monkeypatch.setattr(session_runner, "_group_rss_bytes", rss_after_exit)
    result = run_one_job(
        _session(),
        _job(tmp_path, command=[sys.executable, "-c", "import time; time.sleep(.05); print('done')"]),
        workspace_root=tmp_path,
        quota_monitor=SequenceQuota([_quota_result()]),
    )
    assert reads
    assert result["status"] == "completed"
    assert result["reason"] == "exit_0"
    assert result["observed_resources"]["exit_code"] == 0
    assert result["observed_resources"]["artifact_bytes"] >= len("done\n")


def test_short_exit_transition_without_rss_completes(tmp_path, monkeypatch):
    monkeypatch.setattr(session_runner, "_group_rss_bytes", lambda _pid: None)
    result = run_one_job(
        _session(),
        _job(tmp_path, command=[sys.executable, "-c", "import time; time.sleep(.15)"]),
        workspace_root=tmp_path,
        quota_monitor=SequenceQuota([_quota_result()]),
    )
    assert result["status"] == "completed"
    assert result["reason"] == "exit_0"


def test_missing_rss_for_live_child_still_stops(tmp_path, monkeypatch):
    monkeypatch.setattr(session_runner, "_group_rss_bytes", lambda _pid: None)
    result = run_one_job(
        _session(),
        _job(tmp_path, command=[sys.executable, "-c", "import time; time.sleep(5)"]),
        workspace_root=tmp_path,
        quota_monitor=SequenceQuota([_quota_result()]),
    )
    assert result["status"] == "resource_stop"
    assert result["reason"] == "memory_usage_unavailable"
    assert result["observed_resources"]["exit_code"] is not None


def test_quota_staleness_during_run_terminates_owned_child_group(tmp_path):
    stale = _quota_result("stop_stale", 20, snapshot_at=(dt.datetime.now(UTC) - dt.timedelta(seconds=121)).isoformat())
    quota = SequenceQuota([_quota_result(), _quota_result(), stale])
    child = "import time; time.sleep(5)"
    result = run_one_job(
        _session(wall_seconds=15),
        _job(
            tmp_path,
            command=[sys.executable, "-c", child],
            timeout_seconds=10,
            quota_poll_seconds=0.02,
        ),
        workspace_root=tmp_path,
        quota_monitor=quota,
    )

    assert result["status"] == "quota_stop"
    assert result["reason"] == "quota_stop_stale"
    assert quota.calls >= 3
    events = [json.loads(line) for line in (tmp_path / "autoresearch/results.jsonl").read_text().splitlines()]
    started = [event for event in events if event["event"] == "run_started"][-1]
    assert started["pid"] == started["pgid"]
    assert result["observed_resources"]["exit_code"] is not None


def test_timeout_terminates_owned_child_group_and_waits(tmp_path):
    quota = SequenceQuota([_quota_result(), _quota_result()])
    result = run_one_job(
        _session(wall_seconds=10),
        _job(
            tmp_path,
            command=[sys.executable, "-c", "import time; time.sleep(5)"],
            timeout_seconds=0.15,
            terminate_grace_seconds=0.1,
        ),
        workspace_root=tmp_path,
        quota_monitor=quota,
    )
    assert result["status"] == "timeout"
    assert result["exit_code"] is not None
    assert result["observed_resources"]["wall_seconds"] < 2


def test_preflight_time_counts_against_absolute_session_wall_budget(tmp_path, monkeypatch):
    base = dt.datetime(2026, 10, 21, 20, 0, tzinfo=UTC)
    clock = [base]
    session = _session(wall_seconds=5)
    session["started_at"] = base.isoformat()
    session["execution_started_at"] = base.isoformat()

    class AdvancingQuota(SequenceQuota):
        def check(self):
            clock[0] += dt.timedelta(seconds=3)
            return super().check()

    def unexpected_launch(*_args, **_kwargs):
        pytest.fail("expired session wall budget must block subprocess launch")

    monkeypatch.setattr(session_runner.subprocess, "Popen", unexpected_launch)
    quota = AdvancingQuota([_quota_result(), _quota_result()])
    result = run_one_job(
        session,
        _job(tmp_path, timeout_seconds=2),
        workspace_root=tmp_path,
        quota_monitor=quota,
        now_fn=lambda: clock[0],
    )

    assert result["status"] == "stopped_before_launch"
    assert result["reason"] == "session_wall_budget_exhausted"
    events = [json.loads(line) for line in (tmp_path / "autoresearch/results.jsonl").read_text().splitlines()]
    assert not any(event["event"] == "run_launching" for event in events)
    budget = [event for event in events if event["event"] == "budget_check"][-1]
    assert budget["wall_used_seconds"] == 6
    assert budget["wall_remaining_seconds"] == 0
    assert budget["required_allowance_seconds"] == 2
    assert budget["metrics"] is None


def test_gpu_job_rechecks_and_blocks_external_compute_processes(tmp_path):
    quota = SequenceQuota([_quota_result(), _quota_result()])
    result = run_one_job(
        _session(),
        _job(tmp_path, uses_gpu=True, gpu_wall_limit_seconds=3, min_free_gpu_mb=512),
        workspace_root=tmp_path,
        quota_monitor=quota,
        gpu_probe_fn=lambda: {"status": "busy", "compute_process_count": 1},
    )
    assert result["status"] == "stopped_before_launch"
    assert result["reason"] == "gpu_busy"
    events = [json.loads(line) for line in (tmp_path / "autoresearch/results.jsonl").read_text().splitlines()]
    assert not any(event["event"] == "run_launching" for event in events)
    handoff = json.loads((tmp_path / "autoresearch/handoff.json").read_text())
    assert handoff["process_id"] is None
    assert handoff["gpu_preflight"]["status"] == "busy"


def test_training_is_blocked_by_preparation_contract_and_repairs_are_bounded(tmp_path):
    session = _session()
    session["pilot_training_enabled"] = False
    training_job = _job(tmp_path, mode="training")
    with pytest.raises(RunnerError, match="training jobs are disabled"):
        run_one_job(
            session,
            training_job,
            workspace_root=tmp_path,
            quota_monitor=SequenceQuota([_quota_result()]),
        )

    ledger = tmp_path / "autoresearch/results.jsonl"
    ledger.parent.mkdir(parents=True)
    events = [
        {"event": "run_launching", "trial_id": "base", "repair_of": None},
        {"event": "run_launching", "trial_id": "repair-1", "repair_of": "base"},
        {"event": "run_launching", "trial_id": "repair-2", "repair_of": "base"},
    ]
    ledger.write_text("".join(json.dumps(event) + "\n" for event in events), encoding="utf-8")
    session["max_repairs_per_trial"] = 2
    with pytest.raises(RunnerError, match="repair attempt limit is exhausted"):
        run_one_job(
            session,
            _job(tmp_path, "repair-3", repair_of="base"),
            workspace_root=tmp_path,
            quota_monitor=SequenceQuota([_quota_result()]),
        )


def test_startup_reconciliation_refuses_an_active_runner_without_launching(tmp_path):
    import socket

    ledger = tmp_path / "autoresearch/results.jsonl"
    ledger.parent.mkdir(parents=True)
    active = {
        "event": "run_launching",
        "run_id": "existing-run",
        "trial_id": "existing-trial",
        "host": socket.gethostname(),
        "runner_pid": os.getpid(),
        "runner_start_ticks": _pid_start_ticks(os.getpid()),
        "metrics": None,
    }
    ledger.write_text(json.dumps(active) + "\n", encoding="utf-8")
    quota = SequenceQuota([_quota_result()])
    marker = tmp_path / "must-not-launch"
    job = _job(
        tmp_path,
        command=[sys.executable, "-c", f"from pathlib import Path; Path({str(marker)!r}).touch()"],
    )

    with pytest.raises(RunnerError, match="launch_in_progress"):
        run_one_job(_session(), job, workspace_root=tmp_path, quota_monitor=quota)
    assert quota.calls == 0
    assert not marker.exists()

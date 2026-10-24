from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import uuid

import pytest

from scripts import autoresearch_dashboard as dashboard_module
from scripts.autoresearch_dashboard import (
    DashboardError,
    ProposalStore,
    build_dashboard,
    main,
    render_dashboard,
)


def _stamp(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _write_registry(root: Path, snapshot: dict, *, host: str = "node-local", boot_id: str = "boot-local") -> None:
    scripts = root / "scripts"
    scripts.mkdir(parents=True, exist_ok=True)
    source = """
import json

SNAPSHOT = json.loads(%r)

def local_identity():
    return {"host": %r, "boot_id": %r, "pid": 100, "start_ticks": 200}

class Registry:
    def __init__(self, root):
        self.root = root
    def snapshot(self):
        return SNAPSHOT
""" % (json.dumps(snapshot), host, boot_id)
    (scripts / "autoresearch_coordination.py").write_text(source, encoding="utf-8")


def _proposal(proposal_id: str = "study-1") -> dict:
    return {
        "id": proposal_id,
        "title": "Check event order sensitivity",
        "question": "Does the result survive the alternate event order?",
        "hypothesis": "The paired contrast remains directionally stable.",
        "evidence": ["autoresearch/event_protocol.md", "https://example.invalid/paper"],
        "expected_outcome": "A bounded estimate with the same paired units.",
        "cost": "CPU review, then one existing-checkpoint inference pass",
        "next_action": "Review the frozen input manifest and define a stop rule.",
        "command": ["python", "suggested_script.py", "--dry-run"],
    }


def test_dashboard_normalizes_ownership_and_never_exposes_result_payloads(tmp_path: Path) -> None:
    now = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
    snapshot = {
        "observed_at": _stamp(now),
        "claims": [
            {
                "kind": "trial", "session_id": "remote-session", "trial_id": "remote-trial",
                "host": "gpu-remote", "owner": {"host": "gpu-remote", "boot_id": "remote-boot", "pid": 99, "start_ticks": 321},
                "status": "running", "heartbeat_at": _stamp(now - timedelta(seconds=12)),
                "created_at": _stamp(now - timedelta(minutes=2)), "updated_at": _stamp(now),
                "resource_key": "gpu:GPU-REMOTE-UUID", "device_uuid": "GPU-REMOTE-UUID",
                "source_pins": {"scripts/train.py": "abcdef1234567890"},
                "dependencies": ["sealed-parent"], "owner_observation": {"state": "dead", "basis": "test remote observation"},
                "details": {"metrics": {"secret": "result-payload-marker"}},
            },
            {
                "kind": "trial", "session_id": "stale-session", "trial_id": "stale-trial",
                "host": "gpu-old", "owner": {"host": "gpu-old", "boot_id": "old-boot", "pid": 42},
                "status": "running", "heartbeat_at": _stamp(now - timedelta(minutes=20)),
                "resource_key": "gpu:GPU-OLD-UUID", "device_uuid": "GPU-OLD-UUID",
                "owner_observation": {"state": "unverified", "basis": "remote node"},
            },
            {
                "kind": "trial", "session_id": "unreported-session", "trial_id": "unreported-trial",
                "host": "gpu-next", "owner": {"host": "gpu-next", "boot_id": "next-boot", "pid": 43},
                "status": "claimed", "heartbeat_at": None,
                "resource_key": "gpu:GPU-NEXT-UUID", "device_uuid": "GPU-NEXT-UUID",
            },
        ],
        "resources": [
            {"kind": "gpu", "resource_key": "gpu:GPU-REMOTE-UUID", "device_uuid": "GPU-REMOTE-UUID",
             "host": "gpu-remote", "session_id": "remote-session", "trial_id": "remote-trial", "status": "claimed"}
        ],
        "warnings": [],
    }
    _write_registry(tmp_path, snapshot)
    (tmp_path / "STATUS.md").write_text("# Research status\n\nCurated operator narrative.\n", encoding="utf-8")
    data = build_dashboard(tmp_path, now=now)
    by_trial = {row.get("trial_id"): row for row in data["jobs"]}

    assert by_trial["remote-trial"]["health"] == "reported recently; remote owner unverified"
    assert "dead" not in by_trial["remote-trial"]["health"]
    assert "stale report" in by_trial["stale-trial"]["health"]
    assert "uncertain" in by_trial["unreported-trial"]["health"]
    assert data["resources"][0]["device_uuid"] == "GPU-REMOTE-UUID"
    assert data["totals"]["reported_running"] == 2

    serialized = json.dumps(data)
    rendered = render_dashboard(data, color=False)
    detailed = render_dashboard(data, color=False, all_jobs=True)
    assert "result-payload-marker" not in serialized
    assert "result-payload-marker" not in rendered
    assert "result-payload-marker" not in detailed
    assert "GPU-REMOTE-UUID" in rendered
    assert "owner process: PID 99" in detailed
    assert "scripts/train.py=abcdef12" in detailed
    assert "autoresearch/event_protocol.md" not in rendered
    assert "Curated operator narrative." in rendered


def test_dashboard_reads_only_selected_legacy_lifecycle_fields_and_skips_partial_tail(tmp_path: Path) -> None:
    now = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
    results = tmp_path / "autoresearch" / "results.jsonl"
    results.parent.mkdir(parents=True)
    records = [
        {
            "event": "run_started", "run_id": "old-run", "session_id": "old-session",
            "trial_id": "old-trial", "status": "running", "recorded_at": _stamp(now - timedelta(seconds=20)),
            "metrics": {"secret": "do-not-decode-marker"},
        },
        {
            "event": "quota_check", "run_id": "old-run", "session_id": "old-session",
            "trial_id": "old-trial", "status": "continue", "recorded_at": _stamp(now - timedelta(seconds=2)),
            "metrics": {"secret": "do-not-decode-marker"},
        },
        {
            "event": "run_finished", "run_id": "closed-run", "session_id": "closed-session",
            "trial_id": "closed-trial", "status": "completed", "recorded_at": _stamp(now - timedelta(minutes=5)),
            "finished_at": _stamp(now - timedelta(minutes=5)),
            "metrics": {"secret": "do-not-decode-marker"},
        },
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n{\"event\":\"run_finished\",\"metrics\":{\"partial\":", encoding="utf-8")
    data = build_dashboard(tmp_path, now=now)
    by_trial = {row.get("trial_id"): row for row in data["jobs"]}

    assert by_trial["old-trial"]["status"] == "running"
    assert "reported recently" in by_trial["old-trial"]["health"]
    assert by_trial["closed-trial"]["status"] == "completed"
    assert by_trial["closed-trial"]["health"].startswith("closed")
    assert any("partial tail" in warning for warning in data["warnings"])
    assert "do-not-decode-marker" not in json.dumps(data)


def test_legacy_terminal_events_keep_launch_owner_and_ignore_orphan_annotations(tmp_path: Path) -> None:
    now = datetime(2026, 10, 23, 21, 30, tzinfo=timezone.utc)
    results = tmp_path / "autoresearch" / "results.jsonl"
    results.parent.mkdir(parents=True)
    records = [
        {
            "event": "run_started", "run_id": "crashed-run", "session_id": "gpu-session",
            "trial_id": "crashed-trial", "status": "running", "recorded_at": _stamp(now - timedelta(hours=4)),
            "host": "gpu-node-7", "pid": 8812, "start_ticks": 34421, "boot_id": "gpu-boot-7",
            "metrics": {"secret": "never-read"},
        },
        {
            "event": "run_finished", "run_id": "crashed-run", "session_id": "gpu-session",
            "trial_id": "crashed-trial", "status": "crash", "recorded_at": _stamp(now - timedelta(hours=1)),
            "finished_at": _stamp(now - timedelta(hours=1)), "reason": "worker crash",
            "metrics": {"secret": "never-read"},
        },
        {
            "event": "launch", "run_id": "stopped-run", "session_id": "gpu-session-2",
            "trial_id": "stopped-trial", "status": "running", "recorded_at": _stamp(now - timedelta(hours=3)),
            "owner": {"host": "gpu-node-8", "pid": 9912, "start_ticks": 55421, "boot_id": "gpu-boot-8"},
        },
        {
            "event": "run_finished", "run_id": "stopped-run", "session_id": "gpu-session-2",
            "trial_id": "stopped-trial", "status": "resource_stop", "recorded_at": _stamp(now - timedelta(minutes=30)),
        },
        {
            "event": "run_finished", "run_id": "timeout-run", "session_id": "gpu-session-3",
            "trial_id": "timeout-trial", "status": "timeout", "recorded_at": _stamp(now - timedelta(hours=2)),
        },
        {
            "event": "run_finished", "run_id": "interrupted-run", "session_id": "gpu-session-4",
            "trial_id": "interrupted-trial", "status": "interrupted", "recorded_at": _stamp(now - timedelta(hours=2)),
        },
        {
            "event": "run_finished", "run_id": "unknown-outcome-run", "session_id": "gpu-session-5",
            "trial_id": "unknown-outcome-trial", "recorded_at": _stamp(now - timedelta(hours=2)),
        },
        {"event": "quota_check", "session_id": "gpu-session", "status": "continue", "recorded_at": _stamp(now)},
    ]
    results.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    data = build_dashboard(tmp_path, now=now)
    by_trial = {row.get("trial_id"): row for row in data["jobs"]}

    assert set(by_trial) == {
        "crashed-trial", "stopped-trial", "timeout-trial", "interrupted-trial", "unknown-outcome-trial",
    }
    assert by_trial["crashed-trial"]["status"] == "failed"
    assert by_trial["crashed-trial"]["health"] == "closed (legacy lifecycle event)"
    assert by_trial["crashed-trial"]["host"] == "gpu-node-7"
    assert by_trial["crashed-trial"]["owner_pid"] == 8812
    assert by_trial["crashed-trial"]["owner_start_ticks"] == 34421
    assert by_trial["stopped-trial"]["status"] == "failed"
    assert by_trial["stopped-trial"]["health"] == "closed (legacy lifecycle event)"
    assert by_trial["stopped-trial"]["host"] == "gpu-node-8"
    assert by_trial["stopped-trial"]["owner_pid"] == 9912
    assert by_trial["timeout-trial"]["status"] == "failed"
    assert by_trial["timeout-trial"]["health"] == "closed (legacy lifecycle event)"
    assert by_trial["interrupted-trial"]["status"] == "stopped"
    assert by_trial["interrupted-trial"]["health"] == "closed (legacy lifecycle event)"
    assert by_trial["unknown-outcome-trial"]["status"] == "closed (outcome unknown)"
    assert by_trial["unknown-outcome-trial"]["health"] == "closed (legacy lifecycle event)"
    assert data["totals"]["failed"] == 3
    assert "never-read" not in json.dumps(data)


def test_nodes_panel_keeps_active_uncertain_and_two_newest_closed_sessions() -> None:
    now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
    closed = [
        {"session_id": f"closed-{index}", "status": "closed", "health": "closed",
         "last_reported": _stamp(now - timedelta(minutes=index))}
        for index in range(8)
    ]
    attention = [
        {"session_id": "active-session", "status": "active", "health": "reported active",
         "last_reported": _stamp(now)},
        {"session_id": "uncertain-session", "status": "uncertain", "health": "process uncertain",
         "last_reported": None},
    ]
    sessions = closed + attention
    data = {
        "generated_at": _stamp(now), "registry_observed_at": _stamp(now), "local_host": "node",
        "freshness_threshold_seconds": 300,
        "nodes": [{"host": "node", "sessions": sessions, "resources": []}],
        "sessions": sessions, "jobs": [], "resources": [], "proposals": [], "warnings": [],
        "totals": {"nodes": 1, "sessions": 10, "jobs": 0, "reported_running": 0,
                   "completed": 0, "failed": 0, "uncertain": 0, "resources": 0,
                   "pending_proposals": 0},
    }

    compact = render_dashboard(data)
    full = render_dashboard(data, all_jobs=True)
    assert "active-session" in compact and "uncertain-session" in compact
    assert "closed-0" in compact and "closed-1" in compact
    assert all(f"closed-{index}" not in compact for index in range(2, 8))
    assert "6 older closed session(s) hidden" in compact
    assert all(f"closed-{index}" in full for index in range(8))


def test_dashboard_without_sources_is_read_only_and_non_tty_watch_is_finite(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = build_dashboard(tmp_path, now=datetime(2026, 10, 23, tzinfo=timezone.utc))
    assert not (tmp_path / "autoresearch").exists()
    assert data["totals"]["jobs"] == 0
    assert data["warnings"]

    calls = []
    monkeypatch.setattr("scripts.autoresearch_dashboard.build_dashboard", lambda root: calls.append(root) or data)
    output = io.StringIO()
    error = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)
    monkeypatch.setattr("sys.stderr", error)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    monkeypatch.chdir(tmp_path)
    exit_code = main(["--workspace", str(tmp_path), "show", "--watch", "--interval", "0.01"])
    assert exit_code == 0
    assert len(calls) == 1
    assert "Non-TTY watch prints one snapshot and exits." in error.getvalue()


def test_proposal_store_records_decisions_and_preserves_append_only_history(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    created = store.create(_proposal(), actor="cedric")
    original_event = dict(created["history"][0])
    approved = store.decide("study-1", "approve", priority=2, note="Good bounded next step", actor="reviewer")
    deferred = store.decide("study-1", "defer", note="Wait for current capture review", actor="reviewer-2")
    loaded = store.get("study-1")

    assert approved["decision"] == {
        "decision": "approve", "priority": 2, "note": "Good bounded next step",
        "actor": "reviewer", "at": approved["decision"]["at"],
    }
    assert loaded["decision"]["decision"] == "defer"
    assert loaded["decision"]["actor"] == "reviewer-2"
    assert [event["action"] for event in loaded["history"]] == ["created", "decision", "decision"]
    assert loaded["history"][0] == original_event
    assert loaded["revision"] == 3
    assert deferred["revision"] == 3
    assert store.list()[0]["id"] == "study-1"
    overview = _render_output_for_store(tmp_path)
    assert "study-1" in overview and "cost:" in overview
    assert "suggested_script.py" not in overview
    assert loaded["command"][1] == "suggested_script.py"


def _render_output_for_store(root: Path) -> str:
    return render_dashboard(build_dashboard(root), color=False)


def test_proposal_store_serializes_concurrent_decisions_and_checks_stale_review(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    store.create(_proposal("parallel"))
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [
            pool.submit(store.decide, "parallel", decision, actor=actor)
            for decision, actor in (("approve", "one"), ("defer", "two"))
        ]
        for future in futures:
            future.result(timeout=10)
    record = store.get("parallel")
    assert len(record["history"]) == 3
    assert {event.get("actor") for event in record["history"][1:]} == {"one", "two"}
    with pytest.raises(DashboardError, match="changed during review"):
        store.decide("parallel", "reject", expected_revision=1)


def test_proposal_store_rejects_unsafe_ids_symlinks_and_invalid_cards(tmp_path: Path) -> None:
    store = ProposalStore(tmp_path)
    with pytest.raises(ValueError, match="safe filename"):
        store.create(_proposal("../escape"))
    with pytest.raises(ValueError, match="evidence"):
        store.create({**_proposal("bad-card"), "evidence": "not-a-list"})

    store.create(_proposal("linked"))
    proposal_file = tmp_path / "autoresearch" / "concurrency" / "proposals" / "linked.json"
    outside = tmp_path / "outside.json"
    outside.write_text("{}", encoding="utf-8")
    proposal_file.unlink()
    proposal_file.symlink_to(outside)
    with pytest.raises(DashboardError, match="symlink"):
        store.get("linked")
    with pytest.raises(DashboardError, match="symlink"):
        store.list()


def test_review_without_tty_only_displays_card_and_does_not_claim_approval(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProposalStore(tmp_path)
    store.create(_proposal("show-only"))
    output = io.StringIO()
    monkeypatch.setattr("sys.stdout", output)
    monkeypatch.setattr("sys.stdin", io.StringIO("approve\n"))
    monkeypatch.chdir(tmp_path)

    assert main(["--workspace", str(tmp_path), "review", "show-only"]) == 0
    assert store.get("show-only")["decision"] is None
    assert "Use `decide` with an explicit --decision" in output.getvalue()
    assert "stored only; never executed" in output.getvalue()


def test_default_terminal_view_keeps_all_active_and_only_six_recent_closed_jobs() -> None:
    now = datetime(2026, 10, 23, 22, 0, tzinfo=timezone.utc)
    jobs = [
        {
            "trial_id": f"terminal-{index}", "session_id": "old-session",
            "status": "failed" if index == 0 else "completed",
            "health": "closed", "last_reported": _stamp(now - timedelta(minutes=index)),
            "reason": "nonzero exit" if index == 0 else None,
            "owner_pid": 88 if index == 0 else None,
            "owner_start_ticks": 900 if index == 0 else None,
            "source_pin_count": 1 if index == 0 else 0,
            "source_pins": [{"path": "scripts/train.py", "sha256": "a" * 64}] if index == 0 else [],
        }
        for index in range(10)
    ]
    jobs.extend([
        {"trial_id": "active-job", "session_id": "active-session", "status": "running",
         "health": "reported recently; process unverified", "last_reported": _stamp(now),
         "device_uuid": "GPU-active", "dependencies": ["reviewed-parent"],
         "next_action": "Monitor the next heartbeat", "owner_pid": 77,
         "owner_start_ticks": 901, "source_pin_count": 1,
         "source_pins": [{"path": "scripts/train.py", "sha256": "b" * 64}]},
        {"trial_id": "uncertain-job", "session_id": "uncertain-session", "status": "uncertain",
         "health": "uncertain (no real heartbeat)", "last_reported": None},
    ])
    data = {
        "generated_at": _stamp(now), "registry_observed_at": _stamp(now), "local_host": "node",
        "freshness_threshold_seconds": 300, "nodes": [], "sessions": [], "jobs": jobs,
        "resources": [],
        "proposals": [{"id": "proposal-token", "title": "Review the next step",
                       "question": "Is the bounded check ready?", "cost": "small", "decision": None}],
        "status_summary": {"path": "STATUS.md", "text": "\n".join(f"curated-{index}" for index in range(1, 27))},
        "warnings": ["fixture warning"],
        "totals": {"nodes": 0, "sessions": 12, "jobs": 12, "reported_running": 1,
                   "completed": 9, "failed": 1, "uncertain": 1, "uncertain_sessions": 1,
                   "resources": 0, "pending_proposals": 1},
    }

    compact = render_dashboard(data)
    full = render_dashboard(data, all_jobs=True)
    assert "active-job" in compact and "uncertain-job" in compact
    assert all(f"terminal-{index}" in compact for index in range(6))
    assert all(f"terminal-{index}" not in compact for index in range(6, 10))
    assert "4 older terminal job(s) hidden" in compact
    assert "terminal-9" in full
    assert "execution completed" in compact
    assert "failure reason: nonzero exit" in compact
    assert "dependencies: reviewed-parent" in compact
    assert "Monitor the next heartbeat" in compact
    assert "owner process: PID" not in compact and "source pins (" not in compact
    assert "owner process: PID 77" in full and "source pins (1)" in full
    assert "curated-24" in compact and "curated-25" not in compact
    assert "1 job needs owner/state verification" in compact
    assert (compact.index("Research status") < compact.index("Proposal review")
            < compact.index("Nodes and sessions") < compact.index("Trials and ownership")
            < compact.index("Attention"))


def test_proposal_store_explicitly_repairs_only_dead_same_host_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProposalStore(tmp_path)
    monkeypatch.setattr(dashboard_module, "_local_process_identity", lambda: {
        "host": "local-node", "boot_id": "boot-now", "pid": 10, "start_ticks": 20,
    })
    monkeypatch.setattr(dashboard_module, "_owner_process_state", lambda owner: {
        901: "dead", 902: "alive", 903: "dead", 904: "unknown",
    }.get(owner.get("pid"), "unknown"))

    directory = tmp_path / "autoresearch" / "concurrency" / "proposals"
    directory.mkdir(parents=True)
    lock_owners = {
        "dead-local": {"host": "local-node", "boot_id": "boot-now", "pid": 901, "start_ticks": 1},
        "live-local": {"host": "local-node", "boot_id": "boot-now", "pid": 902, "start_ticks": 2},
        "remote": {"host": "node-remote", "boot_id": "boot-now", "pid": 903, "start_ticks": 3},
        "unknown": {"host": "local-node", "boot_id": "boot-now", "pid": 904, "start_ticks": 4},
    }
    for proposal_id, identity in lock_owners.items():
        lock = directory / f".{proposal_id}.lock"
        lock.mkdir()
        (lock / "owner.json").write_text(json.dumps({**identity, "token": uuid.uuid4().hex}), encoding="utf-8")

    result = store.repair_local_locks()
    assert [item["proposal_id"] for item in result["removed"]] == ["dead-local"]
    assert {item["proposal_id"] for item in result["retained"]} == {"live-local", "remote", "unknown"}
    assert not (directory / ".dead-local.lock").exists()
    assert (directory / ".live-local.lock").exists()
    assert (directory / ".remote.lock").exists()


def test_proposal_lock_repair_rechecks_token_before_removal(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    store = ProposalStore(tmp_path)
    monkeypatch.setattr(dashboard_module, "_local_process_identity", lambda: {
        "host": "local-node", "boot_id": "boot-now", "pid": 10, "start_ticks": 20,
    })
    monkeypatch.setattr(dashboard_module, "_owner_process_state", lambda owner: "dead")
    directory = tmp_path / "autoresearch" / "concurrency" / "proposals"
    directory.mkdir(parents=True)
    lock = directory / ".changing.lock"
    lock.mkdir()
    (lock / "owner.json").write_text(json.dumps({
        "host": "local-node", "boot_id": "boot-now", "pid": 901, "start_ticks": 1,
        "token": uuid.uuid4().hex,
    }), encoding="utf-8")
    original_read = store._read_lock_owner
    count = 0

    def changed_on_recheck(path: Path) -> dict:
        nonlocal count
        owner = original_read(path)
        count += 1
        if count == 2:
            return {**owner, "token": "f" * 32}
        return owner

    monkeypatch.setattr(store, "_read_lock_owner", changed_on_recheck)
    result = store.repair_local_locks()
    assert not result["removed"]
    assert result["retained"][0]["reason"] == "lock ownership changed during repair; retained"
    assert (lock / "owner.json").exists()


def test_proposal_store_mutations_reject_protected_workspace_overlaps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    protected = tmp_path / "manual"
    protected.mkdir()
    nested = protected / "child"
    nested.mkdir()
    monkeypatch.setattr(dashboard_module, "PROTECTED_WORKSPACE_ROOTS", (protected.resolve(),))

    with pytest.raises(DashboardError, match="protected workspace"):
        ProposalStore(protected).create(_proposal("manual-root"))
    with pytest.raises(DashboardError, match="protected workspace"):
        ProposalStore(nested).create(_proposal("manual-child"))
    with pytest.raises(DashboardError, match="protected workspace"):
        ProposalStore(tmp_path).repair_local_locks()
    assert not (protected / "autoresearch").exists()

"""Cross-process safety tests for shared-NFS autoresearch coordination."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pytest

from scripts.autoresearch_coordination import (
    CoordinationError,
    Registry,
    local_identity,
)


def _workspace(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    (path / "autoresearch").mkdir(exist_ok=True)
    return path


def _owner(host: str, *, boot_id: str | None = None, pid: int = 100, ticks: int = 1):
    return {
        "host": host,
        "boot_id": boot_id or f"boot-{host}",
        "pid": pid,
        "start_ticks": ticks,
    }


def _reserve(
    registry: Registry,
    trial_id: str,
    artifact_dir: str,
    *,
    session_id: str = "session-a",
    device_uuid: str | None = "GPU-a",
    owner: dict | None = None,
    **kwargs,
):
    return registry.reserve(
        session_id=session_id,
        trial_id=trial_id,
        artifact_dir=artifact_dir,
        device_uuid=device_uuid,
        owner=owner,
        **kwargs,
    )


def test_two_processes_racing_for_one_trial_have_only_one_winner(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "race-workspace")
    barrier = tmp_path / "race-start"
    module_root = Path(__file__).resolve().parents[1]
    worker = r"""
import json, os, sys, time
from pathlib import Path
from scripts.autoresearch_coordination import CoordinationError, Registry
root, barrier, trial, device, suffix = sys.argv[1:]
deadline = time.monotonic() + 10
while not Path(barrier).exists():
    if time.monotonic() >= deadline:
        raise SystemExit("barrier timeout")
    time.sleep(0.002)
try:
    reservation = Registry(Path(root)).reserve(
        session_id="race-session-" + suffix,
        trial_id=trial,
        artifact_dir="outputs/" + suffix,
        device_uuid=device,
    )
    print(json.dumps({"won": True, "token": reservation["token"]}))
except CoordinationError as exc:
    print(json.dumps({"won": False, "error": str(exc)}))
"""
    child_env = os.environ.copy()
    child_env["PYTHONDONTWRITEBYTECODE"] = "1"
    child_env["OMP_NUM_THREADS"] = "1"
    child_env["MKL_NUM_THREADS"] = "1"
    child_env["OPENBLAS_NUM_THREADS"] = "1"
    child_env["PYTHONPATH"] = os.pathsep.join(
        [str(module_root), child_env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    children = [
        subprocess.Popen(
            [
                sys.executable,
                "-B",
                "-c",
                worker,
                str(root),
                str(barrier),
                "race-trial",
                f"GPU-{suffix}",
                suffix,
            ],
            cwd=module_root,
            env=child_env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        for suffix in ("one", "two")
    ]
    barrier.write_text("go", encoding="ascii")
    outputs = [child.communicate(timeout=30) for child in children]
    assert [child.returncode for child in children] == [0, 0], outputs
    outcomes = [json.loads(stdout.strip()) for stdout, _ in outputs]
    assert sum(item["won"] for item in outcomes) == 1
    assert sum(not item["won"] for item in outcomes) == 1
    snapshot = Registry(root).snapshot()
    assert [claim["trial_id"] for claim in snapshot["claims"]] == ["race-trial"]


def test_distinct_gpu_and_host_resources_coexist_but_same_host_device_is_exclusive(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    _reserve(
        registry,
        "trial-one",
        "outputs/one",
        session_id="session-one",
        owner=_owner("node-a"),
        device_uuid="GPU-1",
    )
    _reserve(
        registry,
        "trial-two",
        "outputs/two",
        session_id="session-two",
        owner=_owner("node-a"),
        device_uuid="GPU-2",
    )
    _reserve(
        registry,
        "trial-three",
        "outputs/three",
        session_id="session-three",
        owner=_owner("node-b"),
        device_uuid="GPU-1",
    )

    with pytest.raises(CoordinationError, match="already leased"):
        _reserve(
            registry,
            "trial-four",
            "outputs/four",
            session_id="session-four",
            owner=_owner("node-a"),
            device_uuid="GPU-1",
        )
    with pytest.raises(CoordinationError, match="active job"):
        _reserve(
            registry,
            "trial-five",
            "outputs/five",
            session_id="session-one",
            owner=_owner("node-a"),
            device_uuid="GPU-3",
        )
    with pytest.raises(CoordinationError, match="active job"):
        _reserve(
            registry,
            "trial-six",
            "outputs/six",
            session_id="session-one",
            owner=_owner("node-a"),
            device_uuid="cpu",
        )

    snapshot = registry.snapshot()
    assert len(snapshot["claims"]) == 3
    gpu_resources = [item for item in snapshot["resources"] if item["kind"] == "gpu"]
    session_resources = [item for item in snapshot["resources"] if item["kind"] == "session"]
    assert len(gpu_resources) == 3
    assert len(session_resources) == 3
    assert {item["resource_key"] for item in gpu_resources} == {
        "gpu:node-a:GPU-1",
        "gpu:node-a:GPU-2",
        "gpu:node-b:GPU-1",
    }


@pytest.mark.parametrize(
    ("first_output", "second_output"),
    [
        ("outputs/shared", "outputs/shared"),
        ("outputs/shared", "outputs/shared/nested"),
        ("outputs/tree/leaf", "outputs/tree"),
    ],
)
def test_canonical_artifact_claims_reject_duplicate_and_ancestor_overlap(
    tmp_path: Path, first_output: str, second_output: str
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    _reserve(
        registry,
        "first",
        first_output,
        session_id="first-session",
        device_uuid="GPU-1",
        owner=_owner("node-a"),
    )
    with pytest.raises(CoordinationError, match="overlaps"):
        _reserve(
            registry,
            "second",
            second_output,
            session_id="second-session",
            device_uuid="GPU-2",
            owner=_owner("node-a"),
        )


def test_hostile_paths_symlinks_nonempty_outputs_and_registry_overlap_are_rejected(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "escape").symlink_to(outside, target_is_directory=True)

    for trial, output in (
        ("outside", "../outside/result"),
        ("symlink", "escape/result"),
        ("registry", "autoresearch/concurrency/registry/claims"),
    ):
        with pytest.raises(CoordinationError):
            _reserve(registry, trial, output, owner=_owner("node-a"))

    with pytest.raises(CoordinationError, match="parent traversal"):
        _reserve(registry, "traversal", "outputs/a/../b", owner=_owner("node-a"))

    nonempty = root / "outputs" / "nonempty"
    nonempty.mkdir(parents=True)
    (nonempty / "existing.txt").write_text("already published", encoding="utf-8")
    with pytest.raises(CoordinationError, match="already contains output"):
        _reserve(registry, "nonempty", "outputs/nonempty", owner=_owner("node-a"))

    (root / "registry-link").symlink_to(outside, target_is_directory=True)
    with pytest.raises(CoordinationError, match="symlink"):
        Registry(root, registry_relative="registry-link/registry")


def test_cpu_leases_are_per_session_and_do_not_claim_a_gpu_resource(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    first = _reserve(
        registry,
        "cpu-one",
        "outputs/cpu-one",
        session_id="cpu-session",
        device_uuid="cpu",
        owner={},
    )
    with pytest.raises(CoordinationError, match="active job"):
        _reserve(
            registry,
            "cpu-two",
            "outputs/cpu-two",
            session_id="cpu-session",
            device_uuid=None,
            owner={},
        )
    _reserve(
        registry,
        "cpu-three",
        "outputs/cpu-three",
        session_id="other-cpu-session",
        device_uuid="cpu",
        owner={},
    )
    snapshot = registry.snapshot()
    assert len(snapshot["resources"]) == 2
    assert all(item["kind"] == "cpu" for item in snapshot["resources"])
    assert first["resource_key"] == "session:cpu-session"


def test_dependencies_require_completed_predecessors_and_keep_source_pins(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    predecessor = _reserve(
        registry,
        "baseline",
        "outputs/baseline",
        source_pins={"source_commit": "abc123", "config_hash": "cfg-7"},
        owner=local_identity(),
    )
    with pytest.raises(CoordinationError, match="not completed"):
        _reserve(
            registry,
            "candidate-blocked",
            "outputs/candidate-blocked",
            dependencies=("baseline",),
            owner=local_identity(),
        )

    registry.heartbeat(predecessor, {"phase": "evaluation", "elapsed_seconds": 2.5})
    registry.finish(predecessor, "completed", {"score": 0.875})
    dependent = _reserve(
        registry,
        "candidate",
        "outputs/candidate",
        dependencies=("baseline",),
        source_pins={"source_commit": "def456"},
        owner=local_identity(),
    )
    registry.finish(dependent, "failed", {"reason": "validation"})
    with pytest.raises(CoordinationError, match="not completed"):
        _reserve(
            registry,
            "after-failure",
            "outputs/after-failure",
            dependencies=("candidate",),
            owner=local_identity(),
        )

    snapshot = registry.snapshot()
    by_id = {claim["trial_id"]: claim for claim in snapshot["claims"]}
    baseline = by_id["baseline"]
    assert baseline["status"] == "completed"
    assert baseline["source_pins"] == {"source_commit": "abc123", "config_hash": "cfg-7"}
    assert baseline["heartbeat_event"] == {"phase": "evaluation", "elapsed_seconds": 2.5}
    assert baseline["heartbeat_at"] is not None
    datetime.fromisoformat(baseline["heartbeat_at"].replace("Z", "+00:00"))
    assert baseline["finished_at"] is not None
    assert baseline["details"] == {"score": 0.875}
    assert by_id["candidate"]["status"] == "failed"
    assert by_id["candidate"]["source_pins"] == {"source_commit": "def456"}
    assert snapshot["resources"] == []


def test_stale_or_unknown_owner_is_reported_without_stealing_or_releasing_claim(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    here = local_identity()
    stale_owner = {
        "host": here["host"],
        "boot_id": here["boot_id"],
        "pid": 2_000_000_000,
        "start_ticks": 1,
    }
    reservation = _reserve(
        registry,
        "stale-trial",
        "outputs/stale",
        device_uuid="GPU-stale",
        owner=stale_owner,
    )
    before = registry.snapshot()
    claim_before = before["claims"][0]
    assert claim_before["owner_observation"]["state"] == "stale_heuristic"
    assert claim_before["owner_observation"]["observed_at"] == before["observed_at"]
    assert claim_before["status"] == "reserved"

    with pytest.raises(CoordinationError, match="already leased"):
        _reserve(
            registry,
            "another-trial",
            "outputs/another",
            session_id="another-session",
            device_uuid="GPU-stale",
            owner=here,
        )
    with pytest.raises(CoordinationError, match="already has a claim"):
        _reserve(
            registry,
            "stale-trial",
            "outputs/new-stale-output",
            session_id="new-session",
            device_uuid="GPU-other",
            owner=here,
        )

    unknown = _reserve(
        registry,
        "unknown-owner",
        "outputs/unknown",
        session_id="unknown-session",
        device_uuid="cpu",
        owner={},
    )
    snapshot = registry.snapshot()
    unknown_claim = next(item for item in snapshot["claims"] if item["trial_id"] == "unknown-owner")
    assert unknown_claim["owner_observation"]["state"] == "unknown_owner"
    assert next(item for item in snapshot["claims"] if item["trial_id"] == "stale-trial")["status"] == "reserved"
    assert reservation["token"] != unknown["token"]


def test_token_mismatch_cannot_heartbeat_or_finish_another_owner(tmp_path: Path) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    reservation = _reserve(
        registry,
        "owned-trial",
        "outputs/owned",
        owner=local_identity(),
    )
    forged = dict(reservation, token="0" * 32)
    with pytest.raises(CoordinationError, match="does not own"):
        registry.heartbeat(forged, {"phase": "forged"})
    with pytest.raises(CoordinationError, match="does not own"):
        registry.finish(forged, "completed")
    snapshot = registry.snapshot()
    assert snapshot["claims"][0]["token"] == reservation["token"]
    assert snapshot["claims"][0]["status"] == "reserved"
    assert snapshot["claims"][0]["heartbeat_at"] is None
    assert len(snapshot["resources"]) == 2


def test_repeated_finish_releases_only_the_original_tokens_remaining_lease(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    reservation = _reserve(
        registry,
        "retry-finish",
        "outputs/retry-finish",
        owner=local_identity(),
    )
    lease_path = registry.resources_dir / (
        hashlib.sha256(reservation["resource_key"].encode()).hexdigest() + ".json"
    )
    original_lease = json.loads(lease_path.read_text(encoding="utf-8"))
    session_key = next(key for key in reservation["resource_keys"] if key != reservation["resource_key"])
    session_lease_path = registry.resources_dir / (hashlib.sha256(session_key.encode()).hexdigest() + ".json")
    session_lease = json.loads(session_lease_path.read_text(encoding="utf-8"))
    registry.finish(reservation, "completed")

    # Simulate an interrupted cleanup after terminal status publication.
    lease_path.write_text(json.dumps(original_lease), encoding="utf-8")
    registry.finish(reservation, "completed")
    assert not lease_path.exists()

    foreign_lease = dict(original_lease, token="f" * 32)
    lease_path.write_text(json.dumps(foreign_lease), encoding="utf-8")
    session_lease_path.write_text(json.dumps(session_lease), encoding="utf-8")
    registry.finish(reservation, "completed")
    assert lease_path.exists()
    assert json.loads(lease_path.read_text())["token"] == "f" * 32
    assert not session_lease_path.exists()


def test_legacy_launched_trial_ids_block_reuse_but_quota_rows_do_not(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    ledger = root / "autoresearch" / "results.jsonl"
    ledger.write_text(
        "\n".join(
            [
                json.dumps({"event": "run_started", "trial_id": "legacy-launched"}),
                json.dumps({"event": "quota_check", "trial_id": "quota-only"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    registry = Registry(root)
    with pytest.raises(CoordinationError, match="legacy launch metadata"):
        _reserve(registry, "legacy-launched", "outputs/legacy", owner=_owner("node-a"))
    _reserve(registry, "quota-only", "outputs/quota", owner=_owner("node-a"))


def test_snapshot_is_read_only_and_reports_missing_owner_without_fabrication(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    absent_registry = root / "autoresearch" / "concurrency" / "registry"
    snapshot = registry.snapshot()
    assert snapshot["observed_at"]
    assert snapshot["claims"] == []
    assert snapshot["resources"] == []
    assert snapshot["warnings"]
    assert not absent_registry.exists()

    reservation = _reserve(
        registry,
        "missing-owner",
        "outputs/missing-owner",
        owner={},
        device_uuid="cpu",
    )
    claim_path = registry.claims_dir / "missing-owner.json"
    claim = json.loads(claim_path.read_text(encoding="utf-8"))
    del claim["owner"]
    claim_path.write_text(json.dumps(claim), encoding="utf-8")
    before = claim_path.stat().st_mtime_ns
    read_only_view = registry.snapshot()
    after = claim_path.stat().st_mtime_ns
    observed = next(item for item in read_only_view["claims"] if item["trial_id"] == "missing-owner")
    assert observed["owner_observation"]["state"] == "unknown_owner"
    assert observed["owner_observation"]["observed_at"] == read_only_view["observed_at"]
    assert before == after
    assert reservation["token"] == observed["token"]


def _install_admission_mutex(registry: Registry, owner_record: dict) -> Path:
    registry._ensure_tree()
    registry.lock_dir.mkdir()
    token = "a" * 32
    owner_path = registry.lock_dir / f"owner-{token}.json"
    owner_path.write_text(
        json.dumps(
            {
                "token": token,
                "owner": owner_record,
                "created_at": "2026-01-01T00:00:00Z",
            }
        ),
        encoding="utf-8",
    )
    return owner_path


def test_explicit_repair_removes_only_proven_dead_local_mutex_and_keeps_claims(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)
    reservation = _reserve(
        registry,
        "retained-claim",
        "outputs/retained-claim",
        owner=local_identity(),
        device_uuid="GPU-repair-test",
    )
    dead_owner = {
        **local_identity(),
        "pid": 2_000_000_000,
        "start_ticks": 1,
    }
    owner_path = _install_admission_mutex(registry, dead_owner)

    snapshot = registry.snapshot()
    assert any("no automatic repair" in warning for warning in snapshot["warnings"])
    assert owner_path.exists()
    claim_files = sorted(registry.claims_dir.glob("*.json"))
    resource_files = sorted(registry.resources_dir.glob("*.json"))
    before = {path: path.read_bytes() for path in claim_files + resource_files}

    repaired = registry.repair_local_admission_lock()

    assert repaired["status"] == "repaired"
    assert repaired["claims_changed"] is False
    assert "PID is absent" in repaired["basis"]
    assert not registry.lock_dir.exists()
    assert {path: path.read_bytes() for path in claim_files + resource_files} == before
    assert json.loads(claim_files[0].read_text())["token"] == reservation["token"]


def test_explicit_repair_refuses_live_remote_and_missing_mutex_owners(
    tmp_path: Path,
) -> None:
    root = _workspace(tmp_path / "workspace")
    registry = Registry(root)

    live_owner_path = _install_admission_mutex(registry, local_identity())
    with pytest.raises(CoordinationError, match="still present"):
        registry.repair_local_admission_lock()
    assert live_owner_path.exists()
    live_owner_path.unlink()
    registry.lock_dir.rmdir()

    here = local_identity()
    remote_owner = {
        **here,
        "host": here["host"] + ".remote",
        "pid": 2_000_000_000,
        "start_ticks": 1,
    }
    remote_owner_path = _install_admission_mutex(registry, remote_owner)
    with pytest.raises(CoordinationError, match="remote-host"):
        registry.repair_local_admission_lock()
    assert remote_owner_path.exists()
    remote_owner_path.unlink()
    registry.lock_dir.rmdir()

    registry._ensure_tree()
    registry.lock_dir.mkdir()
    with pytest.raises(CoordinationError, match="exactly one owner record"):
        registry.repair_local_admission_lock()
    assert registry.lock_dir.exists()

    registry.lock_dir.rmdir()
    owner_path = _install_admission_mutex(registry, {})
    with pytest.raises(CoordinationError, match="incomplete"):
        registry.repair_local_admission_lock()
    assert owner_path.exists()


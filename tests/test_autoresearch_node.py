"""Node session preparation does not allocate hardware or launch training."""
import json
from pathlib import Path

import pytest

from scripts import autoresearch_node as node


def workspace(tmp_path):
    for name in node.SOURCE_FILES:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# fixture source\n")
    return tmp_path


def test_init_creates_unique_bounded_sessions_with_source_identity(tmp_path):
    root = workspace(tmp_path)
    first = node.init_session(root, question="Does scheduling isolate results?", hours=2, max_trials=3, device="cpu")
    second = node.init_session(root, question="Second independent session", device="cpu")
    assert first["session_id"] != second["session_id"]
    session = json.loads(Path(first["session"]).read_text())
    assert session["question"] == "Does scheduling isolate results?"
    assert session["pilot_execution_wall_seconds"] == 7200
    assert session["pilot_max_analysis_trials"] == 3
    assert session["max_parallel_gpu_jobs"] == 1
    assert len(session["source_files_sha256"]) == len(node.SOURCE_FILES)
    assert session["host"] and session["boot_id"]
    assert first["jobs_launched"] == 0


def test_prepared_smoke_has_unique_outputs_and_exact_completion_requirements(tmp_path):
    root = workspace(tmp_path)
    session = node.init_session(root, question="CPU smoke", device="cpu")
    result = node.cpu_smoke_job(root, Path(session["session"]))
    job = json.loads(Path(result["job_spec"]).read_text())
    assert job["uses_gpu"] is False
    assert job["expected_metric_values"] == {"status": "completed", "value": 1}
    assert session["session_id"] in job["artifact_dir"]
    assert not (root / job["artifact_dir"]).exists()


def test_bad_budget_and_external_paths_fail_before_creation(tmp_path):
    root = workspace(tmp_path)
    for updates in ({"hours": float("inf")}, {"hours": 0}, {"max_trials": True}, {"question": ""}):
        args = {"question": "valid", **updates}
        with pytest.raises(ValueError):
            node.init_session(root, **args)
    with pytest.raises(ValueError):
        node.safe_workspace_path(root, "../outside")
    target = root / "linked"
    target.symlink_to(root / "scripts", target_is_directory=True)
    with pytest.raises(ValueError):
        node.safe_workspace_path(root, "linked/file")


def test_read_only_checkout_is_rejected_before_writes(tmp_path, monkeypatch):
    root = workspace(tmp_path)
    monkeypatch.setattr(node, "READ_ONLY_ROOTS", (root,))
    with pytest.raises(ValueError, match="read-only"):
        node.init_session(root, question="must not create output")
    assert not (root / "autoresearch/concurrency/sessions").exists()


def test_dashboard_options_and_review_decisions_are_forwarded_in_order(tmp_path, monkeypatch):
    from scripts import autoresearch_dashboard

    seen = []
    monkeypatch.setattr(autoresearch_dashboard, "main", lambda argv: seen.append(argv) or 0)
    assert node.main(["--workspace", str(tmp_path), "dashboard", "--watch", "--interval", "3"]) == 0
    assert seen[-1] == ["--workspace", str(tmp_path), "show", "--watch", "--interval", "3"]
    assert node.main(["--workspace", str(tmp_path), "decide", "candidate", "--decision", "defer", "--priority", "2"]) == 0
    assert seen[-1][-5:] == ["candidate", "--decision", "defer", "--priority", "2"]

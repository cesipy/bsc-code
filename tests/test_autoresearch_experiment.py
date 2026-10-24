from __future__ import annotations

import json
import hashlib
import io
import sys
import tempfile
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
VERIFICATION_ROOT = ROOT / "autoresearch" / "concurrency" / "verification"
sys.path.insert(0, str(ROOT / "scripts"))

import autoresearch_experiment as experiment_module  # noqa: E402
from autoresearch_experiment import ExperimentError, main, run_experiment  # noqa: E402


class IsolatedExperimentTests(unittest.TestCase):
    def setUp(self) -> None:
        VERIFICATION_ROOT.mkdir(parents=True, exist_ok=True)
        self._temporary = tempfile.TemporaryDirectory(
            prefix="autoresearch-experiment-", dir=VERIFICATION_ROOT
        )
        self.addCleanup(self._temporary.cleanup)
        self.temp_root = Path(self._temporary.name)
        self.workspace = self.temp_root / "workspace"
        (self.workspace / "src").mkdir(parents=True)
        self._write_fake_config()
        self._write_fake_main()

    def _write_fake_config(self) -> None:
        path = self.workspace / "src" / "config.py"
        path.write_text(
            """from dataclasses import dataclass
PREPROCESSED_PATH = "res/preprocessed.pkl"
PRETRAIN_CHECKPOINTS_DIR = "res/checkpoints/pretrains"
FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/finetunes"
MLFLOW_TRACKING_URI = "sqlite:///original.db"
@dataclass
class ViLBERTConfig:
    preprocessed_path: str = PREPROCESSED_PATH
""",
            encoding="utf-8",
        )

    def _write_fake_main(self) -> None:
        path = self.workspace / "src" / "main.py"
        path.write_text(
            """import json
import os
import sys
from pathlib import Path
import config
report = {
    "cwd": os.getcwd(),
    "main_file": __file__,
    "config_file": config.__file__,
    "argv": sys.argv[1:],
    "pretrain_dir": config.PRETRAIN_CHECKPOINTS_DIR,
    "finetune_dir": config.FINETUNE_CHECKPOINTS_DIR,
    "preprocessed_path": config.PREPROCESSED_PATH,
    "dataclass_preprocessed_path": config.ViLBERTConfig().preprocessed_path,
    "mlflow_uri": config.MLFLOW_TRACKING_URI,
    "pythonpath": os.environ.get("PYTHONPATH"),
    "tmpdir": os.environ.get("TMPDIR"),
    "hf_home": os.environ.get("HF_HOME"),
    "hf_hub_cache": os.environ.get("HF_HUB_CACHE"),
    "transformers_cache": os.environ.get("TRANSFORMERS_CACHE"),
    "hf_hub_offline": os.environ.get("HF_HUB_OFFLINE"),
    "transformers_offline": os.environ.get("TRANSFORMERS_OFFLINE"),
    "data_link": str(Path("res/data/conceptual-captions").resolve()),
}
Path("child-report.json").write_text(json.dumps(report), encoding="utf-8")
""",
            encoding="utf-8",
        )

    def test_child_has_private_outputs_environment_and_data_view(self) -> None:
        source_data = self.workspace / "res" / "data" / "conceptual-captions"
        source_data.mkdir(parents=True)
        (source_data / "fixture.csv").write_text("fixture", encoding="utf-8")
        source_bytes = {
            name: (self.workspace / "src" / name).read_bytes()
            for name in ("main.py", "config.py")
        }
        output = self.workspace / "autoresearch" / "concurrency" / "runs" / "run-a"

        result = run_experiment(
            output,
            ["--fusion", "early", "--pretrain-path", "res/checkpoints/input.pt"],
            workspace_root=self.workspace,
        )

        self.assertEqual(result, 0)
        report = json.loads((output / "child-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["cwd"], str(output.resolve()))
        self.assertEqual(report["main_file"], str(output / ".source/src/main.py"))
        self.assertEqual(report["config_file"], str(output / ".source/src/config.py"))
        self.assertEqual(
            report["argv"],
            [
                "--fusion",
                "early",
                "--pretrain-path",
                str((self.workspace / "res/checkpoints/input.pt").resolve()),
            ],
        )
        self.assertEqual(report["pretrain_dir"], str(output / "res/checkpoints/pretrains"))
        self.assertEqual(report["finetune_dir"], str(output / "res/checkpoints/finetunes"))
        self.assertEqual(report["preprocessed_path"], str(output / "res/preprocessed.pkl"))
        self.assertEqual(report["dataclass_preprocessed_path"], str(output / "res/preprocessed.pkl"))
        self.assertEqual(report["mlflow_uri"], f"sqlite:///{output / 'mlflow.db'}")
        self.assertEqual(report["pythonpath"], str(output / ".source/src"))
        self.assertEqual(report["tmpdir"], str(output / "cache/tmp"))
        self.assertEqual(report["hf_home"], str(output / "cache/huggingface"))
        self.assertEqual(report["hf_hub_cache"], str(output / "cache/huggingface/hub"))
        self.assertEqual(report["transformers_cache"], str(output / "cache/huggingface/hub"))
        self.assertEqual(report["hf_hub_offline"], "1")
        self.assertEqual(report["transformers_offline"], "1")
        self.assertEqual(report["data_link"], str(source_data.resolve()))
        self.assertTrue((output / "res/data/conceptual-captions").is_symlink())
        self.assertFalse((output / "res/checkpoints").is_symlink())
        status = json.loads((output / "experiment_status.json").read_text(encoding="utf-8"))
        self.assertEqual(status["status"], "completed")
        self.assertEqual(status["exit_code"], 0)
        self.assertTrue(status["snapshot_verified_after_run"])
        self.assertEqual(
            status["source_snapshot"]["original"]["inventory_sha256"],
            status["source_snapshot"]["snapshot"]["inventory_sha256"],
        )
        self.assertEqual(status["source_snapshot"]["file_count"], 2)
        self.assertEqual(status["data_views"]["res/data/conceptual-captions"], str(source_data.resolve()))
        self.assertEqual(
            source_bytes,
            {
                name: (self.workspace / "src" / name).read_bytes()
                for name in ("main.py", "config.py")
            },
        )

    def test_source_mutation_after_snapshot_is_rejected_before_child_launch(self) -> None:
        config_path = self.workspace / "src/config.py"
        original_config = config_path.read_bytes()
        original_create_cache_views = experiment_module._create_cache_views
        output = self.workspace / "autoresearch/concurrency/runs/mutated-source"

        def mutate_after_snapshot(output_root: Path, workspace_root: Path):
            views = original_create_cache_views(output_root, workspace_root)
            config_path.write_bytes(original_config + b"\n# concurrent edit\n")
            return views

        try:
            with mock.patch.object(
                experiment_module, "_create_cache_views", side_effect=mutate_after_snapshot
            ):
                with self.assertRaisesRegex(ExperimentError, "changed before launch"):
                    run_experiment(output, [], workspace_root=self.workspace)
        finally:
            config_path.write_bytes(original_config)
        self.assertFalse((output / "child-report.json").exists())
        self.assertEqual(
            (output / ".source/src/config.py").read_bytes(), original_config
        )

    def test_snapshot_mutation_during_child_invalidates_an_otherwise_successful_exit(self) -> None:
        output = self.workspace / "autoresearch/concurrency/runs/changed-copy"

        def mutate_copy(*args, **kwargs):
            source = output / ".source/src/main.py"
            source.chmod(0o644)
            source.write_text(source.read_text() + "\n# changed by child\n")
            return mock.Mock(returncode=0)

        with mock.patch.object(experiment_module.subprocess, "run", side_effect=mutate_copy):
            with self.assertRaisesRegex(ExperimentError, "snapshot changed during execution"):
                run_experiment(output, [], workspace_root=self.workspace)
        status = json.loads((output / "experiment_status.json").read_text())
        self.assertEqual(status["status"], "failed")

    def test_private_model_cache_view_reuses_links_and_copies_only_refs(self) -> None:
        repo = (
            self.workspace
            / "autoresearch/cache/huggingface/hub/models--fixture--tiny"
        )
        refs = repo / "refs" / "main"
        blob = repo / "blobs" / "blob-fixture"
        snapshot_leaf = repo / "snapshots" / "commit-fixture" / "config.json"
        refs.parent.mkdir(parents=True)
        blob.parent.mkdir(parents=True)
        snapshot_leaf.parent.mkdir(parents=True)
        refs.write_text("commit-fixture\n", encoding="utf-8")
        blob.write_bytes(b"small fake cache fixture")
        snapshot_leaf.symlink_to("../../blobs/blob-fixture")

        torch_checkpoint = (
            self.workspace
            / "autoresearch/cache/torch/hub/checkpoints/fake-fixture.pth"
        )
        torch_checkpoint.parent.mkdir(parents=True)
        torch_checkpoint.write_bytes(b"small fake torch-cache fixture")
        source_hashes = {
            "refs": hashlib.sha256(refs.read_bytes()).hexdigest(),
            "blob": hashlib.sha256(blob.read_bytes()).hexdigest(),
            "torch": hashlib.sha256(torch_checkpoint.read_bytes()).hexdigest(),
        }
        output = self.workspace / "autoresearch/concurrency/runs/cache-view"

        result = run_experiment(output, [], workspace_root=self.workspace)

        self.assertEqual(result, 0)
        report = json.loads((output / "child-report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["hf_hub_cache"], str(output / "cache/huggingface/hub"))
        self.assertEqual(report["transformers_cache"], report["hf_hub_cache"])
        status = json.loads((output / "experiment_status.json").read_text(encoding="utf-8"))
        cache_views = status["cache_views"]
        hf_view = cache_views["huggingface_hub"]
        self.assertEqual(hf_view["source"], str(repo.parent.resolve()))
        self.assertEqual(hf_view["copied_refs"], 1)
        self.assertEqual(hf_view["linked_blobs"], 1)
        self.assertEqual(hf_view["linked_snapshot_files"], 1)
        self.assertTrue(hf_view["locks_private"])
        private_hub = output / "cache/huggingface/hub"
        private_repo = private_hub / "models--fixture--tiny"
        private_ref = private_repo / "refs/main"
        private_blob = private_repo / "blobs/blob-fixture"
        private_snapshot = private_repo / "snapshots/commit-fixture/config.json"
        self.assertFalse(private_ref.is_symlink())
        self.assertTrue(private_blob.is_symlink())
        self.assertTrue(private_snapshot.is_symlink())
        self.assertEqual(private_snapshot.read_bytes(), b"small fake cache fixture")
        self.assertTrue((private_hub / ".locks").is_dir())
        self.assertFalse((private_hub / ".locks").is_symlink())

        private_ref.write_text("commit-updated\n", encoding="utf-8")
        self.assertEqual(refs.read_text(encoding="utf-8"), "commit-fixture\n")
        self.assertEqual(
            source_hashes,
            {
                "refs": hashlib.sha256(refs.read_bytes()).hexdigest(),
                "blob": hashlib.sha256(blob.read_bytes()).hexdigest(),
                "torch": hashlib.sha256(torch_checkpoint.read_bytes()).hexdigest(),
            },
        )
        torch_view = cache_views["torch_checkpoints"]
        private_torch = output / "cache/torch/hub/checkpoints/fake-fixture.pth"
        self.assertEqual(torch_view["linked_checkpoints"], 1)
        self.assertTrue(private_torch.is_symlink())
        self.assertEqual(private_torch.read_bytes(), b"small fake torch-cache fixture")

    def test_equals_form_of_relative_checkpoint_path_is_resolved(self) -> None:
        output = self.workspace / "autoresearch" / "concurrency" / "runs" / "run-equals"
        result = run_experiment(
            output,
            ["--fusion", "early", "--pretrain-path=res/checkpoints/parent.pt"],
            workspace_root=self.workspace,
        )
        self.assertEqual(result, 0)
        report = json.loads((output / "child-report.json").read_text(encoding="utf-8"))
        self.assertEqual(
            report["argv"][-1],
            f"--pretrain-path={(self.workspace / 'res/checkpoints/parent.pt').resolve()}",
        )

    def test_rejects_output_inside_shared_data_before_creating_it(self) -> None:
        input_dir = self.workspace / "res" / "data" / "conceptual-captions"
        input_dir.mkdir(parents=True)
        output = input_dir / "run"
        with self.assertRaisesRegex(ExperimentError, "overlaps shared input"):
            run_experiment(output, [], workspace_root=self.workspace)
        self.assertFalse(output.exists())

    def test_rejects_output_that_already_exists(self) -> None:
        output = self.workspace / "autoresearch" / "existing"
        output.mkdir(parents=True)
        sentinel = output / "sentinel.txt"
        sentinel.write_text("keep", encoding="utf-8")
        with self.assertRaisesRegex(ExperimentError, "must not already exist"):
            run_experiment(output, [], workspace_root=self.workspace)
        self.assertEqual(sentinel.read_text(encoding="utf-8"), "keep")

    def test_rejects_output_symlink_and_paths_outside_workspace(self) -> None:
        real_parent = self.workspace / "autoresearch" / "real"
        real_parent.mkdir(parents=True)
        symlink_parent = self.workspace / "autoresearch" / "linked"
        symlink_parent.symlink_to(real_parent, target_is_directory=True)
        with self.assertRaisesRegex(ExperimentError, "traverses a symlink"):
            run_experiment(
                symlink_parent / "run", [], workspace_root=self.workspace
            )
        with self.assertRaisesRegex(ExperimentError, "inside the workspace"):
            run_experiment(self.temp_root / "outside", [], workspace_root=self.workspace)
        self.assertFalse((real_parent / "run").exists())

    def test_cli_requires_a_separator_before_main_arguments(self) -> None:
        stderr = io.StringIO()
        with redirect_stderr(stderr):
            result = main(
                ["--output-root", "autoresearch/runs/cli", "--fusion", "early"]
            )
        self.assertEqual(result, 2)
        self.assertIn("--output-root PATH --", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()

"""CPU-only contract tests for the opt-in full-training prefetch wrapper."""

from __future__ import annotations

import hashlib
import os
import gc
import multiprocessing as mp
from pathlib import Path
import runpy
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch.utils.data import Dataset

import autoresearch_full_training as base
import autoresearch_full_training_prefetch as prefetch


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def configured_backend(monkeypatch):
    # Configure explicitly for this test, then let monkeypatch restore the
    # base driver's pre-wrapper globals/functions for the wider test suite.
    for name in (
        "MAX_WORKERS", "CPU_THREADS", "SOURCE_PATHS", "_source_hashes",
        "_make_run_spec", "_build_parser", "_build_loaders", "run_training",
    ):
        monkeypatch.setattr(base, name, getattr(base, name))
    return prefetch.configure_backend()


def test_import_does_not_configure_or_mutate_base_driver():
    code = r"""
import autoresearch_full_training as base
before = (base.MAX_WORKERS, base.CPU_THREADS, base.SOURCE_PATHS,
          base._source_hashes, base._make_run_spec, base._build_parser,
          base._build_loaders, base.run_training)
import autoresearch_full_training_prefetch
after = (base.MAX_WORKERS, base.CPU_THREADS, base.SOURCE_PATHS,
         base._source_hashes, base._make_run_spec, base._build_parser,
         base._build_loaders, base.run_training)
assert before == after
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT / "src")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = ""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=REPO_ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_backend_provenance_spec_defaults_and_worker_bounds(configured_backend, tmp_path):
    metadata = configured_backend
    assert base.MAX_WORKERS == 1
    assert base.CPU_THREADS == 4
    assert metadata == {
        "id": "objective-loader-prefetch-v1",
        "version": 1,
        "objective_loader_count": 3,
        "workers_per_objective_loader": 1,
        "loader_worker_processes": 3,
        "main_cpu_threads": 4,
        "torch_threads_per_loader_worker": 1,
        "declared_cpu_execution_slots": 7,
        "prefetch_factor": 2,
        "persistent_workers": False,
        "pin_memory": "enabled only when CUDA is available, matching base driver",
        "cpu_image_transform_parity_vs_workers_zero": (
            "not guaranteed bitwise: per-row RNG keys are stable, but the 4-thread main process "
            "and 1-thread DataLoader workers have different CPU execution settings; actual-input "
            "image tensor and generated-mask parity requires separate measurement"
        ),
    }

    parser = base._build_parser()
    train_parser = next(action for action in parser._actions if action.dest == "command").choices["train"]
    default_args = parser.parse_args(["train", "--arm", "df2", "--run-id", "x", "--output-root", "autoresearch/x"])
    assert default_args.workers == 1
    worker_action = next(action for action in train_parser._actions if action.dest == "workers")
    assert worker_action.default == 1

    hashes = base._source_hashes()
    assert set(prefetch.SOURCE_PATHS).issubset(hashes)
    wrapper_path = REPO_ROOT / "src/autoresearch_full_training_prefetch.py"
    base_path = REPO_ROOT / "src/autoresearch_full_training.py"
    assert hashes["full_training_driver"] == hashlib.sha256(wrapper_path.read_bytes()).hexdigest()
    assert hashes["src/autoresearch_full_training.py"] == hashlib.sha256(base_path.read_bytes()).hexdigest()

    anchor = {"initial_model_state_sha256_by_arm": {"df2": "same-init", "rev": "same-init"}}
    spec = base._make_run_spec("df2", base.EXPECTED_SEED, 1, {"fixture": "same-inputs"}, hashes, anchor)
    assert spec["schema_version"] == "matched-full-pretraining-run-prefetch-v1"
    assert spec["execution_backend"] == metadata
    assert spec["training"]["workers"] == 1
    assert spec["training"]["cpu_threads"] == 4
    assert spec["training"]["prefetch_factor"] == 2
    assert spec["training"]["persistent_workers"] is False
    assert spec["initial_model_state_sha256"] == "same-init"
    for invalid in (0, 2, True):
        with pytest.raises(base.FullTrainingError, match="requires exactly 1 worker"):
            base._make_run_spec("df2", base.EXPECTED_SEED, invalid, {}, hashes, anchor)
    invalid_args = SimpleNamespace(
        arm="df2",
        seed=base.EXPECTED_SEED,
        workers=0,
        device="cpu",
        run_id="invalid-worker-bound",
        output_root=str(tmp_path / "must-not-be-created"),
        resume=False,
    )
    with pytest.raises(base.FullTrainingError, match="workers must be exactly 1"):
        base.run_training(invalid_args, injected={})
    assert not Path(invalid_args.output_root).exists()


class _RandomRows(Dataset):
    def __init__(self, task: str, size: int):
        self.task = task
        self.split = "train"
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        import random

        features = torch.tensor(
            [random.random(), float(np.random.random()), float(torch.rand(()))],
            dtype=torch.float32,
        )
        return {"index": torch.tensor(index, dtype=torch.long), "features": features}


class _PidRows(Dataset):
    def __init__(self, task: str, split: str, size: int = 8):
        self.task = task
        self.split = split
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        return {
            "index": torch.tensor(index, dtype=torch.long),
            "worker_pid": torch.tensor(os.getpid(), dtype=torch.long),
        }


class _ErrorRows(Dataset):
    def __init__(self, task: str, split: str, size: int = 8):
        self.task = task
        self.split = split
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        raise RuntimeError(f"fixture decode failure at {index}")


def _wait_for_child_count(expected: int, timeout: float = 5.0) -> set[int]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        children = {child.pid for child in mp.active_children() if child.is_alive()}
        if len(children) == expected:
            return children
        time.sleep(0.01)
    children = {child.pid for child in mp.active_children() if child.is_alive()}
    assert len(children) == expected, f"expected {expected} active loader workers, got {children}"
    return children


def _active_child_ids() -> set[int]:
    return {child.pid for child in mp.active_children() if child.is_alive()}


def _consume_phase(split: str, epoch: int) -> tuple[set[int], int]:
    datasets = {
        task: base.IsolatedRandomnessDataset(
            _PidRows(task, split), seed=1804, split=split, task=task
        )
        for task in base.TASK_KEYS
    }
    loaders = base._build_loaders(
        datasets,
        split=split,
        epoch=epoch,
        start_batch=0,
        seed=1804,
        workers=1,
        torch=torch,
    )
    iterators = {task: iter(loader) for task, loader in loaders.items()}
    _wait_for_child_count(3)
    maximum_live = len(_active_child_ids())
    seen_pids: set[int] = set()
    batch_count = len(loaders["ap"])
    for batch_index in range(batch_count):
        for task in base.TASK_KEYS:
            batch = next(iterators[task])
            pids = set(int(pid) for pid in batch["worker_pid"].tolist())
            assert len(pids) == 1
            pid = next(iter(pids))
            seen_pids.add(pid)
            if batch_index == batch_count - 1:
                # Worker exit is synchronous with delivery of the final batch;
                # no extra StopIteration request is needed to trigger cleanup.
                assert pid not in _active_child_ids()
            maximum_live = max(maximum_live, len(_active_child_ids()))
    assert _active_child_ids().isdisjoint(seen_pids)
    assert len(seen_pids) == 3
    del iterators, loaders
    gc.collect()
    return seen_pids, maximum_live


def _tiny_update(workers: int):
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        datasets = {
            task: base.IsolatedRandomnessDataset(
                _RandomRows(task, 8), seed=1804, split="train", task=task
            )
            for task in base.TASK_KEYS
        }
        loaders = base._build_loaders(
            datasets,
            split="train",
            epoch=2,
            start_batch=0,
            seed=1804,
            workers=workers,
            torch=torch,
        )
        assert set(loaders) == set(base.TASK_KEYS)
        assert all(loader.num_workers == workers for loader in loaders.values())
        if workers:
            assert all(loader.prefetch_factor == 2 for loader in loaders.values())

        iterators = {task: iter(loader) for task, loader in loaders.items()}
        parameter = torch.nn.Parameter(torch.tensor([0.125], dtype=torch.float32))
        optimizer = torch.optim.SGD([parameter], lr=0.01)
        seen = {task: [] for task in base.TASK_KEYS}
        task_scale = {"ap": 0.5, "mlm": 0.25, "mim": 0.75}
        for batch_index in range(4):
            for task in base.TASK_KEYS:
                batch = next(iterators[task])
                seen[task].append((batch["index"].clone(), batch["features"].clone()))
                loss = parameter.mul(batch["features"].sum()).sum() * task_scale[task]
                loss.backward()
            if batch_index % 2 == 1:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
        result = parameter.detach().clone()
        del iterators, loaders
        return seen, result
    finally:
        torch.set_num_threads(old_threads)


def test_fixed_thread_tiny_loader_rng_order_and_update_match_workers0_vs1(configured_backend, monkeypatch):
    monkeypatch.setattr(base, "BATCH_SIZE", 2)
    baseline_batches, baseline_parameter = _tiny_update(workers=0)
    candidate_batches, candidate_parameter = _tiny_update(workers=1)
    assert torch.equal(baseline_parameter, candidate_parameter)
    for task in base.TASK_KEYS:
        assert len(baseline_batches[task]) == len(candidate_batches[task]) == 4
        for (baseline_indices, baseline_features), (candidate_indices, candidate_features) in zip(
            baseline_batches[task], candidate_batches[task]
        ):
            assert torch.equal(baseline_indices, candidate_indices)
            assert torch.equal(baseline_features, candidate_features)


def test_loader_workers_close_at_final_batch_without_epoch_overlap(configured_backend, monkeypatch):
    monkeypatch.setattr(base, "BATCH_SIZE", 2)
    assert _active_child_ids() == set()
    train_pids, train_maximum = _consume_phase("train", epoch=0)
    assert _active_child_ids() == set()
    validation_pids, validation_maximum = _consume_phase("validation", epoch=0)
    assert _active_child_ids() == set()
    next_epoch_pids, next_epoch_maximum = _consume_phase("train", epoch=1)
    assert _active_child_ids() == set()
    assert train_pids.isdisjoint(validation_pids)
    assert validation_pids.isdisjoint(next_epoch_pids)
    assert train_pids.isdisjoint(next_epoch_pids)
    assert max(train_maximum, validation_maximum, next_epoch_maximum) == 3


def test_loader_worker_is_closed_when_iterator_errors_or_is_deleted(configured_backend, monkeypatch):
    monkeypatch.setattr(base, "BATCH_SIZE", 2)
    failing = {
        task: base.IsolatedRandomnessDataset(
            _ErrorRows(task, "train"), seed=1804, split="train", task=task
        )
        for task in base.TASK_KEYS
    }
    loaders = base._build_loaders(
        failing, split="train", epoch=0, start_batch=0, seed=1804, workers=1, torch=torch
    )
    iterator = iter(loaders["ap"])
    pids = _wait_for_child_count(1)
    with pytest.raises(RuntimeError, match="fixture decode failure"):
        next(iterator)
    assert _active_child_ids().isdisjoint(pids)

    regular = {
        task: base.IsolatedRandomnessDataset(
            _PidRows(task, "train"), seed=1804, split="train", task=task
        )
        for task in base.TASK_KEYS
    }
    loaders = base._build_loaders(
        regular, split="train", epoch=0, start_batch=0, seed=1804, workers=1, torch=torch
    )
    iterator = iter(loaders["ap"])
    pids = _wait_for_child_count(1)
    del iterator
    gc.collect()
    deadline = time.monotonic() + 5.0
    while _active_child_ids().intersection(pids) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _active_child_ids().isdisjoint(pids)


def test_run_training_exception_closes_all_started_objective_workers(
    configured_backend, monkeypatch, tmp_path
):
    helper = runpy.run_path(str(REPO_ROOT / "tests" / "test_autoresearch_full_training.py"))
    monkeypatch.setattr(base, "AUTORESEARCH", tmp_path / "autoresearch")
    base.AUTORESEARCH.mkdir()
    monkeypatch.setattr(base, "EPOCHS", 1)
    monkeypatch.setattr(base, "EXPECTED_TRAIN_ROWS", 8)
    monkeypatch.setattr(base, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(base, "BATCH_SIZE", 2)
    monkeypatch.setattr(base, "GRADIENT_ACCUMULATION", 2)
    monkeypatch.setattr(base, "CHECKPOINT_EVERY_UPDATES", 100)
    helper["_install_signal_state"](monkeypatch)
    args, injected, trainer = helper["_injected_run"](
        output_root=base.AUTORESEARCH / "runs", run_id="worker-error"
    )
    args.workers = 1
    observed_children: set[int] = set()

    def fail_after_workers_start(_batch, *, flag_optimizer):
        assert flag_optimizer is False
        observed_children.update(_active_child_ids())
        raise RuntimeError("fixture trainer failure")

    trainer.train_alignment_prediction_batch = fail_after_workers_start
    with pytest.raises(RuntimeError, match="fixture trainer failure"):
        base.run_training(args, injected=injected)
    assert len(observed_children) == 3
    deadline = time.monotonic() + 5.0
    while _active_child_ids().intersection(observed_children) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert _active_child_ids().isdisjoint(observed_children)


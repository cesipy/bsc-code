#!/usr/bin/env python3
"""Versioned one-worker-per-objective wrapper for full matched pretraining.

Importing this module is inert. The backend is applied only by an explicit
``configure_backend()`` call or by ``main()`` when launched as a command.
"""

from __future__ import annotations

import argparse
from contextvars import ContextVar
import os
from pathlib import Path
from typing import Any, Sequence

import autoresearch_full_training as _base


BACKEND_ID = "objective-loader-prefetch-v1"
WORKERS_PER_OBJECTIVE_LOADER = 1
OBJECTIVE_LOADER_COUNT = 3
MAIN_CPU_THREADS = 4
WORKER_CPU_THREADS = 1
PREFETCH_FACTOR = 2
PERSISTENT_WORKERS = False
DECLARED_CPU_EXECUTION_SLOTS = MAIN_CPU_THREADS + (
    OBJECTIVE_LOADER_COUNT * WORKERS_PER_OBJECTIVE_LOADER * WORKER_CPU_THREADS
)

# Keep this tuple literal for the runner's AST provenance checks. The base
# driver remains an explicit dependency alongside every file it hashes.
SOURCE_PATHS = (
    "src/autoresearch_full_training_prefetch.py",
    "src/autoresearch_full_training.py",
    "src/config.py",
    "src/vilbert.py",
    "src/attention.py",
    "src/trainer/trainers.py",
    "src/trainer/mlm_loss.py",
    "src/datasets/dataset_pretrain.py",
    "src/datasets/dataset_utils.py",
    "src/datasets/get_datasets.py",
    "src/augments_transforms.py",
    "src/utils.py",
    "src/task.py",
    "src/autoresearch_training_study.py",
)

_ORIGINALS: dict[str, Any] | None = None
_ACTIVE_ITERATORS: ContextVar[set[Any] | None] = ContextVar(
    "full_training_prefetch_active_iterators", default=None
)


def execution_backend_metadata() -> dict[str, Any]:
    """Return the immutable, JSON-serializable execution contract."""
    return {
        "id": BACKEND_ID,
        "version": 1,
        "objective_loader_count": OBJECTIVE_LOADER_COUNT,
        "workers_per_objective_loader": WORKERS_PER_OBJECTIVE_LOADER,
        "loader_worker_processes": OBJECTIVE_LOADER_COUNT * WORKERS_PER_OBJECTIVE_LOADER,
        "main_cpu_threads": MAIN_CPU_THREADS,
        "torch_threads_per_loader_worker": WORKER_CPU_THREADS,
        "declared_cpu_execution_slots": DECLARED_CPU_EXECUTION_SLOTS,
        "prefetch_factor": PREFETCH_FACTOR,
        "persistent_workers": PERSISTENT_WORKERS,
        "pin_memory": "enabled only when CUDA is available, matching base driver",
        "cpu_image_transform_parity_vs_workers_zero": (
            "not guaranteed bitwise: per-row RNG keys are stable, but the 4-thread main process "
            "and 1-thread DataLoader workers have different CPU execution settings; actual-input "
            "image tensor and generated-mask parity requires separate measurement"
        ),
    }


def _validated_workers(workers: int) -> int:
    if isinstance(workers, bool) or int(workers) != WORKERS_PER_OBJECTIVE_LOADER:
        raise _base.FullTrainingError(
            f"{BACKEND_ID} requires exactly {WORKERS_PER_OBJECTIVE_LOADER} worker per objective loader"
        )
    return int(workers)


def _source_hashes() -> dict[str, str]:
    assert _ORIGINALS is not None
    hashes = _ORIGINALS["source_hashes"]()
    # The base helper emits this alias for its own __file__. Keep the normal
    # alias meaning consistent with a directly launched driver: it identifies
    # the executable wrapper; the base driver is also hashed by SOURCE_PATHS.
    hashes["full_training_driver"] = _base.sha256_file(Path(__file__).resolve())
    return hashes


def _make_run_spec(
    arm: str,
    seed: int,
    workers: int,
    inputs: dict[str, Any],
    source_hashes: dict[str, str],
    anchor: dict[str, Any],
) -> dict[str, Any]:
    assert _ORIGINALS is not None
    workers = _validated_workers(workers)
    spec = _ORIGINALS["make_run_spec"](arm, seed, workers, inputs, source_hashes, anchor)
    metadata = execution_backend_metadata()
    spec["schema_version"] = "matched-full-pretraining-run-prefetch-v1"
    spec["execution_backend"] = dict(metadata)
    spec["training"]["execution_backend"] = BACKEND_ID
    spec["training"]["workers"] = workers
    spec["training"]["cpu_threads"] = MAIN_CPU_THREADS
    spec["training"]["prefetch_factor"] = PREFETCH_FACTOR
    spec["training"]["persistent_workers"] = PERSISTENT_WORKERS
    return spec


def _build_parser() -> argparse.ArgumentParser:
    assert _ORIGINALS is not None
    parser = _ORIGINALS["build_parser"]()
    subparser_action = next(
        action for action in parser._actions if isinstance(action, argparse._SubParsersAction)
    )
    train_parser = subparser_action.choices["train"]
    worker_action = next(action for action in train_parser._actions if action.dest == "workers")
    worker_action.default = WORKERS_PER_OBJECTIVE_LOADER
    return parser


class _ManagedDataLoaderIterator:
    """Close a worker-backed DataLoader as soon as its final item is yielded."""

    def __init__(self, loader: "_ManagedDataLoader"):
        self._loader = loader
        self._iterator = iter(loader._loader)
        self._remaining = len(loader)
        self._closed = False
        self._owner_pid = os.getpid()
        registry = _ACTIVE_ITERATORS.get()
        if registry is not None:
            registry.add(self)

    def __iter__(self) -> "_ManagedDataLoaderIterator":
        return self

    def __next__(self) -> Any:
        if self._closed:
            raise StopIteration
        try:
            batch = next(self._iterator)
        except StopIteration:
            self.close(suppress_errors=True)
            raise
        except BaseException:
            self.close(suppress_errors=True)
            raise
        self._remaining -= 1
        if self._remaining <= 0:
            # The delivered final batch is already materialized in the parent;
            # workers no longer need to stay alive for an extra StopIteration.
            self.close()
        return batch

    def close(self, *, suppress_errors: bool = False) -> None:
        if self._closed:
            return
        self._closed = True
        registry = _ACTIVE_ITERATORS.get()
        if registry is not None:
            registry.discard(self)
        # DataLoader workers may inherit this object's ContextVar registry
        # under fork; only the parent process owns/shuts down the iterator.
        if os.getpid() != self._owner_pid:
            return
        shutdown = getattr(self._iterator, "_shutdown_workers", None)
        if callable(shutdown):
            try:
                shutdown()
            except BaseException:
                if not suppress_errors:
                    raise

    def __del__(self) -> None:
        try:
            self.close(suppress_errors=True)
        except BaseException:
            pass


class _ManagedDataLoader:
    """Transparent loader proxy preserving the base driver's accessed API."""

    def __init__(self, loader: Any):
        self._loader = loader

    def __iter__(self) -> _ManagedDataLoaderIterator:
        return _ManagedDataLoaderIterator(self)

    def __len__(self) -> int:
        return len(self._loader)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._loader, name)


def _build_loaders(*args: Any, **kwargs: Any) -> dict[str, _ManagedDataLoader]:
    assert _ORIGINALS is not None
    loaders = _ORIGINALS["build_loaders"](*args, **kwargs)
    return {task: _ManagedDataLoader(loader) for task, loader in loaders.items()}


def _run_training(args: Any, *, injected: Any = None) -> dict[str, Any]:
    assert _ORIGINALS is not None
    active: set[Any] = set()
    token = _ACTIVE_ITERATORS.set(active)
    try:
        return _ORIGINALS["run_training"](args, injected=injected)
    finally:
        # Base-loop exceptions can happen between batch deliveries. Shut down
        # every still-owned worker iterator before the run returns or raises.
        for iterator in tuple(active):
            iterator.close(suppress_errors=True)
        _ACTIVE_ITERATORS.reset(token)


def configure_backend() -> dict[str, Any]:
    """Explicitly configure the base runner for the versioned worker backend."""
    global _ORIGINALS
    if _ORIGINALS is None:
        _ORIGINALS = {
            "source_hashes": _base._source_hashes,
            "make_run_spec": _base._make_run_spec,
            "build_parser": _base._build_parser,
            "build_loaders": _base._build_loaders,
            "run_training": _base.run_training,
        }

    _base.MAX_WORKERS = WORKERS_PER_OBJECTIVE_LOADER
    _base.CPU_THREADS = MAIN_CPU_THREADS
    _base.SOURCE_PATHS = SOURCE_PATHS
    _base._source_hashes = _source_hashes
    _base._make_run_spec = _make_run_spec
    _base._build_parser = _build_parser
    _base._build_loaders = _build_loaders
    _base.run_training = _run_training
    return execution_backend_metadata()


def main(argv: Sequence[str] | None = None) -> int:
    configure_backend()
    return _base.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())

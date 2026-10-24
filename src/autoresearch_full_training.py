#!/usr/bin/env python3
"""Matched, resumable full-recipe DF2 versus REV pretraining runner.

This runner uses the repository's AP/MLM/MIM datasets and PretrainingTrainer
batch methods, but owns its input selection, sampler, checkpointing, and
validation loop. It never constructs HM/test loaders or starts MLflow.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import re
import signal
import sys
import tempfile
import time
from types import MethodType
from typing import Any, Iterable, Mapping, Sequence


def _find_root(module_path: str | Path) -> Path:
    for parent in Path(module_path).resolve().parents:
        if (parent / "src").is_dir() and (parent / "autoresearch").is_dir():
            return parent
    raise RuntimeError(f"cannot locate project root from {module_path}")


ROOT = _find_root(__file__)
AUTORESEARCH = ROOT / "autoresearch"
FEASIBILITY_PATH = AUTORESEARCH / "training_feasibility.json"
DEFAULT_OUTPUT_ROOT = AUTORESEARCH / "full-training/runs"
DEFAULT_INITIALIZATION_ANCHOR = AUTORESEARCH / "full-training/initialization-seed1804-v2.json"
DEFAULT_PREFLIGHT_OUTPUT = DEFAULT_INITIALIZATION_ANCHOR
EXPECTED_TRAIN_ROWS = 500_000
EXPECTED_VALIDATION_ROWS = 8_436
TRAIN_CSV_SHA256 = "a7494b1aca2aadfe63324d1148a9f4fe6636de52a887ccadec3238a0f5923e12"
VALIDATION_CSV_SHA256 = "3cad6a5821c081940f84f9c7e55c9fbe0cc33d43ab19aac430db2de3b1a0a839"
EXPECTED_SEED = 1804
EPOCHS = 7
BATCH_SIZE = 20
GRADIENT_ACCUMULATION = 26
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.01
CHECKPOINT_EVERY_UPDATES = 100
MAX_WORKERS = 0
CPU_THREADS = 4
GPU_MEMORY_FRACTION = 0.90
TASK_KEYS = ("ap", "mlm", "mim")
TASK_NAMES = ("alignment_prediction", "mlm", "mim")
ARM_CONFIGS = {
    "df2": {"text": [3, 4, 5], "vision": [9, 10, 11]},
    "rev": {"text": [9, 10, 11], "vision": [3, 4, 5]},
}
SOURCE_PATHS = (
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


class FullTrainingError(RuntimeError):
    """Raised when a run violates its frozen inputs or execution contract."""


def _fail_closed_invalid_sample(dataset: Any) -> None:
    raise FullTrainingError(
        "a pinned Conceptual Captions image or caption could not be decoded; "
        "refusing the legacy random-row fallback"
    )


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _source_hashes() -> dict[str, str]:
    hashes: dict[str, str] = {}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FullTrainingError(f"required source disappeared: {relative}")
        hashes[relative] = sha256_file(path)
    hashes["full_training_driver"] = sha256_file(__file__)
    return hashes


def _write_json_atomic(path: Path, value: Mapping[str, Any], *, create_only: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FullTrainingError(f"refusing symlink output: {path}")
    if create_only and path.exists():
        raise FullTrainingError(f"refusing to overwrite output: {path}")
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if create_only:
            os.link(temporary_path, path)
        else:
            os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise FullTrainingError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary_path.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _safe_autoresearch_path(path: str | Path, *, must_exist: bool = False) -> Path:
    target = Path(path)
    if target.is_symlink():
        raise FullTrainingError("output paths may not be symlinks")
    resolved = target.resolve(strict=must_exist)
    allowed = AUTORESEARCH.resolve(strict=True)
    if not resolved.is_relative_to(allowed):
        raise FullTrainingError("all driver outputs must remain under autoresearch/")
    manual = Path("/home/cedric.sillaber/coding/bsc-code").resolve(strict=False)
    if resolved.is_relative_to(manual) or manual.is_relative_to(resolved):
        raise FullTrainingError("output path overlaps the protected manual checkout")
    return resolved


def _ensure_project_imports() -> None:
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)


def _runtime_modules() -> dict[str, Any]:
    _ensure_project_imports()
    import numpy as np
    import torch
    torch.set_num_threads(CPU_THREADS)
    import utils
    from config import VIT_MODEL_NAME, ViLBERTConfig
    from task import Task
    from trainer import PretrainingTrainer

    return {
        "np": np,
        "torch": torch,
        "utils": utils,
        "ViLBERTConfig": ViLBERTConfig,
        "VIT_MODEL_NAME": VIT_MODEL_NAME,
        "Task": Task,
        "PretrainingTrainer": PretrainingTrainer,
    }


def configure_runtime(threads: int = 4) -> None:
    if isinstance(threads, bool) or not 1 <= int(threads) <= 4:
        raise FullTrainingError("CPU thread count must be in [1,4]")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(threads)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True


def _make_config(arm: str, seed: int, workers: int, modules: Mapping[str, Any]) -> Any:
    if arm not in ARM_CONFIGS:
        raise FullTrainingError("arm must be df2 or rev")
    if seed != EXPECTED_SEED:
        raise FullTrainingError(f"this paired comparison is fixed to seed {EXPECTED_SEED}")
    if isinstance(workers, bool) or not 0 <= int(workers) <= MAX_WORKERS:
        raise FullTrainingError(f"workers must be exactly {MAX_WORKERS} for this run contract")
    placement = ARM_CONFIGS[arm]
    Task = modules["Task"]
    return modules["ViLBERTConfig"](
        text_cross_attention_layers=list(placement["text"]),
        vision_cross_attention_layers=list(placement["vision"]),
        cross_attention_backend="eager",
        epochs=EPOCHS,
        learning_rate=LEARNING_RATE,
        seed=seed,
        use_contrastive_loss=False,
        mlm_selected_tokens_only=True,
        pretrain_batch_size=BATCH_SIZE,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        pretraining_tasks=[Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM],
        num_workers=workers,
        prefetch=2 if workers > 0 else None,
        persistent_workers=False,
        pin_memory=True,
    )


def _hash_model_state(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(str(tensor.dtype).encode("ascii") + b"\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
        digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))
    return digest.hexdigest()


def _hash_torch_rng(torch: Any) -> str:
    return hashlib.sha256(torch.get_rng_state().cpu().numpy().tobytes()).hexdigest()


def _load_assets(feasibility_path: str | Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    feasibility_file = Path(feasibility_path).resolve(strict=True)
    feasibility = json.loads(feasibility_file.read_text(encoding="utf-8"))
    if feasibility.get("schema_version") != "training-feasibility-v2":
        raise FullTrainingError("unsupported local training-feasibility record")
    training = feasibility["training_data"]
    train_record = training["full_train_csv"]
    validation_record = training["pilot_cohorts"]["validation_manifest"]
    train_path = Path(train_record["path"]).resolve(strict=True)
    validation_path = Path(validation_record["source_csv_path"]).resolve(strict=True)
    if sha256_file(train_path) != TRAIN_CSV_SHA256 or train_record.get("sha256") != TRAIN_CSV_SHA256:
        raise FullTrainingError("full CC training CSV does not match its frozen SHA-256")
    if sha256_file(validation_path) != VALIDATION_CSV_SHA256 or validation_record.get("source_csv_sha256") != VALIDATION_CSV_SHA256:
        raise FullTrainingError("CC validation CSV does not match its frozen SHA-256")
    inputs = {
        "feasibility_path": str(feasibility_file),
        "feasibility_sha256": sha256_file(feasibility_file),
        "train_csv_path": str(train_path),
        "train_csv_sha256": TRAIN_CSV_SHA256,
        "validation_csv_path": str(validation_path),
        "validation_csv_sha256": VALIDATION_CSV_SHA256,
        "image_root": str(Path(training["image_root"]).resolve(strict=True)),
        "image_integrity_policy": "all pinned CSV paths are resolved and every row is required to exist; individual image bytes are read-only inputs and are not pre-hashed",
    }
    if len(feasibility["training_data"].get("integrity_check", {})) == 0:
        raise FullTrainingError("feasibility record lacks the local input integrity audit")
    return feasibility, inputs, {"train": train_record, "validation": validation_record}


def _source_model_factory() -> tuple[Any, dict[str, Any]]:
    _ensure_project_imports()
    from autoresearch_training_study import construct_fresh_model, load_feasibility_inputs, validate_local_assets

    feasibility = load_feasibility_inputs(FEASIBILITY_PATH)
    assets = validate_local_assets(feasibility)
    return construct_fresh_model, assets


def _set_seed(seed: int, modules: Mapping[str, Any], *, cuda: bool = False) -> None:
    np = modules["np"]
    torch = modules["torch"]
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.random.default_generator.manual_seed(seed)
    if cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _construct_model(arm: str, seed: int, modules: Mapping[str, Any], assets: Mapping[str, Any]) -> Any:
    from config import VIT_MODEL_NAME

    config = _make_config(arm, seed, workers=0, modules=modules)
    construct_fresh_model, _ = _source_model_factory()
    _set_seed(seed, modules, cuda=False)
    model = construct_fresh_model(
        config,
        bert_snapshot=Path(assets["bert_snapshot"]),
        vit_weights=Path(assets["vision_weights_path"]),
        vit_model_name=VIT_MODEL_NAME,
    )
    return model


def create_initialization_preflight(
    *,
    output_path: str | Path = DEFAULT_PREFLIGHT_OUTPUT,
    seed: int = EXPECTED_SEED,
    modules: Mapping[str, Any] | None = None,
    model_factory: Any | None = None,
    asset_record: Mapping[str, Any] | None = None,
    source_hashes: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if seed != EXPECTED_SEED:
        raise FullTrainingError(f"paired full training is fixed to seed {EXPECTED_SEED}")
    output = _safe_autoresearch_path(output_path)
    if output.exists():
        raise FullTrainingError(f"refusing to overwrite initialization preflight: {output}")
    if modules is None:
        configure_runtime()
        modules = _runtime_modules()
    if model_factory is None or asset_record is None:
        _feasibility, _inputs, _records = _load_assets(FEASIBILITY_PATH)
        construct_fresh_model, real_assets = _source_model_factory()

        def model_factory(config: Any, _arm: str, assets: Mapping[str, Any]) -> Any:
            return construct_fresh_model(
                config,
                bert_snapshot=Path(assets["bert_snapshot"]),
                vit_weights=Path(assets["vision_weights_path"]),
                vit_model_name=assets["vision_model_name"],
            )

        asset_record = real_assets
    if source_hashes is None:
        source_hashes = _source_hashes()

    state_hashes: dict[str, str] = {}
    post_init_rng_hashes: dict[str, str] = {}
    config_records: dict[str, Any] = {}
    for arm in ("df2", "rev"):
        _set_seed(seed, modules, cuda=False)
        config = _make_config(arm, seed, workers=0, modules=modules)
        model = model_factory(config, arm, asset_record)
        state_hashes[arm] = _hash_model_state(model.state_dict(), modules["torch"])
        post_init_rng_hashes[arm] = _hash_torch_rng(modules["torch"])
        config_records[arm] = {
            "text_cross_attention_layers_zero_based": ARM_CONFIGS[arm]["text"],
            "vision_cross_attention_layers_zero_based": ARM_CONFIGS[arm]["vision"],
            "backend": "eager",
            "mlm_projection": "selected-token; same CrossEntropy objective as dense projection",
            "pretrain_epochs": EPOCHS,
            "learning_rate": LEARNING_RATE,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "pretraining_tasks": list(TASK_NAMES),
        }
        del model
    if state_hashes["df2"] != state_hashes["rev"]:
        raise FullTrainingError("DF2 and REV initial model tensors differ for the paired seed")
    if post_init_rng_hashes["df2"] != post_init_rng_hashes["rev"]:
        raise FullTrainingError("DF2 and REV initialization consumed different CPU RNG sequences")
    artifact = {
        "schema_version": "full-training-paired-initialization-v1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "passed",
        "seed": seed,
        "initial_model_state_sha256_by_arm": state_hashes,
        "post_initialization_cpu_rng_sha256_by_arm": post_init_rng_hashes,
        "configs": config_records,
        "offline_backbones": dict(asset_record),
        "source_sha256": dict(source_hashes),
        "pairing_claim": "fresh fusion/cross-attention/task parameters and local BERT/ViT weights are tensor-identical across DF2 and REV before training; placement metadata differs",
    }
    _write_json_atomic(output, artifact, create_only=True)
    return artifact


def _dataset_setup_rng(seed: int, np: Any, torch: Any):
    @contextlib.contextmanager
    def context():
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state()
        random.seed(seed)
        np.random.seed(seed % (2**32))
        torch.random.default_generator.manual_seed(seed)
        try:
            yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)
            torch.random.set_rng_state(torch_state)
    return context()


def _resolve_image_path(raw_path: str, root: Path, image_root: Path) -> Path | None:
    raw = Path(raw_path)
    candidates = [raw] if raw.is_absolute() else [root / raw, image_root / raw]
    repository_images = Path("res/data/conceptual-captions/images")
    if not raw.is_absolute() and raw.is_relative_to(repository_images):
        candidates.insert(0, image_root / raw.relative_to(repository_images))
    for candidate in candidates:
        try:
            resolved = candidate.resolve(strict=True)
        except OSError:
            continue
        if resolved.is_file():
            return resolved
    return None


def load_cc_pairs(
    csv_path: str | Path,
    *,
    root: str | Path,
    image_root: str | Path,
    expected_count: int,
    limit: int | None = None,
) -> tuple[list[tuple[str, str]], dict[str, Any]]:
    """Read only the pinned CC CSV and resolve paths without changing CWD."""
    csv_file = Path(csv_path).resolve(strict=True)
    base = Path(root).resolve(strict=True)
    images = Path(image_root).resolve(strict=True)
    pairs: list[tuple[str, str]] = []
    source_rows = 0
    missing_images = 0
    with csv_file.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.reader(stream, quotechar='"')
        try:
            header = next(reader)
        except StopIteration as exc:
            raise FullTrainingError(f"empty CSV: {csv_file}") from exc
        normalized_header = [name.strip().lstrip("\ufeff").casefold() for name in header]
        if len(normalized_header) != len(set(normalized_header)):
            raise FullTrainingError(f"CC CSV contains duplicate column names: {csv_file}")
        header_indices = {name: index for index, name in enumerate(normalized_header)}
        if "text" not in header_indices or "filepath" not in header_indices:
            raise FullTrainingError(
                f"CC CSV must contain named 'text' and 'filepath' columns: {csv_file}"
            )
        caption_index = header_indices["text"]
        image_index = header_indices["filepath"]
        required_width = max(caption_index, image_index) + 1
        for row in reader:
            source_rows += 1
            if len(row) < required_width:
                raise FullTrainingError(f"malformed CSV row {source_rows + 1} in {csv_file.name}")
            image = _resolve_image_path(row[image_index], base, images)
            if image is None:
                missing_images += 1
                continue
            pairs.append((str(image), row[caption_index]))
            if limit is not None and len(pairs) >= limit:
                break
    if len(pairs) != expected_count:
        raise FullTrainingError(
            f"{csv_file.name} resolved {len(pairs)} usable pairs; expected {expected_count}"
        )
    if source_rows != expected_count or missing_images:
        raise FullTrainingError(
            f"{csv_file.name} must resolve every pinned row exactly once; "
            f"read {source_rows} rows and skipped {missing_images} missing images"
        )
    digest = hashlib.sha256()
    for image_path, caption in pairs:
        digest.update(image_path.encode("utf-8") + b"\0" + caption.encode("utf-8") + b"\0")
    return pairs, {
        "csv_path": str(csv_file),
        "csv_sha256": sha256_file(csv_file),
        "source_rows_read": source_rows,
        "usable_pair_count": len(pairs),
        "missing_image_rows_skipped": missing_images,
        "ordered_resolved_pairs_sha256": digest.hexdigest(),
        "csv_header": header,
    }


def _stable_item_seed(base_seed: int, split: str, task: str, epoch: int, index: int) -> int:
    value = f"{base_seed}:{split}:{task}:{epoch}:{index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(value).digest()[:8], "big") & 0x7FFFFFFF


class IsolatedRandomnessDataset:
    """Wrap a legacy stochastic dataset with independent per-row RNG streams."""

    def __init__(self, dataset: Any, *, seed: int, split: str, task: str):
        self.dataset = dataset
        self.seed = seed
        self.split = split
        self.task = task
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> Any:
        import numpy as np
        import torch

        numpy_state = np.random.get_state()
        python_state = random.getstate()
        seed = _stable_item_seed(self.seed, self.split, self.task, self.epoch, int(index))
        random.seed(seed)
        np.random.seed(seed % (2**32))
        with torch.random.fork_rng(devices=[]):
            torch.random.default_generator.manual_seed(seed)
            try:
                return self.dataset[index]
            finally:
                random.setstate(python_state)
                np.random.set_state(numpy_state)


def _dataset_identity(dataset: Any) -> str:
    digest = hashlib.sha256()
    for record in dataset.data:
        values = tuple(str(value.value if hasattr(value, "value") else value) for value in record)
        digest.update("\0".join(values).encode("utf-8") + b"\n")
    return digest.hexdigest()


def _construct_datasets(
    train_pairs: Sequence[tuple[str, str]],
    validation_pairs: Sequence[tuple[str, str]],
    *,
    seed: int,
    modules: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    _ensure_project_imports()
    from transformers import BertTokenizerFast
    from datasets.dataset_pretrain import PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
    import augments_transforms
    from autoresearch_training_study import validate_local_assets, load_feasibility_inputs

    feasibility = load_feasibility_inputs(FEASIBILITY_PATH)
    asset_record = validate_local_assets(feasibility)
    tokenizer_path = asset_record["tokenizer_path"]
    tokenizer = BertTokenizerFast.from_pretrained(tokenizer_path, local_files_only=True)
    datasets_by_split: dict[str, dict[str, Any]] = {"train": {}, "validation": {}}
    for split, pairs in (("train", train_pairs), ("validation", validation_pairs)):
        with _dataset_setup_rng(seed, modules["np"], modules["torch"]):
            raw_datasets = {
                "ap": PretrainDatasetAP(pairs, tokenizer, image_processor=None, is_train=(split == "train")),
                "mlm": PretrainDatasetMLM(pairs, tokenizer, image_processor=None),
                "mim": PretrainDatasetMIM(
                    pairs,
                    tokenizer,
                    image_processor=None,
                    transforms_weak=augments_transforms.get_transform_unmasked(),
                    transforms_strong=augments_transforms.get_transform_masked(),
                ),
            }
        for task, dataset in raw_datasets.items():
            # Legacy datasets replace corrupt rows with a random fallback. For
            # a paired full-data run, that silently changes sample identity, so
            # fail at the first corrupt image instead.
            dataset.handle_fallback = MethodType(_fail_closed_invalid_sample, dataset)
            datasets_by_split[split][task] = IsolatedRandomnessDataset(
                dataset,
                seed=seed,
                split=split,
                task=task,
            )
    order_hashes = {
        split: {task: _dataset_identity(wrapper.dataset) for task, wrapper in split_data.items()}
        for split, split_data in datasets_by_split.items()
    }
    return datasets_by_split["train"], datasets_by_split["validation"], {
        "tokenizer_path": tokenizer_path,
        "tokenizer_files_sha256": asset_record["tokenizer_files_sha256"],
        "dataset_order_sha256": order_hashes,
        "rng_policy": "per-row stateless Python/NumPy/CPU-Torch RNG keyed by independent data seed, split, task, epoch, and dataset index; training model RNG is restored around data access",
    }


def _epoch_order(length: int, seed: int, epoch: int, torch: Any) -> list[int]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed((seed + 1_000_003 * epoch) % (2**63 - 1))
    return torch.randperm(length, generator=generator).tolist()


def _make_loader(dataset: Any, *, order: Sequence[int], start_batch: int, epoch: int, task: str,
                 seed: int, workers: int, torch: Any, pin_memory: bool) -> Any:
    from torch.utils.data import DataLoader

    dataset.epoch = epoch if dataset.split == "train" else 0
    sampler = order[start_batch * BATCH_SIZE:]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_item_seed(seed, dataset.split, task, epoch, 0))
    options: dict[str, Any] = {
        "dataset": dataset,
        "batch_size": BATCH_SIZE,
        "sampler": sampler,
        "drop_last": False,
        "num_workers": workers,
        "pin_memory": pin_memory,
        "persistent_workers": False,
        "generator": generator,
    }
    if workers > 0:
        options["prefetch_factor"] = 2
    return DataLoader(**options)


def _capture_rng(torch: Any, np: Any) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def _restore_rng(state: Mapping[str, Any], torch: Any, np: Any) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    cuda_state = state.get("torch_cuda")
    if cuda_state is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(cuda_state)


def _atomic_save_checkpoint(path: Path, payload: Mapping[str, Any]) -> int:
    """Keep latest and previous full states; hardlink the old latest during rotation."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FullTrainingError(f"refusing symlink checkpoint: {path}")
    previous = path.with_name("resume_previous.pt")
    if previous.is_symlink():
        raise FullTrainingError(f"refusing symlink checkpoint: {previous}")
    if path.exists():
        previous.unlink(missing_ok=True)
        os.link(path, previous)
    fd, name = tempfile.mkstemp(prefix=".resume_latest.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        import torch

        torch.save(dict(payload), temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)
    return path.stat().st_size


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink():
        raise FullTrainingError(f"refusing symlink log: {path}")
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(encoded + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _reconcile_jsonl(path: Path, *, expected_records: int, sequence_key: str) -> None:
    """Make append-only logs agree with the last durable optimizer checkpoint."""
    if expected_records < 0:
        raise FullTrainingError("checkpoint log cursor must be nonnegative")
    if path.is_symlink():
        raise FullTrainingError(f"refusing symlink log: {path}")
    if not path.exists():
        if expected_records:
            raise FullTrainingError(f"checkpoint expects missing log records: {path.name}")
        return
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) < expected_records:
        raise FullTrainingError(f"{path.name} is shorter than its durable checkpoint cursor")
    orphaned = b"".join(raw_lines[expected_records:])
    if orphaned:
        digest = hashlib.sha256(orphaned).hexdigest()
        archive = path.with_name(f"orphaned_{path.stem}_{digest[:16]}.jsonl")
        if archive.exists():
            if archive.is_symlink() or archive.read_bytes() != orphaned:
                raise FullTrainingError(f"orphan archive collision: {archive.name}")
        else:
            _write_bytes_create_only(archive, orphaned)
        _append_jsonl(path.parent / "resume_reconciliation.jsonl", {
            "log": path.name,
            "checkpoint_records": expected_records,
            "orphaned_bytes": len(orphaned),
            "orphaned_sha256": digest,
            "archive": archive.name,
        })
    retained: list[bytes] = []
    for index, raw_line in enumerate(raw_lines[:expected_records], start=1):
        try:
            record = json.loads(raw_line)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise FullTrainingError(f"invalid durable row {index} in {path.name}") from exc
        if int(record.get(sequence_key, -1)) != index:
            raise FullTrainingError(f"noncontiguous {sequence_key} in {path.name} at row {index}")
        retained.append(raw_line.rstrip(b"\r\n") + b"\n")
    if len(raw_lines) != expected_records or any(not line.endswith(b"\n") for line in raw_lines[:expected_records]):
        fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
        temporary = Path(name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.writelines(retained)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
            _fsync_directory(path.parent)
        finally:
            temporary.unlink(missing_ok=True)


def _write_bytes_create_only(path: Path, content: bytes) -> None:
    if path.exists() or path.is_symlink():
        raise FullTrainingError(f"refusing to overwrite output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise FullTrainingError(f"refusing to overwrite output: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _ensure_first20_gate(run_dir: Path, *, attempts: int) -> dict[str, Any] | None:
    path = run_dir / "first20_gate.json"
    if path.is_symlink():
        raise FullTrainingError("refusing symlink first20 gate")
    if attempts < 20:
        path.unlink(missing_ok=True)
        return None
    if path.is_file():
        gate = json.loads(path.read_text(encoding="utf-8"))
        if gate.get("attempts") != 20:
            raise FullTrainingError("first20 gate artifact has the wrong attempt count")
        return gate
    rows_path = run_dir / "optimizer_attempts.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text(encoding="utf-8").splitlines()[:20]]
    if len(rows) < 20:
        raise FullTrainingError("first20 attempt records are missing")
    first20_applied = sum(row.get("actual_optimizer_step") is True for row in rows)
    first20_skipped = sum(row.get("actual_optimizer_step") is False for row in rows)
    gate = {
        "status": "passed" if first20_applied > 0 else "zero_applied_updates",
        "attempts": 20,
        "applied_updates": first20_applied,
        "skipped_updates": first20_skipped,
        "wall_seconds": sum(float(row.get("attempt_wall_seconds", 0.0)) for row in rows),
        "peak_gpu_allocated_bytes": max((int(row.get("gpu_peak_allocated_bytes", 0)) for row in rows), default=0),
        "peak_gpu_reserved_bytes": max((int(row.get("gpu_peak_reserved_bytes", 0)) for row in rows), default=0),
        "reconstructed_from_durable_attempt_log": True,
        "interpretation": "engineering gate only; no scientific outcome or automatic response to ordinary AMP skips",
    }
    _write_json_atomic(path, gate, create_only=True)
    return gate


def _optimizer_step_counter(optimizer: Any, model: Any) -> int:
    for parameter in model.parameters():
        state = optimizer.state.get(parameter)
        if state and "step" in state:
            value = state["step"]
            return int(value.item() if hasattr(value, "item") else value)
    return 0


def _write_model_checkpoint(path: Path, model: Any) -> None:
    payload = {"model_state_dict": model.state_dict(), "config": dict(model.config.__dict__)}
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise FullTrainingError(f"refusing to overwrite model checkpoint: {path}")
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        import torch

        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    except FileExistsError as exc:
        raise FullTrainingError(f"refusing to overwrite model checkpoint: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def _atomic_replace_model_checkpoint(path: Path, model: Any) -> None:
    if path.is_symlink():
        raise FullTrainingError(f"refusing symlink model checkpoint: {path}")
    payload = {"model_state_dict": model.state_dict(), "config": dict(model.config.__dict__)}
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        import torch

        torch.save(payload, temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _make_run_spec(arm: str, seed: int, workers: int, inputs: Mapping[str, Any],
                   source_hashes: Mapping[str, str], anchor: Mapping[str, Any]) -> dict[str, Any]:
    placement = ARM_CONFIGS[arm]
    loader_batches = EXPECTED_TRAIN_ROWS // BATCH_SIZE
    historical_scheduler_steps = EPOCHS * (3 * loader_batches) // GRADIENT_ACCUMULATION
    actual_attempts = EPOCHS * math.ceil(loader_batches / GRADIENT_ACCUMULATION)
    warmup_iterations = int(0.1 * historical_scheduler_steps)
    decay_iterations = int(0.9 * historical_scheduler_steps)
    min_lr_fraction = 0.2
    if actual_attempts <= warmup_iterations:
        final_lr_fraction = (actual_attempts + 1) / (warmup_iterations + 1)
    elif actual_attempts > decay_iterations:
        final_lr_fraction = min_lr_fraction
    else:
        decay_ratio = (actual_attempts - warmup_iterations) / (decay_iterations - warmup_iterations)
        final_lr_fraction = min_lr_fraction + 0.5 * (1 + math.cos(math.pi * decay_ratio)) * (1 - min_lr_fraction)
    return {
        "schema_version": "matched-full-pretraining-run-v1",
        "arm": arm,
        "seed": seed,
        "text_cross_attention_layers_zero_based": placement["text"],
        "vision_cross_attention_layers_zero_based": placement["vision"],
        "initial_model_state_sha256": anchor["initial_model_state_sha256_by_arm"][arm],
        "inputs": dict(inputs),
        "source_sha256": dict(source_hashes),
        "training": {
            "epochs": EPOCHS,
            "training_examples": EXPECTED_TRAIN_ROWS,
            "validation_examples": EXPECTED_VALIDATION_ROWS,
            "tasks": list(TASK_NAMES),
            "contrastive_ap": False,
            "batch_size": BATCH_SIZE,
            "gradient_accumulation": GRADIENT_ACCUMULATION,
            "learning_rate": LEARNING_RATE,
            "weight_decay": WEIGHT_DECAY,
            "optimizer": "AdamW",
            "attention_backend": "eager",
            "mlm_projection": "selected-token",
            "mim_preprocessing": "legacy dataset pipeline, no scale modification",
            "precision": "existing PretrainingTrainer autocast float16 + GradScaler on CUDA",
            "grad_scaler": {
                "initial_scale": 65536.0,
                "growth_factor": 2.0,
                "backoff_factor": 0.5,
                "growth_interval": 2000,
                "gradient_clipping": None,
            },
            "scheduler": {
                "source": "ExperimentTracker._run_pretrain summed three loader lengths; preserved here",
                "train_loader_batches_per_task_per_epoch": loader_batches,
                "scheduler_total_steps": historical_scheduler_steps,
                "warmup_fraction": 0.1,
                "decay_fraction": 0.9,
                "minimum_lr_fraction": min_lr_fraction,
                "warmup_iterations": warmup_iterations,
                "decay_iterations": decay_iterations,
                "actual_optimizer_attempts_per_epoch": math.ceil(loader_batches / GRADIENT_ACCUMULATION),
                "actual_optimizer_attempts_total": actual_attempts,
                "tail_accumulation_microbatches_per_epoch": loader_batches % GRADIENT_ACCUMULATION,
                "tail_loss_divisor": GRADIENT_ACCUMULATION,
                "attempt_includes_amp_skips": True,
                "lr_fraction_at_last_actual_attempt": final_lr_fraction,
            },
            "data_seed": seed,
            "data_rng_independent_from_model_dropout_rng": True,
            "workers": workers,
            "cpu_threads": CPU_THREADS,
            "gpu_memory_fraction_limit": GPU_MEMORY_FRACTION,
            "prefetch_factor": 2 if workers > 0 else None,
            "persistent_workers": False,
            "checkpoint_every_optimizer_attempts": CHECKPOINT_EVERY_UPDATES,
            "signal_handling": "SIGTERM/SIGINT sets request; save at next optimizer boundary and exit with interrupted status",
            "initialization_pair_verification": "full model state hashes match the same-seed preflight anchor before optimizer creation",
            "validation": "CC validation.csv only; all three task losses each epoch; no HM/test/MLflow data",
            "primary_checkpoint": "final epoch, saved as ViLBERT.load_model-compatible resume_latest.pt",
            "auxiliary_best_checkpoint_selection": "minimum sum of the three per-task CC validation losses; final epoch remains primary",
            "first_20_attempt_gate": "report wall, CUDA peak allocated/reserved, loss, scale and applied/skipped counts; abort only if zero optimizer calls applied in all first 20 attempts",
        },
        "interpretation": "matched full-exposure placement comparison with selected-token MLM optimization; source scheduler and epoch-tail behavior are preserved, but exact historical run replication is unverified",
    }


def _load_initialization_anchor(path: str | Path, source_hashes: Mapping[str, str]) -> dict[str, Any]:
    anchor_path = Path(path).resolve(strict=True)
    anchor = json.loads(anchor_path.read_text(encoding="utf-8"))
    if anchor.get("schema_version") != "full-training-paired-initialization-v1" or anchor.get("status") != "passed":
        raise FullTrainingError("initialization anchor is missing or not passed")
    if anchor.get("seed") != EXPECTED_SEED:
        raise FullTrainingError("initialization anchor seed differs from the frozen run seed")
    if anchor.get("source_sha256") != dict(source_hashes):
        raise FullTrainingError("initialization anchor source hashes differ from the frozen driver source")
    return anchor


def _build_loaders(
    datasets_by_task: Mapping[str, Any], *,
    split: str,
    epoch: int,
    start_batch: int,
    seed: int,
    workers: int,
    torch: Any,
) -> dict[str, Any]:
    if split == "train":
        order = _epoch_order(len(next(iter(datasets_by_task.values()))), seed, epoch, torch)
    else:
        order = list(range(len(next(iter(datasets_by_task.values())))))
    return {
        task: _make_loader(
            dataset,
            order=order,
            start_batch=(start_batch if split == "train" else 0),
            epoch=epoch,
            task=task,
            seed=seed,
            workers=workers,
            torch=torch,
            pin_memory=(torch.cuda.is_available()),
        )
        for task, dataset in datasets_by_task.items()
    }


def _validation(trainer: Any, loaders: Mapping[str, Any]) -> dict[str, Any]:
    ap_loss, ap_accuracy = trainer.evaluate_ap(loaders["ap"])
    mlm_loss = trainer.evaluate_mlm(loaders["mlm"])
    mim_loss = trainer.evaluate_mim(loaders["mim"])
    values = {"ap_loss": float(ap_loss), "ap_accuracy": float(ap_accuracy),
              "mlm_loss": float(mlm_loss), "mim_loss": float(mim_loss)}
    values["selection_sum_loss"] = values["ap_loss"] + values["mlm_loss"] + values["mim_loss"]
    values["all_finite"] = all(math.isfinite(value) for value in values.values() if isinstance(value, float))
    return values


def _checkpoint_payload(
    *, model: Any, trainer: Any, progress: Mapping[str, Any], spec: Mapping[str, Any],
    source_hashes: Mapping[str, str], np: Any, torch: Any, data_identity: Mapping[str, Any],
    totals: Mapping[str, Any], best_score: float | None, best_epoch: int | None,
) -> dict[str, Any]:
    return {
        "format": "matched-full-pretraining-resume-v1",
        "model_state_dict": model.state_dict(),
        "config": dict(model.config.__dict__),
        "optimizer_state_dict": trainer.optimizer.state_dict(),
        "scaler_state_dict": trainer.scaler.state_dict(),
        "scheduler_state": dict(vars(trainer.scheduler)),
        "rng_state": _capture_rng(torch, np),
        "progress": dict(progress),
        "run_spec_sha256": sha256_json(spec),
        "source_sha256": dict(source_hashes),
        "data_identity": dict(data_identity),
        "totals": dict(totals),
        "best_validation_score": best_score,
        "best_validation_epoch": best_epoch,
        "saved_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _restore_checkpoint(
    path: Path, *, model: Any, trainer: Any, spec: Mapping[str, Any],
    source_hashes: Mapping[str, str], data_identity: Mapping[str, Any], np: Any, torch: Any,
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "matched-full-pretraining-resume-v1":
        raise FullTrainingError("resume checkpoint has an unsupported format")
    if checkpoint.get("run_spec_sha256") != sha256_json(spec):
        raise FullTrainingError("resume checkpoint run configuration differs")
    if checkpoint.get("source_sha256") != dict(source_hashes):
        raise FullTrainingError("resume checkpoint source hashes differ")
    if checkpoint.get("data_identity") != dict(data_identity):
        raise FullTrainingError("resume checkpoint data identity differs")
    model.load_state_dict(checkpoint["model_state_dict"])
    trainer.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    trainer.scaler.load_state_dict(checkpoint["scaler_state_dict"])
    for name, value in checkpoint["scheduler_state"].items():
        setattr(trainer.scheduler, name, value)
    _restore_rng(checkpoint["rng_state"], torch, np)
    return checkpoint


def _install_signal_handlers() -> tuple[Any, dict[str, Any]]:
    state: dict[str, Any] = {"requested": None}
    previous: dict[int, Any] = {}

    def handler(signum: int, _frame: Any) -> None:
        state["requested"] = signum

    for signum in (signal.SIGTERM, signal.SIGINT):
        previous[signum] = signal.getsignal(signum)
        signal.signal(signum, handler)
    return previous, state


def _restore_signal_handlers(previous: Mapping[int, Any]) -> None:
    for signum, handler in previous.items():
        signal.signal(signum, handler)


def _logical_cuda_device(value: str) -> str:
    """Use the first accelerator exposed by CUDA_VISIBLE_DEVICES explicitly."""
    if value not in {"cuda", "cuda:0"}:
        raise FullTrainingError(
            "select the physical accelerator with CUDA_VISIBLE_DEVICES and pass --device cuda or cuda:0"
        )
    return "cuda:0"


def run_training(args: argparse.Namespace, *, injected: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if args.arm not in ARM_CONFIGS:
        raise FullTrainingError("arm must be df2 or rev")
    if args.seed != EXPECTED_SEED:
        raise FullTrainingError(f"seed must be exactly {EXPECTED_SEED}")
    logical_device = _logical_cuda_device(args.device) if injected is None else None
    if isinstance(args.workers, bool) or int(args.workers) != MAX_WORKERS:
        raise FullTrainingError(f"workers must be exactly {MAX_WORKERS} for this run contract")
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", args.run_id):
        raise FullTrainingError("run-id must use only letters, digits, underscore, and dash")
    output_root = _safe_autoresearch_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    run_dir = output_root / args.run_id
    if run_dir.is_symlink():
        raise FullTrainingError("run directory may not be a symlink")
    if args.resume:
        resume_checkpoint = run_dir / "resume_latest.pt"
        if not run_dir.is_dir() or resume_checkpoint.is_symlink() or not resume_checkpoint.is_file():
            raise FullTrainingError("--resume requires an existing run with resume_latest.pt")
        result_path = run_dir / "result.json"
        if result_path.is_symlink():
            raise FullTrainingError("result.json may not be a symlink")
        if result_path.is_file():
            prior_result = json.loads(result_path.read_text(encoding="utf-8"))
            if prior_result.get("status") == "completed":
                raise FullTrainingError("completed run IDs cannot be resumed")
            if prior_result.get("status") in {"no_applied_updates_gate", "persistent_skipped_updates"}:
                raise FullTrainingError("the persistent skipped-update stop requires parent adjudication")
    elif run_dir.exists():
        raise FullTrainingError(f"run directory already exists: {run_dir}")
    else:
        run_dir.mkdir(parents=True, exist_ok=False)

    previous_handlers, signal_state = _install_signal_handlers()
    started = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    original_cwd = Path.cwd()
    try:
        os.chdir(run_dir)
        if injected is None:
            configure_runtime()
            modules = _runtime_modules()
            torch = modules["torch"]
            np = modules["np"]
            if not torch.cuda.is_available():
                raise FullTrainingError("full training requires CUDA; use CPU-only tests for validation")
            device = torch.device(logical_device)
            if device.type != "cuda":
                raise FullTrainingError("full training device must be cuda")
            torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=device)
            _feasibility, inputs, _records = _load_assets(FEASIBILITY_PATH)
            source_hashes = _source_hashes()
            anchor = _load_initialization_anchor(args.initialization_anchor, source_hashes)
            train_pairs, train_identity = load_cc_pairs(
                inputs["train_csv_path"], root=ROOT, image_root=inputs["image_root"],
                expected_count=EXPECTED_TRAIN_ROWS, limit=EXPECTED_TRAIN_ROWS,
            )
            val_pairs, val_identity = load_cc_pairs(
                inputs["validation_csv_path"], root=ROOT, image_root=inputs["image_root"],
                expected_count=EXPECTED_VALIDATION_ROWS,
            )
            datasets_train, datasets_val, dataset_metadata = _construct_datasets(
                train_pairs, val_pairs, seed=args.seed, modules=modules
            )
            data_identity = {
                "inputs": inputs,
                "train_pairs": train_identity,
                "validation_pairs": val_identity,
                "dataset_metadata": dataset_metadata,
            }
            construct_fresh_model, assets = _source_model_factory()
            config = _make_config(args.arm, args.seed, args.workers, modules)
            _set_seed(args.seed, modules, cuda=False)
            model = construct_fresh_model(
                config,
                bert_snapshot=Path(assets["bert_snapshot"]),
                vit_weights=Path(assets["vision_weights_path"]),
                vit_model_name=modules["VIT_MODEL_NAME"],
            )
            initial_hash = _hash_model_state(model.state_dict(), torch)
            if initial_hash != anchor["initial_model_state_sha256_by_arm"][args.arm]:
                raise FullTrainingError("fresh run initialization differs from paired preflight")
            model.to(device)
            trainer = modules["PretrainingTrainer"](
                model=model,
                config=config,
                tasks=config.pretraining_tasks,
                use_contrastive_loss=False,
                gradient_accumulation=GRADIENT_ACCUMULATION,
            )
            trainer.scaler = torch.amp.GradScaler(
                device=device.type,
                init_scale=65536.0,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=True,
            )
            train_lengths = {name: len(value) for name, value in datasets_train.items()}
            val_lengths = {name: len(value) for name, value in datasets_val.items()}
            if set(train_lengths.values()) != {EXPECTED_TRAIN_ROWS} or set(val_lengths.values()) != {EXPECTED_VALIDATION_ROWS}:
                raise FullTrainingError("constructed task datasets differ in size")
            # Match ExperimentTracker._run_pretrain: three loader lengths feed
            # the LR schedule, while one zipped triad is one microbatch.
            trainer.setup_scheduler(
                EPOCHS,
                total_training_steps=sum(math.ceil(size / BATCH_SIZE) for size in train_lengths.values()),
                lr=LEARNING_RATE,
            )
            expected_scheduler_steps = EPOCHS * sum(
                math.ceil(size / BATCH_SIZE) for size in train_lengths.values()
            ) // GRADIENT_ACCUMULATION
            if (
                trainer.scheduler.warmup_iterations != int(0.1 * expected_scheduler_steps)
                or trainer.scheduler.decay_iterations != int(0.9 * expected_scheduler_steps)
                or trainer.scheduler.min_lr != LEARNING_RATE * 0.2
            ):
                raise FullTrainingError("trainer scheduler differs from the frozen matched-run schedule")
            spec = _make_run_spec(args.arm, args.seed, args.workers, inputs, source_hashes, anchor)
        else:
            modules = injected["modules"]
            torch = modules["torch"]
            torch.set_num_threads(CPU_THREADS)
            np = modules["np"]
            device = torch.device("cpu")
            source_hashes = dict(injected["source_hashes"])
            inputs = dict(injected["inputs"])
            anchor = dict(injected["anchor"])
            datasets_train = injected["datasets_train"]
            datasets_val = injected["datasets_val"]
            data_identity = dict(injected["data_identity"])
            model = injected["model"]
            trainer = injected["trainer"]
            spec = _make_run_spec(args.arm, args.seed, args.workers, inputs, source_hashes, anchor)

        identity_path = run_dir / "data_identity.json"
        if args.resume:
            if identity_path.is_symlink() or not identity_path.is_file():
                raise FullTrainingError("resume run is missing its data identity record")
            stored_identity = json.loads(identity_path.read_text(encoding="utf-8"))
            if stored_identity != data_identity:
                raise FullTrainingError("resume input identity differs from the original run")
        else:
            _write_json_atomic(identity_path, data_identity, create_only=True)

        if args.resume:
            run_spec_path = run_dir / "run_spec.json"
            if run_spec_path.is_symlink():
                raise FullTrainingError("run_spec.json may not be a symlink")
            existing_spec = json.loads(run_spec_path.read_text(encoding="utf-8"))
            if existing_spec != spec:
                raise FullTrainingError("existing run_spec differs from current frozen configuration")
            checkpoint = _restore_checkpoint(
                run_dir / "resume_latest.pt", model=model, trainer=trainer, spec=spec,
                source_hashes=source_hashes, data_identity=data_identity, np=np, torch=torch,
            )
            progress = dict(checkpoint["progress"])
            totals = dict(checkpoint["totals"])
            best_score = checkpoint.get("best_validation_score")
            best_epoch = checkpoint.get("best_validation_epoch")
            _reconcile_jsonl(
                run_dir / "optimizer_attempts.jsonl",
                expected_records=int(totals["optimizer_attempts"]),
                sequence_key="global_attempt",
            )
            _reconcile_jsonl(
                run_dir / "epochs.jsonl",
                expected_records=int(progress["epoch_index"]),
                sequence_key="epoch",
            )
        else:
            _write_json_atomic(run_dir / "run_spec.json", spec, create_only=True)
            progress = {"epoch_index": 0, "next_batch_index": 0, "validation_pending": False}
            totals = {
                "optimizer_attempts": 0,
                "optimizer_steps_applied": 0,
                "optimizer_steps_skipped": 0,
                "consecutive_skipped_updates": 0,
                "microbatches_completed": 0,
                "examples_per_objective": 0,
                "empty_mlm_batches": 0,
                "loss_nonfinite_batches": 0,
            }
            best_score = None
            best_epoch = None
            trainer.optimizer.zero_grad(set_to_none=True)
            _set_seed(args.seed, modules, cuda=(device.type == "cuda"))

        optimizer_hook_calls = {"count": 0}
        hook_handle = None
        if hasattr(trainer.optimizer, "register_step_post_hook"):
            hook_handle = trainer.optimizer.register_step_post_hook(
                lambda *_args, **_kwargs: optimizer_hook_calls.__setitem__("count", optimizer_hook_calls["count"] + 1)
            )
        log_path = run_dir / "optimizer_attempts.jsonl"
        loader_batches_per_epoch = EXPECTED_TRAIN_ROWS // BATCH_SIZE
        source_hash = sha256_json(source_hashes)
        total_started = time.monotonic()
        status = "completed"
        failure_reason = None
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        restored_gate = _ensure_first20_gate(run_dir, attempts=int(totals["optimizer_attempts"]))
        if restored_gate is not None and restored_gate.get("status") == "zero_applied_updates":
            status = "persistent_skipped_updates"
            failure_reason = "zero optimizer updates were applied in the first 20 attempts"
        if int(totals.get("consecutive_skipped_updates", 0)) >= 20:
            status = "persistent_skipped_updates"
            failure_reason = "20 consecutive optimizer attempts were skipped by GradScaler"

        def save_boundary(next_progress: Mapping[str, Any]) -> None:
            payload = _checkpoint_payload(
                model=model, trainer=trainer, progress=next_progress, spec=spec,
                source_hashes=source_hashes, np=np, torch=torch, data_identity=data_identity,
                totals=totals, best_score=best_score, best_epoch=best_epoch,
            )
            _atomic_save_checkpoint(run_dir / "resume_latest.pt", payload)

        def run_validation(epoch_index: int) -> dict[str, Any]:
            loaders = _build_loaders(
                datasets_val, split="validation", epoch=0, start_batch=0, seed=args.seed,
                workers=args.workers, torch=torch,
            )
            model.eval()
            with torch.no_grad():
                return _validation(trainer, loaders)

        try:
            while status == "completed" and progress["epoch_index"] < EPOCHS:
                epoch_index = int(progress["epoch_index"])
                if progress.get("validation_pending"):
                    epoch_metrics = run_validation(epoch_index)
                    if not epoch_metrics["all_finite"]:
                        raise FullTrainingError(f"non-finite validation metrics at epoch {epoch_index + 1}")
                    _append_jsonl(run_dir / "epochs.jsonl", {
                        "epoch": epoch_index + 1,
                        "validation": epoch_metrics,
                        "optimizer_attempts_total": totals["optimizer_attempts"],
                        "optimizer_steps_applied_total": totals["optimizer_steps_applied"],
                        "optimizer_steps_skipped_total": totals["optimizer_steps_skipped"],
                        "scheduler_iteration": trainer.scheduler.iteration,
                    })
                    if best_score is None or epoch_metrics["selection_sum_loss"] < best_score:
                        best_score = epoch_metrics["selection_sum_loss"]
                        best_epoch = epoch_index + 1
                        _atomic_replace_model_checkpoint(run_dir / "best_validation_model.pt", model)
                    progress = {"epoch_index": epoch_index + 1, "next_batch_index": 0, "validation_pending": False}
                    save_boundary(progress)
                    if signal_state["requested"] is not None:
                        status = "interrupted"
                        break
                    continue

                start_batch = int(progress["next_batch_index"])
                loaders = _build_loaders(
                    datasets_train, split="train", epoch=epoch_index, start_batch=start_batch,
                    seed=args.seed, workers=args.workers, torch=torch,
                )
                iterators = {task: iter(loaders[task]) for task in TASK_KEYS}
                trainer.model.train()
                accumulated = 0
                accumulated_losses = {task: 0.0 for task in TASK_NAMES}
                accumulated_nonfinite = False
                window_started_at = time.monotonic()
                batch_index = start_batch
                while batch_index < loader_batches_per_epoch:
                    batches = {task: next(iterators[task]) for task in TASK_KEYS}
                    ap_loss = trainer.train_alignment_prediction_batch(batches["ap"], flag_optimizer=False)
                    mlm_loss = trainer.train_mlm_batch(batches["mlm"], flag_optimizer=False)
                    mim_loss = trainer.train_mim_batch(batches["mim"], flag_optimizer=False)
                    current_losses = {
                        "alignment_prediction": float(ap_loss),
                        "mlm": float(mlm_loss),
                        "mim": float(mim_loss),
                    }
                    if not all(math.isfinite(value) for value in current_losses.values()):
                        accumulated_nonfinite = True
                        totals["loss_nonfinite_batches"] += 1
                    if int(batches["mlm"]["label"].ne(-100).sum().item()) == 0:
                        totals["empty_mlm_batches"] += 1
                    for task, value in current_losses.items():
                        accumulated_losses[task] += value
                    accumulated += 1
                    batch_index += 1
                    totals["microbatches_completed"] += 1
                    totals["examples_per_objective"] += int(batches["mlm"]["label"].shape[0])

                    end_of_epoch = batch_index == loader_batches_per_epoch
                    if accumulated == GRADIENT_ACCUMULATION or end_of_epoch:
                        next_optimizer_step = totals["optimizer_attempts"] + 1
                        lr = trainer.scheduler.get_lr()
                        for group in trainer.optimizer.param_groups:
                            group["lr"] = lr
                        scale_before = float(trainer.scaler.get_scale())
                        hook_before = optimizer_hook_calls["count"]
                        trainer.scaler.step(trainer.optimizer)
                        trainer.scaler.update()
                        trainer.optimizer.zero_grad(set_to_none=True)
                        scale_after = float(trainer.scaler.get_scale())
                        applied = optimizer_hook_calls["count"] > hook_before
                        if not hasattr(trainer.optimizer, "register_step_post_hook"):
                            applied = scale_after >= scale_before
                        totals["optimizer_attempts"] += 1
                        if applied:
                            totals["optimizer_steps_applied"] += 1
                            totals["consecutive_skipped_updates"] = 0
                        else:
                            totals["optimizer_steps_skipped"] += 1
                            totals["consecutive_skipped_updates"] = int(
                                totals.get("consecutive_skipped_updates", 0)
                            ) + 1
                        step_row = {
                            "global_attempt": next_optimizer_step,
                            "epoch": epoch_index + 1,
                            "triad_end_index_zero_based": batch_index - 1,
                            "microbatches_accumulated": accumulated,
                            "examples_per_objective": accumulated * BATCH_SIZE,
                            "loss_mean_by_task": {name: value / accumulated for name, value in accumulated_losses.items()},
                            "loss_nonfinite_in_window": accumulated_nonfinite,
                            "learning_rate": float(lr),
                            "scaler_scale_before": scale_before,
                            "scaler_scale_after": scale_after,
                            "actual_optimizer_step": bool(applied),
                            "optimizer_step_status": "applied" if applied else "skipped_by_grad_scaler",
                            "attempt_wall_seconds": time.monotonic() - window_started_at,
                            "wall_seconds_since_start": time.monotonic() - total_started,
                            "scheduler_iteration": int(trainer.scheduler.iteration),
                        }
                        if device.type == "cuda" and torch.cuda.is_available():
                            step_row["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
                            step_row["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
                        _append_jsonl(log_path, step_row)
                        progress = {
                            "epoch_index": epoch_index,
                            "next_batch_index": batch_index,
                            "validation_pending": bool(end_of_epoch),
                        }
                        if (
                            totals["optimizer_attempts"] == 1
                            or totals["optimizer_attempts"] % CHECKPOINT_EVERY_UPDATES == 0
                            or end_of_epoch
                            or signal_state["requested"] is not None
                        ):
                            save_boundary(progress)
                        accumulated = 0
                        accumulated_losses = {task: 0.0 for task in TASK_NAMES}
                        accumulated_nonfinite = False
                        window_started_at = time.monotonic()

                        if totals["optimizer_attempts"] == 20:
                            gate = _ensure_first20_gate(run_dir, attempts=int(totals["optimizer_attempts"]))
                            if gate is not None and gate.get("status") == "zero_applied_updates":
                                status = "persistent_skipped_updates"
                                failure_reason = "zero optimizer updates were applied in the first 20 attempts"
                            save_boundary(progress)
                            if status != "completed":
                                break
                        if int(totals.get("consecutive_skipped_updates", 0)) >= 20:
                            status = "persistent_skipped_updates"
                            failure_reason = "20 consecutive optimizer attempts were skipped by GradScaler"
                            save_boundary(progress)
                            break
                        if signal_state["requested"] is not None:
                            status = "interrupted"
                            save_boundary(progress)
                            break
                    if signal_state["requested"] is not None and accumulated == 0:
                        status = "interrupted"
                        save_boundary(progress)
                        break
                if status != "completed":
                    break
                progress = {"epoch_index": epoch_index, "next_batch_index": loader_batches_per_epoch, "validation_pending": True}
                save_boundary(progress)
                # Validation can be repeated safely after a signal because the
                # current model and optimizer boundary are already checkpointed.
                if signal_state["requested"] is not None:
                    status = "interrupted"
                    break

            if status == "completed" and progress["epoch_index"] >= EPOCHS:
                final_model = run_dir / "resume_latest.pt"
                if not final_model.is_file():
                    raise FullTrainingError("completed run has no final model checkpoint")
                # Drop the older resume state on clean completion; latest is also
                # a standard ViLBERT.load_model-compatible checkpoint.
                (run_dir / "resume_previous.pt").unlink(missing_ok=True)
        except BaseException as exc:
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            status = "failed"
            failure_reason = f"{type(exc).__name__}: {exc}"
            # The last complete optimizer-boundary checkpoint remains recoverable.
            raise
        finally:
            if hook_handle is not None:
                hook_handle.remove()

        elapsed = time.monotonic() - started
        result = {
            "schema_version": "matched-full-pretraining-result-v1",
            "status": status,
            "run_id": args.run_id,
            "arm": args.arm,
            "seed": args.seed,
            "started_utc": started_utc,
            "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "wall_seconds": elapsed,
            "output_dir": str(run_dir),
            "run_spec_sha256": sha256_json(spec),
            "training_totals": totals,
            "progress": progress,
            "best_validation_score": best_score,
            "best_validation_epoch": best_epoch,
            "latest_checkpoint_path": str(run_dir / "resume_latest.pt"),
            "previous_checkpoint_path": str(run_dir / "resume_previous.pt") if (run_dir / "resume_previous.pt").exists() else None,
            "best_checkpoint_path": str(run_dir / "best_validation_model.pt") if (run_dir / "best_validation_model.pt").exists() else None,
            "failure_reason": failure_reason,
            "termination_signal": signal_state["requested"],
            "source_sha256": source_hashes,
            "data_identity_sha256": sha256_json(data_identity),
        }
        if status == "interrupted":
            result["exit_code"] = 143 if signal_state["requested"] == signal.SIGTERM else 130
        _write_json_atomic(run_dir / "result.json", result)
        return result
    except BaseException as exc:
        # Keep a machine-readable failure summary even when model construction,
        # input loading, or a training operation fails before a resumable state.
        if run_dir.exists():
            failure = {
                "schema_version": "matched-full-pretraining-result-v1",
                "status": "failed",
                "run_id": args.run_id,
                "arm": args.arm,
                "seed": args.seed,
                "started_utc": started_utc,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "wall_seconds": time.monotonic() - started,
                "failure_reason": f"{type(exc).__name__}: {exc}",
                "latest_checkpoint_path": str(run_dir / "resume_latest.pt") if (run_dir / "resume_latest.pt").exists() else None,
            }
            try:
                _write_json_atomic(run_dir / "result.json", failure)
            except Exception:
                pass
        raise
    finally:
        _restore_signal_handlers(previous_handlers)
        os.chdir(original_cwd)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight = commands.add_parser("preflight", help="verify matched DF2/REV initialization from offline backbones")
    preflight.add_argument("--seed", type=int, default=EXPECTED_SEED)
    preflight.add_argument("--output", default=str(DEFAULT_PREFLIGHT_OUTPUT))
    train = commands.add_parser("train", help="run or resume one full 7-epoch arm")
    train.add_argument("--arm", required=True, choices=tuple(ARM_CONFIGS))
    train.add_argument("--seed", type=int, default=EXPECTED_SEED)
    train.add_argument("--run-id", required=True)
    train.add_argument("--output-root", required=True, help="runner-owned artifact directory under autoresearch/")
    train.add_argument("--initialization-anchor", default=str(DEFAULT_INITIALIZATION_ANCHOR))
    train.add_argument("--device", default="cuda")
    train.add_argument("--workers", type=int, default=0)
    train.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        if args.command == "preflight":
            artifact = create_initialization_preflight(output_path=args.output, seed=args.seed)
            print(json.dumps(artifact, sort_keys=True))
            return 0
        result = run_training(args)
        print(json.dumps(result, sort_keys=True))
        if result.get("status") == "interrupted":
            return int(result.get("exit_code", 143))
        if result.get("status") not in {"completed"}:
            return 2
        return 0
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}, sort_keys=True), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

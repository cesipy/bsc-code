#!/usr/bin/env python3
"""Bounded fresh-initialization ViLBERT study runner (staging copy).

This module intentionally does not use the application's pretraining entry
point: that entry point loads extra validation/test sources.  The runner reads
only the frozen CC manifests, materializes fixed AP/MLM/MIM batches, constructs
the model from local checkpoint files, and writes create-only artifacts below
``autoresearch``.  It is staged here while the source tree is frozen.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import resource
import shutil
import sys
import tempfile
import time
from types import MethodType
from typing import Any, Callable, Mapping, Sequence


def _find_project_root(module_path: str | Path) -> Path:
    path = Path(module_path).resolve()
    for parent in path.parents:
        if (parent / "src").is_dir() and (parent / "autoresearch").is_dir():
            return parent
    raise RuntimeError(f"cannot locate project root from {path}")


ROOT = _find_project_root(__file__)
AUTORESEARCH_ROOT = ROOT / "autoresearch"
DEFAULT_TRAIN_MANIFEST = AUTORESEARCH_ROOT / "manifests/pilot-cc-train200-v2.json"
DEFAULT_VALIDATION_MANIFEST = AUTORESEARCH_ROOT / "manifests/pilot-cc-validation64-v2.json"
DEFAULT_FEASIBILITY = AUTORESEARCH_ROOT / "training_feasibility.json"
DEFAULT_OUTPUT_ROOT = AUTORESEARCH_ROOT / "training-study/runs"
DEFAULT_INITIALIZATION_ANCHOR_ROOT = AUTORESEARCH_ROOT / "training-study/seed-initializations"
DEFAULT_PREFLIGHT_OUTPUT = AUTORESEARCH_ROOT / "verification/study-input-preflight-v2.json"

TRAIN_MANIFEST_SHA256 = "fbc69ede01394ec591408b1944be92daaad1567d59b86fcdbe44199664e3cf24"
VALIDATION_MANIFEST_SHA256 = "3af597c8564768ae3035730dd57eca8fb258d8f8e481595215ba39ac8876aedc"
EXPECTED_SEEDS = (1801, 1802, 1803)
ORDERED_RUNS = (
    (1801, "legacy"),
    (1801, "unit_scale"),
    (1802, "unit_scale"),
    (1802, "legacy"),
    (1803, "legacy"),
    (1803, "unit_scale"),
)
STUDY_ARM = "DF2"
STUDY_BACKEND = "eager"
STUDY_MLM_MODE = "dense"
GRAD_SCALER_INIT_SCALE = 65536.0
DATA_SEED = 424242
TRAINING_SEED_OFFSET = 100_000
ARM_PLACEMENTS = {
    "DF1": {"text": [3, 4, 5], "vision": [5, 6, 7]},
    "DF2": {"text": [3, 4, 5], "vision": [9, 10, 11]},
    "REV": {"text": [9, 10, 11], "vision": [3, 4, 5]},
}
BALANCED_ARM_ORDER = {
    1801: ("DF1", "DF2", "REV"),
    1802: ("DF2", "REV", "DF1"),
    1803: ("REV", "DF1", "DF2"),
}
EXPECTED_EMPTY_MLM_BATCHES = (0, 11, 48, 70, 80)
VERSION = "mim-input-scale-study-v1"
BATCH_SIZE = 2
MAX_ATTEMPTS = 100
THREADS = 4
MAX_WALL_SECONDS = 600.0
FINALIZE_RESERVE_SECONDS = 30.0
MAX_RSS_BYTES = 8 * 1024**3
MAX_TOTAL_ARTIFACT_BYTES = 10 * 1024**3
MAX_RUN_ARTIFACT_BYTES = int(1.5 * 1024**3)
EXPECTED_IMAGE_SIZE = (224, 224)
TOKENIZER_MAX_LENGTH = 197
PROTECTED_READ_ONLY_ROOTS = (
    Path("/home/cedric.sillaber/coding/bsc-code"),
    Path("/home/cedric.sillaber/coding/bsc-code-research-material"),
)
SOURCE_PATHS = (
    "src/config.py",
    "src/vilbert.py",
    "src/attention.py",
    "src/trainer/trainers.py",
    "src/trainer/mlm_loss.py",
    "src/datasets/dataset_pretrain.py",
    "src/datasets/dataset_utils.py",
    "src/augments_transforms.py",
    "src/utils.py",
    "src/task.py",
    "src/autoresearch.py",
    "autoresearch/short_training_study_protocol.md",
    "autoresearch/mim_preprocessing_protocol.md",
)


class StudyError(RuntimeError):
    """Raised when inputs, resources, or run invariants are invalid."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _update_tensor_hash(digest: Any, value: Any, torch: Any) -> None:
    tensor = value.detach().contiguous().cpu()
    digest.update(b"tensor\0")
    digest.update(str(tensor.dtype).encode("ascii") + b"\0")
    digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii") + b"\0")
    digest.update(memoryview(tensor.view(torch.uint8).numpy()).cast("B"))


def hash_tensor_tree(value: Any, torch: Any) -> str:
    """Hash nested cached batches without serializing or copying whole trees."""
    digest = hashlib.sha256()

    def visit(item: Any) -> None:
        if isinstance(item, torch.Tensor):
            _update_tensor_hash(digest, item, torch)
        elif isinstance(item, Mapping):
            digest.update(b"mapping\0")
            for key in sorted(item, key=lambda value: str(value)):
                digest.update(str(key).encode("utf-8") + b"\0")
                visit(item[key])
        elif isinstance(item, (list, tuple)):
            digest.update(b"sequence\0")
            for child in item:
                visit(child)
        elif item is None or isinstance(item, (str, int, float, bool)):
            digest.update(json.dumps(item, separators=(",", ":")).encode("utf-8") + b"\0")
        else:
            raise TypeError(f"unsupported input-hash value: {type(item).__name__}")

    visit(value)
    return digest.hexdigest()


def hash_model_state(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        digest.update(name.encode("utf-8") + b"\0")
        _update_tensor_hash(digest, state[name], torch)
    return digest.hexdigest()


def validate_run_bounds(
    *,
    attempts: int,
    max_wall_seconds: float,
    batch_size: int = BATCH_SIZE,
    threads: int = THREADS,
    seed: int | None = None,
    arm: str | None = None,
    backend: str | None = None,
    mlm_mode: str | None = None,
    mim_policy: str | None = None,
    empty_mlm_policy: str | None = None,
) -> None:
    if isinstance(attempts, bool) or not 1 <= int(attempts) <= MAX_ATTEMPTS:
        raise StudyError(f"attempts must be in [1,{MAX_ATTEMPTS}]")
    if not math.isfinite(float(max_wall_seconds)) or not 1 <= float(max_wall_seconds) <= MAX_WALL_SECONDS:
        raise StudyError(f"max wall seconds must be in (0,{MAX_WALL_SECONDS}]")
    if float(max_wall_seconds) <= FINALIZE_RESERVE_SECONDS:
        raise StudyError(f"max wall seconds must exceed the {FINALIZE_RESERVE_SECONDS:.0f}s finalization reserve")
    if batch_size != BATCH_SIZE:
        raise StudyError(f"batch size is fixed at {BATCH_SIZE}")
    if threads != THREADS:
        raise StudyError(f"CPU thread count is fixed at {THREADS}")
    if seed is not None and seed not in EXPECTED_SEEDS:
        raise StudyError(f"seed must be one of {EXPECTED_SEEDS}")
    if arm is not None and arm != STUDY_ARM:
        raise StudyError(f"this paired preprocessing study is fixed to {STUDY_ARM}")
    if backend is not None and backend != STUDY_BACKEND:
        raise StudyError(f"this study is fixed to the profiled {STUDY_BACKEND} backend")
    if mlm_mode is not None and mlm_mode != STUDY_MLM_MODE:
        raise StudyError(f"this study is fixed to the profiled {STUDY_MLM_MODE} MLM implementation")
    if mim_policy is not None and mim_policy not in {"legacy", "unit_scale"}:
        raise StudyError("MIM policy must be legacy or unit_scale")
    if empty_mlm_policy is not None and empty_mlm_policy not in {"fail", "zero_gradient"}:
        raise StudyError("empty MLM policy must be fail or zero_gradient")


def derive_training_seed(init_seed: int) -> int:
    if init_seed not in EXPECTED_SEEDS:
        raise StudyError(f"unsupported initialization seed {init_seed}")
    return init_seed + TRAINING_SEED_OFFSET


def seed_all(seed: int, torch: Any, np: Any, *, include_cuda: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if include_cuda and getattr(torch, "cuda", None) is not None and torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalized_manifest_row_hash(caption: str) -> str:
    """Match cohort selection's whitespace-collapsed lowercase caption key."""
    normalized = " ".join(caption.casefold().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def validate_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None,
    split: str,
    sample_count: int,
    expected_data_seed: int = DATA_SEED,
    verify_sources: bool = False,
    project_root: str | Path = ROOT,
) -> tuple[dict[str, Any], str]:
    path = Path(manifest_path).resolve(strict=True)
    actual_sha = sha256_file(path)
    if expected_sha256 is not None and actual_sha != expected_sha256:
        raise StudyError(f"manifest SHA-256 mismatch for {path.name}")
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"could not read manifest {path}: {exc}") from exc

    samples = manifest.get("samples")
    if manifest.get("dataset") != "Conceptual Captions":
        raise StudyError("only the frozen Conceptual Captions cohort is allowed")
    if manifest.get("split") != split:
        raise StudyError(f"manifest split must be {split!r}")
    # The v2 manifest filenames update the initialization-seed metadata; the
    # schema_version field remains v1 because the row schema did not change.
    if manifest.get("schema_version") != f"pilot-cc-{('train200' if split == 'training-only' else 'validation64')}-v1":
        raise StudyError("unsupported pilot manifest schema")
    if manifest.get("sample_count") != sample_count or not isinstance(samples, list) or len(samples) != sample_count:
        raise StudyError(f"manifest must contain exactly {sample_count} rows")
    if manifest.get("data_seed") != expected_data_seed:
        raise StudyError("manifest data seed differs from the frozen study contract")
    if manifest.get("proposed_independent_initialization_seeds") != list(EXPECTED_SEEDS):
        raise StudyError("manifest initialization-seed roles differ from the study contract")
    plan = manifest.get("study_plan", {})
    if plan.get("independent_initialization_seeds") != list(EXPECTED_SEEDS):
        raise StudyError("manifest study-plan initialization seeds differ")
    if plan.get("data_seed") != DATA_SEED or plan.get("arms") != {
        name: {"text_insertions": spec["text"], "vision_insertions": spec["vision"]}
        for name, spec in ARM_PLACEMENTS.items()
    }:
        raise StudyError("manifest study plan differs from the frozen arm/data plan")

    ids: set[str] = set()
    caption_hashes: set[str] = set()
    normalized_caption_hashes: set[str] = set()
    image_hashes: set[str] = set()
    source_indices: set[int] = set()
    for sample in samples:
        try:
            sample_id = str(sample["sample_id"])
            caption = str(sample["caption"])
            image_path = Path(sample["image_path_abs"]).resolve(strict=True)
            source_index = int(sample["source_row_index_zero_based"])
            image_hash = str(sample["image_sha256"])
            caption_hash = str(sample["caption_sha256"])
            normalized_hash = str(sample["normalized_caption_sha256"])
        except (KeyError, OSError, ValueError) as exc:
            raise StudyError(f"invalid sample record in {path.name}: {exc}") from exc
        if not caption or not image_path.is_file():
            raise StudyError(f"empty caption or missing image for {sample_id}")
        if hashlib.sha256(caption.encode("utf-8")).hexdigest() != caption_hash:
            raise StudyError(f"caption hash mismatch for {sample_id}")
        if _normalized_manifest_row_hash(caption) != normalized_hash:
            raise StudyError(f"normalized-caption hash mismatch for {sample_id}")
        if len(image_hash) != 64 or len(caption_hash) != 64 or len(normalized_hash) != 64:
            raise StudyError(f"malformed hash in {sample_id}")
        if sample.get("csv_row") != source_index + 2:
            raise StudyError(f"CSV physical row/index disagreement for {sample_id}")
        if tuple(sample.get("image_dimensions", ())) != EXPECTED_IMAGE_SIZE:
            raise StudyError(f"image dimensions differ from 224x224 for {sample_id}")
        ids.add(sample_id)
        caption_hashes.add(caption_hash)
        normalized_caption_hashes.add(normalized_hash)
        image_hashes.add(image_hash)
        source_indices.add(source_index)
        if verify_sources:
            image_root = Path(manifest["image_root"]).resolve(strict=True)
            if not image_path.is_relative_to(image_root):
                raise StudyError(f"image path escapes the frozen image root for {sample_id}")
            if image_path.stat().st_size != int(sample["image_size_bytes"]):
                raise StudyError(f"image size mismatch for {sample_id}")
            if sha256_file(image_path) != image_hash:
                raise StudyError(f"image SHA-256 mismatch for {sample_id}")
            try:
                from PIL import Image

                with Image.open(image_path) as image:
                    if image.size != EXPECTED_IMAGE_SIZE:
                        raise StudyError(f"decoded image dimensions differ for {sample_id}")
                    if image.mode != sample["image_mode_on_disk"]:
                        raise StudyError(f"image mode mismatch for {sample_id}")
            except StudyError:
                raise
            except Exception as exc:
                raise StudyError(f"image decode failed for {sample_id}: {exc}") from exc
    if any(len(values) != sample_count for values in (
        ids, caption_hashes, normalized_caption_hashes, image_hashes, source_indices
    )):
        raise StudyError("manifest sample IDs, captions, images, and source rows must be unique")

    if verify_sources:
        csv_path = Path(manifest["source_csv_path"]).resolve(strict=True)
        csv_sha = str(manifest["source_csv_sha256"])
        if sha256_file(csv_path) != csv_sha:
            raise StudyError(f"source CSV SHA-256 mismatch for {csv_path.name}")
        needed = {int(sample["source_row_index_zero_based"]) for sample in samples}
        found: dict[int, dict[str, str]] = {}
        with csv_path.open("r", encoding="utf-8", newline="") as stream:
            for index, row in enumerate(csv.DictReader(stream)):
                if index in needed:
                    found[index] = row
                if len(found) == len(needed):
                    break
        if set(found) != needed:
            raise StudyError("manifest references a source CSV row that is not present")
        for sample in samples:
            row = found[int(sample["source_row_index_zero_based"])]
            if row.get("text") != sample["caption"]:
                raise StudyError(f"source CSV caption differs for {sample['sample_id']}")
            if row.get("filepath") != sample["image_path_rel"]:
                raise StudyError(f"source CSV image path differs for {sample['sample_id']}")

    manifest["_identity"] = {
        "manifest_path": str(path),
        "manifest_sha256": actual_sha,
        "schema_version": manifest["schema_version"],
        "split": split,
        "sample_count": sample_count,
        "data_seed": expected_data_seed,
        "sample_ids_ordered": [sample["sample_id"] for sample in samples],
        "ordered_sample_ids_sha256": hashlib.sha256(
            "\n".join(sample["sample_id"] for sample in samples).encode("utf-8")
        ).hexdigest(),
        "source_csv_path": str(Path(manifest["source_csv_path"]).resolve()),
        "source_csv_sha256": manifest["source_csv_sha256"],
    }
    return manifest, actual_sha


def validate_disjoint_cohorts(train: Mapping[str, Any], validation: Mapping[str, Any]) -> None:
    train_rows = train["samples"]
    val_rows = validation["samples"]
    if {row["image_sha256"] for row in train_rows} & {row["image_sha256"] for row in val_rows}:
        raise StudyError("training and validation manifests share image content")
    if {row["normalized_caption_sha256"] for row in train_rows} & {
        row["normalized_caption_sha256"] for row in val_rows
    }:
        raise StudyError("training and validation manifests share normalized captions")


def apply_mim_input_policy(batch: Mapping[str, Any], policy: str) -> dict[str, Any]:
    """Return an MIM batch under the chosen scale policy without mutating input.

    ``legacy`` keeps the current dataset output (masked pixels in raw 0..255
    scale). ``unit_scale`` divides only that masked strong-view tensor by 255.
    It intentionally preserves the existing weak/strong augmentation mismatch,
    clipping/quantization, and patch indices; it is not a normalization fix.
    """
    if policy not in {"legacy", "unit_scale"}:
        raise StudyError("MIM policy must be legacy or unit_scale")
    result = dict(batch)
    masked = batch.get("masked_img")
    if not isinstance(masked, Mapping) or "pixel_values" not in masked:
        raise StudyError("MIM batch is missing masked_img.pixel_values")
    copied_masked = dict(masked)
    pixels = masked["pixel_values"]
    if policy == "unit_scale":
        copied_masked["pixel_values"] = pixels / 255.0
    result["masked_img"] = copied_masked
    return result


def _materialize_dataset_types(
    *,
    root: Path = ROOT,
    dataset_types: Mapping[str, Any] | None = None,
    data_loader_factory: Callable[..., Any] | None = None,
    transforms_module: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve the real offline dataset classes lazily (injectable in CPU tests)."""
    if dataset_types is None:
        source = str(root / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        _install_silent_logger()
        from datasets.dataset_pretrain import PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
        import augments_transforms
        from torch.utils.data import DataLoader

        dataset_types = {"ap": PretrainDatasetAP, "mlm": PretrainDatasetMLM, "mim": PretrainDatasetMIM}
        transforms_module = augments_transforms
        data_loader_factory = DataLoader
    if data_loader_factory is None:
        import torch

        data_loader_factory = torch.utils.data.DataLoader
    extras: dict[str, dict[str, Any]] = {"ap": {"is_train": True}, "mlm": {}, "mim": {}}
    if transforms_module is not None:
        extras["mim"] = {
            "transforms_weak": transforms_module.get_transform_unmasked(),
            "transforms_strong": transforms_module.get_transform_masked(),
        }
    classes = {name: dataset_types[name] for name in ("ap", "mlm", "mim")}
    return classes, {name: extras[name] for name in classes}


def _install_silent_logger() -> None:
    import types

    class _SilentLogger:
        def __init__(self, *_: Any, **__: Any) -> None:
            pass

        def info(self, *_: Any, **__: Any) -> None:
            pass

        warn = info
        error = info
        debug = info

    module = types.ModuleType("logger")
    module.Logger = _SilentLogger
    sys.modules["logger"] = module


def materialize_training_batches(
    manifest: Mapping[str, Any],
    tokenizer: Any,
    torch: Any,
    np: Any,
    *,
    mim_policy: str,
    data_seed: int = DATA_SEED,
    batch_size: int = BATCH_SIZE,
    expected_sample_count: int = 200,
    planned_batches: int = MAX_ATTEMPTS,
    dataset_types: Mapping[str, Any] | None = None,
    data_loader_factory: Callable[..., Any] | None = None,
    transforms_module: Any | None = None,
    root: Path = ROOT,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Freeze sequential AP/MLM/MIM DataLoader batches before model allocation."""
    if mim_policy not in {"legacy", "unit_scale"}:
        raise StudyError("MIM policy must be legacy or unit_scale")
    if batch_size != BATCH_SIZE or planned_batches > MAX_ATTEMPTS or planned_batches < 1:
        raise StudyError("training data batch/update bounds are invalid")
    samples = list(manifest.get("samples", []))
    if len(samples) != expected_sample_count or expected_sample_count != planned_batches * batch_size:
        raise StudyError("fixed training cohort must provide exactly one batch per planned attempt")
    seed_all(data_seed, torch, np, include_cuda=False)
    classes, extras = _materialize_dataset_types(
        root=root,
        dataset_types=dataset_types,
        data_loader_factory=data_loader_factory,
        transforms_module=transforms_module,
    )
    if data_loader_factory is None:
        data_loader_factory = torch.utils.data.DataLoader
    data = [(str(row["image_path_abs"]), str(row["caption"])) for row in samples]
    datasets: dict[str, Any] = {}
    for name in ("ap", "mlm", "mim"):
        # The source datasets shuffle their rows in __init__; resetting the
        # same Python seed before each constructor reproduces one shared order.
        random.seed(data_seed)
        datasets[name] = classes[name](data, tokenizer, None, **extras[name])
    path_to_id = {str(row["image_path_abs"]): str(row["sample_id"]) for row in samples}
    orders = {
        name: [path_to_id[str(record[1])] for record in dataset.data]
        for name, dataset in datasets.items()
    }
    if any(order != orders["ap"] for order in orders.values()):
        raise StudyError("AP/MLM/MIM datasets did not preserve one identical shuffled row order")
    loaders = {
        name: data_loader_factory(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
        )
        for name, dataset in datasets.items()
    }
    if any(len(loader) != planned_batches for loader in loaders.values()):
        raise StudyError("each objective loader must yield exactly the fixed number of batches")
    iterators = {name: iter(loader) for name, loader in loaders.items()}
    triads: list[dict[str, Any]] = []
    for _ in range(planned_batches):
        triad: dict[str, Any] = {}
        for name in ("ap", "mlm", "mim"):
            batch = next(iterators[name])
            triad[name] = apply_mim_input_policy(batch, mim_policy) if name == "mim" else batch
        triads.append(triad)

    empty_mlm = [
        index for index, triad in enumerate(triads)
        if int(triad["mlm"]["label"].ne(-100).sum().item()) == 0
    ]
    mim_masks = [triad["mim"]["masked_patches_idxs"] for triad in triads]
    if any(tuple(mask.shape[-1:]) != (196,) for mask in mim_masks):
        raise StudyError("MIM patch mask must have 196 entries per sample")
    per_sample_masked = [int(mask.ne(0).sum().item()) for batch in mim_masks for mask in batch]
    per_sample_unmasked = [int(mask.eq(0).sum().item()) for batch in mim_masks for mask in batch]
    if any(value == 0 for value in per_sample_masked) or any(value == 0 for value in per_sample_unmasked):
        raise StudyError("each MIM example must have at least one masked and one unmasked patch")

    def without_policy_pixels(triad: Mapping[str, Any]) -> dict[str, Any]:
        mim = dict(triad["mim"])
        masked = dict(mim["masked_img"])
        masked.pop("pixel_values", None)
        mim["masked_img"] = masked
        return {"ap": triad["ap"], "mlm": triad["mlm"], "mim": mim}

    report = {
        "data_seed": data_seed,
        "batch_size": batch_size,
        "planned_batches": planned_batches,
        "sample_order_ids": orders,
        "sample_order_sha256": hashlib.sha256(
            "\n".join(orders["ap"]).encode("utf-8")
        ).hexdigest(),
        "empty_mlm_batch_indices_zero_based": empty_mlm,
        "empty_mlm_batch_count": len(empty_mlm),
        "mlm_supervised_tokens_per_batch": [
            int(triad["mlm"]["label"].ne(-100).sum().item()) for triad in triads
        ],
        "mim_masked_patches_per_sample_min": min(per_sample_masked),
        "mim_masked_patches_per_sample_max": max(per_sample_masked),
        "mim_unmasked_patches_per_sample_min": min(per_sample_unmasked),
        "mim_unmasked_patches_per_sample_max": max(per_sample_unmasked),
        "mim_policy": mim_policy,
        "input_batch_sha256": hash_tensor_tree(triads, torch),
        "policy_invariant_batch_sha256": hash_tensor_tree(
            [without_policy_pixels(triad) for triad in triads], torch
        ),
        "mim_patch_mask_sha256": hash_tensor_tree(mim_masks, torch),
        "mim_masked_tensor_sha256_after_policy": hash_tensor_tree(
            [triad["mim"]["masked_img"]["pixel_values"] for triad in triads], torch
        ),
        "loader_settings": {
            "num_workers": 0,
            "pin_memory": False,
            "persistent_workers": False,
            "prefetch_factor": None,
            "drop_last": False,
            "shuffle": False,
            "objective_order": ["ap", "mlm", "mim"],
            "cache": "all fixed CPU triads materialized before model allocation",
        },
    }
    return triads, report


def build_paired_triad_identity_records(
    legacy_batches: Sequence[Mapping[str, Any]],
    unit_scale_batches: Sequence[Mapping[str, Any]],
    legacy_report: Mapping[str, Any],
    unit_scale_report: Mapping[str, Any],
    torch: Any,
    *,
    expected_batch_count: int = MAX_ATTEMPTS,
) -> dict[str, Any]:
    """Hash every fixed AP/MLM/MIM triad and prove policy pairing invariants."""
    if (
        len(legacy_batches) != expected_batch_count
        or len(unit_scale_batches) != expected_batch_count
        or not 1 <= expected_batch_count <= MAX_ATTEMPTS
    ):
        raise StudyError(f"paired input proof requires exactly {expected_batch_count} triads per policy")
    order = legacy_report["sample_order_ids"]["ap"]
    if order != unit_scale_report["sample_order_ids"]["ap"]:
        raise StudyError("legacy and unit_scale use different sample orders")
    if legacy_report["input_batch_sha256"] == unit_scale_report["input_batch_sha256"]:
        raise StudyError("paired MIM policies unexpectedly have identical complete input hashes")
    if legacy_report["policy_invariant_batch_sha256"] != unit_scale_report["policy_invariant_batch_sha256"]:
        raise StudyError("paired MIM policies differ outside masked strong-view pixels")

    def hash_value(value: Any) -> str:
        return hash_tensor_tree(value, torch)

    records = []
    for index, (legacy, scaled) in enumerate(zip(legacy_batches, unit_scale_batches)):
        legacy_mim = legacy["mim"]
        scaled_mim = scaled["mim"]
        expected_scaled = legacy_mim["masked_img"]["pixel_values"] / 255.0
        scale_matches = torch.equal(
            expected_scaled, scaled_mim["masked_img"]["pixel_values"]
        )
        if not scale_matches:
            raise StudyError(f"unit_scale MIM tensor is not legacy masked pixels divided by 255 in batch {index}")

        item = {
            "batch_index_zero_based": index,
            "sample_ids_ordered": order[index * BATCH_SIZE:(index + 1) * BATCH_SIZE],
            "ap": {
                "image_tensor_sha256": hash_value(legacy["ap"]["img"]),
                "text_tensors_sha256": hash_value(legacy["ap"]["text"]),
                "labels_sha256": hash_value(legacy["ap"]["label"]),
                "batch_sha256": hash_value(legacy["ap"]),
            },
            "mlm": {
                "image_tensor_sha256": hash_value(legacy["mlm"]["img"]),
                "text_tensors_sha256": hash_value(legacy["mlm"]["text"]),
                "labels_sha256": hash_value(legacy["mlm"]["label"]),
                "batch_sha256": hash_value(legacy["mlm"]),
            },
            "mim": {
                "unmasked_image_legacy_sha256": hash_value(legacy_mim["img"]),
                "unmasked_image_unit_scale_sha256": hash_value(scaled_mim["img"]),
                "text_legacy_sha256": hash_value(legacy_mim["text"]),
                "text_unit_scale_sha256": hash_value(scaled_mim["text"]),
                "patch_mask_legacy_sha256": hash_value(legacy_mim["masked_patches_idxs"]),
                "patch_mask_unit_scale_sha256": hash_value(scaled_mim["masked_patches_idxs"]),
                "masked_image_legacy_sha256": hash_value(legacy_mim["masked_img"]["pixel_values"]),
                "masked_image_unit_scale_sha256": hash_value(scaled_mim["masked_img"]["pixel_values"]),
                "unit_scale_equals_legacy_divided_by_255": scale_matches,
            },
        }
        invariant_legacy = {
            "ap": legacy["ap"],
            "mlm": legacy["mlm"],
            "mim": {
                "img": legacy_mim["img"],
                "text": legacy_mim["text"],
                "masked_patches_idxs": legacy_mim["masked_patches_idxs"],
            },
        }
        invariant_scaled = {
            "ap": scaled["ap"],
            "mlm": scaled["mlm"],
            "mim": {
                "img": scaled_mim["img"],
                "text": scaled_mim["text"],
                "masked_patches_idxs": scaled_mim["masked_patches_idxs"],
            },
        }
        item["invariant_inputs_legacy_sha256"] = hash_value(invariant_legacy)
        item["invariant_inputs_unit_scale_sha256"] = hash_value(invariant_scaled)
        if item["invariant_inputs_legacy_sha256"] != item["invariant_inputs_unit_scale_sha256"]:
            raise StudyError(f"paired inputs differ outside masked MIM pixels in batch {index}")
        records.append(item)

    return {
        "schema_version": "paired-training-triad-input-hashes-v1",
        "batch_count": len(records),
        "batch_size": BATCH_SIZE,
        "data_seed": DATA_SEED,
        "sample_order_ids": list(order),
        "sample_order_sha256": legacy_report["sample_order_sha256"],
        "empty_mlm_batch_indices_zero_based": legacy_report["empty_mlm_batch_indices_zero_based"],
        "mlm_supervised_tokens_per_batch": legacy_report["mlm_supervised_tokens_per_batch"],
        "policy_input_batch_sha256": {
            "legacy": legacy_report["input_batch_sha256"],
            "unit_scale": unit_scale_report["input_batch_sha256"],
        },
        "policy_invariant_batch_sha256": legacy_report["policy_invariant_batch_sha256"],
        "mim_patch_mask_sha256": legacy_report["mim_patch_mask_sha256"],
        "paired_invariant_sha256": hashlib.sha256(
            json.dumps(
                [
                    (item["sample_ids_ordered"], item["invariant_inputs_legacy_sha256"])
                    for item in records
                ],
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest(),
        "policy_definition": "unit_scale masked MIM strong view equals exact legacy tensor divided out-of-place by 255; no other tensor differs",
        "triads": records,
    }


def preflight_empty_mlm_batches(
    batches: Sequence[Mapping[str, Any]],
    *,
    policy: str,
    expected_indices: Sequence[int] | None = None,
) -> list[int]:
    if policy not in {"fail", "zero_gradient"}:
        raise StudyError("empty MLM policy must be fail or zero_gradient")
    empty = [
        index for index, triad in enumerate(batches)
        if int(triad["mlm"]["label"].ne(-100).sum().item()) == 0
    ]
    if expected_indices is not None and tuple(empty) != tuple(expected_indices):
        raise StudyError(f"fixed MLM mask schedule changed: expected {list(expected_indices)}, got {empty}")
    if empty and policy == "fail":
        raise StudyError(f"fixed MLM batches have no supervised tokens: {empty}")
    return empty


def safe_mlm_projection_loss(
    hidden_states: Any,
    labels: Any,
    projection: Any,
    loss_fn: Any,
    *,
    selected_tokens_only: bool,
    empty_policy: str,
    mlm_projection_loss_fn: Callable[..., Any] | None = None,
) -> tuple[Any, str]:
    """Use a zero-gradient MLM term for all-ignored batches, without NaN.

    For empty supervision, the same normal head projection still runs (on all
    tokens in dense mode, or the empty selected tensor in selected mode), then
    ``logits.sum()*0`` supplies zero gradients to the head and encoder path.
    This preserves AdamW's zero-gradient treatment while avoiding mean-CE NaN.
    """
    if empty_policy not in {"fail", "zero_gradient"}:
        raise StudyError("empty MLM policy must be fail or zero_gradient")
    supervised = labels.ne(loss_fn.ignore_index)
    if not bool(supervised.any().item()):
        if empty_policy == "fail":
            raise StudyError("MLM batch contains no supervised tokens")
        if selected_tokens_only:
            flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])
            logits = projection(flat_hidden[supervised.reshape(-1)])
        else:
            logits = projection(hidden_states)
        return logits.sum() * 0.0, "empty_zero_gradient"
    if mlm_projection_loss_fn is None:
        from trainer.mlm_loss import mlm_projection_loss as mlm_projection_loss_fn
    value = mlm_projection_loss_fn(
        hidden_states,
        labels,
        projection,
        loss_fn,
        selected_tokens_only=selected_tokens_only,
    )
    return value, "supervised"


def _squeeze_inputs(batch: Mapping[str, Any], torch: Any, device: Any) -> tuple[dict[str, Any], dict[str, Any]]:
    text = {
        key: value.squeeze(1).to(device, non_blocking=device.type == "cuda")
        for key, value in batch["text"].items()
    }
    image = {
        key: value.squeeze(1).to(device, non_blocking=device.type == "cuda")
        for key, value in batch["img"].items()
    }
    return text, image


def _call_model(model: Any, text: Mapping[str, Any], image: Mapping[str, Any], task: str) -> Any:
    return model.forward_pretrain(
        text_input_ids=text["input_ids"],
        text_attention_mask=text.get("attention_mask"),
        text_token_type_ids=text.get("token_type_ids"),
        image_pixel_values=image["pixel_values"],
        image_attention_mask=image.get("attention_mask"),
        tasks=[task],
    )


def _finite_tensor_collection(tensors: Sequence[Any], torch_module: Any) -> bool:
    checks = [torch_module.isfinite(tensor).all() for tensor in tensors]
    if not checks:
        return True
    return bool(torch_module.stack(checks).all().item())


def _tracked_optimizer_step_value(optimizer: Any, model: Any) -> float | None:
    tracked = next((parameter for parameter in model.parameters() if parameter.requires_grad), None)
    if tracked is None:
        return None
    step = optimizer.state.get(tracked, {}).get("step")
    if step is None:
        return 0.0 if type(optimizer).__name__ == "AdamW" else None
    return float(step.item() if hasattr(step, "item") else step)


def joint_optimizer_step(
    model: Any,
    optimizer: Any,
    scaler: Any,
    triad: Mapping[str, Any],
    *,
    device: Any,
    selected_tokens_only: bool,
    empty_mlm_policy: str,
    synchronize: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
    loss_fn_mim: Any | None = None,
    torch_module: Any | None = None,
    mlm_projection_loss_fn: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    if torch_module is None:
        import torch as torch_module
    from torch import nn
    if loss_fn_mim is None:
        from info_nce import InfoNCE

        loss_fn_mim = InfoNCE()
    loss_fn_ap = nn.BCEWithLogitsLoss()
    loss_fn_mlm = nn.CrossEntropyLoss(ignore_index=-100)
    optimizer.zero_grad(set_to_none=True)
    optimizer._autoresearch_step_phase = "before_optimizer_step"
    optimizer._autoresearch_step_counter_before = None
    optimizer._autoresearch_grad_scaler_scale_before = (
        float(scaler.get_scale()) if hasattr(scaler, "get_scale") else 1.0
    )
    synchronize()
    started = clock()
    amp_enabled = device.type == "cuda"
    def autocast_context():
        return torch_module.autocast(
            device_type=device.type,
            dtype=torch_module.float16 if amp_enabled else torch_module.bfloat16,
            enabled=amp_enabled,
        )

    losses: dict[str, float] = {}
    statuses: dict[str, str] = {}
    scale_before = optimizer._autoresearch_grad_scaler_scale_before

    def backward_objective(name: str, loss: Any) -> None:
        scalar = float(loss.detach().float().item())
        if not math.isfinite(scalar):
            raise StudyError(f"nonfinite {name} loss before backward: {scalar}")
        losses[name] = scalar
        scaler.scale(loss).backward()

    # Separate backward calls release each objective graph before the next
    # forward pass while accumulating exactly the gradient of their sum.
    with autocast_context():
        ap_text, ap_img = _squeeze_inputs(triad["ap"], torch_module, device)
        ap_text_features, _ = _call_model(model, ap_text, ap_img, "alignment_prediction")
        ap_logits = model.alignment_fc(ap_text_features).squeeze(-1)
        ap_labels = triad["ap"]["label"].to(device, non_blocking=amp_enabled).float().reshape(-1)
        ap_loss = loss_fn_ap(ap_logits, ap_labels)
    backward_objective("ap", ap_loss)
    statuses["ap"] = "computed"
    del ap_text, ap_img, ap_text_features, ap_logits, ap_labels, ap_loss

    with autocast_context():
        mlm_text, mlm_img = _squeeze_inputs(triad["mlm"], torch_module, device)
        mlm_labels = triad["mlm"]["label"].to(device, non_blocking=amp_enabled)
        mlm_hidden, _ = _call_model(model, mlm_text, mlm_img, "mlm")
        mlm_loss, statuses["mlm"] = safe_mlm_projection_loss(
            mlm_hidden,
            mlm_labels,
            model.mlm,
            loss_fn_mlm,
            selected_tokens_only=selected_tokens_only,
            empty_policy=empty_mlm_policy,
            mlm_projection_loss_fn=mlm_projection_loss_fn,
        )
    backward_objective("mlm", mlm_loss)
    del mlm_text, mlm_img, mlm_labels, mlm_hidden, mlm_loss

    with autocast_context():
        mim_text, mim_img = _squeeze_inputs(triad["mim"], torch_module, device)
        masked_img = {
            key: value.squeeze(1).to(device, non_blocking=amp_enabled)
            for key, value in triad["mim"]["masked_img"].items()
        }
        patch_mask = triad["mim"]["masked_patches_idxs"].to(device, non_blocking=amp_enabled)
        _, masked_features = _call_model(model, mim_text, masked_img, "mim")
        _, unmasked_features = _call_model(model, mim_text, mim_img, "mim")
        unmasked_patch_mask = patch_mask.eq(0)
        cls_mask = torch_module.ones(
            patch_mask.shape[0], 1, device=device, dtype=torch_module.bool
        )
        full_mask = torch_module.cat((cls_mask, unmasked_patch_mask), dim=1)
        mim_loss = loss_fn_mim(masked_features[full_mask], unmasked_features[full_mask])
    backward_objective("mim", mim_loss)
    statuses["mim"] = "computed_two_branches"
    del mim_text, mim_img, masked_img, patch_mask, masked_features, unmasked_features
    del unmasked_patch_mask, cls_mask, full_mask, mim_loss

    total_loss = sum(losses.values())
    if not math.isfinite(total_loss):
        raise StudyError("nonfinite summed joint loss")

    tracked = next((parameter for parameter in model.parameters() if parameter.requires_grad), None)
    if tracked is None:
        raise StudyError("model has no trainable parameters")
    before = optimizer.state.get(tracked, {}).get("step", 0)
    before_value = float(before.item() if hasattr(before, "item") else before)
    if not any(parameter.grad is not None for parameter in model.parameters() if parameter.requires_grad):
        raise StudyError("the joint objectives produced no model gradients")
    scaler.unscale_(optimizer)
    grad_tensors = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    gradients_finite = _finite_tensor_collection(grad_tensors, torch_module)
    optimizer._autoresearch_step_counter_before = before_value
    optimizer._autoresearch_step_phase = "calling_optimizer_step"
    scaler.step(optimizer)
    optimizer._autoresearch_step_phase = "optimizer_step_returned"
    scaler.update()
    optimizer._autoresearch_step_phase = "scaler_updated"
    scale_after = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else 1.0
    parameters_finite = _finite_tensor_collection(
        [parameter.detach() for parameter in model.parameters()],
        torch_module,
    )
    optimizer._autoresearch_step_phase = "post_step_diagnostics"
    synchronize()
    after = optimizer.state.get(tracked, {}).get("step", 0)
    after_value = float(after.item() if hasattr(after, "item") else after)
    applied = after_value > before_value
    elapsed = clock() - started
    optimizer.zero_grad(set_to_none=True)
    optimizer._autoresearch_step_phase = "complete"
    return {
        "losses": losses,
        "mlm_status": statuses["mlm"],
        "mim_status": statuses["mim"],
        "total_loss": total_loss,
        "actual_optimizer_step": applied,
        "optimizer_step_status": (
            "applied"
            if applied
            else "skipped_by_grad_scaler"
            if not gradients_finite
            else "skipped_no_step_with_finite_gradients"
        ),
        "gradients_finite_after_unscale": gradients_finite,
        "parameters_finite_after_step": parameters_finite,
        "fatal_error": None if parameters_finite else "model parameters became nonfinite after optimizer/scaler step",
        "tracked_parameter_step_before": before_value,
        "tracked_parameter_step_after": after_value,
        "grad_scaler_scale_before": scale_before,
        "grad_scaler_scale_after": scale_after,
        "optimizer_skip_reason": (
            None
            if applied
            else "nonfinite_gradients_grad_scaler_skip"
            if not gradients_finite
            else "adamw_step_counter_not_advanced_with_finite_gradients"
        ),
        "step_wall_seconds": float(elapsed),
    }


def run_training_attempts(
    model: Any,
    optimizer: Any,
    scaler: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    attempts: int,
    max_wall_seconds: float,
    started_monotonic: float,
    device: Any,
    selected_tokens_only: bool,
    empty_mlm_policy: str,
    synchronize: Callable[[], None] = lambda: None,
    clock: Callable[[], float] = time.perf_counter,
    monotonic: Callable[[], float] = time.monotonic,
    rss_bytes: Callable[[], int] = lambda: 0,
    max_rss_bytes: int = MAX_RSS_BYTES,
    loss_fn_mim: Any | None = None,
    torch_module: Any | None = None,
    update_fn: Callable[..., dict[str, Any]] = joint_optimizer_step,
    on_attempt: Callable[[Mapping[str, Any]], None] | None = None,
) -> tuple[list[dict[str, Any]], str]:
    if not 1 <= attempts <= MAX_ATTEMPTS or attempts > len(batches):
        raise StudyError("attempt budget exceeds fixed batch availability or hard maximum")
    stop_at = started_monotonic + max_wall_seconds - FINALIZE_RESERVE_SECONDS
    results: list[dict[str, Any]] = []
    stop_reason = "attempt_budget_reached"
    for index, triad in enumerate(batches[:attempts]):
        if monotonic() >= stop_at:
            stop_reason = "wall_limit_reserve_reached"
            break
        rss_before = rss_bytes()
        if rss_before > max_rss_bytes:
            raise StudyError(f"host RSS exceeds {max_rss_bytes} bytes before attempt {index + 1}")
        if device.type == "cuda":
            torch_module.cuda.reset_peak_memory_stats(device)
        optimizer._autoresearch_step_phase = None
        optimizer._autoresearch_step_counter_before = None
        optimizer._autoresearch_grad_scaler_scale_before = None
        try:
            result = update_fn(
                model,
                optimizer,
                scaler,
                triad,
                device=device,
                selected_tokens_only=selected_tokens_only,
                empty_mlm_policy=empty_mlm_policy,
                synchronize=synchronize,
                clock=clock,
                loss_fn_mim=loss_fn_mim,
                torch_module=torch_module,
            )
        except Exception as exc:
            step_phase = getattr(optimizer, "_autoresearch_step_phase", None)
            step_before = getattr(optimizer, "_autoresearch_step_counter_before", None)
            step_after = _tracked_optimizer_step_value(optimizer, model)
            if step_before is not None and step_after is not None and step_after > float(step_before):
                actual_step: bool | None = True
                step_status = "optimizer_step_advanced_before_exception"
            elif step_phase == "before_optimizer_step":
                actual_step = False
                step_status = "failed_before_optimizer_step"
            elif step_phase in {"optimizer_step_returned", "scaler_updated", "post_step_diagnostics"} and step_after is not None:
                actual_step = False
                step_status = "no_optimizer_step_counter_advance_before_exception"
            else:
                actual_step = None
                step_status = "optimizer_step_outcome_unknown"
            try:
                scale_after_error = float(scaler.get_scale()) if hasattr(scaler, "get_scale") else None
            except Exception:
                scale_after_error = None
            failed_attempt = {
                "attempt_index_zero_based": index,
                "attempt": index + 1,
                "phase": "training",
                "attempt_record_status": "failed_before_step_result",
                "processed_samples_per_objective": BATCH_SIZE,
                "actual_optimizer_step": actual_step,
                "optimizer_step_status": step_status,
                "optimizer_step_phase_at_exception": step_phase,
                "tracked_parameter_step_before": step_before,
                "tracked_parameter_step_after": step_after,
                "grad_scaler_scale_before": getattr(
                    optimizer, "_autoresearch_grad_scaler_scale_before", None
                ),
                "grad_scaler_scale_after": scale_after_error,
                "optimizer_skip_reason": None,
                "parameters_finite_after_step": None,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "rss_peak_bytes": max(rss_before, rss_bytes()),
            }
            results.append(failed_attempt)
            if on_attempt is not None:
                on_attempt(failed_attempt)
            raise
        result.update({
            "attempt_index_zero_based": index,
            "attempt": index + 1,
            "phase": "training",
            "attempt_record_status": "completed",
            "processed_samples_per_objective": BATCH_SIZE,
            "rss_peak_bytes": max(rss_before, rss_bytes()),
        })
        if device.type == "cuda":
            result["gpu_peak_allocated_bytes"] = int(torch_module.cuda.max_memory_allocated(device))
            result["gpu_peak_reserved_bytes"] = int(torch_module.cuda.max_memory_reserved(device))
        else:
            result["gpu_peak_allocated_bytes"] = None
            result["gpu_peak_reserved_bytes"] = None
        results.append(result)
        if on_attempt is not None:
            on_attempt(result)
        if result.get("fatal_error"):
            raise StudyError(str(result["fatal_error"]))
        if result["rss_peak_bytes"] > max_rss_bytes:
            raise StudyError(f"host RSS exceeds {max_rss_bytes} bytes after attempt {index + 1}")
    return results, stop_reason


def _install_offline_backbone_loaders(
    bert_model_class: Any,
    timm_module: Any,
    *,
    bert_snapshot: str | Path,
    vit_weights: str | Path,
    vit_model_name: str,
    load_vit_state: Callable[[str | Path], Mapping[str, Any]],
):
    """Temporarily force ViLBERT's constructors to load only explicit local files."""
    import inspect

    original_bert_descriptor = inspect.getattr_static(bert_model_class, "from_pretrained")
    original_bert_bound = bert_model_class.from_pretrained
    original_timm_factory = timm_module.create_model

    def local_bert_loader(cls: Any, *_args: Any, **kwargs: Any) -> Any:
        kwargs.pop("local_files_only", None)
        kwargs.pop("use_safetensors", None)
        return original_bert_bound(
            str(Path(bert_snapshot).resolve(strict=True)),
            local_files_only=True,
            use_safetensors=True,
            **kwargs,
        )

    def local_timm_factory(name: str, *args: Any, **kwargs: Any) -> Any:
        if name != vit_model_name:
            raise StudyError(f"unexpected vision model constructor request {name!r}")
        if kwargs.get("num_classes") != 0:
            raise StudyError("the frozen ViT constructor must omit the pretrained classifier with num_classes=0")
        kwargs["pretrained"] = False
        vision_model = original_timm_factory(name, *args, **kwargs)
        state = load_vit_state(vit_weights)
        load_info = _load_pretrained_vit_body_without_classifier(vision_model, state)
        vision_model._autoresearch_pretrained_load_info = load_info
        del state
        return vision_model

    setattr(bert_model_class, "from_pretrained", classmethod(local_bert_loader))
    timm_module.create_model = local_timm_factory

    @contextmanager
    def _restore():
        try:
            yield
        finally:
            setattr(bert_model_class, "from_pretrained", original_bert_descriptor)
            timm_module.create_model = original_timm_factory

    return _restore()


def _load_pretrained_vit_body_without_classifier(
    vision_model: Any,
    checkpoint_state: Mapping[str, Any],
) -> dict[str, Any]:
    """Strictly load the ViT body, dropping only its known 1k-class head.

    The local ``augreg2_in21k_ft_in1k`` checkpoint has an ImageNet-1k
    classifier, while ``timm.create_model(..., num_classes=0)`` intentionally
    constructs the feature extractor used by ViLBERT.  Every body key must
    still match strictly; only the two classifier tensors with the recorded
    1000-by-768 and 1000 shapes may be ignored.
    """
    target_state = vision_model.state_dict()
    target_keys = set(target_state)
    source_keys = set(checkpoint_state)
    missing = target_keys - source_keys
    extra = source_keys - target_keys
    expected_head_keys = {"head.weight", "head.bias"}
    if missing:
        raise StudyError(f"local ViT checkpoint is missing backbone keys: {sorted(missing)[:8]}")
    if extra != expected_head_keys:
        raise StudyError(
            "local ViT checkpoint has unexpected keys beyond its known classifier: "
            f"{sorted(extra)}"
        )
    head_weight_shape = tuple(checkpoint_state["head.weight"].shape)
    head_bias_shape = tuple(checkpoint_state["head.bias"].shape)
    if head_weight_shape != (1000, 768) or head_bias_shape != (1000,):
        raise StudyError(
            "local ViT classifier shape differs from the expected ImageNet-1k head "
            f"(1000,768)/(1000,): {head_weight_shape}/{head_bias_shape}"
        )
    body_state = {key: checkpoint_state[key] for key in target_state}
    vision_model.load_state_dict(body_state, strict=True)
    return {
        "strict_backbone_load": True,
        "classifier_keys_dropped": ["head.bias", "head.weight"],
        "classifier_shapes_dropped": {
            "head.weight": list(head_weight_shape),
            "head.bias": list(head_bias_shape),
        },
        "classifier_reason": "feature extractor constructed with num_classes=0; local source checkpoint includes its ImageNet-1k classifier",
    }


@contextmanager
def offline_environment() -> Any:
    names = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "TOKENIZERS_PARALLELISM")
    old = {name: os.environ.get(name) for name in names}
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    try:
        yield
    finally:
        for name, value in old.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def construct_fresh_model(
    config: Any,
    *,
    bert_snapshot: str | Path,
    vit_weights: str | Path,
    vit_model_name: str,
    model_class: Any | None = None,
    bert_model_class: Any | None = None,
    timm_module: Any | None = None,
    load_vit_state: Callable[[str | Path], Mapping[str, Any]] | None = None,
) -> Any:
    if model_class is None or bert_model_class is None or timm_module is None or load_vit_state is None:
        source = str(ROOT / "src")
        if source not in sys.path:
            sys.path.insert(0, source)
        from transformers import BertModel
        import timm
        from safetensors.torch import load_file
        from vilbert import ViLBERT

        model_class = model_class or ViLBERT
        bert_model_class = bert_model_class or BertModel
        timm_module = timm_module or timm
        load_vit_state = load_vit_state or (lambda path: load_file(str(path), device="cpu"))
    with offline_environment():
        with _install_offline_backbone_loaders(
            bert_model_class,
            timm_module,
            bert_snapshot=bert_snapshot,
            vit_weights=vit_weights,
            vit_model_name=vit_model_name,
            load_vit_state=load_vit_state,
        ):
            model = model_class(config)
            vision = getattr(model, "vit", None)
            if vision is not None and hasattr(vision, "_autoresearch_pretrained_load_info"):
                model._autoresearch_pretrained_load_info = vision._autoresearch_pretrained_load_info
            return model


def configure_runtime_environment() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True


def load_feasibility_inputs(path: str | Path) -> dict[str, Any]:
    file = Path(path).resolve(strict=True)
    value = json.loads(file.read_text(encoding="utf-8"))
    if value.get("schema_version") != "training-feasibility-v2":
        raise StudyError("unsupported training-feasibility record")
    return {"path": str(file), "sha256": sha256_file(file), "data": value}


def validate_local_assets(feasibility: Mapping[str, Any]) -> dict[str, Any]:
    data = feasibility["data"]
    offline = data["offline_initialization"]
    bert = offline["bert"]
    vit = offline["vit"]
    tokenizer = offline["tokenizer"]
    # Resolve the snapshot directory itself, not the model.safetensors link;
    # resolving that file first would move its parent to Hugging Face's blobs/.
    bert_snapshot = Path(bert["weights_path"]).parent.resolve(strict=True)
    bert_weights = Path(bert["weights_path"]).resolve(strict=True)
    bert_config = Path(bert["config_path"]).resolve(strict=True)
    vit_weights = Path(vit["weights_path"]).resolve(strict=True)
    tokenizer_dir = Path(tokenizer["path"]).resolve(strict=True)
    if not (bert_snapshot / "config.json").is_file() or not bert_config.is_file():
        raise StudyError("local BERT config is missing")
    if not bert_weights.is_file() or not vit_weights.is_file():
        raise StudyError("local offline backbone weights are missing")
    tokenizer_hashes = tokenizer["source_file_sha256"]
    actual_tokenizer_hashes = {}
    for name, expected in tokenizer_hashes.items():
        file = tokenizer_dir / name
        if sha256_file(file) != expected:
            raise StudyError(f"local tokenizer file SHA-256 mismatch: {name}")
        actual_tokenizer_hashes[name] = expected
    return {
        "bert_model_id": bert["model_id"],
        "bert_revision": bert["revision"],
        "bert_snapshot": str(bert_snapshot),
        "bert_config_path": str(bert_config),
        "bert_config_sha256": sha256_file(bert_config),
        "bert_weights_path": str(bert_weights),
        "bert_weights_sha256": sha256_file(bert_weights),
        "bert_weights_size_bytes": bert_weights.stat().st_size,
        "vision_model_id": vit["cached_model_id"],
        "vision_model_name": vit["timm_model_name"],
        "vision_revision": vit["revision"],
        "vision_weights_path": str(vit_weights),
        "vision_weights_sha256": sha256_file(vit_weights),
        "vision_weights_size_bytes": vit_weights.stat().st_size,
        "tokenizer_model_id": tokenizer["model_id"],
        "tokenizer_path": str(tokenizer_dir),
        "tokenizer_files_sha256": actual_tokenizer_hashes,
    }


def _source_hashes() -> dict[str, str]:
    result = {}
    for relative in SOURCE_PATHS:
        file = ROOT / relative
        if file.is_file():
            result[relative] = sha256_file(file)
        else:
            raise StudyError(f"source file disappeared during freeze: {relative}")
    result["study_driver"] = sha256_file(__file__)
    return result


def _version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _runtime_versions(torch: Any, np: Any, tokenizer: Any) -> dict[str, Any]:
    import PIL
    import torchvision
    import transformers
    import timm
    import safetensors

    return {
        "python": sys.version.split()[0],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
        "pillow": str(PIL.__version__),
        "torchvision": str(torchvision.__version__),
        "transformers": str(transformers.__version__),
        "timm": str(timm.__version__),
        "safetensors": str(safetensors.__version__),
        "tokenizer_class": type(tokenizer).__name__,
        "info_nce_distribution": _version("info-nce-pytorch"),
    }


def _config_dict(config: Any) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Mapping):
            return {str(key): convert(child) for key, child in value.items()}
        if isinstance(value, (list, tuple)):
            return [convert(child) for child in value]
        enum_value = getattr(value, "value", None)
        if enum_value is not None:
            return convert(enum_value)
        return str(value)

    return {key: convert(value) for key, value in vars(config).items()}


def _safe_output_root(path: str | Path, root: Path = ROOT) -> Path:
    allowed = (root / "autoresearch").resolve(strict=True)
    requested = Path(path)
    if requested.is_symlink():
        raise StudyError("output root must not be a symlink")
    resolved = requested.resolve(strict=False)
    if not resolved.is_relative_to(allowed):
        raise StudyError("all run outputs must remain below autoresearch/")
    for protected in PROTECTED_READ_ONLY_ROOTS:
        protected_path = protected.resolve(strict=False)
        if resolved.is_relative_to(protected_path) or protected_path.is_relative_to(resolved):
            raise StudyError("output root overlaps a protected read-only workspace")
    return resolved


def _write_bytes_create_only(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise StudyError(f"refusing to overwrite existing output: {path}")
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temp_path, path)
    except FileExistsError as exc:
        raise StudyError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def write_json_create_only(path: Path, value: Mapping[str, Any]) -> None:
    payload = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    _write_bytes_create_only(path, payload)


def create_jsonl_log(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise StudyError(f"refusing to overwrite attempt log: {path}")
    with path.open("x", encoding="utf-8") as stream:
        stream.flush()
        os.fsync(stream.fileno())


def append_jsonl_durable(path: Path, value: Mapping[str, Any]) -> None:
    if path.is_symlink() or not path.is_file():
        raise StudyError(f"attempt log is missing or not a regular file: {path}")
    line = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def read_jsonl_steps(path: Path) -> tuple[list[dict[str, Any]], bool]:
    if not path.is_file() or path.is_symlink():
        return [], False
    rows = []
    corrupt_tail = False
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            corrupt_tail = True
            break
        if not isinstance(row, dict):
            corrupt_tail = True
            break
        rows.append(row)
    return rows, corrupt_tail


def _save_torch_create_only(torch: Any, path: Path, value: Any) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise StudyError(f"refusing to overwrite existing output: {path}")
    temp_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp"
    try:
        torch.save(value, temp_path)
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temp_path, path)
        return path.stat().st_size
    except FileExistsError as exc:
        raise StudyError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def _save_npz_create_only(np: Any, path: Path, arrays: Mapping[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise StudyError(f"refusing to overwrite existing output: {path}")
    temp_path = path.parent / f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp.npz"
    try:
        np.savez_compressed(temp_path, **arrays)
        with temp_path.open("rb") as stream:
            os.fsync(stream.fileno())
        os.link(temp_path, path)
        return path.stat().st_size
    except FileExistsError as exc:
        raise StudyError(f"refusing to overwrite existing output: {path}") from exc
    finally:
        try:
            temp_path.unlink()
        except FileNotFoundError:
            pass


def register_seed_initialization(
    output_root: Path,
    *,
    seed: int,
    state_sha256: str,
    backbone_hashes: Mapping[str, str],
    source_hashes: Mapping[str, str],
    anchor_root: Path | None = None,
    create_only_writer: Callable[[Path, Mapping[str, Any]], None] = write_json_create_only,
) -> dict[str, Any]:
    """Create or verify one tensor-hash anchor shared by every arm/policy."""
    path = (anchor_root or (output_root / "seed-initializations")) / f"seed-{seed}.json"
    expected = {
        "schema_version": "fresh-initialization-anchor-v1",
        "seed": seed,
        "initial_model_state_sha256": state_sha256,
        "offline_backbone_weight_sha256": dict(backbone_hashes),
        "source_sha256": dict(source_hashes),
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in expected.items():
            if existing.get(key) != value:
                raise StudyError(f"within-seed initial tensor/source mismatch in {path}")
        return existing
    create_only_writer(path, expected)
    return expected


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _tree_bytes(path: Path) -> int:
    total = 0
    if not path.exists():
        return 0
    for child in path.rglob("*"):
        if child.is_symlink():
            raise StudyError(f"symlinks are not allowed below output root: {child}")
        if child.is_file():
            total += child.stat().st_size
    return total


def check_storage_budget(output_root: Path, model_state_bytes: int, planned_runs: int) -> dict[str, int]:
    if planned_runs < 1 or planned_runs > 9:
        raise StudyError("planned run count must be between 1 and 9")
    if model_state_bytes <= 0 or model_state_bytes + 32 * 1024**2 > MAX_RUN_ARTIFACT_BYTES:
        raise StudyError("one FP32 model state exceeds the 1.5-GiB per-run artifact bound")
    existing_bytes = _tree_bytes(output_root)
    run_dirs = [
        child for child in output_root.iterdir()
        if child.is_dir() and (
            (child / "result.json").is_file() or (child / "failure.json").is_file()
        )
    ] if output_root.exists() else []
    remaining_runs = max(0, planned_runs - len(run_dirs))
    reserve_per_run = model_state_bytes + 32 * 1024**2
    estimated_total = existing_bytes + remaining_runs * reserve_per_run
    if estimated_total > MAX_TOTAL_ARTIFACT_BYTES:
        raise StudyError(
            f"estimated campaign artifacts {estimated_total} exceed {MAX_TOTAL_ARTIFACT_BYTES} bytes"
        )
    if output_root.exists():
        free_bytes = shutil.disk_usage(output_root).free
    else:
        nearest = output_root.parent
        while not nearest.exists():
            nearest = nearest.parent
        free_bytes = shutil.disk_usage(nearest).free
    required_free = remaining_runs * reserve_per_run
    if free_bytes < required_free:
        raise StudyError(f"free disk space {free_bytes} is below required reserve {required_free}")
    return {
        "existing_artifact_bytes": existing_bytes,
        "run_count_seen": len(run_dirs),
        "remaining_planned_runs": remaining_runs,
        "estimated_total_artifact_bytes": estimated_total,
        "filesystem_free_bytes": free_bytes,
        "required_free_bytes": required_free,
        "per_run_artifact_limit_bytes": MAX_RUN_ARTIFACT_BYTES,
        "campaign_artifact_limit_bytes": MAX_TOTAL_ARTIFACT_BYTES,
    }


def _seeded_config(arm: str, backend: str, mlm_mode: str) -> Any:
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from config import ViLBERTConfig

    placement = ARM_PLACEMENTS[arm]
    return ViLBERTConfig(
        text_cross_attention_layers=list(placement["text"]),
        vision_cross_attention_layers=list(placement["vision"]),
        cross_attention_backend=backend,
        mlm_selected_tokens_only=(mlm_mode == "selected"),
        dropout_prob=0.1,
        learning_rate=1e-4,
        pretrain_batch_size=BATCH_SIZE,
        gradient_accumulation=1,
    )


def _load_tokenizer(path: str | Path) -> Any:
    from transformers import BertTokenizerFast

    tokenizer = BertTokenizerFast.from_pretrained(str(path), local_files_only=True)
    if tokenizer.vocab_size != 30522 or tokenizer.mask_token_id is None:
        raise StudyError("offline BERT tokenizer vocabulary/mask token differs from contract")
    return tokenizer


def materialize_validation_batches(
    manifest: Mapping[str, Any],
    tokenizer: Any,
    *,
    torch: Any,
    batch_size: int = BATCH_SIZE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if batch_size != BATCH_SIZE or len(manifest["samples"]) != 64:
        raise StudyError("validation must use the frozen 64-row cohort in batch-size-two order")
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from datasets.dataset_utils import get_text_embedding, process_image

    rows: list[dict[str, Any]] = []
    for sample in manifest["samples"]:
        image = process_image(str(sample["image_path_abs"]))
        if image is None:
            raise StudyError(f"validation image decode failed: {sample['sample_id']}")
        text = get_text_embedding(sample["caption"], tokenizer)
        rows.append({
            "sample_id": sample["sample_id"],
            "img": image["pixel_values"],
            "text": text,
        })
    batches = []
    for start in range(0, len(rows), batch_size):
        subset = rows[start:start + batch_size]
        if len(subset) != batch_size:
            raise StudyError("validation cohort is not divisible by batch size")
        batches.append({
            "sample_ids": [row["sample_id"] for row in subset],
            "text": {
                key: torch.cat([row["text"][key] for row in subset], dim=0)
                for key in subset[0]["text"]
            },
            "img": {
                "pixel_values": torch.cat([row["img"] for row in subset], dim=0),
            },
        })
    return batches, {
        "ordered_sample_ids": [row["sample_id"] for row in rows],
        "input_sha256": hash_tensor_tree(batches, torch),
        "batch_size": batch_size,
        "count": len(rows),
        "transform": "source dataset_utils.process_image minimal ImageNet normalization; BertTokenizerFast padding=max_length,truncation,max_length=197",
    }


def extract_validation_cls_features(
    model: Any,
    batches: Sequence[Mapping[str, Any]],
    *,
    torch: Any,
    np: Any,
    device: Any,
) -> tuple[Any, Any]:
    model.eval()
    text_layers: list[list[Any]] = [[] for _ in range(12)]
    vision_layers: list[list[Any]] = [[] for _ in range(12)]
    with torch.inference_mode():
        for batch in batches:
            text = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch["text"].items()
            }
            image = {
                key: value.to(device, non_blocking=device.type == "cuda")
                for key, value in batch["img"].items()
            }
            _, _, intermediate = model(
                text_input_ids=text["input_ids"],
                text_attention_mask=text.get("attention_mask"),
                text_token_type_ids=text.get("token_type_ids"),
                image_pixel_values=image["pixel_values"],
                save_intermediate_representations=True,
            )
            if len(intermediate) != 12:
                raise StudyError(f"expected 12 paired layers, got {len(intermediate)}")
            for layer in intermediate:
                index = int(layer["layer"])
                if index not in range(12):
                    raise StudyError(f"unexpected layer index {index}")
                text_layers[index].append(layer["text_embedding"][:, 0, :].detach().cpu().numpy())
                vision_layers[index].append(layer["vision_embedding"][:, 0, :].detach().cpu().numpy())
    text = np.stack([np.concatenate(parts, axis=0) for parts in text_layers], axis=0).astype(np.float32)
    vision = np.stack([np.concatenate(parts, axis=0) for parts in vision_layers], axis=0).astype(np.float32)
    if text.shape != (12, 64, 768) or vision.shape != (12, 64, 768):
        raise StudyError(f"unexpected validation feature shapes: text={text.shape}, vision={vision.shape}")
    if not np.isfinite(text).all() or not np.isfinite(vision).all():
        raise StudyError("validation CLS features contain nonfinite values")
    return text, vision


def summarize_geometry(text_features: Any, vision_features: Any) -> dict[str, Any]:
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from autoresearch import _linear_cka, _rank_diagnostics

    metrics = []
    for layer in range(12):
        cka, cka_status = _linear_cka(text_features[layer], vision_features[layer])
        text_rank = _rank_diagnostics(text_features[layer])["participation_ratio_rank"]
        vision_rank = _rank_diagnostics(vision_features[layer])["participation_ratio_rank"]
        metrics.append({
            "layer": layer,
            "raw_linear_cka": cka,
            "cka_status": cka_status,
            "text_participation_ratio_rank": text_rank,
            "vision_participation_ratio_rank": vision_rank,
        })
    return {
        "definition": "raw centered linear CKA and participation ratio of centered covariance eigenvalues; no null calibration",
        "per_layer": metrics,
    }


def _working_directory(path: Path):
    @contextmanager
    def _context():
        previous = Path.cwd()
        os.chdir(path)
        try:
            yield
        finally:
            os.chdir(previous)
    return _context()


def _atomic_run_dir(output_root: Path, run_id: str) -> Path:
    if not run_id or any(char not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_." for char in run_id):
        raise StudyError("run ID contains unsupported characters")
    output_root.mkdir(parents=True, exist_ok=True)
    target = output_root / run_id
    target.mkdir(exist_ok=False)
    return target.resolve(strict=True)


def _model_state_snapshot(model: Any, torch: Any) -> dict[str, Any]:
    state = {}
    for key, value in model.state_dict().items():
        item = value.detach().cpu().clone()
        if item.is_floating_point() and item.dtype != torch.float32:
            raise StudyError(f"floating model state is not FP32: {key} ({item.dtype})")
        state[key] = item
    return state


def _compare_state_changes(initial: Mapping[str, Any], final: Mapping[str, Any], torch: Any) -> dict[str, Any]:
    if set(initial) != set(final):
        raise StudyError("model state keys changed during training")
    changed_tensors = 0
    changed_elements = 0
    for name in initial:
        if initial[name].shape != final[name].shape or initial[name].dtype != final[name].dtype:
            raise StudyError(f"model state schema changed for {name}")
        changed = initial[name].ne(final[name])
        count = int(changed.sum().item())
        changed_elements += count
        changed_tensors += int(count > 0)
    return {
        "states_changed": changed_elements > 0,
        "changed_tensor_count": changed_tensors,
        "changed_element_count": changed_elements,
    }


def _check_elapsed(started: float, max_wall_seconds: float, phase: str) -> None:
    elapsed = time.monotonic() - started
    if elapsed > max_wall_seconds:
        raise StudyError(f"{phase} crossed the {max_wall_seconds:.0f}-second driver wall bound")
    if _rss_bytes() > MAX_RSS_BYTES:
        raise StudyError(f"{phase} crossed the 8-GiB host RSS bound")


def build_run_spec(args: argparse.Namespace) -> dict[str, Any]:
    paired_input_path = Path(args.paired_input_artifact).resolve(strict=True)
    order_index = ORDERED_RUNS.index((args.seed, args.mim_policy)) + 1
    return {
        "schema_version": VERSION,
        "seed": args.seed,
        "campaign_order_index": order_index,
        "ordered_campaign_run": [args.seed, args.mim_policy],
        "derived_training_dropout_seed": derive_training_seed(args.seed),
        "arm": args.arm,
        "text_cross_attention_layers": list(ARM_PLACEMENTS[args.arm]["text"]),
        "vision_cross_attention_layers": list(ARM_PLACEMENTS[args.arm]["vision"]),
        "cross_attention_backend": args.backend,
        "mlm_mode": args.mlm_mode,
        "mim_input_policy": args.mim_policy,
        "empty_mlm_policy": args.empty_mlm_policy,
        "data_mask_seed": DATA_SEED,
        "attempt_budget": args.attempts,
        "batch_size": BATCH_SIZE,
        "gradient_accumulation": 1,
        "joint_objective_backward_order": ["AP", "MLM", "MIM"],
        "objective_backward_memory_policy": "backward each scaled objective immediately to release its graph; unscale once and apply one AdamW update",
        "exception_step_accounting": "failed attempts compare the tracked AdamW step counter; an update is false only when failure is known pre-step or a returned scaler step did not advance it, true when it advanced, and null when uncertain",
        "learning_rate": 1e-4,
        "weight_decay": 0.01,
        "gradient_clipping": None,
        "amp": "CUDA autocast float16 + GradScaler; CPU smoke tests disable autocast",
        "grad_scaler": {
            "enabled_on_cuda": True,
            "init_scale": GRAD_SCALER_INIT_SCALE,
            "growth_factor": 2.0,
            "backoff_factor": 0.5,
            "growth_interval": 2000,
        },
        "device": args.device,
        "threads": THREADS,
        "max_driver_wall_seconds": args.max_wall_seconds,
        "max_rss_bytes": MAX_RSS_BYTES,
        "max_total_artifact_bytes": MAX_TOTAL_ARTIFACT_BYTES,
        "allocator_fraction": 0.30 if str(args.device).startswith("cuda") else None,
        "planned_run_count_for_storage_preflight": args.planned_runs,
        "paired_input_artifact_path": str(paired_input_path),
        "paired_input_artifact_sha256": sha256_file(paired_input_path),
        "initialization_anchor_root": str(_safe_output_root(args.initialization_anchor_root)),
        "caveat": "DF2 paired MIM input-scale pilot with fresh fusion/task-head initialization on shared offline pretrained backbones; not new BERT/ViT pretraining.",
    }


def perform_preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Validate exact inputs and local fresh initialization without training or GPU use."""
    started = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    configure_runtime_environment()
    train_manifest, train_sha = validate_manifest(
        args.train_manifest,
        expected_sha256=TRAIN_MANIFEST_SHA256,
        split="training-only",
        sample_count=200,
        verify_sources=True,
    )
    validation_manifest, validation_sha = validate_manifest(
        args.validation_manifest,
        expected_sha256=VALIDATION_MANIFEST_SHA256,
        split="validation-only",
        sample_count=64,
        verify_sources=True,
    )
    validate_disjoint_cohorts(train_manifest, validation_manifest)
    feasibility = load_feasibility_inputs(args.feasibility)
    assets = validate_local_assets(feasibility)

    import numpy as np
    import torch

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass
    tokenizer = _load_tokenizer(assets["tokenizer_path"])
    legacy_batches, legacy_data_report = materialize_training_batches(
        train_manifest,
        tokenizer,
        torch,
        np,
        mim_policy="legacy",
    )
    unit_batches, unit_data_report = materialize_training_batches(
        train_manifest,
        tokenizer,
        torch,
        np,
        mim_policy="unit_scale",
    )
    for policy, batches in (("legacy", legacy_batches), ("unit_scale", unit_batches)):
        actual_empty = preflight_empty_mlm_batches(batches, policy="zero_gradient")
        if tuple(actual_empty) != EXPECTED_EMPTY_MLM_BATCHES:
            raise StudyError(
                f"{policy} MLM empty-label schedule differs from frozen indices: {actual_empty}"
            )
    paired_inputs = build_paired_triad_identity_records(
        legacy_batches,
        unit_batches,
        legacy_data_report,
        unit_data_report,
        torch,
    )
    validation_batches, validation_report = materialize_validation_batches(
        validation_manifest,
        tokenizer,
        torch=torch,
    )
    del legacy_batches, unit_batches, validation_batches

    source_hashes = _source_hashes()
    construction_checks = []
    for seed in EXPECTED_SEEDS:
        seed_all(seed, torch, np, include_cuda=False)
        model = construct_fresh_model(
            _seeded_config(STUDY_ARM, STUDY_BACKEND, STUDY_MLM_MODE),
            bert_snapshot=assets["bert_snapshot"],
            vit_weights=assets["vision_weights_path"],
            vit_model_name=assets["vision_model_name"],
        )
        vision_load_info = getattr(model, "_autoresearch_pretrained_load_info", None)
        if not isinstance(vision_load_info, dict) or vision_load_info.get("strict_backbone_load") is not True:
            raise StudyError("fresh model construction did not report a strict local ViT backbone load")
        state_sha = hash_model_state(model.state_dict(), torch)
        state_bytes = sum(value.numel() * value.element_size() for value in model.state_dict().values())
        all_fp32 = all(
            not value.is_floating_point() or value.dtype == torch.float32
            for value in model.state_dict().values()
        )
        if not all_fp32:
            raise StudyError(f"fresh model for seed {seed} contains non-FP32 floating state")
        construction_checks.append({
            "seed": seed,
            "initial_model_state_sha256": state_sha,
            "model_state_bytes": state_bytes,
            "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
            "all_floating_state_fp32": all_fp32,
            "local_backbone_weight_sha256": {
                "bert": assets["bert_weights_sha256"],
                "vision": assets["vision_weights_sha256"],
            },
            "vision_pretrained_load": vision_load_info,
            "network_access": "disabled by offline environment; both constructor requests redirected to explicit local assets",
        })
        del model
        import gc

        gc.collect()
        _check_elapsed(started, MAX_WALL_SECONDS, "offline construction preflight")

    output_path = Path(args.preflight_output)
    output_parent = _safe_output_root(output_path.parent)
    output_path = output_parent / output_path.name
    if not output_path.parent.is_dir():
        output_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": "offline-fresh-training-preflight-v1",
        "status": "passed",
        "started_utc": started_utc,
        "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight_wall_seconds": time.monotonic() - started,
        "device": "cpu",
        "threads": THREADS,
        "training_attempts_executed": 0,
        "gpu_launched": False,
        "training_manifest": train_manifest["_identity"],
        "validation_manifest": validation_manifest["_identity"],
        "feasibility": feasibility,
        "offline_assets": assets,
        "fixed_data": {
            "legacy": legacy_data_report,
            "unit_scale": unit_data_report,
            "paired_triad_identity": paired_inputs,
            "empty_mlm_policy": "zero_gradient",
            "expected_empty_mlm_batch_indices_zero_based": list(EXPECTED_EMPTY_MLM_BATCHES),
            "loader_settings": legacy_data_report["loader_settings"],
        },
        "validation_input": validation_report,
        "fresh_local_initialization_checks": construction_checks,
        "source_sha256": source_hashes,
        "limits": {
            "max_wall_seconds": MAX_WALL_SECONDS,
            "max_host_rss_bytes": MAX_RSS_BYTES,
            "gpu_use": "none",
            "output_create_only": True,
        },
    }
    write_json_create_only(output_path, record)
    record["artifact_path"] = str(output_path)
    record["artifact_sha256"] = sha256_file(output_path)
    return record


def load_paired_input_preflight(path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    artifact_path = Path(path).resolve(strict=True)
    try:
        record = json.loads(artifact_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise StudyError(f"could not read paired-input preflight artifact: {exc}") from exc
    paired = record.get("fixed_data", {}).get("paired_triad_identity")
    if record.get("schema_version") != "offline-fresh-training-preflight-v1" or record.get("status") != "passed":
        raise StudyError("paired-input preflight is not a completed passing artifact")
    if not isinstance(paired, dict) or paired.get("batch_count") != MAX_ATTEMPTS:
        raise StudyError("paired-input preflight does not contain the exact 100-triad proof")
    if paired.get("empty_mlm_batch_indices_zero_based") != list(EXPECTED_EMPTY_MLM_BATCHES):
        raise StudyError("paired-input preflight has a different empty-label MLM schedule")
    return record, {
        "path": str(artifact_path),
        "sha256": sha256_file(artifact_path),
        "paired_invariant_sha256": paired["paired_invariant_sha256"],
        "policy_input_batch_sha256": paired["policy_input_batch_sha256"],
    }


def execute_run(args: argparse.Namespace) -> dict[str, Any]:
    validate_run_bounds(
        attempts=args.attempts,
        max_wall_seconds=args.max_wall_seconds,
        seed=args.seed,
        arm=args.arm,
        backend=args.backend,
        mlm_mode=args.mlm_mode,
        mim_policy=args.mim_policy,
        empty_mlm_policy=args.empty_mlm_policy,
    )
    if args.attempts != MAX_ATTEMPTS:
        raise StudyError(f"study attempt budget is frozen at exactly {MAX_ATTEMPTS}")
    if args.planned_runs != 6:
        raise StudyError("this study has exactly six planned paired runs")
    preflight_record, paired_input_artifact = load_paired_input_preflight(args.paired_input_artifact)
    if preflight_record["training_manifest"]["manifest_sha256"] != TRAIN_MANIFEST_SHA256:
        raise StudyError("paired-input preflight has a different training manifest")
    if preflight_record["validation_manifest"]["manifest_sha256"] != VALIDATION_MANIFEST_SHA256:
        raise StudyError("paired-input preflight has a different validation manifest")
    current_source_hashes = _source_hashes()
    if preflight_record.get("source_sha256") != current_source_hashes:
        raise StudyError("source/protocol files changed after paired-input preflight")
    started_monotonic = time.monotonic()
    started_utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_root = _safe_output_root(args.output_root)
    run_id = args.run_id or (
        f"seed{args.seed}-{args.arm}-{args.mim_policy}-{args.backend}-{args.mlm_mode}"
    )
    run_dir = _atomic_run_dir(output_root, run_id)
    spec = build_run_spec(args)
    write_json_create_only(run_dir / "run_spec.json", spec)
    partial: dict[str, Any] = {"run_spec": spec, "run_dir": str(run_dir), "phase": "preflight"}
    failure_path = run_dir / "failure.json"

    try:
        with _working_directory(run_dir):
            configure_runtime_environment()
            train_manifest, train_sha = validate_manifest(
                args.train_manifest,
                expected_sha256=TRAIN_MANIFEST_SHA256,
                split="training-only",
                sample_count=200,
                verify_sources=True,
            )
            validation_manifest, validation_sha = validate_manifest(
                args.validation_manifest,
                expected_sha256=VALIDATION_MANIFEST_SHA256,
                split="validation-only",
                sample_count=64,
                verify_sources=True,
            )
            validate_disjoint_cohorts(train_manifest, validation_manifest)
            partial["phase"] = "offline_assets"
            feasibility = load_feasibility_inputs(args.feasibility)
            assets = validate_local_assets(feasibility)

            import numpy as np
            import torch

            torch.set_num_threads(THREADS)
            try:
                torch.set_num_interop_threads(THREADS)
            except RuntimeError:
                pass
            if args.device == "cpu":
                device = torch.device("cpu")
                synchronize = lambda: None
            else:
                if not args.device.startswith("cuda:") or not torch.cuda.is_available():
                    raise StudyError("device must be cpu or an available explicit cuda:N")
                device = torch.device(args.device)
                if device.index is None or device.index >= torch.cuda.device_count():
                    raise StudyError(f"invalid CUDA device {args.device}")
                torch.cuda.set_device(device)
                torch.cuda.set_per_process_memory_fraction(0.30, device=device)
                synchronize = lambda: torch.cuda.synchronize(device)
            _check_elapsed(started_monotonic, args.max_wall_seconds, "runtime setup")

            tokenizer = _load_tokenizer(assets["tokenizer_path"])
            versions = _runtime_versions(torch, np, tokenizer)
            partial["runtime_versions"] = versions
            partial["phase"] = "fixed_data_materialization"
            data_started = time.monotonic()
            fixed_batches, data_report = materialize_training_batches(
                train_manifest,
                tokenizer,
                torch,
                np,
                mim_policy=args.mim_policy,
                data_seed=DATA_SEED,
            )
            expected_policy_hash = paired_input_artifact["policy_input_batch_sha256"][args.mim_policy]
            if data_report["input_batch_sha256"] != expected_policy_hash:
                raise StudyError("runtime materialized inputs differ from paired preflight hashes")
            expected_invariant_hash = preflight_record["fixed_data"][args.mim_policy]["policy_invariant_batch_sha256"]
            if data_report["policy_invariant_batch_sha256"] != expected_invariant_hash:
                raise StudyError("runtime invariant input hashes differ from paired preflight")
            if args.attempts == MAX_ATTEMPTS and train_sha == TRAIN_MANIFEST_SHA256:
                if data_report["empty_mlm_batch_indices_zero_based"] != list(EXPECTED_EMPTY_MLM_BATCHES):
                    raise StudyError("fixed all-ignored MLM mask indices changed from preflight")
            empty_batches = preflight_empty_mlm_batches(
                fixed_batches[:args.attempts],
                policy=args.empty_mlm_policy,
            )
            data_report["attempt_window_empty_mlm_indices_zero_based"] = empty_batches
            data_report["materialization_seconds"] = time.monotonic() - data_started
            _check_elapsed(started_monotonic, args.max_wall_seconds, "training-data materialization")
            validation_batches, validation_report = materialize_validation_batches(
                validation_manifest,
                tokenizer,
                torch=torch,
            )
            _check_elapsed(started_monotonic, args.max_wall_seconds, "validation-data materialization")

            source_hashes = current_source_hashes
            config = _seeded_config(args.arm, args.backend, args.mlm_mode)
            seed_all(args.seed, torch, np)
            partial["phase"] = "fresh_model_construction"
            model = construct_fresh_model(
                config,
                bert_snapshot=assets["bert_snapshot"],
                vit_weights=assets["vision_weights_path"],
                vit_model_name=assets["vision_model_name"],
            )
            vision_load_info = getattr(model, "_autoresearch_pretrained_load_info", None)
            if not isinstance(vision_load_info, dict) or vision_load_info.get("strict_backbone_load") is not True:
                raise StudyError("fresh model construction did not report a strict local ViT backbone load")
            initial_state = _model_state_snapshot(model, torch)
            initial_state_hash = hash_model_state(initial_state, torch)
            expected_seed_state_hash = next(
                item["initial_model_state_sha256"]
                for item in preflight_record["fresh_local_initialization_checks"]
                if item["seed"] == args.seed
            )
            if initial_state_hash != expected_seed_state_hash:
                raise StudyError("fresh model tensor hash differs from the offline preflight for this seed")
            model_state_bytes = sum(value.numel() * value.element_size() for value in initial_state.values())
            storage = check_storage_budget(output_root, model_state_bytes, args.planned_runs)
            seed_anchor = register_seed_initialization(
                output_root,
                seed=args.seed,
                state_sha256=initial_state_hash,
                backbone_hashes={
                    "bert_weights_sha256": assets["bert_weights_sha256"],
                    "vision_weights_sha256": assets["vision_weights_sha256"],
                },
                source_hashes=source_hashes,
                anchor_root=_safe_output_root(args.initialization_anchor_root),
            )
            init_record = {
                "seed": args.seed,
                "initial_model_state_sha256": initial_state_hash,
                "preflight_initial_model_state_sha256": expected_seed_state_hash,
                "matches_offline_preflight": True,
                "seed_anchor_path": str(_safe_output_root(args.initialization_anchor_root) / f"seed-{args.seed}.json"),
                "seed_anchor_verified": True,
                "model_state_bytes_fp32": model_state_bytes,
                "trainable_parameter_count": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad),
                "all_model_parameters_trainable": all(parameter.requires_grad for parameter in model.parameters()),
                "backbone_state_is_original_local_weights": True,
                "vision_pretrained_load": vision_load_info,
            }
            write_json_create_only(run_dir / "initialization.json", init_record)
            partial["initialization"] = init_record
            _check_elapsed(started_monotonic, args.max_wall_seconds, "model initialization")

            model.to(device)
            model.eval()
            partial["phase"] = "initial_validation_features"
            initial_text, initial_vision = extract_validation_cls_features(
                model, validation_batches, torch=torch, np=np, device=device
            )
            initial_metrics = summarize_geometry(initial_text, initial_vision)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
                torch.cuda.reset_peak_memory_stats(device)
            _check_elapsed(started_monotonic, args.max_wall_seconds, "initial validation features")

            partial["phase"] = "training"
            step_log_path = run_dir / "steps.jsonl"
            create_jsonl_log(step_log_path)
            partial["attempt_log_path"] = str(step_log_path)
            partial["steps_completed"] = 0
            partial["attempt_records_written"] = 0
            partial["failed_attempt_records"] = 0
            partial["actual_optimizer_steps"] = 0
            partial["skipped_optimizer_steps"] = 0

            def persist_attempt(step_result: Mapping[str, Any]) -> None:
                append_jsonl_durable(step_log_path, step_result)
                partial["attempt_records_written"] += 1
                partial["last_durable_attempt"] = dict(step_result)
                if step_result.get("attempt_record_status") == "completed":
                    partial["steps_completed"] += 1
                    partial["actual_optimizer_steps"] += int(bool(step_result["actual_optimizer_step"]))
                    if str(step_result.get("optimizer_step_status", "")).startswith("skipped"):
                        partial["skipped_optimizer_steps"] += 1
                else:
                    partial["failed_attempt_records"] += 1

            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
            scaler = torch.amp.GradScaler(
                device=device.type,
                init_scale=GRAD_SCALER_INIT_SCALE,
                growth_factor=2.0,
                backoff_factor=0.5,
                growth_interval=2000,
                enabled=device.type == "cuda",
            )
            seed_all(derive_training_seed(args.seed), torch, np)
            model.train()
            step_results, stop_reason = run_training_attempts(
                model,
                optimizer,
                scaler,
                fixed_batches,
                attempts=args.attempts,
                max_wall_seconds=args.max_wall_seconds,
                started_monotonic=started_monotonic,
                device=device,
                selected_tokens_only=(args.mlm_mode == "selected"),
                empty_mlm_policy=args.empty_mlm_policy,
                synchronize=synchronize,
                rss_bytes=_rss_bytes,
                torch_module=torch,
                on_attempt=persist_attempt,
            )
            partial["steps_completed"] = len(step_results)
            _check_elapsed(started_monotonic, args.max_wall_seconds, "training")
            if hash_tensor_tree(fixed_batches, torch) != data_report["input_batch_sha256"]:
                raise StudyError("fixed cached training batches mutated during training")

            partial["phase"] = "final_validation_features"
            final_text, final_vision = extract_validation_cls_features(
                model, validation_batches, torch=torch, np=np, device=device
            )
            final_metrics = summarize_geometry(final_text, final_vision)
            _check_elapsed(started_monotonic, args.max_wall_seconds, "final validation features")

            final_state = _model_state_snapshot(model, torch)
            final_state_hash = hash_model_state(final_state, torch)
            state_changes = _compare_state_changes(initial_state, final_state, torch)
            feature_path = run_dir / "validation_features.npz"
            feature_bytes = _save_npz_create_only(np, feature_path, {
                "initial_text_cls": initial_text,
                "initial_vision_cls": initial_vision,
                "final_text_cls": final_text,
                "final_vision_cls": final_vision,
            })
            checkpoint_path = run_dir / "final_model_fp32_cpu.pt"
            checkpoint_bytes = _save_torch_create_only(torch, checkpoint_path, {
                "model_state_dict": final_state,
                "config": _config_dict(config),
            })
            per_run_bytes = _tree_bytes(run_dir)
            if checkpoint_bytes > MAX_RUN_ARTIFACT_BYTES or per_run_bytes > MAX_RUN_ARTIFACT_BYTES:
                raise StudyError("run artifacts exceed the 1.5-GiB per-run storage bound")

            actual_steps = sum(bool(item["actual_optimizer_step"]) for item in step_results)
            skipped_steps = sum(not bool(item["actual_optimizer_step"]) for item in step_results)
            seen_ids = data_report["sample_order_ids"]["ap"][:2 * len(step_results)]
            train_results = {
                "schema_version": VERSION,
                "status": "completed" if len(step_results) == args.attempts else "partial_wall_limit",
                "started_utc": started_utc,
                "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "driver_wall_seconds": time.monotonic() - started_monotonic,
                "gpu_job_wall_seconds_note": "This value is process/driver wall time; use the owned-job ledger for scheduler-inclusive GPU-job wall time.",
                "run_spec": spec,
                "run_id": run_id,
                "output_dir": str(run_dir),
                "inputs": {
                    "train_manifest": train_manifest["_identity"],
                    "validation_manifest": validation_manifest["_identity"],
                    "validation_input": validation_report,
                    "training_data": data_report,
                    "train_validation_selected_content_disjoint": True,
                    "full_training_csv_accessed": True,
                    "test_or_hateful_memes_data_accessed": False,
                    "feasibility_record": feasibility,
                    "offline_assets": assets,
                    "paired_input_preflight": paired_input_artifact,
                },
                "initialization": init_record,
                "training": {
                    "attempts_requested": args.attempts,
                    "attempts_started": len(step_results),
                    "actual_optimizer_steps": actual_steps,
                    "skipped_optimizer_steps": skipped_steps,
                    "stop_reason": stop_reason,
                    "unique_training_rows_exposed": len(set(seen_ids)),
                    "training_sample_ids_exposed_ordered": seen_ids,
                    "processed_samples_per_objective": len(step_results) * BATCH_SIZE,
                    "mlm_empty_batches_in_attempt_window": len(empty_batches),
                    "mim_input_policy": args.mim_policy,
                    "step_results": step_results,
                    "durable_attempt_log_path": str(step_log_path),
                    "durable_attempt_log_sha256": sha256_file(step_log_path),
                    "durable_attempt_log_count": partial["steps_completed"],
                    "constant_learning_rate": 1e-4,
                    "weight_decay": 0.01,
                    "gradient_accumulation": 1,
                    "joint_objective_backward_order": ["AP", "MLM", "MIM"],
                    "gradient_clipping": None,
                    "optimizer": "AdamW over every trainable model parameter",
                    "precision": "CUDA float16 autocast with GradScaler; all model weights FP32; CPU tests run full precision",
                "grad_scaler": {
                    "init_scale": GRAD_SCALER_INIT_SCALE,
                    "growth_factor": 2.0,
                    "backoff_factor": 0.5,
                    "growth_interval": 2000,
                    "step_statuses": "scale before/after, finite gradients after one unscale, parameter finiteness after step, and actual AdamW counter advance recorded per attempt",
                },
                },
                "validation": {
                    "batch_size": BATCH_SIZE,
                    "initial": initial_metrics,
                    "final": final_metrics,
                    "readout": "all 12 paired CLS layer features, raw centered linear CKA, text/vision participation ratio only; no null calibration",
                    "feature_artifact": str(feature_path),
                    "feature_artifact_bytes": feature_bytes,
                },
                "model_state": {
                    "initial_sha256": initial_state_hash,
                    "final_sha256": final_state_hash,
                    **state_changes,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "checkpoint_bytes": checkpoint_bytes,
                    "format": "FP32 CPU state_dict plus config only; no optimizer/scaler",
                },
                "source_sha256": source_hashes,
                "runtime_versions": versions,
                "resources": {
                    "peak_host_rss_bytes": _rss_bytes(),
                    "peak_gpu_allocated_bytes": max(
                        (int(item["gpu_peak_allocated_bytes"] or 0) for item in step_results),
                        default=0,
                    ) if device.type == "cuda" else None,
                    "peak_gpu_reserved_bytes": max(
                        (int(item["gpu_peak_reserved_bytes"] or 0) for item in step_results),
                        default=0,
                    ) if device.type == "cuda" else None,
                    "storage_preflight": storage,
                    "run_artifact_bytes": per_run_bytes,
                },
                "seed_anchor": seed_anchor,
                "reproducibility_caveats": [
                    "Numeric dropout seeds are derived consistently, but arm/policy runs can consume stochastic calls differently.",
                    "AMP/GradScaler can skip optimizer updates; attempted and applied updates are both reported.",
                    "Exceptions after optimizer execution compare the tracked AdamW counter; unresolvable outcomes are null and require adjudication, not automatic repair.",
                    "GPU kernels and library versions may prevent bitwise reproducibility.",
                    "This is a short fixed-cohort pilot, not full encoder pretraining or a downstream utility estimate.",
                    "MIM unit_scale changes only masked-view input scale; weak/strong augmentations remain different and masked zeros stay zero.",
                ],
            }
            write_json_create_only(run_dir / "steps.json", {"steps": step_results})
            write_json_create_only(run_dir / "result.json", train_results)
            return train_results
    except Exception as exc:
        attempt_log_info: dict[str, Any] | None = None
        attempt_log_value = partial.get("attempt_log_path")
        if attempt_log_value:
            attempt_log_path = Path(attempt_log_value)
            durable_rows, corrupt_tail = read_jsonl_steps(attempt_log_path)
            if attempt_log_path.is_file():
                completed_rows = [
                    row for row in durable_rows
                    if row.get("attempt_record_status") == "completed"
                ]
                attempt_log_info = {
                    "path": str(attempt_log_path),
                    "sha256": sha256_file(attempt_log_path),
                    "durable_record_count": len(durable_rows),
                    "completed_attempt_result_count": len(completed_rows),
                    "failed_attempt_record_count": sum(
                        row.get("attempt_record_status") != "completed" for row in durable_rows
                    ),
                    "failed_attempt_optimizer_steps_observed": sum(
                        row.get("attempt_record_status") != "completed"
                        and row.get("actual_optimizer_step") is True
                        for row in durable_rows
                    ),
                    "failed_attempt_optimizer_step_outcome_unknown_count": sum(
                        row.get("attempt_record_status") != "completed"
                        and row.get("actual_optimizer_step") is None
                        for row in durable_rows
                    ),
                    "actual_optimizer_steps": sum(
                        bool(row.get("actual_optimizer_step")) for row in completed_rows
                    ),
                    "skipped_optimizer_steps": sum(
                        str(row.get("optimizer_step_status", "")).startswith("skipped")
                        for row in completed_rows
                    ),
                    "last_durable_attempt": durable_rows[-1] if durable_rows else None,
                    "corrupt_tail": corrupt_tail,
                }
                partial["steps_completed"] = len(completed_rows)
        failure = {
            "schema_version": VERSION,
            "status": "failed",
            "started_utc": started_utc,
            "failed_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "driver_wall_seconds": time.monotonic() - started_monotonic,
            "run_spec": spec,
            "run_dir": str(run_dir),
            "phase": partial.get("phase"),
            "steps_completed": partial.get("steps_completed", 0),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "durable_attempt_log": attempt_log_info,
            "partial_provenance": partial,
        }
        try:
            write_json_create_only(failure_path, failure)
        except Exception as write_exc:
            raise StudyError(f"run failed ({exc}); failure artifact could not be saved ({write_exc})") from exc
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", required=True, type=int, choices=EXPECTED_SEEDS)
    parser.add_argument("--arm", required=True, choices=(STUDY_ARM,))
    parser.add_argument("--mim-policy", required=True, choices=("legacy", "unit_scale"))
    parser.add_argument("--backend", required=True, choices=(STUDY_BACKEND,))
    parser.add_argument("--mlm-mode", required=True, choices=(STUDY_MLM_MODE,))
    parser.add_argument("--empty-mlm-policy", required=True, choices=("zero_gradient",))
    parser.add_argument("--attempts", type=int, default=MAX_ATTEMPTS, choices=(MAX_ATTEMPTS,))
    parser.add_argument("--max-wall-seconds", type=float, default=MAX_WALL_SECONDS)
    parser.add_argument("--device", default="cuda:0", help="cpu or an explicit available cuda:N")
    parser.add_argument("--planned-runs", type=int, default=6, choices=(6,), help="fixed paired run count used for the storage reserve check")
    parser.add_argument("--run-id")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--validation-manifest", type=Path, default=DEFAULT_VALIDATION_MANIFEST)
    parser.add_argument("--feasibility", type=Path, default=DEFAULT_FEASIBILITY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--initialization-anchor-root", type=Path, default=DEFAULT_INITIALIZATION_ANCHOR_ROOT)
    parser.add_argument("--paired-input-artifact", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT)
    parser.add_argument("--preflight-only", action="store_true", help="verify real inputs and local model construction without a training step")
    parser.add_argument("--preflight-output", type=Path, default=DEFAULT_PREFLIGHT_OUTPUT)
    args = parser.parse_args(argv)
    try:
        validate_run_bounds(
            attempts=args.attempts,
            max_wall_seconds=args.max_wall_seconds,
            seed=args.seed,
            arm=args.arm,
            backend=args.backend,
            mlm_mode=args.mlm_mode,
            mim_policy=args.mim_policy,
            empty_mlm_policy=args.empty_mlm_policy,
        )
    except StudyError as exc:
        parser.error(str(exc))
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.preflight_only:
        try:
            result = perform_preflight(args)
        except Exception as exc:
            print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
            return 2
        print(json.dumps({
            "status": result["status"],
            "preflight_path": result["artifact_path"],
            "preflight_sha256": result["artifact_sha256"],
            "elapsed_seconds": result["preflight_wall_seconds"],
            "init_seeds_checked": [item["seed"] for item in result["fresh_local_initialization_checks"]],
            "empty_mlm_batches": result["fixed_data"]["paired_triad_identity"]["empty_mlm_batch_indices_zero_based"],
        }, sort_keys=True))
        return 0
    try:
        result = execute_run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "error": str(exc)}))
        return 2
    print(json.dumps({
        "status": result["status"],
        "run_id": result["run_id"],
        "result_path": str(Path(result["output_dir"]) / "result.json"),
        "attempts_started": result["training"]["attempts_started"],
        "actual_optimizer_steps": result["training"]["actual_optimizer_steps"],
        "skipped_optimizer_steps": result["training"]["skipped_optimizer_steps"],
    }, sort_keys=True))
    return 0 if result["status"] == "completed" else 3


if __name__ == "__main__":
    raise SystemExit(main())

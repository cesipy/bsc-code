"""Read-only validation cohort indexing for checkpoint-only analysis.

This module keeps data discovery and immutable sample selection independent
from model loading.  It deliberately supports only named validation-family
splits; it never guesses a split from the files that happen to exist.
"""

from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


MANIFEST_VERSION = 1
MAX_SAMPLES = 4096
SUPPORTED_TASKS = ("hateful_memes", "mm_imdb", "upmc_food")


class ValidationDataError(ValueError):
    """Raised when the requested validation cohort is missing or ambiguous."""


@dataclass(frozen=True)
class ValidationRecord:
    sample_id: str
    group_id: str
    source_key: str
    pair_sha256: str
    image_sha256: str
    text_sha256: str
    position: int
    payload: dict[str, Any]


@dataclass
class ValidationIndex:
    task: str
    split: str
    data_root: Path
    source_files: dict[str, str]
    records: list[ValidationRecord]
    dataset: Any = None
    validation_role: str = "validation_discovery"

    def by_id(self) -> dict[str, ValidationRecord]:
        return {record.sample_id: record for record in self.records}


def canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        while chunk := file.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_array(value: Any) -> str:
    """Hash an image array with its shape and dtype, without serializing it."""
    import numpy as np

    array = np.ascontiguousarray(np.asarray(value))
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(canonical_json(list(array.shape)))
    digest.update(memoryview(array).cast("B"))
    return digest.hexdigest()


def _safe_data_file(root: Path, relative_path: str) -> Path:
    try:
        candidate = (root / relative_path).resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationDataError(
            f"required validation source file is missing: {relative_path}"
        ) from exc
    resolved_root = root.resolve(strict=True)
    if not candidate.is_relative_to(resolved_root):
        raise ValidationDataError(
            f"data path escapes --data-root: {relative_path!r}"
        )
    if not candidate.is_file():
        raise ValidationDataError(f"expected a file at {candidate}")
    return candidate


def _require_data_root(path: str | Path) -> Path:
    root = Path(path)
    if not root.is_absolute():
        raise ValidationDataError("--data-root must be an absolute path")
    try:
        resolved = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise ValidationDataError(f"data root does not exist: {root}") from exc
    if not resolved.is_dir():
        raise ValidationDataError(f"data root is not a directory: {resolved}")
    return resolved


def _pair_hash(image_hash: str, text: str, label: Any) -> tuple[str, str]:
    text_hash = sha256_bytes(text.encode("utf-8"))
    label_value = label.item() if hasattr(label, "item") else label
    pair_hash = sha256_bytes(
        canonical_json(
            {
                "image_sha256": image_hash,
                "text_sha256": text_hash,
                "label": str(label_value),
            }
        )
    )
    return text_hash, pair_hash


def _make_record(
    *,
    sample_id: str,
    source_key: str,
    image_hash: str,
    text: str,
    label: Any,
    position: int,
    payload: dict[str, Any],
) -> ValidationRecord:
    text_hash, pair_hash = _pair_hash(image_hash, text, label)
    return ValidationRecord(
        sample_id=sample_id,
        group_id=f"image-sha256:{image_hash}",
        source_key=source_key,
        pair_sha256=pair_hash,
        image_sha256=image_hash,
        text_sha256=text_hash,
        position=position,
        payload=payload,
    )


def _hateful_memes_index(root: Path, split: str) -> ValidationIndex:
    if split != "dev_seen":
        raise ValidationDataError(
            "Hateful Memes requires explicit --split dev_seen for this pilot; "
            "no other split is selected automatically"
        )
    metadata_path = _safe_data_file(root, "dev_seen.jsonl")
    source_hash = sha256_file(metadata_path)
    records: list[ValidationRecord] = []
    seen_ids: set[str] = set()
    with metadata_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                source = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValidationDataError(
                    f"invalid JSON in {metadata_path} at line {line_number}"
                ) from exc
            if not isinstance(source, dict):
                raise ValidationDataError(
                    f"expected a JSON object at {metadata_path}:{line_number}"
                )
            if "id" not in source or source["id"] is None or "img" not in source:
                raise ValidationDataError(
                    f"missing stable id or image key at {metadata_path}:{line_number}"
                )
            if "text" not in source or "label" not in source:
                raise ValidationDataError(
                    f"missing text or label at {metadata_path}:{line_number}"
                )
            sample_id = str(source["id"]).strip()
            if not sample_id:
                raise ValidationDataError(
                    f"empty source id at {metadata_path}:{line_number}"
                )
            if sample_id in seen_ids:
                raise ValidationDataError(
                    f"duplicate source id {sample_id!r} in {metadata_path}"
                )
            seen_ids.add(sample_id)
            image_relpath = str(source["img"])
            image_path = _safe_data_file(root, image_relpath)
            image_hash = sha256_file(image_path)
            text = str(source["text"])
            records.append(
                _make_record(
                    sample_id=f"hateful_memes:{sample_id}",
                    source_key=image_relpath,
                    image_hash=image_hash,
                    text=text,
                    label=source["label"],
                    position=len(records),
                    payload={
                        "image_path": str(image_path),
                        "image_relpath": image_relpath,
                        "text": text,
                        "label": source["label"],
                    },
                )
            )
    if not records:
        raise ValidationDataError(f"validation split is empty: {metadata_path}")
    return ValidationIndex(
        task="hateful_memes",
        split=split,
        data_root=root,
        source_files={"dev_seen.jsonl": source_hash},
        records=records,
        validation_role="validation_discovery_seen_historical",
    )


def _mm_imdb_index(root: Path, split: str) -> ValidationIndex:
    if split != "validation":
        raise ValidationDataError(
            "MM-IMDB requires explicit --split validation"
        )
    csv_path = _safe_data_file(root, "mmimdb_trainval.csv")
    image_store_path = _safe_data_file(root, "images.h5")

    # Import only for this selected task. This constructor reads its HDF5 input
    # in read-only mode and applies the repository's documented 80/20 split.
    from datasets.dataset_mm_imdb import MM_IMDB_Dataset

    dataset = MM_IMDB_Dataset(
        csv_path=str(csv_path),
        img_path=str(image_store_path),
        tokenizer=None,
        image_processor=None,
        is_train=False,
        is_test=False,
        train_test_ratio=0.8,
        transform=None,
    )
    records: list[ValidationRecord] = []
    for position in range(len(dataset.csv_data)):
        row = dataset.csv_data.iloc[position]
        image_index = row["img_index"]
        raw_image = dataset.img_data["images"][image_index]
        image_hash = _hash_array(raw_image)
        text = str(row["caption"])
        text_hash = sha256_bytes(text.encode("utf-8"))
        source_key = str(image_index)
        sample_id = f"mm_imdb:{source_key}:{text_hash[:16]}:row:{position}"
        records.append(
            _make_record(
                sample_id=sample_id,
                source_key=source_key,
                image_hash=image_hash,
                text=text,
                label=row["genre"],
                position=position,
                payload={"dataset_position": position, "source_key": source_key},
            )
        )
    if not records:
        dataset.img_data.close()
        raise ValidationDataError("MM-IMDB validation split is empty")
    return ValidationIndex(
        task="mm_imdb",
        split=split,
        data_root=root,
        source_files={
            "mmimdb_trainval.csv": sha256_file(csv_path),
            "images.h5": sha256_file(image_store_path),
        },
        records=records,
        dataset=dataset,
    )


def _upmc_index(root: Path, split: str) -> ValidationIndex:
    if split != "validation":
        raise ValidationDataError("UPMC Food-101 requires --split validation")
    csv_path = _safe_data_file(root, "upmcfood_trainval.csv")
    image_root = (root / "images").resolve(strict=True)
    if not image_root.is_dir() or not image_root.is_relative_to(root.resolve()):
        raise ValidationDataError("UPMC image directory must stay below --data-root")

    from datasets.dataset_upmc import UPMC_Dataset

    dataset = UPMC_Dataset(
        csv_path=str(csv_path),
        img_path=str(image_root),
        tokenizer=None,
        image_processor=None,
        is_train=False,
        is_test=False,
        train_test_ratio=0.8,
        transform=None,
    )
    records: list[ValidationRecord] = []
    for position in range(len(dataset.data)):
        row = dataset.data.iloc[position]
        image_relpath, label, _class_name, text = row.tolist()
        image_relpath = str(image_relpath)
        image_path = _safe_data_file(image_root, image_relpath)
        image_hash = sha256_file(image_path)
        text = str(text)
        text_hash = sha256_bytes(text.encode("utf-8"))
        source_key = image_relpath
        sample_id = f"upmc_food:{source_key}:{text_hash[:16]}:row:{position}"
        records.append(
            _make_record(
                sample_id=sample_id,
                source_key=source_key,
                image_hash=image_hash,
                text=text,
                label=label,
                position=position,
                payload={"dataset_position": position, "source_key": source_key},
            )
        )
    if not records:
        raise ValidationDataError("UPMC Food validation split is empty")
    return ValidationIndex(
        task="upmc_food",
        split=split,
        data_root=root,
        source_files={"upmcfood_trainval.csv": sha256_file(csv_path)},
        records=records,
        dataset=dataset,
    )


def open_validation_index(
    task: str, split: str, data_root: str | Path
) -> ValidationIndex:
    """Index one named validation-family split and its paired source rows."""
    if task not in SUPPORTED_TASKS:
        raise ValidationDataError(
            f"unsupported task {task!r}; choose one of {SUPPORTED_TASKS}"
        )
    root = _require_data_root(data_root)
    if task == "hateful_memes":
        return _hateful_memes_index(root, split)
    if task == "mm_imdb":
        return _mm_imdb_index(root, split)
    return _upmc_index(root, split)


def _unique_image_records(records: list[ValidationRecord]) -> tuple[list[ValidationRecord], int]:
    by_group: dict[str, ValidationRecord] = {}
    for record in records:
        existing = by_group.get(record.group_id)
        if existing is None or record.sample_id < existing.sample_id:
            by_group[record.group_id] = record
    unique = sorted(by_group.values(), key=lambda record: record.sample_id)
    return unique, len(records) - len(unique)


def create_ordered_manifest(
    index: ValidationIndex, *, n_samples: int, seed: int
) -> dict[str, Any]:
    """Create a deterministic sample manifest after exact-image deduplication."""
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise ValidationDataError("n_samples must be an integer")
    if not 2 <= n_samples <= MAX_SAMPLES:
        raise ValidationDataError(
            f"n_samples must be between 2 and {MAX_SAMPLES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValidationDataError("seed must be a non-negative integer")
    unique, removed = _unique_image_records(index.records)
    if n_samples > len(unique):
        raise ValidationDataError(
            f"requested {n_samples} samples, but only {len(unique)} distinct "
            f"image groups remain after deduplication"
        )
    selected = random.Random(seed).sample(unique, n_samples)
    selected_text_counts: dict[str, int] = {}
    for record in selected:
        selected_text_counts[record.text_sha256] = (
            selected_text_counts.get(record.text_sha256, 0) + 1
        )
    duplicate_caption_rows = sum(count - 1 for count in selected_text_counts.values())
    samples = [
        {
            "id": record.sample_id,
            "group_id": record.group_id,
            "caption_group_id": f"text-sha256:{record.text_sha256}",
            "source_key": record.source_key,
            "pair_sha256": record.pair_sha256,
            "image_sha256": record.image_sha256,
            "text_sha256": record.text_sha256,
        }
        for record in selected
    ]
    return {
        "format": "autoresearch-validation-manifest",
        "manifest_version": MANIFEST_VERSION,
        "task": index.task,
        "split": index.split,
        "validation_role": index.validation_role,
        "selection": {
            "method": "python-random-sample-after-deterministic-image-dedup-v1",
            "seed": seed,
            "n_samples": n_samples,
            "records_before_deduplication": len(index.records),
            "duplicate_image_records_removed": removed,
            "distinct_image_groups": len(unique),
            "unique_caption_groups_selected": len(selected_text_counts),
            "repeated_caption_groups_selected": sum(
                count > 1 for count in selected_text_counts.values()
            ),
            "duplicate_caption_rows_selected": duplicate_caption_rows,
        },
        "source_data_sha256": dict(sorted(index.source_files.items())),
        "ordered_sample_ids": [sample["id"] for sample in samples],
        "samples": samples,
        "exchangeability_note": (
            "Exact duplicate image content was deduplicated globally within the "
            "selected validation split. Exact caption groups are recorded but are "
            "not deduplicated or used as permutation blocks; the unrestricted row "
            "null therefore relies on row exchangeability and may not account for "
            "caption or other source dependence."
        ),
    }


def validate_manifest(manifest: Any) -> dict[str, Any]:
    if not isinstance(manifest, dict):
        raise ValidationDataError("sample manifest must be a JSON object")
    if manifest.get("format") != "autoresearch-validation-manifest":
        raise ValidationDataError("unrecognized sample manifest format")
    if manifest.get("manifest_version") != MANIFEST_VERSION:
        raise ValidationDataError("unsupported sample manifest version")
    task = manifest.get("task")
    split = manifest.get("split")
    if task not in SUPPORTED_TASKS:
        raise ValidationDataError("manifest must name a supported task")
    expected_split = "dev_seen" if task == "hateful_memes" else "validation"
    if split != expected_split:
        raise ValidationDataError("manifest must name a validation-family split")
    samples = manifest.get("samples")
    ordered_ids = manifest.get("ordered_sample_ids")
    if not isinstance(samples, list) or not isinstance(ordered_ids, list):
        raise ValidationDataError("manifest needs samples and ordered_sample_ids")
    ids = [sample.get("id") for sample in samples if isinstance(sample, dict)]
    groups = [sample.get("group_id") for sample in samples if isinstance(sample, dict)]
    caption_groups = [
        sample.get("caption_group_id") for sample in samples if isinstance(sample, dict)
    ]
    if (
        len(ids) != len(samples)
        or len(groups) != len(samples)
        or len(caption_groups) != len(samples)
    ):
        raise ValidationDataError(
            "each manifest sample needs id, image group, and caption group"
        )
    if ids != ordered_ids:
        raise ValidationDataError("ordered_sample_ids do not match samples order")
    if not ids or len(ids) != len(set(ids)):
        raise ValidationDataError("manifest sample IDs must be non-empty and unique")
    if len(groups) != len(set(groups)):
        raise ValidationDataError(
            "manifest contains duplicate image groups; create a deduplicated manifest"
        )
    selection = manifest.get("selection")
    if not isinstance(selection, dict) or selection.get("n_samples") != len(ids):
        raise ValidationDataError("manifest selection count does not match its IDs")
    if not isinstance(selection.get("seed"), int) or selection["seed"] < 0:
        raise ValidationDataError("manifest selection needs a non-negative integer seed")
    return manifest


def validate_manifest_against_index(
    manifest: dict[str, Any], index: ValidationIndex
) -> None:
    validate_manifest(manifest)
    if manifest.get("task") != index.task or manifest.get("split") != index.split:
        raise ValidationDataError("manifest task/split differs from the requested data")
    if manifest.get("source_data_sha256") != dict(sorted(index.source_files.items())):
        raise ValidationDataError("validation source data hashes differ from manifest")
    expected = create_ordered_manifest(
        index,
        n_samples=int(manifest["selection"]["n_samples"]),
        seed=int(manifest["selection"]["seed"]),
    )
    if manifest.get("ordered_sample_ids") != expected["ordered_sample_ids"]:
        raise ValidationDataError(
            "ordered sample IDs do not match the frozen seed and deduplication rule"
        )
    current = index.by_id()
    for sample in manifest["samples"]:
        record = current.get(sample["id"])
        if record is None:
            raise ValidationDataError(
                f"manifest sample is absent from requested validation split: {sample['id']}"
            )
        for key in (
            "group_id",
            "caption_group_id",
            "source_key",
            "pair_sha256",
            "image_sha256",
            "text_sha256",
        ):
            expected_value = (
                f"text-sha256:{record.text_sha256}"
                if key == "caption_group_id"
                else getattr(record, key)
            )
            if sample.get(key) != expected_value:
                raise ValidationDataError(
                    f"validation pairing changed for sample {sample['id']} ({key})"
                )


def manifest_sha256(manifest: dict[str, Any]) -> str:
    return sha256_bytes(canonical_json(validate_manifest(manifest)))


def read_manifest(path: str | Path) -> dict[str, Any]:
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            manifest = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise ValidationDataError(f"could not read sample manifest {path}: {exc}") from exc
    return validate_manifest(manifest)


def write_manifest_once(
    manifest: dict[str, Any],
    path: str | Path,
    *,
    authorized_output_root: str | Path,
    data_root: str | Path,
) -> Path:
    """Write an immutable manifest beneath the authorized output root only."""
    validate_manifest(manifest)
    output_root = Path(authorized_output_root)
    if output_root.is_symlink():
        raise ValidationDataError("authorized output root must not be a symlink")
    resolved_root = output_root.resolve(strict=True)
    target = Path(path)
    if not target.is_absolute():
        raise ValidationDataError("manifest output path must be absolute")
    resolved_target = target.resolve(strict=False)
    if not resolved_target.is_relative_to(resolved_root):
        raise ValidationDataError("manifest output must stay inside the authorized output root")
    resolved_data = Path(data_root).resolve(strict=True)
    if resolved_target.is_relative_to(resolved_data) or resolved_data.is_relative_to(resolved_target):
        raise ValidationDataError("manifest output must not overlap the read-only data root")
    if target.exists() and target.is_symlink():
        raise ValidationDataError("manifest path must not be a symlink")
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with target.open("x", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2, ensure_ascii=False)
            file.write("\n")
    except FileExistsError as exc:
        raise ValidationDataError(
            f"sample manifests are immutable and will not be overwritten: {target}"
        ) from exc
    return target.resolve(strict=True)


def close_validation_index(index: ValidationIndex) -> None:
    dataset = index.dataset
    image_file = getattr(dataset, "img_data", None)
    if image_file is not None:
        try:
            image_file.close()
        except Exception:
            pass
        dataset.img_data = None


def load_hateful_memes_item(record: ValidationRecord) -> tuple[str, str]:
    """Return the selected caption and image path without the legacy HM loader."""
    image_path = record.payload["image_path"]
    text = record.payload["text"]
    return text, image_path


def load_existing_dataset_item(
    index: ValidationIndex, record: ValidationRecord
) -> tuple[Any, Any]:
    """Read one MM-IMDB/UPMC validation example through its existing dataset."""
    if index.dataset is None:
        raise ValidationDataError("this task uses the side-effect-free HM reader")
    item = index.dataset[record.position]
    return item["text"], item["img"]


"""Selected-row validation manifests for MM-IMDB and UPMC Food.

The original evaluator indexes and hashes every validation image.  This entry
point preserves its manifest and evaluation APIs while selecting rows through
a deterministic random candidate order and hashing only the selected unique
images.  The protocol is row-uniform with duplicate replenishment; it does not
claim uniform sampling over all distinct image groups.
"""

from __future__ import annotations

import csv
import random
import sys
from pathlib import Path
from typing import Any, Callable

import autoresearch

# ``analyses.__init__`` imports a legacy module that may create relative logs.
# Import it only while the base evaluator has moved into its owned output root,
# and restore the caller's working directory immediately after import.
with autoresearch._working_directory(autoresearch.authorized_output_root()):
    from analyses import autoresearch_data as data


SUBSET_INDEX_PROTOCOL_VERSION = "random-row-candidates-image-dedup-v1.1"
SUBSET_TASKS = ("mm_imdb", "upmc_food")
VALIDATION_RATIO = 0.8
TASK_SPLIT_SHUFFLE_SEEDS = {"mm_imdb": 42, "upmc_food": 46}


def _validate_sampling_parameters(n_samples: int, seed: int) -> None:
    if isinstance(n_samples, bool) or not isinstance(n_samples, int):
        raise data.ValidationDataError("n_samples must be an integer")
    if not 2 <= n_samples <= data.MAX_SAMPLES:
        raise data.ValidationDataError(
            f"n_samples must be between 2 and {data.MAX_SAMPLES}"
        )
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise data.ValidationDataError("seed must be a non-negative integer")


def _count_csv_rows(path: Path) -> int:
    """Count nonblank logical CSV rows, including the header."""
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as source:
            return sum(
                1
                for row in csv.reader(source)
                if row and any(cell.strip() for cell in row)
            )
    except (OSError, UnicodeError, csv.Error) as exc:
        raise data.ValidationDataError(
            f"could not count rows in validation metadata CSV {path}: {exc}"
        ) from exc


def _candidate_positions(row_count: int, seed: int) -> list[int]:
    positions = list(range(row_count))
    random.Random(seed).shuffle(positions)
    return positions


def _selected_pair_fingerprint(records: list[data.ValidationRecord]) -> str:
    pairs = [
        {
            "position": record.position,
            "sample_id": record.sample_id,
            "source_key": record.source_key,
            "image_sha256": record.image_sha256,
            "text_sha256": record.text_sha256,
            "pair_sha256": record.pair_sha256,
        }
        for record in records
    ]
    return data.sha256_bytes(data.canonical_json(pairs))


def _source_hashes(
    csv_path: Path,
    records: list[data.ValidationRecord],
    *,
    wrapper_path: Path | None = None,
    data_code_path: Path | None = None,
) -> dict[str, str]:
    wrapper_path = wrapper_path or Path(__file__).resolve(strict=True)
    data_code_path = data_code_path or Path(data.__file__).resolve(strict=True)
    return {
        csv_path.name: data.sha256_file(csv_path),
        "src/analyses/autoresearch_data.py": data.sha256_file(data_code_path),
        "src/autoresearch_subset_index.py": data.sha256_file(wrapper_path),
        "selected_validation_pairs": _selected_pair_fingerprint(records),
    }


def _file_metadata(
    path: Path, *, content_hash_scope: str, **extra: Any
) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path),
        "size_bytes": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "content_hash_scope": content_hash_scope,
        **extra,
    }


def _select_records(
    *,
    task: str,
    row_count: int,
    n_samples: int,
    seed: int,
    row_fields: Callable[[int], tuple[str, str, Any]],
    image_hash: Callable[[int, str], str],
) -> tuple[list[data.ValidationRecord], dict[str, Any]]:
    """Scan a seeded row permutation until N distinct image hashes are found."""
    _validate_sampling_parameters(n_samples, seed)
    if row_count < 1:
        raise data.ValidationDataError("validation split is empty")

    candidates = _candidate_positions(row_count, seed)
    selected: list[data.ValidationRecord] = []
    seen_image_hashes: set[str] = set()
    duplicate_rows_skipped = 0
    rows_scanned = 0
    for position in candidates:
        rows_scanned += 1
        source_key, text, label = row_fields(position)
        image_digest = image_hash(position, source_key)
        if image_digest in seen_image_hashes:
            duplicate_rows_skipped += 1
            continue
        seen_image_hashes.add(image_digest)
        text = str(text)
        text_digest = data.sha256_bytes(text.encode("utf-8"))
        sample_id = (
            f"{task}:{source_key}:{text_digest[:16]}:row:{position}"
        )
        selected.append(
            data._make_record(
                sample_id=sample_id,
                source_key=source_key,
                image_hash=image_digest,
                text=text,
                label=label,
                position=position,
                payload={"dataset_position": position, "source_key": source_key},
            )
        )
        if len(selected) == n_samples:
            break

    if len(selected) != n_samples:
        raise data.ValidationDataError(
            f"requested {n_samples} distinct images, but found only "
            f"{len(selected)} after scanning all {row_count} validation rows"
        )

    metadata = {
        "version": SUBSET_INDEX_PROTOCOL_VERSION,
        "seed": seed,
        "candidate_order": "python-random.Random(seed).shuffle(validation-row-positions)",
        "candidate_permutation_sha256": data.sha256_bytes(
            data.canonical_json(candidates)
        ),
        "scanned_candidate_prefix_sha256": data.sha256_bytes(
            data.canonical_json(candidates[:rows_scanned])
        ),
        "validation_row_count": row_count,
        "candidate_rows_scanned": rows_scanned,
        "duplicate_image_candidate_rows_skipped": duplicate_rows_skipped,
        "selected_distinct_image_count": len(selected),
        "selected_candidate_validation_positions": [
            record.position for record in selected
        ],
        "full_validation_image_inventory_hashed": False,
        "sampling_claim": (
            "Rows are visited in a seeded random permutation; repeated image "
            "content is skipped until N unique images are selected. This is "
            "not uniform sampling over the unknown set of all distinct image "
            "groups, because group multiplicity may affect inclusion."
        ),
    }
    return selected, metadata


def _open_subset_index(
    task: str,
    split: str,
    data_root: str | Path,
    *,
    n_samples: int,
    seed: int,
) -> data.ValidationIndex:
    if task not in SUBSET_TASKS:
        raise data.ValidationDataError(
            f"selected-row indexing supports only {SUBSET_TASKS}"
        )
    if split != "validation":
        raise data.ValidationDataError(
            f"{task} selected-row indexing requires --split validation"
        )
    _validate_sampling_parameters(n_samples, seed)
    root = data._require_data_root(data_root)

    if task == "mm_imdb":
        csv_path = data._safe_data_file(root, "mmimdb_trainval.csv")
        image_store_path = data._safe_data_file(root, "images.h5")
        from datasets.dataset_mm_imdb import MM_IMDB_Dataset

        dataset = MM_IMDB_Dataset(
            csv_path=str(csv_path),
            img_path=str(image_store_path),
            tokenizer=None,
            image_processor=None,
            is_train=False,
            is_test=False,
            train_test_ratio=VALIDATION_RATIO,
            transform=None,
        )
        rows = dataset.csv_data
        image_store = _file_metadata(
            image_store_path,
            content_hash_scope=(
                "no whole-file hash; selected image arrays are hashed with "
                "shape and dtype and appear in the manifest samples"
            ),
        )
    else:
        csv_path = data._safe_data_file(root, "upmcfood_trainval.csv")
        try:
            image_root = (root / "images").resolve(strict=True)
        except FileNotFoundError as exc:
            raise data.ValidationDataError(
                "required UPMC image directory is missing: images"
            ) from exc
        if not image_root.is_dir() or not image_root.is_relative_to(root):
            raise data.ValidationDataError(
                "UPMC image directory must stay below --data-root"
            )
        from datasets.dataset_upmc import UPMC_Dataset

        dataset = UPMC_Dataset(
            csv_path=str(csv_path),
            img_path=str(image_root),
            tokenizer=None,
            image_processor=None,
            is_train=False,
            is_test=False,
            train_test_ratio=VALIDATION_RATIO,
            transform=None,
        )
        rows = dataset.data
        image_store = _file_metadata(
            image_root,
            content_hash_scope=(
                "no recursive image inventory; only candidate files examined "
                "before N unique images were selected are hashed"
            ),
        )

    try:
        logical_csv_rows = _count_csv_rows(csv_path)
        if logical_csv_rows < 2:
            raise data.ValidationDataError(
                f"metadata CSV has no data rows: {csv_path}"
            )
        source_data_rows = logical_csv_rows - 1
        validation_start = int(VALIDATION_RATIO * source_data_rows)
        expected_validation_rows = source_data_rows - validation_start
        if len(rows) != expected_validation_rows:
            raise data.ValidationDataError(
                "dataset validation split row count differs from the "
                f"documented 80/20 split ({len(rows)} != {expected_validation_rows})"
            )

        if task == "mm_imdb":
            image_hash_cache: dict[str, str] = {}
            image_reads = 0

            def row_fields(position: int) -> tuple[str, str, Any]:
                row = rows.iloc[position]
                return str(row["img_index"]), str(row["caption"]), row["genre"]

            def selected_image_hash(position: int, source_key: str) -> str:
                nonlocal image_reads
                if source_key not in image_hash_cache:
                    row = rows.iloc[position]
                    try:
                        image = dataset.img_data["images"][row["img_index"]]
                        image_hash_cache[source_key] = data._hash_array(image)
                    except (KeyError, IndexError, OSError, TypeError, ValueError) as exc:
                        raise data.ValidationDataError(
                            "could not read selected MM-IMDB image at validation "
                            f"row {position}, source key {source_key!r}: {exc}"
                        ) from exc
                    image_reads += 1
                return image_hash_cache[source_key]

        else:
            image_hash_cache = {}
            image_reads = 0

            def row_fields(position: int) -> tuple[str, str, Any]:
                row = rows.iloc[position]
                image_relpath, label, _class_name, text = row.tolist()
                return str(image_relpath), str(text), label

            def selected_image_hash(position: int, source_key: str) -> str:
                nonlocal image_reads
                if source_key not in image_hash_cache:
                    image_path = data._safe_data_file(image_root, source_key)
                    image_hash_cache[source_key] = data.sha256_file(image_path)
                    image_reads += 1
                return image_hash_cache[source_key]

        selected_records, candidate_metadata = _select_records(
            task=task,
            row_count=len(rows),
            n_samples=n_samples,
            seed=seed,
            row_fields=row_fields,
            image_hash=selected_image_hash,
        )

        source_files = _source_hashes(csv_path, selected_records)
        source_csv_metadata = _file_metadata(
            csv_path,
            content_hash_scope="full CSV content SHA256 is recorded in source_data_sha256",
            logical_rows_including_header=logical_csv_rows,
            data_rows=source_data_rows,
            validation_rows=len(rows),
            validation_start_after_task_shuffle=validation_start,
        )
        candidate_metadata.update(
            {
                "repository_split_shuffle_seed": TASK_SPLIT_SHUFFLE_SEEDS[task],
                "repository_train_validation_ratio": VALIDATION_RATIO,
                "source_csv_data_rows": source_data_rows,
                "source_csv_logical_rows_including_header": logical_csv_rows,
                "validation_start_after_task_shuffle": validation_start,
                "image_hash_reads": image_reads,
                "selected_pair_fingerprint_sha256": source_files[
                    "selected_validation_pairs"
                ],
            }
        )
        index = data.ValidationIndex(
            task=task,
            split=split,
            data_root=root,
            source_files=source_files,
            records=selected_records,
            dataset=dataset,
        )
        index.subset_protocol_metadata = {
            "version": SUBSET_INDEX_PROTOCOL_VERSION,
            "candidate_row_selection": candidate_metadata,
            "source_file_metadata": {
                "metadata_csv": source_csv_metadata,
                "image_source": image_store,
            },
        }
        return index
    except BaseException:
        image_file = getattr(dataset, "img_data", None)
        if image_file is not None:
            try:
                image_file.close()
            except Exception:
                pass
        raise


def _annotate_subset_manifest(
    manifest: dict[str, Any], index: data.ValidationIndex
) -> dict[str, Any]:
    metadata = getattr(index, "subset_protocol_metadata", None)
    if metadata is None:
        return manifest

    candidate_selection = metadata["candidate_row_selection"]
    positions_by_id = {record.sample_id: record.position for record in index.records}
    candidate_selection["ordered_sample_validation_positions"] = [
        positions_by_id[sample_id] for sample_id in manifest["ordered_sample_ids"]
    ]
    manifest["subset_index_protocol"] = metadata["version"]
    manifest["selection"]["candidate_row_sampling"] = candidate_selection
    manifest["source_file_metadata"] = metadata["source_file_metadata"]
    manifest["exchangeability_note"] = (
        "Candidate validation rows were visited in a deterministic random "
        "permutation and exact duplicate image content was skipped until the "
        "requested number of unique images was reached. Only the scanned prefix "
        "was image-hashed; the remainder of the validation split was not "
        "inventoried. This procedure samples rows with duplicate replenishment, "
        "not uniformly from all distinct image groups, and source or caption "
        "dependence may remain. Caption groups are recorded but are not used as "
        "permutation blocks."
    )
    return manifest


def _row_seed_from_manifest(manifest: dict[str, Any], requested_seed: int) -> int:
    selection = manifest.get("selection")
    row_sampling = (
        selection.get("candidate_row_sampling") if isinstance(selection, dict) else None
    )
    if manifest.get("subset_index_protocol") != SUBSET_INDEX_PROTOCOL_VERSION:
        raise data.ValidationDataError(
            "MM-IMDB/UPMC evaluation requires a manifest created by "
            "autoresearch_subset_index.py"
        )
    if not isinstance(row_sampling, dict) or row_sampling.get("version") != SUBSET_INDEX_PROTOCOL_VERSION:
        raise data.ValidationDataError("manifest is missing its selected-row protocol")
    row_seed = row_sampling.get("seed")
    if isinstance(row_seed, bool) or not isinstance(row_seed, int) or row_seed < 0:
        raise data.ValidationDataError("manifest has an invalid candidate-row seed")
    if row_seed != requested_seed:
        raise data.ValidationDataError(
            "--seed must match the candidate-row seed frozen in the manifest"
        )
    return row_seed


def main(argv: list[str] | None = None) -> int:
    args = autoresearch._make_parser().parse_args(argv)
    if args.task not in SUBSET_TASKS:
        return autoresearch.main(argv)

    if args.command == "manifest":
        candidate_seed = args.seed
    else:
        try:
            manifest = data.read_manifest(args.manifest)
            candidate_seed = _row_seed_from_manifest(manifest, args.seed)
        except data.ValidationDataError as exc:
            print(f"autoresearch: error: {exc}", file=sys.stderr)
            return 2

    original_open = data.open_validation_index
    original_create = data.create_ordered_manifest

    def open_validation_index(
        task: str, split: str, data_root: str | Path
    ) -> data.ValidationIndex:
        if task in SUBSET_TASKS:
            return _open_subset_index(
                task,
                split,
                data_root,
                n_samples=args.n,
                seed=candidate_seed,
            )
        return original_open(task, split, data_root)

    def create_ordered_manifest(
        index: data.ValidationIndex, *, n_samples: int, seed: int
    ) -> dict[str, Any]:
        manifest = original_create(index, n_samples=n_samples, seed=seed)
        return _annotate_subset_manifest(manifest, index)

    data.open_validation_index = open_validation_index
    data.create_ordered_manifest = create_ordered_manifest
    try:
        return autoresearch.main(argv)
    finally:
        data.open_validation_index = original_open
        data.create_ordered_manifest = original_create


if __name__ == "__main__":
    raise SystemExit(main())

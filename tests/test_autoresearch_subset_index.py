"""Tests for deterministic, selected-row-only validation indexing."""

from __future__ import annotations

import csv
import os
import random
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analyses import autoresearch_data as data
import autoresearch_subset_index as subset_index


def _row_loader(rows, reads):
    def load(position):
        reads.append(position)
        row = rows[position]
        if isinstance(row, Exception):
            raise row
        return row

    return load


def test_random_row_candidates_skip_duplicate_images_and_replenish():
    seed = 17
    candidate_order = subset_index._candidate_positions(7, seed)
    image_keys = {
        candidate_order[0]: "image-a",
        candidate_order[1]: "image-a",
        candidate_order[2]: "image-b",
        candidate_order[3]: "image-c",
        candidate_order[4]: "image-d",
        candidate_order[5]: "image-e",
        candidate_order[6]: "image-f",
    }
    rows = [
        (f"source-{position}", f"caption-{position}", position % 2)
        for position in range(7)
    ]

    def run_once():
        image_reads = []
        selected, metadata = subset_index._select_records(
            task="mm_imdb",
            row_count=len(rows),
            n_samples=3,
            seed=seed,
            row_fields=_row_loader(rows, []),
            image_hash=lambda position, _source: image_reads.append(position)
            or image_keys[position],
        )
        return selected, metadata, image_reads

    first, metadata, image_reads = run_once()
    second, second_metadata, second_reads = run_once()
    assert [record.position for record in first] == [
        candidate_order[0], candidate_order[2], candidate_order[3]
    ]
    assert metadata["candidate_rows_scanned"] == 4
    assert metadata["duplicate_image_candidate_rows_skipped"] == 1
    assert metadata["full_validation_image_inventory_hashed"] is False
    assert image_reads == candidate_order[:4]
    assert [record.sample_id for record in first] == [record.sample_id for record in second]
    assert metadata == second_metadata
    assert image_reads == second_reads


def test_missing_encountered_candidate_fails_instead_of_being_dropped():
    seed = 9
    candidate_order = subset_index._candidate_positions(5, seed)
    missing_position = candidate_order[1]
    rows = [
        (f"source-{position}", f"caption-{position}", 0)
        for position in range(5)
    ]
    reads = []

    def load_image(position, _source):
        reads.append(position)
        if position == missing_position:
            raise data.ValidationDataError("missing image")
        return f"image-{position}"

    with pytest.raises(data.ValidationDataError, match="missing image"):
        subset_index._select_records(
            task="upmc_food",
            row_count=5,
            n_samples=4,
            seed=seed,
            row_fields=_row_loader(rows, []),
            image_hash=load_image,
        )
    assert reads == candidate_order[:2]


def test_selected_row_source_hashes_change_with_csv_wrapper_or_selected_pair(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    wrapper_path = tmp_path / "wrapper.py"
    data_code_path = tmp_path / "data.py"
    csv_path.write_text("id,text\n1,one\n", encoding="utf-8")
    wrapper_path.write_text("version = 1\n", encoding="utf-8")
    data_code_path.write_text("version = 1\n", encoding="utf-8")
    record = data._make_record(
        sample_id="mm_imdb:1:abc:row:0",
        source_key="1",
        image_hash="image-a",
        text="caption",
        label="label",
        position=0,
        payload={"dataset_position": 0},
    )

    first = subset_index._source_hashes(
        csv_path,
        [record],
        wrapper_path=wrapper_path,
        data_code_path=data_code_path,
    )
    csv_path.write_text("id,text\n1,changed\n", encoding="utf-8")
    changed_csv = subset_index._source_hashes(
        csv_path,
        [record],
        wrapper_path=wrapper_path,
        data_code_path=data_code_path,
    )
    assert first["metadata.csv"] != changed_csv["metadata.csv"]

    wrapper_path.write_text("version = 2\n", encoding="utf-8")
    changed_wrapper = subset_index._source_hashes(
        csv_path,
        [record],
        wrapper_path=wrapper_path,
        data_code_path=data_code_path,
    )
    assert changed_csv["src/autoresearch_subset_index.py"] != changed_wrapper[
        "src/autoresearch_subset_index.py"
    ]

    changed_record = data._make_record(
        sample_id=record.sample_id,
        source_key=record.source_key,
        image_hash="image-b",
        text="caption",
        label="label",
        position=record.position,
        payload=record.payload,
    )
    changed_pair = subset_index._source_hashes(
        csv_path,
        [changed_record],
        wrapper_path=wrapper_path,
        data_code_path=data_code_path,
    )
    assert changed_wrapper["selected_validation_pairs"] != changed_pair[
        "selected_validation_pairs"
    ]


def test_mm_index_reads_only_selected_arrays_and_does_not_hash_h5_whole_file(
    tmp_path, monkeypatch
):
    data_root = tmp_path / "mm"
    data_root.mkdir()
    csv_path = data_root / "mmimdb_trainval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["img_index", "genre", "caption"])
        for position in range(20):
            writer.writerow([position, "[Drama]", f"caption {position}"])
    h5_path = data_root / "images.h5"
    h5_path.write_bytes(b"placeholder; fake dataset constructor does not open HDF5")

    reads = []

    class FakeImages:
        def __getitem__(self, image_index):
            reads.append(int(image_index))
            return np.asarray([[[int(image_index) % 255, 2, 3]]], dtype=np.uint8)

    class FakeH5:
        def __init__(self):
            self.images = FakeImages()
            self.closed = False

        def __getitem__(self, key):
            assert key == "images"
            return self.images

        def close(self):
            self.closed = True

    store = FakeH5()

    class FakeMMImdbDataset:
        def __init__(
            self,
            *,
            csv_path,
            img_path,
            tokenizer,
            image_processor,
            is_train,
            is_test,
            train_test_ratio,
            transform,
        ):
            assert img_path == str(h5_path)
            assert tokenizer is None and image_processor is None and transform is None
            assert not is_train and not is_test and train_test_ratio == 0.8
            rows = pd.read_csv(csv_path).sample(frac=1, random_state=42).reset_index(
                drop=True
            )
            self.csv_data = rows[int(0.8 * len(rows)) :]
            self.img_data = store

    package = types.ModuleType("datasets")
    package.__path__ = []
    dataset_module = types.ModuleType("datasets.dataset_mm_imdb")
    dataset_module.MM_IMDB_Dataset = FakeMMImdbDataset
    monkeypatch.setitem(sys.modules, "datasets", package)
    monkeypatch.setitem(sys.modules, "datasets.dataset_mm_imdb", dataset_module)

    index = subset_index._open_subset_index(
        "mm_imdb", "validation", data_root, n_samples=2, seed=42
    )
    try:
        metadata = index.subset_protocol_metadata
        assert len(index.records) == 2
        assert metadata["candidate_row_selection"]["validation_row_count"] == 4
        assert metadata["candidate_row_selection"]["image_hash_reads"] == len(reads)
        assert len(reads) == metadata["candidate_row_selection"]["candidate_rows_scanned"]
        assert len(reads) < metadata["candidate_row_selection"]["validation_row_count"]
        assert "images.h5" not in index.source_files
        assert index.source_files["selected_validation_pairs"]
        assert "no whole-file hash" in metadata["source_file_metadata"][
            "image_source"
        ]["content_hash_scope"]
        assert metadata["candidate_row_selection"][
            "full_validation_image_inventory_hashed"
        ] is False
    finally:
        data.close_validation_index(index)
    assert store.closed


def test_upmc_index_hashes_only_selected_contained_image_files(tmp_path, monkeypatch):
    data_root = tmp_path / "upmc"
    image_root = data_root / "images"
    image_root.mkdir(parents=True)
    csv_path = data_root / "upmcfood_trainval.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as target:
        writer = csv.writer(target)
        writer.writerow(["img_path", "label", "class", "text"])
        for position in range(20):
            relative = f"class-{position % 4}/{position}.bin"
            writer.writerow([relative, position % 4, f"class-{position % 4}", f"caption {position}"])
            image = image_root / relative
            image.parent.mkdir(parents=True, exist_ok=True)
            image.write_bytes(f"image bytes {position}".encode())

    class FakeUPMCDataset:
        def __init__(
            self,
            *,
            csv_path,
            img_path,
            tokenizer,
            image_processor,
            is_train,
            is_test,
            train_test_ratio,
            transform,
        ):
            assert img_path == str(image_root.resolve())
            assert tokenizer is None and image_processor is None and transform is None
            assert not is_train and not is_test and train_test_ratio == 0.8
            rows = pd.read_csv(csv_path).sample(frac=1, random_state=46).reset_index(
                drop=True
            )
            self.data = rows[int(0.8 * len(rows)) :]

    package = types.ModuleType("datasets")
    package.__path__ = []
    dataset_module = types.ModuleType("datasets.dataset_upmc")
    dataset_module.UPMC_Dataset = FakeUPMCDataset
    monkeypatch.setitem(sys.modules, "datasets", package)
    monkeypatch.setitem(sys.modules, "datasets.dataset_upmc", dataset_module)

    index = subset_index._open_subset_index(
        "upmc_food", "validation", data_root, n_samples=2, seed=42
    )
    try:
        metadata = index.subset_protocol_metadata
        candidate = metadata["candidate_row_selection"]
        assert len(index.records) == 2
        assert candidate["validation_row_count"] == 4
        assert candidate["candidate_rows_scanned"] == 2
        assert candidate["image_hash_reads"] == 2
        assert candidate["full_validation_image_inventory_hashed"] is False
        assert all(
            (image_root / record.source_key).is_file() for record in index.records
        )
        assert metadata["source_file_metadata"]["image_source"][
            "content_hash_scope"
        ].startswith("no recursive image inventory")
    finally:
        data.close_validation_index(index)


def test_selected_row_index_rejects_wrong_split_and_out_of_bounds_n(tmp_path):
    with pytest.raises(data.ValidationDataError, match="requires --split validation"):
        subset_index._open_subset_index(
            "mm_imdb", "test", tmp_path / "does-not-need-to-exist", n_samples=2, seed=1
        )
    with pytest.raises(data.ValidationDataError, match="between 2"):
        subset_index._validate_sampling_parameters(1, 1)
    with pytest.raises(data.ValidationDataError, match="non-negative"):
        subset_index._validate_sampling_parameters(2, -1)


def test_wrapper_import_restores_external_cwd_and_keeps_side_effects_owned(tmp_path):
    caller = tmp_path / "caller"
    caller.mkdir()
    owned_output = tmp_path / "owned" / "autoresearch"
    owned_output.mkdir(parents=True)
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    environment["AUTORESEARCH_TEST_OUTPUT"] = str(owned_output)
    code = (
        "import os; from pathlib import Path; import autoresearch; "
        "autoresearch.authorized_output_root = "
        "lambda: Path(os.environ['AUTORESEARCH_TEST_OUTPUT']); "
        "before = Path.cwd(); import autoresearch_subset_index; "
        "assert Path.cwd() == before"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=caller,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert list(caller.iterdir()) == []
    assert all(
        path.resolve().is_relative_to(owned_output.resolve())
        for path in owned_output.rglob("*")
    )

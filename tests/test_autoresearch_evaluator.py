"""Boundary tests for the checkpoint-only autoresearch validation path."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from analyses.autoresearch_data import (
    ValidationDataError,
    create_ordered_manifest,
    open_validation_index,
    validate_manifest,
    validate_manifest_against_index,
    write_manifest_once,
)
import autoresearch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC = PROJECT_ROOT / "src"


def _make_hateful_memes_root(root: Path, *, duplicate_image: bool = True) -> Path:
    root.mkdir(parents=True)
    (root / "images").mkdir()
    rows = []
    for index in range(6):
        relative = f"images/{index}.bin"
        content = b"identical-image" if duplicate_image and index == 5 else f"image-{index}".encode()
        if duplicate_image and index == 0:
            content = b"identical-image"
        (root / relative).write_bytes(content)
        rows.append(
            {
                "id": f"hm-{index}",
                "img": relative,
                "text": "same caption" if index in (0, 1) else f"caption {index}",
                "label": index % 2,
            }
        )
    (root / "dev_seen.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
    )
    return root


def test_hm_manifest_is_deterministic_paired_and_records_duplicate_groups(tmp_path):
    data_root = _make_hateful_memes_root(tmp_path / "hm")
    index = open_validation_index("hateful_memes", "dev_seen", data_root)
    try:
        first = create_ordered_manifest(index, n_samples=5, seed=29)
        second = create_ordered_manifest(index, n_samples=5, seed=29)
        assert first == second
        assert first["validation_role"] == "validation_discovery_seen_historical"
        assert len(first["ordered_sample_ids"]) == 5
        assert len(set(first["ordered_sample_ids"])) == 5
        assert len({sample["group_id"] for sample in first["samples"]}) == 5
        assert first["selection"]["duplicate_image_records_removed"] == 1
        assert first["selection"]["unique_caption_groups_selected"] == 4
        assert first["selection"]["repeated_caption_groups_selected"] == 1
        assert first["selection"]["duplicate_caption_rows_selected"] == 1
        validate_manifest_against_index(first, index)
    finally:
        from analyses.autoresearch_data import close_validation_index

        close_validation_index(index)


def test_hm_reader_rejects_missing_requested_split_without_fallback(tmp_path):
    data_root = tmp_path / "hm"
    data_root.mkdir()
    (data_root / "validation.jsonl").write_text("{}\n", encoding="utf-8")
    (data_root / "test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValidationDataError, match="missing"):
        open_validation_index("hateful_memes", "dev_seen", data_root)
    with pytest.raises(ValidationDataError, match="dev_seen"):
        open_validation_index("hateful_memes", "validation", data_root)


def test_manifest_detects_changed_validation_pairing(tmp_path):
    data_root = _make_hateful_memes_root(tmp_path / "hm", duplicate_image=False)
    index = open_validation_index("hateful_memes", "dev_seen", data_root)
    try:
        manifest = create_ordered_manifest(index, n_samples=4, seed=13)
    finally:
        from analyses.autoresearch_data import close_validation_index

        close_validation_index(index)
    image_path = data_root / manifest["samples"][0]["source_key"]
    image_path.write_bytes(b"changed-image-content")
    changed_index = open_validation_index("hateful_memes", "dev_seen", data_root)
    try:
        with pytest.raises(ValidationDataError, match="pairing changed|ordered sample IDs"):
            validate_manifest_against_index(manifest, changed_index)
    finally:
        from analyses.autoresearch_data import close_validation_index

        close_validation_index(changed_index)


def test_manifest_write_is_exclusive_and_confined_to_output_root(tmp_path):
    data_root = _make_hateful_memes_root(tmp_path / "hm", duplicate_image=False)
    output_root = tmp_path / "project" / "autoresearch"
    output_root.mkdir(parents=True)
    index = open_validation_index("hateful_memes", "dev_seen", data_root)
    try:
        manifest = create_ordered_manifest(index, n_samples=3, seed=4)
    finally:
        from analyses.autoresearch_data import close_validation_index

        close_validation_index(index)

    target = output_root / "manifests" / "hm.json"
    written = write_manifest_once(
        manifest,
        target,
        authorized_output_root=output_root,
        data_root=data_root,
    )
    assert written == target
    with pytest.raises(ValidationDataError, match="will not be overwritten"):
        write_manifest_once(
            manifest,
            target,
            authorized_output_root=output_root,
            data_root=data_root,
        )
    with pytest.raises(ValidationDataError, match="inside the authorized output root"):
        write_manifest_once(
            manifest,
            data_root / "should-not-write.json",
            authorized_output_root=output_root,
            data_root=data_root,
        )


def test_legacy_log_directory_symlink_is_rejected(tmp_path):
    data_root = tmp_path / "data"
    data_root.mkdir()
    output_root = tmp_path / "project" / "autoresearch"
    run_directory = output_root / "run"
    run_directory.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (run_directory / "logs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(autoresearch.EvaluatorError, match="symlink|below"):
        autoresearch._prepare_legacy_log_directory(
            run_directory, authorized_root=output_root, data_root=data_root
        )


def test_import_and_cli_help_do_not_load_model_or_training_test_paths():
    environment = dict(os.environ)
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = str(SRC)
    help_result = subprocess.run(
        [sys.executable, str(SRC / "autoresearch.py"), "--help"],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "saved ViLBERT checkpoints" in help_result.stdout

    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, autoresearch; "
                "forbidden = {'torch', 'transformers', 'timm', 'vilbert', 'metrics', "
                "'main', 'experiment_tracker', 'performance_metric_collection', "
                "'probing', 'datasets.get_datasets'}; "
                "assert not forbidden.intersection(sys.modules), "
                "forbidden.intersection(sys.modules)"
            ),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert import_result.returncode == 0, import_result.stderr


def test_raw_cka_and_rank_diagnostics_define_degenerate_cases():
    x = np.asarray([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    cka, status = autoresearch._linear_cka(x, x.copy())
    assert status == "ok"
    assert cka == pytest.approx(1.0)

    zero = autoresearch._rank_diagnostics(np.zeros((4, 3)))
    assert zero["status"] == "zero_variance"
    assert zero["numerical_rank"] == 0
    assert zero["participation_ratio_rank"] is None

    rank_one = autoresearch._rank_diagnostics(np.column_stack((x[:, 0], x[:, 0], x[:, 0])))
    assert rank_one["status"] == "rank_deficient"
    assert rank_one["numerical_rank"] == 1
    assert rank_one["participation_ratio_rank"] == pytest.approx(1.0)
    assert "covariance eigenvalues" in rank_one["definition"]


def test_cached_cka_gram_metric_matches_direct_observed_value():
    rng = np.random.default_rng(17)
    text = rng.normal(size=(12, 5))
    vision = rng.normal(size=(12, 7))
    direct, status = autoresearch._linear_cka(text, vision)
    assert status == "ok"
    cka_x, indexed_vision, metric = autoresearch._cached_cka_metric(text, vision)
    assert metric(cka_x, indexed_vision) == pytest.approx(direct, abs=1e-12)


def test_validate_manifest_rejects_repeated_image_groups(tmp_path):
    data_root = _make_hateful_memes_root(tmp_path / "hm")
    index = open_validation_index("hateful_memes", "dev_seen", data_root)
    try:
        manifest = create_ordered_manifest(index, n_samples=5, seed=29)
    finally:
        from analyses.autoresearch_data import close_validation_index

        close_validation_index(index)
    manifest["samples"][1]["group_id"] = manifest["samples"][0]["group_id"]
    with pytest.raises(ValidationDataError, match="duplicate image groups"):
        validate_manifest(manifest)

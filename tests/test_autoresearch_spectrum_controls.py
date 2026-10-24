"""Synthetic controls for cached-feature spectrum and paired metrics."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/analyses/autoresearch_spectrum_controls.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_spectrum_controls", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def summarize(matrix, *, left_vectors=False):
    centered, _ = MODULE._center_rows(matrix, normalize=False)
    return MODULE._spectrum_summary(centered, left_vectors=left_vectors)


def test_spectrum_ranks_are_invariant_to_orthogonal_rotation_and_scale():
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(48, 7))
    q, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    first, _, singular_first = summarize(x)
    second, _, singular_second = summarize(13.0 * (x @ q))

    for field in (
        "participation_ratio_rank",
        "effective_entropy_rank",
        "stable_rank",
    ):
        assert first[field] == pytest.approx(second[field], rel=1e-10, abs=1e-10)
    assert first["pca95_components"] == second["pca95_components"]
    assert first["pca99_components"] == second["pca99_components"]
    assert first["numerical_rank_float32"] == second["numerical_rank_float32"]
    assert first["numerical_rank_float64"] == second["numerical_rank_float64"]
    np.testing.assert_allclose(
        singular_first / singular_first[0],
        singular_second / singular_second[0],
        rtol=1e-10,
        atol=1e-10,
    )


def test_rank_one_and_constant_features_have_explicit_edge_results():
    left = np.asarray([0.1, 0.3, 0.7, 1.1, 1.5, 2.0, 2.2, 2.8], dtype=np.float32)
    direction = np.asarray([0.2, -0.4, 0.7, 1.3], dtype=np.float32)
    rank_one = np.outer(left, direction).astype(np.float32)
    summary, _, _ = summarize(rank_one)
    assert summary["status"] == "ok"
    assert summary["participation_ratio_rank"] == pytest.approx(1.0, abs=1e-6)
    assert summary["effective_entropy_rank"] == pytest.approx(1.0, abs=1e-6)
    assert summary["stable_rank"] == pytest.approx(1.0, abs=1e-6)
    assert summary["pca95_components"] == 1
    assert summary["pca99_components"] == 1
    assert summary["numerical_rank_float32"] == 1
    assert summary["numerical_rank_float32_relative_threshold"] > (
        summary["numerical_rank_float64_relative_threshold"]
    )

    constant, _, singular = summarize(np.full((8, 4), 3.0, dtype=np.float32))
    assert constant["status"] == "zero_variance"
    assert constant["participation_ratio_rank"] is None
    assert constant["effective_entropy_rank"] is None
    assert constant["stable_rank"] is None
    assert constant["pca95_components"] == constant["pca99_components"] == 0
    assert constant["numerical_rank_float32"] == 0
    np.testing.assert_array_equal(singular, np.zeros(4))


def test_equal_variance_axes_have_two_effective_components():
    equal_axes = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float32,
    )
    summary, _, _ = summarize(equal_axes)
    assert summary["participation_ratio_rank"] == pytest.approx(2.0)
    assert summary["effective_entropy_rank"] == pytest.approx(2.0)
    assert summary["stable_rank"] == pytest.approx(2.0)
    assert summary["pca95_components"] == summary["pca99_components"] == 2


def test_exact_svcca95_and_centered_procrustes_similarity_are_bounded_repeatable():
    rng = np.random.default_rng(7)
    x = rng.normal(size=(40, 6)).astype(np.float32)
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    y = (x @ q).astype(np.float32)

    xc, _ = MODULE._center_rows(x, normalize=False)
    yc, _ = MODULE._center_rows(y, normalize=False)
    sx, ux, singular_x = MODULE._spectrum_summary(xc, left_vectors=True)
    sy, uy, singular_y = MODULE._spectrum_summary(yc, left_vectors=True)
    first = MODULE._svcca95(ux, sx, uy, sy)
    second = MODULE._svcca95(ux, sx, uy, sy)
    procrustes = MODULE._procrustes_similarity(xc, yc)

    assert first == second
    assert first["status"] == "ok"
    assert 0.0 <= first["value"] <= 1.0
    assert all(0.0 <= value <= 1.0 for value in first["canonical_correlations"])
    assert first["value"] == pytest.approx(1.0, abs=1e-6)
    assert procrustes["status"] == "ok"
    assert procrustes["value"] == pytest.approx(1.0, abs=1e-6)


def test_orthogonal_paired_features_have_zero_exact_svcca_and_procrustes():
    x = np.asarray([1.0, -1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)[:, None]
    y = np.asarray([0.0, 0.0, 1.0, -1.0, 0.0, 0.0], dtype=np.float32)[:, None]
    xc, _ = MODULE._center_rows(x, normalize=False)
    yc, _ = MODULE._center_rows(y, normalize=False)
    sx, ux, singular_x = MODULE._spectrum_summary(xc, left_vectors=True)
    sy, uy, singular_y = MODULE._spectrum_summary(yc, left_vectors=True)

    svcca = MODULE._svcca95(ux, sx, uy, sy)
    procrustes = MODULE._procrustes_similarity(xc, yc)
    assert svcca["value"] == pytest.approx(0.0, abs=1e-12)
    assert procrustes["value"] == pytest.approx(0.0, abs=1e-12)


def test_cache_report_hashes_pairing_and_output_is_create_only(tmp_path):
    rng = np.random.default_rng(12)
    text = rng.normal(size=(20, 3, 5)).astype(np.float32)
    vision = text.copy()
    ids = np.asarray([f"sample-{index:03d}" for index in range(20)])
    cache = tmp_path / "features.npz"
    np.savez_compressed(
        cache,
        text_features=text,
        vision_features=vision,
        sample_ids=ids,
        layer_ids=np.asarray([0, 2, 5]),
        metadata=np.asarray(json.dumps({"split": "dev_seen"})),
    )

    report = MODULE.analyze_cache(cache, normalization="both")
    assert report["input"]["feature_shape"] == [20, 3, 5]
    assert report["input"]["sample_id_count"] == 20
    assert len(report["input"]["sample_ids_sha256"]) == 64
    assert len(report["input"]["cache_sha256"]) == 64
    assert len(report["input"]["metadata_sha256"]) == 64
    assert report["input"]["layer_ids"] == [0, 2, 5]
    assert set(report["results"]) == {"raw", "l2_normalized"}
    assert len(report["results"]["raw"]["text"]) == 3
    assert report["results"]["raw"]["paired_final_layer_metrics"]["svcca95"]["value"] == pytest.approx(1.0)
    serialized = json.dumps(report, allow_nan=False)
    assert "sample-000" not in serialized

    output = tmp_path / "spectrum.json"
    MODULE._write_immutable_json(output, report)
    with pytest.raises(FileExistsError):
        MODULE._write_immutable_json(output, report)


def test_cache_rejects_mismatched_pair_ids_and_memory_over_limit(tmp_path):
    x = np.ones((6, 2, 3), dtype=np.float32)
    cache = tmp_path / "mismatch.npz"
    np.savez(
        cache,
        text_features=x,
        vision_features=x.copy(),
        text_ids=np.asarray([f"id-{i}" for i in range(6)]),
        vision_ids=np.asarray([f"other-{i}" for i in range(6)]),
    )
    with pytest.raises(ValueError, match="text_ids and vision_ids must match"):
        MODULE.analyze_cache(cache)

    valid_cache = tmp_path / "valid.npz"
    np.savez(
        valid_cache,
        text_features=x,
        vision_features=x.copy(),
        sample_ids=np.asarray([f"id-{i}" for i in range(6)]),
    )
    with pytest.raises(ValueError, match="estimated peak memory"):
        MODULE.analyze_cache(valid_cache, max_memory_bytes=1)


def test_constant_cache_reports_undefined_paired_metrics_and_valid_json(tmp_path):
    features = np.ones((5, 2, 3), dtype=np.float32)
    cache = tmp_path / "constant.npz"
    np.savez(
        cache,
        text_features=features,
        vision_features=features,
        sample_ids=np.asarray([f"id-{index}" for index in range(5)]),
        layer_ids=np.asarray([b"early", b"final"]),
    )
    report = MODULE.analyze_cache(cache, normalization="raw")
    paired = report["results"]["raw"]["paired_final_layer_metrics"]
    assert paired["svcca95"]["status"].startswith("zero_variance")
    assert paired["svcca95"]["value"] is None
    assert paired["centered_normalized_orthogonal_procrustes_similarity"]["status"] == "zero_variance"
    assert paired["centered_normalized_orthogonal_procrustes_similarity"]["value"] is None
    assert report["input"]["layer_ids"] == ["early", "final"]
    json.dumps(report, allow_nan=False)


def test_cache_rejects_nonfinite_sample_ids(tmp_path):
    features = np.ones((3, 1, 2), dtype=np.float32)
    cache = tmp_path / "bad-ids.npz"
    np.savez(
        cache,
        text_features=features,
        vision_features=features,
        sample_ids=np.asarray([1.0, np.nan, 3.0]),
    )
    with pytest.raises(ValueError, match="non-finite numeric IDs"):
        MODULE.analyze_cache(cache)

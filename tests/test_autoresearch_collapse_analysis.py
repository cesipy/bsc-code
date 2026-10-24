"""Synthetic controls for the layer-wise dimensionality-collapse analysis."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load("autoresearch_collapse_analysis", "src/analyses/autoresearch_collapse_analysis.py")
REFERENCE = _load("autoresearch_spectrum_controls_ref", "src/analyses/autoresearch_spectrum_controls.py")


def test_rank_one_matrix_has_unit_rank_summaries():
    rng = np.random.default_rng(0)
    x = np.outer(rng.normal(size=40), rng.normal(size=12))
    out = MODULE.spectrum_metrics(MODULE.center(x))
    assert out["participation_ratio"] == pytest.approx(1.0, abs=1e-9)
    assert out["stable_rank"] == pytest.approx(1.0, abs=1e-9)
    assert out["entropy_rank"] == pytest.approx(1.0, abs=1e-6)
    assert out["pca95"] == 1.0 and out["top1_pc_share"] == pytest.approx(1.0)


def test_isotropic_gaussian_is_close_to_full_rank():
    x = np.random.default_rng(1).normal(size=(4000, 40))
    out = MODULE.view_metrics(x, "raw")
    assert 36.0 < out["participation_ratio"] <= 40.0
    assert out["pca95"] >= 36.0


def test_raw_metrics_are_invariant_to_rotation_and_global_scale():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(60, 9)) * np.linspace(3.0, 0.2, 9)
    q, _ = np.linalg.qr(rng.normal(size=(9, 9)))
    a = MODULE.view_metrics(x, "raw")
    b = MODULE.view_metrics(7.0 * x @ q, "raw")
    for key in MODULE.METRICS:
        assert a[key] == pytest.approx(b[key], rel=1e-9, abs=1e-9)


def test_standardized_view_ignores_per_column_scale_but_raw_view_does_not():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(300, 20))
    scaled = x * np.concatenate([[200.0], np.ones(19)])
    raw_plain, raw_scaled = MODULE.view_metrics(x, "raw"), MODULE.view_metrics(scaled, "raw")
    std_plain, std_scaled = MODULE.view_metrics(x, "standardized"), MODULE.view_metrics(scaled, "standardized")
    assert raw_scaled["participation_ratio"] < 1.1 < 0.2 * raw_plain["participation_ratio"]
    assert raw_scaled["top_channel_share"] > 0.99
    for key in MODULE.METRICS:
        assert std_plain[key] == pytest.approx(std_scaled[key], rel=1e-9, abs=1e-9)


def test_constant_columns_are_dropped_from_standardized_view_and_reported():
    rng = np.random.default_rng(4)
    x = np.concatenate([rng.normal(size=(80, 5)), np.full((80, 3), 2.5)], axis=1)
    out = MODULE.view_metrics(x, "standardized")
    assert out["n_features_used"] == 5.0
    assert 3.0 < out["participation_ratio"] <= 5.0


def test_zero_variance_matrix_is_reported_as_nan_not_as_low_rank():
    out = MODULE.view_metrics(np.ones((30, 6)), "raw")
    assert all(np.isnan(out[key]) for key in MODULE.METRICS)


def test_matches_independent_reference_implementation():
    rng = np.random.default_rng(5)
    x = rng.normal(size=(120, 30)) * np.exp(-np.arange(30) / 6.0)
    mine = MODULE.view_metrics(x, "raw")
    centered, _ = REFERENCE._center_rows(x, normalize=False)
    ref, _, _ = REFERENCE._spectrum_summary(centered, left_vectors=False)
    assert mine["participation_ratio"] == pytest.approx(ref["participation_ratio_rank"], rel=1e-9)
    assert mine["entropy_rank"] == pytest.approx(ref["effective_entropy_rank"], rel=1e-9)
    assert mine["stable_rank"] == pytest.approx(ref["stable_rank"], rel=1e-9)
    assert mine["pca95"] == ref["pca95_components"]
    assert mine["pca99"] == ref["pca99_components"]


def test_subsets_are_deterministic_unique_and_sorted():
    first = MODULE.draw_subsets(100, (20, 50), reps=5, seed=11)
    second = MODULE.draw_subsets(100, (20, 50), reps=5, seed=11)
    other = MODULE.draw_subsets(100, (20, 50), reps=5, seed=12)
    for size in (20, 50):
        assert first[size].shape == (5, size)
        np.testing.assert_array_equal(first[size], second[size])
        assert not np.array_equal(first[size], other[size])
        assert all(len(set(row)) == size and list(row) == sorted(row) for row in first[size])
    with pytest.raises(ValueError):
        MODULE.draw_subsets(100, (100,), reps=1, seed=0)


@pytest.mark.parametrize(
    "text_ids,vision_ids,expected",
    [
        ([], [], "no-cross-attention"),
        ([3, 4, 5], [3, 4, 5], "early"),
        ([9, 10, 11], [9, 10, 11], "late"),
        ([3, 4, 5], [5, 6, 7], "delayed-vision"),
        ([3, 4, 5], [9, 10, 11], "delayed-vision"),
        ([9, 10, 11], [3, 4, 5], "reverse"),
        ([1, 2], [1, 2], "other"),
    ],
)
def test_family_labels(text_ids, vision_ids, expected):
    assert MODULE.family_of(text_ids, vision_ids) == expected


def _write_cache(path: Path, n=24, layers=4, dim=6, task="hateful_memes", text_ids=(1,), vision_ids=(2,)):
    rng = np.random.default_rng(6)
    metadata = {
        "task": task,
        "cache_identity": {"checkpoint_sha256": "ab" * 32},
        "checkpoint_architecture": {
            "saved_checkpoint_config": {
                "text_cross_attention_layers": list(text_ids),
                "vision_cross_attention_layers": list(vision_ids),
                "seed": 7,
            }
        },
    }
    np.savez(
        path,
        text_features=rng.normal(size=(n, layers, dim)).astype(np.float32),
        vision_features=rng.normal(size=(n, layers, dim)).astype(np.float32),
        layer_ids=np.arange(layers),
        cross_attention_layers=np.array(sorted(set(text_ids) | set(vision_ids))),
        sample_ids=np.array([f"s:{i}" for i in range(n)]),
        metadata=np.array(json.dumps(metadata)),
    )


def test_load_cache_reads_schedule_and_rejects_nonfinite(tmp_path):
    good = tmp_path / "good.npz"
    _write_cache(good)
    cache = MODULE.load_cache(good, "toy")
    assert cache["text_ids"] == [1] and cache["vision_ids"] == [2]
    assert cache["text"].dtype == np.float64 and cache["text"].shape == (24, 4, 6)
    bad = tmp_path / "bad.npz"
    with np.load(good) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays["vision_features"][0, 0, 0] = np.nan
    np.savez(bad, **arrays)
    with pytest.raises(ValueError, match="non-finite"):
        MODULE.load_cache(bad, "bad")


def test_step_table_flags_first_cross_attention_step(tmp_path):
    path = tmp_path / "toy.npz"
    _write_cache(path, vision_ids=(2, 3))
    cache = MODULE.load_cache(path, "toy")
    subsets = MODULE.draw_subsets(24, (12,), reps=2, seed=1)
    full, sub = MODULE.analyze_cache(cache, (12,), subsets)
    steps = MODULE.step_table(cache, full, ("participation_ratio",))
    vision_raw = [s for s in steps if s["stream"] == "vision" and s["view"] == "raw"]
    assert [s["layer"] for s in vision_raw if s["is_cross_attention_step"]] == [2, 3]
    assert [s["layer"] for s in vision_raw if s["is_first_cross_attention_step"]] == [2]
    assert len(vision_raw) == 3
    assert {r["n"] for r in sub} == {12} and all(r["reps"] == 2 for r in sub)


def test_run_writes_create_only_outputs_and_refuses_existing_directory(tmp_path):
    root = tmp_path / "trials"
    (root / "toy" / "features").mkdir(parents=True)
    _write_cache(root / "toy" / "features" / "features-x.npz", n=30)
    args = MODULE.build_parser().parse_args(
        ["--trials-root", str(root), "--trial", "toy", "--output-dir", str(tmp_path / "out"),
         "--sizes", "10", "15", "--reps", "3"]
    )
    summary = MODULE.run(args)
    out = tmp_path / "out"
    assert {p.name for p in out.iterdir()} == {
        "collapse_metrics_full.csv", "collapse_metrics_subsample.csv", "collapse_steps.csv", "collapse_summary.json",
    }
    assert summary["inputs"][0]["family"] == "other"
    json.loads((out / "collapse_summary.json").read_text())
    with pytest.raises(FileExistsError):
        MODULE.run(args)

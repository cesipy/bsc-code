"""Synthetic controls for the label-based retention probe."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "src/analyses/autoresearch_retention_probe.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_retention_probe", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def planted(n=300, d=40, strength=2.0, direction=0, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile([0, 1], n // 2)
    x = rng.normal(size=(n, d))
    x[:, direction] += strength * (2 * y - 1)
    return x, y


def test_planted_signal_is_recovered_and_noise_is_not():
    x, y = planted(strength=1.5)
    folds = MODULE.make_folds(y, 5, 1)
    assert MODULE.probe_auc(x, y, folds, None, 1)["auroc"] > 0.9
    noise = np.random.default_rng(2).normal(size=x.shape)
    assert abs(MODULE.probe_auc(noise, y, folds, None, 1)["auroc"] - 0.5) < 0.12


def test_top_k_keeps_signal_on_the_leading_direction_and_drops_it_elsewhere():
    rng = np.random.default_rng(3)
    n, d = 400, 30
    y = np.tile([0, 1], n // 2)
    sign = 2 * y - 1

    def block_features(shift_latent: float, shift_channel: float):
        latent = rng.normal(size=(n, 1)) + shift_latent * sign[:, None]
        x = rng.normal(size=(n, d))
        x[:, :10] = latent + 0.3 * rng.normal(size=(n, 10))  # ten correlated channels: the leading PC
        x[:, d - 1] += shift_channel * sign
        return x

    on_leading = block_features(1.5, 0.0)
    off_leading = block_features(0.0, 1.5)
    folds = MODULE.make_folds(y, 5, 4)
    assert MODULE.probe_auc(on_leading, y, folds, 2, 4)["auroc"] > 0.85
    assert MODULE.probe_auc(off_leading, y, folds, None, 4)["auroc"] > 0.85
    assert MODULE.probe_auc(off_leading, y, folds, 2, 4)["auroc"] < 0.65


def test_preprocessing_uses_training_rows_only():
    rng = np.random.default_rng(5)
    train = rng.normal(size=(50, 6))
    test = rng.normal(size=(20, 6)) * 100.0 + 50.0
    a, b = MODULE._fit_transform(train, test, None)
    np.testing.assert_allclose(a.mean(axis=0), 0.0, atol=1e-9)
    np.testing.assert_allclose(a.std(axis=0), 1.0, atol=1e-9)
    assert abs(b.mean()) > 5.0  # the test rows are not re-centred with their own statistics


def test_whitened_top_k_components_have_unit_variance_on_training_rows():
    rng = np.random.default_rng(6)
    train = rng.normal(size=(200, 12)) * np.linspace(5, 1, 12)
    a, _ = MODULE._fit_transform(train, train[:5], 4)
    assert a.shape == (200, 4)
    np.testing.assert_allclose(a.var(axis=0, ddof=1), 1.0, atol=1e-9)


def test_svd_falls_back_to_gesvd_when_gesdd_does_not_converge(monkeypatch):
    rng = np.random.default_rng(11)
    train = rng.normal(size=(120, 10)) * np.linspace(4, 1, 10)
    expected, _ = MODULE._fit_transform(train, train[:3], 3)

    def failing(*args, **kwargs):
        raise np.linalg.LinAlgError("SVD did not converge")

    monkeypatch.setattr(MODULE.np.linalg, "svd", failing)
    before = MODULE.SVD_FALLBACKS
    got, _ = MODULE._fit_transform(train, train[:3], 3)
    assert MODULE.SVD_FALLBACKS == before + 1
    # singular vectors are sign-ambiguous; compare the whitened scores up to sign per component
    np.testing.assert_allclose(np.abs(got), np.abs(expected), atol=1e-8)


def test_folds_are_deterministic_and_partition_the_rows():
    y = np.tile([0, 1, 1], 40)
    first, second = MODULE.make_folds(y, 5, 7), MODULE.make_folds(y, 5, 7)
    for (a, b), (c, d) in zip(first, second):
        np.testing.assert_array_equal(a, c)
        np.testing.assert_array_equal(b, d)
    assert sorted(np.concatenate([te for _, te in first])) == list(range(len(y)))


def test_auc_does_not_depend_on_row_order():
    x, y = planted(strength=1.0, seed=8)
    perm = np.random.default_rng(9).permutation(len(y))
    a = MODULE.probe_auc(x, y, MODULE.make_folds(y, 5, 1), None, 1)["auroc"]
    b = MODULE.probe_auc(x[perm], y[perm], MODULE.make_folds(y[perm], 5, 1), None, 1)["auroc"]
    assert abs(a - b) < 0.06  # different fold membership, same data


def test_bootstrap_interval_contains_the_point_estimate_for_a_clean_signal():
    x, y = planted(strength=1.0, seed=10)
    res = MODULE.probe_auc(x, y, MODULE.make_folds(y, 5, 1), None, 1)
    lo, hi = MODULE.bootstrap_interval(y, res["oof"], 100, 1)
    assert lo <= res["auroc"] <= hi and hi - lo < 0.2


def test_select_targets_requires_enough_positives_and_negatives():
    labels = {"kind": "multilabel", "targets": {"a": np.array([1] * 60 + [0] * 40), "b": np.array([1] * 5 + [0] * 95),
                                                 "c": np.array([1] * 99 + [0] * 1)}}
    assert set(MODULE.select_targets(labels, 20)) == {"a"}


def test_label_loaders_join_by_dataset_key(tmp_path):
    hm = tmp_path / "hm"
    hm.mkdir()
    (hm / "dev_seen.jsonl").write_text(
        json.dumps({"id": "00012", "label": 1}) + "\n" + json.dumps({"id": "00034", "label": 0}) + "\n")
    out = MODULE.load_labels("hateful_memes", hm, ["hateful_memes:00034", "hateful_memes:12"])
    np.testing.assert_array_equal(out["targets"]["hateful"], [0, 1])
    mm = tmp_path / "mm"
    mm.mkdir()
    (mm / "mmimdb_trainval.csv").write_text('img_index,genre,caption\n7,"[1, 0, 1]","x"\n9,"[0, 1, 0]","y"\n')
    out = MODULE.load_labels("mm_imdb", mm, ["mm_imdb:9:abc:row:1", "mm_imdb:7:def:row:0"])
    np.testing.assert_array_equal(out["targets"]["genre02"], [0, 1])
    up = tmp_path / "up"
    up.mkdir()
    (up / "upmcfood_trainval.csv").write_text('img_file,label,class,text\n1.jpg,"[1]",pizza,t\n2.jpg,"[0]",sushi,t\n')
    out = MODULE.load_labels("upmc_food", up, ["upmc_food:2.jpg:aa:row:5"])
    assert out["classes"] == ["sushi"]
    with pytest.raises(KeyError):
        MODULE.load_labels("upmc_food", up, ["upmc_food:9.jpg:aa:row:5"])

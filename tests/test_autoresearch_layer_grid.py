import json
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

_MODULE_PATH = Path(__file__).resolve().parents[1] / "src" / "analyses" / "autoresearch_layer_grid.py"
_SPEC = importlib.util.spec_from_file_location("autoresearch_layer_grid_test_module", _MODULE_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)
LayerGridError = _MODULE.LayerGridError
analyze_caches = _MODULE.analyze_caches


def _write_cache(
    path,
    text,
    vision,
    *,
    sample_ids=None,
    manifest_hash="manifest-a",
    image_preprocessing="fixed validation transform v1",
    checkpoint_seed=31,
):
    n_samples, n_layers, _ = text.shape
    sample_ids = sample_ids or [f"sample-{index}" for index in range(n_samples)]
    layer_ids = [2 + 5 * index for index in range(n_layers)]
    cross_layers = layer_ids.copy()
    metadata = {
        "cache_identity": {
            "checkpoint_sha256": f"checkpoint-{path.stem}",
            "manifest_sha256": manifest_hash,
        },
        "checkpoint_path": f"/checkpoints/{path.stem}.pt",
        "checkpoint_architecture": {
            "text_cross_attention_layers_zero_based": [layer_ids[0]],
            "vision_cross_attention_layers_zero_based": [layer_ids[-1]],
            "saved_checkpoint_config": {
                "text_cross_attention_layers": [layer_ids[0]],
                "vision_cross_attention_layers": [layer_ids[-1]],
            },
        },
        "checkpoint_provenance": {
            "embedded_metadata": {"seed": checkpoint_seed},
            "config_seed": checkpoint_seed,
        },
        "task": "hateful_memes",
        "split": "val_discovery",
        "validation_role": "discovery",
        "source_data_sha256": {"metadata.jsonl": "source-hash"},
        "data_manifest_sha256": manifest_hash,
        "image_preprocessing": image_preprocessing,
        "tokenizer_identity": {"name": "local-test-tokenizer", "revision": "test"},
        "text_max_length": 128,
        "extraction_version": "test-extraction-v1",
        "code_sha256": {"src/autoresearch.py": "same-source"},
        "ordered_sample_ids": list(sample_ids),
        "sample_pair_hashes": {sample_id: f"pair-{sample_id}" for sample_id in sample_ids},
        "layer_ids_zero_based": layer_ids,
        "cross_attention_layer_ids_zero_based": cross_layers,
    }
    np.savez_compressed(
        path,
        text_features=np.asarray(text, dtype=np.float32),
        vision_features=np.asarray(vision, dtype=np.float32),
        layer_ids=np.asarray(layer_ids, dtype=np.int64),
        cross_attention_layers=np.asarray(cross_layers, dtype=np.int64),
        sample_ids=np.asarray(sample_ids, dtype=np.str_),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    return path


def test_known_permuted_layer_patterns_preserve_text_rows_and_vision_columns(tmp_path):
    rng = np.random.default_rng(2026)
    x = rng.normal(size=(72, 7))
    z = rng.normal(size=(72, 7))
    path = _write_cache(
        tmp_path / "crossed.npz",
        np.stack((x, z), axis=1),
        np.stack((z, x), axis=1),
    )

    result = analyze_caches([path])
    run = result["runs"][0]
    grid = run["grid"]

    assert grid["text_layer_ids_zero_based"] == [2, 7]
    assert grid["vision_layer_ids_zero_based"] == [2, 7]
    assert grid["values"][0][1] > 0.999
    assert grid["values"][1][0] > 0.999
    assert grid["values"][0][0] < 0.4
    assert run["configured_ordinal_cross_attention_pairs"][0][
        "configured_text_layer_id_zero_based"
    ] == 2
    assert run["configured_ordinal_cross_attention_pairs"][0][
        "configured_vision_layer_id_zero_based"
    ] == 7
    assert run["configured_ordinal_cross_attention_pairs"][0]["linear_cka_raw"] > 0.999
    assert "does not assert" in run["configured_ordinal_cross_attention_pairs"][0]["interpretation"]


def test_constant_layers_are_explicitly_unavailable(tmp_path):
    rng = np.random.default_rng(12)
    x = rng.normal(size=(30, 4))
    constant = np.ones((30, 4))
    path = _write_cache(
        tmp_path / "constant.npz",
        np.stack((constant, x), axis=1),
        np.stack((x, constant), axis=1),
    )

    run = analyze_caches([path])["runs"][0]

    assert run["grid"]["values"][0] == [None, None]
    assert run["grid"]["statuses"][0] == [
        "zero_variance_or_zero_rank",
        "zero_variance_or_zero_rank",
    ]
    assert run["grid"]["values"][1][0] == pytest.approx(1.0)
    assert run["grid"]["values"][1][1] is None
    assert run["max_grid_calibration"]["status"] == "not_requested"


def test_all_constant_grid_skips_max_calibration(tmp_path):
    constant = np.ones((12, 2, 3), dtype=np.float32)
    path = _write_cache(tmp_path / "all-constant.npz", constant, constant)

    run = analyze_caches([path], calibrate_max_grid=True, n_permutations=5)["runs"][0]

    assert run["max_grid_raw"] is None
    assert run["n_valid_grid_cells"] == 0
    assert run["max_grid_calibration"]["status"] == "not_run_degenerate_grid"


def test_comparison_rejects_a_different_sample_manifest(tmp_path):
    rng = np.random.default_rng(18)
    values = rng.normal(size=(20, 2, 4))
    first = _write_cache(tmp_path / "first.npz", values, values)
    second = _write_cache(
        tmp_path / "second.npz", values, values, manifest_hash="manifest-b"
    )

    with pytest.raises(LayerGridError, match="data_manifest_sha256"):
        analyze_caches([first, second])


def test_comparison_rejects_same_ids_in_a_different_row_order(tmp_path):
    rng = np.random.default_rng(19)
    values = rng.normal(size=(20, 2, 4))
    ids = [f"sample-{index}" for index in range(20)]
    first = _write_cache(tmp_path / "first.npz", values, values, sample_ids=ids)
    reverse = list(reversed(ids))
    second = _write_cache(
        tmp_path / "second.npz",
        values[::-1],
        values[::-1],
        sample_ids=reverse,
    )

    with pytest.raises(LayerGridError, match="ordered_sample_ids"):
        analyze_caches([first, second])


def test_comparison_rejects_a_transform_mismatch(tmp_path):
    rng = np.random.default_rng(20)
    values = rng.normal(size=(20, 2, 4))
    first = _write_cache(tmp_path / "first.npz", values, values)
    second = _write_cache(
        tmp_path / "second.npz",
        values,
        values,
        image_preprocessing="different validation transform",
    )

    with pytest.raises(LayerGridError, match="image_preprocessing"):
        analyze_caches([first, second])


def test_aggregate_null_uses_one_draw_for_every_vision_layer(tmp_path):
    rng = np.random.default_rng(20261021)
    text = rng.normal(size=(25, 3, 5))
    vision = rng.normal(size=(25, 3, 5))
    # Make the selected cell strong while keeping the remaining layers distinct.
    vision[:, 2, :] = text[:, 0, :] + 0.05 * vision[:, 2, :]
    path = _write_cache(tmp_path / "calibrate.npz", text, vision)
    seed = 987
    permutations = 9

    run = analyze_caches(
        [path],
        calibrate_max_grid=True,
        n_permutations=permutations,
        seed=seed,
    )["runs"][0]
    calibration = run["max_grid_calibration"]

    def gram_per_layer(features):
        output = []
        for index in range(features.shape[1]):
            centered = features[:, index, :].astype(np.float64)
            centered -= centered.mean(axis=0, keepdims=True)
            output.append(centered @ centered.T)
        return output

    text_grams = gram_per_layer(text.astype(np.float32))
    vision_grams = gram_per_layer(vision.astype(np.float32))
    text_norms = [np.sqrt(np.sum(gram * gram)) for gram in text_grams]
    vision_norms = [np.sqrt(np.sum(gram * gram)) for gram in vision_grams]
    expected_null = []
    null_rng = np.random.default_rng(seed)
    for _ in range(permutations):
        row_order = null_rng.permutation(text.shape[0])
        cell_scores = []
        for text_index, text_gram in enumerate(text_grams):
            for vision_index, vision_gram in enumerate(vision_grams):
                permuted = vision_gram[np.ix_(row_order, row_order)]
                cell_scores.append(
                    np.sum(text_gram * permuted)
                    / (text_norms[text_index] * vision_norms[vision_index])
                )
        expected_null.append(float(max(cell_scores)))

    assert calibration["scope"] == "maximum over all valid text-layer x vision-layer cells"
    assert calibration["n_permutations"] == permutations
    assert calibration["null_scores"] == pytest.approx(expected_null, abs=1e-11)
    assert calibration["pointwise_after_selection_guarantee"] is False
    assert calibration["observed"] == pytest.approx(run["max_grid_raw"], abs=1e-11)

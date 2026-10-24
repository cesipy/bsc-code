"""Exact mathematical checks for the standalone cached-feature controls."""

import itertools
import json

import numpy as np
import pytest

from analyses.autoresearch_cycle_controls import (
    analyze_arrays,
    analyze_npz,
    biased_linear_cka_from_centered_grams,
    centered_linear_gram,
    cycle_knn_score,
    exact_biased_cka_null_mean,
    exact_cycle_null_mean,
    permute_neighbor_graph,
)


def _toy_neighbor_graphs():
    graph_a = np.asarray(
        [[1, 2], [0, 2], [0, 1], [0, 1], [0, 1]], dtype=np.int32
    )
    graph_b = np.asarray(
        [[1, 4], [0, 2], [1, 3], [2, 4], [3, 0]], dtype=np.int32
    )
    return graph_a, graph_b


def test_cycle_hubness_null_means_match_all_n5_permutations_and_delta_is_nonzero():
    graph_a, graph_b = _toy_neighbor_graphs()
    n, k = graph_a.shape
    all_orders = itertools.permutations(range(n))
    scores_ab = []
    scores_ba = []
    for order in all_orders:
        graph_b_permuted = permute_neighbor_graph(graph_b, order)
        scores_ab.append(cycle_knn_score(graph_a, graph_b_permuted, k))
        scores_ba.append(cycle_knn_score(graph_b_permuted, graph_a, k))

    exact_ab = exact_cycle_null_mean(graph_a, k)
    exact_ba = exact_cycle_null_mean(graph_b, k)
    assert np.mean(scores_ab) == pytest.approx(exact_ab, abs=1e-14)
    assert np.mean(scores_ba) == pytest.approx(exact_ba, abs=1e-14)
    assert exact_ab == pytest.approx(0.5666666666666667)
    assert exact_ba == pytest.approx(0.8333333333333334)
    assert exact_ab - exact_ba == pytest.approx(-0.2666666666666667)
    assert abs(exact_ab - exact_ba) > 0.2


def test_cycle_null_truncates_wider_stored_graph_to_requested_k():
    graph_a, graph_b = _toy_neighbor_graphs()
    n = graph_a.shape[0]
    requested_k = 1
    scores_ab = []
    scores_ba = []
    for order in itertools.permutations(range(n)):
        graph_b_permuted = permute_neighbor_graph(graph_b, order)
        scores_ab.append(cycle_knn_score(graph_a, graph_b_permuted, requested_k))
        scores_ba.append(cycle_knn_score(graph_b_permuted, graph_a, requested_k))

    exact_ab = exact_cycle_null_mean(graph_a, requested_k)
    exact_ba = exact_cycle_null_mean(graph_b, requested_k)
    assert np.mean(scores_ab) == pytest.approx(exact_ab, abs=1e-14)
    assert np.mean(scores_ba) == pytest.approx(exact_ba, abs=1e-14)


def test_graph_relabel_matches_direct_row_and_index_relabeling():
    graph, _ = _toy_neighbor_graphs()
    order = np.asarray([2, 4, 1, 0, 3], dtype=np.int64)
    inverse = {old: new for new, old in enumerate(order.tolist())}
    expected = np.asarray(
        [[inverse[int(item)] for item in graph[old]] for old in order],
        dtype=np.int32,
    )
    np.testing.assert_array_equal(permute_neighbor_graph(graph, order), expected)


def test_biased_linear_cka_null_matches_all_n5_permutations():
    features_a = np.asarray(
        [[1, 0], [0, 2], [3, 1], [-1, 4], [2, -2]], dtype=np.float64
    )
    features_b = np.asarray(
        [[1, 2], [3, -1], [0, 4], [2, 2], [-2, 1]], dtype=np.float64
    )
    gram_a = centered_linear_gram(features_a)
    gram_b = centered_linear_gram(features_b)
    exact = exact_biased_cka_null_mean(gram_a, gram_b)
    permutation_scores = []
    for order in itertools.permutations(range(len(features_a))):
        gram_b_permuted = gram_b[np.ix_(order, order)]
        permutation_scores.append(
            biased_linear_cka_from_centered_grams(gram_a, gram_b_permuted)
        )
    assert np.mean(permutation_scores) == pytest.approx(exact, abs=1e-13)


def test_no_permutations_reports_exact_nulls_without_p_values():
    rng = np.random.default_rng(7)
    text = rng.normal(size=(8, 2, 5)).astype(np.float32)
    vision = rng.normal(size=(8, 2, 7)).astype(np.float32)
    result = analyze_arrays(
        text,
        vision,
        np.asarray([f"id-{i}" for i in range(8)]),
        layer_ids=np.asarray([3, 7]),
        k_values=(2, 4),
        permutations=0,
        seed=14,
    )
    json.dumps(result, allow_nan=False)
    assert result["configuration"]["permutations"] == 0
    assert result["layers"][0]["layer_id"] == 3
    for layer in result["layers"]:
        for k_result in layer["k_results"]:
            assert k_result["delta_text_minus_vision"]["exact_unrestricted_permutation_null_mean"] is not None
            assert k_result["delta_text_minus_vision"]["two_sided_add_one_p_about_exact_null_mean"] is None
        assert layer["biased_linear_cka"]["analytic_unrestricted_permutation_null_mean"] is not None
        assert layer["biased_linear_cka"]["permutation_sample_summary"]["mean"] is None


def test_npz_cache_records_hash_and_rejects_mismatched_ids(tmp_path):
    rng = np.random.default_rng(2)
    text = rng.normal(size=(7, 1, 4)).astype(np.float32)
    vision = rng.normal(size=(7, 1, 5)).astype(np.float32)
    ids = np.asarray([f"sample-{i}" for i in range(7)])
    cache = tmp_path / "paired_features.npz"
    np.savez(cache, text=text, vision=vision, ids=ids, layer_ids=np.asarray([11]))

    result = analyze_npz(cache, k_values=(2,), permutations=0)
    json.dumps(result, allow_nan=False)
    assert len(result["input"]["cache_sha256"]) == 64
    assert result["input"]["paired_ids_verified"] is True
    assert result["layers"][0]["layer_id"] == 11

    mismatch = tmp_path / "mismatched_ids.npz"
    np.savez(
        mismatch,
        text=text,
        vision=vision,
        text_ids=ids,
        vision_ids=np.asarray([f"other-{i}" for i in range(7)]),
    )
    with pytest.raises(ValueError, match="text_ids and vision_ids differ"):
        analyze_npz(mismatch, k_values=(2,), permutations=0)


def test_cli_defaults_resolve_feature_keys_and_sample_ids(tmp_path, capsys):
    from analyses.autoresearch_cycle_controls import main

    rng = np.random.default_rng(23)
    text = rng.normal(size=(8, 2, 4)).astype(np.float32)
    vision = rng.normal(size=(8, 2, 5)).astype(np.float32)
    cache = tmp_path / "actual_schema.npz"
    np.savez(
        cache,
        text_features=text,
        vision_features=vision,
        sample_ids=np.asarray([f"sample-{i}" for i in range(8)]),
        layer_ids=np.asarray([2, 5]),
        metadata=np.asarray("unused descriptive metadata"),
    )
    output = tmp_path / "controls.json"
    assert main(
        [str(cache), "--output", str(output), "--k", "2", "--permutations", "0"]
    ) == 0
    assert "controls.json" in capsys.readouterr().out
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["input"]["text_key"] == "text_features"
    assert report["input"]["vision_key"] == "vision_features"
    assert report["input"]["id_key"] == "sample_ids"
    assert report["input"]["paired_ids_verified"] is True
    assert [layer["layer_id"] for layer in report["layers"]] == [2, 5]

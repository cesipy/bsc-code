"""Synthetic controls for paired permutation-null calibration."""

import json
import math
import numpy as np
import pytest

from analyses.null_calibration import calibrate_permutation_null


def exact_row_match(x, y):
    return float(np.mean(np.all(x == y, axis=1)))


def mean_squared_distance(x, y):
    return float(np.mean(np.sum((x - y) ** 2, axis=-1)))


def test_matching_pairs_outscore_a_fixed_mispairing():
    x = np.eye(12)
    mispaired_y = np.roll(x, shift=1, axis=0)

    aligned = calibrate_permutation_null(
        x, x.copy(), exact_row_match, n_permutations=99, seed=13, maximum=1.0
    )
    mispaired = calibrate_permutation_null(
        x, mispaired_y, exact_row_match, n_permutations=99, seed=13, maximum=1.0
    )

    assert aligned["observed"] == 1.0
    assert aligned["permutation_p_value"] == aligned["monte_carlo_resolution"]
    assert aligned["calibrated_effect"] == pytest.approx(1.0)
    assert mispaired["observed"] == 0.0
    assert mispaired["permutation_p_value"] == 1.0
    assert mispaired["calibrated_effect"] == 0.0


def test_less_is_better_uses_lower_tail_and_known_minimum():
    x = np.eye(10)
    result = calibrate_permutation_null(
        x,
        x.copy(),
        mean_squared_distance,
        n_permutations=99,
        seed=4,
        alternative="less",
        quantile=0.95,
        minimum=0.0,
    )

    assert result["observed"] == 0.0
    assert result["threshold_quantile"] == pytest.approx(0.05)
    rank = math.ceil(0.95 * (len(result["null_scores"]) + 1))
    oriented_combined = np.sort(
        -np.asarray([result["observed"], *result["null_scores"]])
    )
    assert result["null_threshold"] == pytest.approx(-oriented_combined[rank - 1])
    assert result["threshold_order_statistic_rank"] == rank
    assert result["permutation_p_value"] == result["monte_carlo_resolution"]
    assert result["permutation_p_value"] <= result["effect_gate_alpha"]
    assert result["calibrated_effect"] == pytest.approx(1.0)


def test_supplied_seed_reproduces_the_null_scores():
    rng = np.random.default_rng(22)
    x = rng.normal(size=(17, 4))
    y = rng.normal(size=(17, 6))  # feature widths can differ across modalities
    metric = lambda left, right: float(np.mean(left) - np.mean(right))

    first = calibrate_permutation_null(x, y, metric, n_permutations=31, seed=711)
    second = calibrate_permutation_null(x, y, metric, n_permutations=31, seed=711)

    np.testing.assert_array_equal(first["null_scores"], second["null_scores"])
    assert first["observed"] == second["observed"]
    assert first["null_count"] == 31
    assert first["monte_carlo_resolution"] == pytest.approx(1 / 32)
    assert first["raw_observed"] == first["observed"]
    json.dumps(first, allow_nan=False)


def test_aggregate_metric_uses_one_row_permutation_for_all_layers():
    rng = np.random.default_rng(8)
    x = rng.normal(size=(14, 3, 5))
    y = rng.normal(size=(14, 3, 5))

    def max_layer_similarity(left, right):
        layer_scores = np.mean(np.sum(left * right, axis=-1), axis=0)
        return float(np.max(layer_scores))

    seed = 91
    draws = 23
    result = calibrate_permutation_null(
        x, y, max_layer_similarity, n_permutations=draws, seed=seed
    )

    manual_rng = np.random.default_rng(seed)
    expected = [
        max_layer_similarity(x, np.take(y, manual_rng.permutation(len(x)), axis=0))
        for _ in range(draws)
    ]
    np.testing.assert_allclose(result["null_scores"], expected)
    assert result["observed"] == max_layer_similarity(x, y)


def test_constant_identical_rows_report_degenerate_effect_denominator():
    x = np.zeros((7, 4))
    result = calibrate_permutation_null(
        x,
        x.copy(),
        exact_row_match,
        n_permutations=12,
        seed=3,
        maximum=1.0,
    )

    assert result["observed"] == 1.0
    assert result["null_mean"] == 1.0
    assert result["null_threshold"] == 1.0
    assert result["permutation_p_value"] == 1.0
    assert result["calibrated_effect"] is None
    assert result["calibrated_effect_status"] == "degenerate_threshold_denominator"
    assert result["threshold_method"] == "combined_observed_null_order_statistic"


def test_missing_known_boundary_does_not_invent_a_calibrated_effect():
    x = np.eye(6)
    result = calibrate_permutation_null(
        x, x.copy(), exact_row_match, n_permutations=10, seed=1
    )
    assert result["calibrated_effect"] is None
    assert result["calibrated_effect_status"] == "no_known_favorable_boundary"


def test_combined_order_statistic_gates_effect_at_add_one_alpha_with_small_k_and_ties():
    x = np.eye(12)
    for n_permutations in (10, 19, 99):
        result = calibrate_permutation_null(
            x,
            x.copy(),
            exact_row_match,
            n_permutations=n_permutations,
            seed=4,
            quantile=0.95,
            maximum=1.0,
        )
        expected_rank = math.ceil(0.95 * (n_permutations + 1))
        assert result["threshold_order_statistic_rank"] == expected_rank
        assert result["threshold_order_statistic_size"] == n_permutations + 1
        has_positive_effect = (
            result["calibrated_effect"] is not None
            and result["calibrated_effect"] > 0.0
        )
        assert has_positive_effect == (
            result["permutation_p_value"] <= result["effect_gate_alpha"]
        )

    tied = calibrate_permutation_null(
        np.zeros((2, 3)),
        np.zeros((2, 3)),
        exact_row_match,
        n_permutations=19,
        seed=5,
        quantile=0.95,
        maximum=1.0,
    )
    assert tied["exceedance_count"] > 0
    assert tied["permutation_p_value"] > tied["effect_gate_alpha"]
    assert tied["calibrated_effect"] is None
    assert tied["calibrated_effect_status"] == "degenerate_threshold_denominator"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_permutations": 0},
        {"n_permutations": True},
        {"seed": -1},
        {"seed": 1.5},
        {"alternative": "two-sided"},
        {"quantile": 1.0},
        {"quantile": float("nan")},
        {"maximum": 1.0, "minimum": 0.0},
        {"alternative": "less", "maximum": 1.0},
        {"alternative": "greater", "minimum": 0.0},
    ],
)
def test_invalid_parameters_are_rejected(kwargs):
    x = np.eye(4)
    with pytest.raises(ValueError):
        calibrate_permutation_null(x, x, exact_row_match, **kwargs)


def test_invalid_inputs_and_nonexchangeable_groups_fail_closed():
    x = np.ones((5, 2))
    with pytest.raises(ValueError, match="same number"):
        calibrate_permutation_null(x, x[:-1], exact_row_match)
    with pytest.raises(ValueError, match="finite"):
        calibrate_permutation_null(
            np.array([[0.0], [np.inf]]), np.ones((2, 1)), mean_squared_distance
        )
    with pytest.raises(ValueError, match="groups are unsupported"):
        calibrate_permutation_null(x, x, exact_row_match, groups=[0, 0, 1, 1, 2])
    with pytest.raises(ValueError, match="finite real scalar"):
        calibrate_permutation_null(x, x, lambda left, right: float("nan"))
    with pytest.raises(ValueError, match="finite real scalar"):
        calibrate_permutation_null(x, x, lambda left, right: np.array([1.0]))

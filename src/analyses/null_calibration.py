"""Permutation-null calibration for paired, sample-level representations.

The caller supplies cached representations and a deterministic scalar metric.
The same metric is evaluated on the observed pairing and on every null draw.
Each draw permutes the sample axis of ``y`` once, so stacked layers in trailing
dimensions share one correspondence permutation and any fixed aggregation in
``metric`` (for example, a maximum over layers) is calibrated as reported.

This implementation assumes unrestricted exchangeability of individual rows.
It does not infer or preserve clusters, repeated examples, labels, or other
strata. Such structure must be handled by a separately justified null scheme.
"""

from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import numpy as np


__all__ = ["calibrate_permutation_null"]


def _as_finite_rows(values: Any, name: str) -> np.ndarray:
    """Return a private numeric array with finite entries."""
    try:
        array = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric array") from exc

    if array.ndim < 1:
        raise ValueError(f"{name} must have a sample axis")
    if array.dtype.kind not in "biuf":
        raise ValueError(f"{name} must contain real numeric values")
    if any(size == 0 for size in array.shape):
        raise ValueError(f"{name} must have non-empty sample and feature axes")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")

    # Keep caller-owned arrays untouched. The metric receives isolated copies
    # below so NumPy-to-Torch adapters can safely request writable storage.
    result = np.array(array, copy=True)
    return result


def _finite_real_scalar(value: Any, name: str) -> float:
    """Validate a scalar metric result or declared finite score boundary."""
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite real scalar") from exc
    if array.ndim != 0 or array.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(array)
    if not np.isfinite(result):
        raise ValueError(f"{name} must be a finite real scalar")
    return result


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{name} must be a positive integer")
    result = int(value)
    if result < 1:
        raise ValueError(f"{name} must be a positive integer")
    return result


def _nonnegative_seed(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError("seed must be a non-negative integer")
    result = int(value)
    if result < 0:
        raise ValueError("seed must be a non-negative integer")
    return result


def calibrate_permutation_null(
    x: Any,
    y: Any,
    metric: Callable[[np.ndarray, np.ndarray], Any],
    *,
    n_permutations: int = 200,
    seed: int = 0,
    alternative: str = "greater",
    quantile: float = 0.95,
    groups: Any = None,
    maximum: float | None = None,
    minimum: float | None = None,
) -> dict[str, Any]:
    """Calibrate a paired metric by permuting row correspondence in ``y``.

    Args:
        x, y: Finite real feature arrays with samples on axis 0. Their
            trailing shapes may differ, as with text and image feature widths.
            To calibrate an aggregate over layers, stack layers on trailing
            axes and compute that same aggregate inside ``metric``.
        metric: Deterministic callable returning one finite real scalar. It is
            evaluated once on the observed pair and once for every null pair.
            It should use cached features only; this routine never performs
            model inference.
        n_permutations: Fixed number K of Monte Carlo null draws.
        seed: Non-negative seed passed to NumPy's default random generator.
        alternative: ``"greater"`` for a larger-is-better metric or ``"less"``
            for a smaller-is-better metric.
        quantile: Favorable-tail coverage q in (0, 1), with alpha = 1-q.
            The threshold is the ``ceil(q * (K + 1))`` order statistic of the
            combined observed and null scores after orienting favorable scores
            as larger. Thus ``less`` uses negated scores and returns the
            threshold in the original metric units.
        groups: Any non-None value is rejected. This routine only implements
            unrestricted row exchangeability; it does not guess a valid
            group-aware permutation scheme.
        maximum: Known finite theoretical maximum of a larger-is-better
            metric. It enables max-preserving calibrated effect size.
        minimum: Known finite theoretical minimum of a smaller-is-better
            metric. It enables the corresponding lower-is-better effect size.

    Returns:
        A JSON-friendly dictionary containing the observed raw score, sampled
        null scores and summary, add-one permutation p-value, Monte Carlo
        resolution, the exact combined-set threshold rank and method, and
        optional max-preserving calibrated effect. The effect gate and p-value
        use the same alpha. The reported p-value is an exploratory Monte Carlo
        estimate, not a claim of confirmatory significance.

    Raises:
        ValueError: If arrays, parameters, metric outputs, or declared bounds
            are invalid, or if groups are supplied without a justified
            group-aware null implementation.
    """
    if groups is not None:
        raise ValueError(
            "groups are unsupported: this calibration assumes unrestricted "
            "exchangeability of individual rows"
        )
    if not callable(metric):
        raise ValueError("metric must be callable")

    count = _positive_int(n_permutations, "n_permutations")
    random_seed = _nonnegative_seed(seed)
    if alternative not in ("greater", "less"):
        raise ValueError("alternative must be 'greater' or 'less'")
    if isinstance(quantile, (bool, np.bool_)):
        raise ValueError("quantile must be strictly between 0 and 1")
    try:
        q = float(quantile)
    except (TypeError, ValueError) as exc:
        raise ValueError("quantile must be strictly between 0 and 1") from exc
    if not np.isfinite(q) or not 0.0 < q < 1.0:
        raise ValueError("quantile must be strictly between 0 and 1")

    if maximum is not None and minimum is not None:
        raise ValueError("provide at most one favorable score boundary")
    if alternative == "greater" and minimum is not None:
        raise ValueError("a larger-is-better metric requires maximum, not minimum")
    if alternative == "less" and maximum is not None:
        raise ValueError("a smaller-is-better metric requires minimum, not maximum")
    boundary = None
    if maximum is not None:
        boundary = _finite_real_scalar(maximum, "maximum")
    elif minimum is not None:
        boundary = _finite_real_scalar(minimum, "minimum")

    x_rows = _as_finite_rows(x, "x")
    y_rows = _as_finite_rows(y, "y")
    if x_rows.shape[0] != y_rows.shape[0]:
        raise ValueError("x and y must have the same number of paired samples")
    n_samples = x_rows.shape[0]
    if n_samples < 2:
        raise ValueError("at least two paired samples are required")

    observed = _finite_real_scalar(
        metric(np.array(x_rows, copy=True), np.array(y_rows, copy=True)),
        "observed metric",
    )
    rng = np.random.default_rng(random_seed)
    null_scores = np.empty(count, dtype=np.float64)
    for index in range(count):
        permutation = rng.permutation(n_samples)
        permuted_y = np.take(y_rows, permutation, axis=0)
        null_scores[index] = _finite_real_scalar(
            metric(np.array(x_rows, copy=True), permuted_y),
            f"null metric at permutation {index}",
        )

    alpha = 1.0 - q
    threshold_order_statistic_rank = math.ceil(q * (count + 1))  # one-based
    if alternative == "greater":
        oriented_observed = observed
        oriented_null_scores = null_scores
    else:
        # A sign flip lets distances use the same right-tail rank and p-value
        # convention as similarities; negate the threshold back for reporting.
        oriented_observed = -observed
        oriented_null_scores = -null_scores

    combined_oriented = np.concatenate(
        (np.asarray([oriented_observed]), oriented_null_scores)
    )
    oriented_threshold = float(
        np.sort(combined_oriented)[threshold_order_statistic_rank - 1]
    )
    threshold = (
        oriented_threshold if alternative == "greater" else -oriented_threshold
    )
    threshold_quantile = q if alternative == "greater" else 1.0 - q
    exceedance_count = int(
        np.count_nonzero(oriented_null_scores >= oriented_observed)
    )
    p_value = (1 + exceedance_count) / (count + 1)

    effect: float | None = None
    effect_status = "no_known_favorable_boundary"
    if boundary is not None:
        scores_to_check = np.concatenate((null_scores, np.asarray([observed])))
        tolerance = 1e-12 * max(1.0, abs(boundary))
        if alternative == "greater":
            if np.any(scores_to_check > boundary + tolerance):
                raise ValueError("a score exceeds the declared maximum")
            oriented_boundary = boundary
        else:
            if np.any(scores_to_check < boundary - tolerance):
                raise ValueError("a score is below the declared minimum")
            oriented_boundary = -boundary

        denominator = oriented_boundary - oriented_threshold
        numerator = oriented_observed - oriented_threshold

        if denominator <= tolerance:
            effect_status = "degenerate_threshold_denominator"
        else:
            # The theoretical bound may be exceeded only by negligible
            # floating-point error; keep the normalized quantity in [0, 1].
            effect = float(np.clip(max(0.0, numerator / denominator), 0.0, 1.0))
            effect_status = "computed_from_declared_favorable_boundary"

    null_scores_list = null_scores.tolist()
    return {
        "observed": observed,
        "raw_observed": observed,
        "alternative": alternative,
        "n_permutations": count,
        "seed": random_seed,
        "null_scores": null_scores_list,
        "null_count": count,
        "null_mean": float(np.mean(null_scores)),
        "null_threshold": threshold,
        "oriented_null_threshold": oriented_threshold,
        "threshold_quantile": threshold_quantile,
        "oriented_threshold_quantile": q,
        "requested_quantile": q,
        "effect_gate_alpha": alpha,
        "threshold_method": "combined_observed_null_order_statistic",
        "threshold_order_statistic_rank": threshold_order_statistic_rank,
        "threshold_order_statistic_size": count + 1,
        "exceedance_count": exceedance_count,
        "permutation_p_value": float(p_value),
        "monte_carlo_resolution": 1.0 / (count + 1),
        "calibrated_effect": effect,
        "calibrated_effect_status": effect_status,
        "favorable_boundary": boundary,
        "exchangeability_assumption": "unrestricted individual-row exchangeability",
    }

#!/usr/bin/env python3
"""CPU-only null controls for paired, cached text/vision feature arrays.

This module is standalone: it does not import the project metrics module,
model code, torch, datasets, or GPU utilities. See autoresearch/cycle_controls_design.md.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import platform
import resource
import time
import zipfile
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

CPU_THREAD_LIMIT = 4
MAX_MEMORY_BYTES = 1 << 30
DEFAULT_K_VALUES = (4, 10, 32)
PRIMARY_K = 10
CKA_DENOMINATOR_EPSILON = 1e-6
MIN_RUNTIME_RESERVE_BYTES = 64 << 20


def _positive_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _nonnegative_int(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _validate_graph(graph: Any, name: str = "neighbor_graph") -> np.ndarray:
    array = np.asarray(graph)
    if array.ndim != 2 or array.shape[0] < 2 or array.shape[1] < 1:
        raise ValueError(f"{name} must have shape [N, k] with N >= 2 and k >= 1")
    if array.dtype.kind not in "iu":
        raise ValueError(f"{name} must contain integer neighbor indices")
    n = array.shape[0]
    if np.any(array < 0) or np.any(array >= n):
        raise ValueError(f"{name} contains an out-of-range index")
    if np.any(array == np.arange(n)[:, None]):
        raise ValueError(f"{name} must exclude each row's self-index")
    if any(np.unique(row).size != array.shape[1] for row in array):
        raise ValueError(f"{name} rows must contain distinct neighbors")
    return array.astype(np.int32, copy=False)


def cycle_knn_score(knn_a: Any, knn_b: Any, k: int) -> float:
    """Match current code: B is hop 1, A is the return hop.

    The score is mean_i 1[i occurs in knn_A[knn_B[i]]].
    """
    graph_a = _validate_graph(knn_a, "knn_a")
    graph_b = _validate_graph(knn_b, "knn_b")
    if graph_a.shape[0] != graph_b.shape[0]:
        raise ValueError("neighbor graphs must have the same N")
    n = graph_a.shape[0]
    neighbor_count = _positive_int(k, "k")
    if neighbor_count > min(graph_a.shape[1], graph_b.shape[1], n - 1):
        raise ValueError("k exceeds a graph's stored neighbor count or N-1")
    return _cycle_knn_score_unchecked(graph_a, graph_b, neighbor_count)


def _cycle_knn_score_unchecked(knn_a: np.ndarray, knn_b: np.ndarray, k: int) -> float:
    n = knn_a.shape[0]
    returned = knn_a[knn_b[:, :k], :k]
    rows = np.arange(n, dtype=np.int32)[:, None, None]
    return float(np.any(returned == rows, axis=(1, 2)).mean(dtype=np.float64))


def exact_cycle_null_mean(return_graph: Any, k: int) -> float:
    """Exact mean under unrestricted uniform relative row relabeling.

    For incoming degree d_i in the return graph, a random first-hop k-subset
    misses all d_i return neighbors with probability
    prod_r (1 - d_i/(N-1-r)). The logarithmic product avoids huge combinations
    and uses expm1 to preserve precision when the hit chance is small.
    """
    graph = _validate_graph(return_graph, "return_graph")
    n = graph.shape[0]
    neighbor_count = _positive_int(k, "k")
    if neighbor_count > min(graph.shape[1], n - 1):
        raise ValueError("k exceeds the graph's stored neighbor count or N-1")
    # The cache may store a wider graph than this k-specific score uses.
    # Match cycle_knn_score's `:k` truncation before computing hubness.
    indegree = np.bincount(graph[:, :neighbor_count].reshape(-1), minlength=n)
    hit_probabilities: list[float] = []
    for degree_value in indegree:
        degree = int(degree_value)
        if neighbor_count > n - 1 - degree:
            hit_probability = 1.0
        else:
            log_no_hit = math.fsum(
                math.log1p(-degree / (n - 1 - draw))
                for draw in range(neighbor_count)
            )
            hit_probability = -math.expm1(log_no_hit)
        hit_probabilities.append(hit_probability)
    return math.fsum(hit_probabilities) / n


def permute_neighbor_graph(graph: Any, permutation: Any) -> np.ndarray:
    """Relabel rows and all neighbor IDs like replacing B with B[permutation]."""
    neighbor_graph = _validate_graph(graph)
    n = neighbor_graph.shape[0]
    order = np.asarray(permutation)
    if order.ndim != 1 or order.shape[0] != n or order.dtype.kind not in "iu":
        raise ValueError("permutation must be an integer vector of length N")
    order = order.astype(np.int64, copy=False)
    if not np.array_equal(np.sort(order), np.arange(n)):
        raise ValueError("permutation must contain every index exactly once")
    return _permute_neighbor_graph_unchecked(neighbor_graph, order)


def _permute_neighbor_graph_unchecked(graph: np.ndarray, order: np.ndarray) -> np.ndarray:
    n = graph.shape[0]
    inverse = np.empty(n, dtype=np.int32)
    inverse[order] = np.arange(n, dtype=np.int32)
    return inverse[graph[order]]


def normalize_rows_l2(features: Any) -> np.ndarray:
    """L2-normalize rows in float32 with the torch F.normalize epsilon."""
    rows = np.asarray(features)
    if rows.ndim != 2 or rows.dtype.kind not in "fiu":
        raise ValueError("features must be a real numeric [N, D] array")
    rows32 = rows.astype(np.float32, copy=False)
    if not np.isfinite(rows32).all():
        raise ValueError("features must remain finite after float32 conversion")
    norms = np.linalg.norm(rows32, axis=1, keepdims=True)
    if not np.isfinite(norms).all():
        raise ValueError("feature row norms overflowed float32")
    return rows32 / np.maximum(norms, np.float32(1e-12))


def _cosine_knn_from_normalized(rows: np.ndarray, k: int) -> np.ndarray:
    n = rows.shape[0]
    neighbor_count = _positive_int(k, "k")
    if neighbor_count >= n:
        raise ValueError("k must be smaller than N because self is excluded")
    similarities = rows @ rows.T
    np.fill_diagonal(similarities, -np.inf)
    # Stable sorting defines ties by original row index while preserving the
    # same nearest-neighbor set as argsort for continuous non-tied features.
    nearest = np.argsort(-similarities, axis=1, kind="stable")[:, :neighbor_count]
    return nearest.astype(np.int32, copy=False)


def cosine_knn(features: Any, k: int) -> np.ndarray:
    """Exhaustive cosine kNN on L2-normalized rows, with stable tie ordering."""
    return _cosine_knn_from_normalized(normalize_rows_l2(features), k)


def centered_linear_gram(features: Any) -> np.ndarray:
    """Centered linear Gram from float32 L2-normalized features, accumulated in float64."""
    rows = np.asarray(features)
    if rows.ndim != 2 or rows.dtype.kind not in "fiu":
        raise ValueError("features must be a real numeric [N, D] array")
    if not np.isfinite(rows).all():
        raise ValueError("features must be finite")
    values = rows.astype(np.float64, copy=False)
    centered = values - values.mean(axis=0, keepdims=True)
    return centered @ centered.T


def biased_linear_cka_from_centered_grams(
    gram_a_centered: Any,
    gram_b_centered: Any,
    epsilon: float = CKA_DENOMINATOR_EPSILON,
) -> float:
    """Biased linear CKA matching hsic_biased and the repository epsilon."""
    gram_a = np.asarray(gram_a_centered, dtype=np.float64)
    gram_b = np.asarray(gram_b_centered, dtype=np.float64)
    if gram_a.ndim != 2 or gram_a.shape[0] != gram_a.shape[1] or gram_a.shape != gram_b.shape:
        raise ValueError("centered Gram matrices must be square and have the same shape")
    if not np.isfinite(gram_a).all() or not np.isfinite(gram_b).all():
        raise ValueError("centered Gram matrices must be finite")
    eps = float(epsilon)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("epsilon must be finite and non-negative")
    numerator = float(np.sum(gram_a * gram_b, dtype=np.float64))
    norm_a_squared = float(np.sum(gram_a * gram_a, dtype=np.float64))
    norm_b_squared = float(np.sum(gram_b * gram_b, dtype=np.float64))
    denominator = math.sqrt(max(0.0, norm_a_squared * norm_b_squared)) + eps
    return numerator / denominator if denominator > 0 else 0.0


def exact_biased_cka_null_mean(
    gram_a_centered: Any,
    gram_b_centered: Any,
    epsilon: float = CKA_DENOMINATOR_EPSILON,
) -> float:
    """Exact mean CKA under unrestricted row permutation of the B features."""
    gram_a = np.asarray(gram_a_centered, dtype=np.float64)
    gram_b = np.asarray(gram_b_centered, dtype=np.float64)
    if gram_a.ndim != 2 or gram_a.shape[0] != gram_a.shape[1] or gram_a.shape != gram_b.shape:
        raise ValueError("centered Gram matrices must be square and have the same shape")
    n = gram_a.shape[0]
    if n < 2:
        raise ValueError("at least two samples are required for CKA null")
    if not np.isfinite(gram_a).all() or not np.isfinite(gram_b).all():
        raise ValueError("centered Gram matrices must be finite")
    eps = float(epsilon)
    if not math.isfinite(eps) or eps < 0:
        raise ValueError("epsilon must be finite and non-negative")
    trace_a = float(np.trace(gram_a))
    trace_b = float(np.trace(gram_b))
    norm_a_squared = float(np.sum(gram_a * gram_a, dtype=np.float64))
    norm_b_squared = float(np.sum(gram_b * gram_b, dtype=np.float64))
    denominator = math.sqrt(max(0.0, norm_a_squared * norm_b_squared)) + eps
    if denominator == 0:
        return 0.0
    return (trace_a * trace_b / (n - 1)) / denominator


def _all_finite_in_chunks(array: np.ndarray, rows_per_chunk: int = 256) -> bool:
    for start in range(0, array.shape[0], rows_per_chunk):
        if not np.isfinite(array[start : start + rows_per_chunk]).all():
            return False
    return True


def _safe_mean_sd(values: Sequence[float]) -> dict[str, float | None]:
    if not values:
        return {"mean": None, "sample_sd": None, "monte_carlo_se": None}
    array = np.asarray(values, dtype=np.float64)
    sample_sd = float(array.std(ddof=1)) if array.size > 1 else 0.0
    return {
        "mean": float(array.mean()),
        "sample_sd": sample_sd,
        "monte_carlo_se": sample_sd / math.sqrt(array.size) if array.size > 1 else 0.0,
    }


def _two_sided_add_one_p(observed: float, null_mean: float, null_samples: Sequence[float]) -> float | None:
    if not null_samples:
        return None
    deviations = np.abs(np.asarray(null_samples, dtype=np.float64) - null_mean)
    observed_deviation = abs(observed - null_mean)
    tolerance = 1e-12 * max(1.0, observed_deviation)
    exceedances = int(np.count_nonzero(deviations >= observed_deviation - tolerance))
    return (1 + exceedances) / (len(null_samples) + 1)


def _canonical_id(value: Any) -> str:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="strict")
    return str(scalar)


def _validate_ids(ids: Any, n: int, name: str = "ids") -> tuple[list[str], str]:
    array = np.asarray(ids)
    if array.ndim != 1 or array.shape[0] != n or array.dtype.kind not in "biufSU":
        raise ValueError(f"{name} must be a one-dimensional non-object ID array of length N")
    values = [_canonical_id(value) for value in array]
    if any(value == "" for value in values):
        raise ValueError(f"{name} cannot contain empty IDs")
    if len(set(values)) != n:
        raise ValueError(f"{name} must contain unique IDs")
    payload = json.dumps(values, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    return values, hashlib.sha256(payload).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            block = stream.read(1 << 20)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _npz_uncompressed_bytes(path: Path) -> int:
    if not zipfile.is_zipfile(path):
        raise ValueError("cache must be a valid .npz archive")
    with zipfile.ZipFile(path, "r") as archive:
        return sum(member.file_size for member in archive.infolist())


def _estimate_peak_bytes(
    cache_uncompressed_bytes: int,
    n: int,
    max_feature_dim: int,
    max_k: int,
) -> int:
    # Includes the loaded cache, a conservative allowance for argsort/Gram
    # matrices, per-layer feature copies, cycle gathers, and interpreter reserve.
    return int(
        cache_uncompressed_bytes
        + MIN_RUNTIME_RESERVE_BYTES
        + 64 * n * n
        + 32 * n * max_feature_dim
        + 64 * n * max_k
        + 4 * n * max_k * max_k
    )


@contextlib.contextmanager
def _limit_blas_threads(limit: int = CPU_THREAD_LIMIT) -> Iterator[list[dict[str, Any]]]:
    try:
        from threadpoolctl import threadpool_info, threadpool_limits
    except ImportError as exc:
        raise RuntimeError("threadpoolctl is required to enforce the four-thread CPU cap") from exc
    with threadpool_limits(limits=limit):
        pools = [
            {
                "user_api": item.get("user_api"),
                "internal_api": item.get("internal_api"),
                "num_threads": int(item.get("num_threads", 0)),
            }
            for item in threadpool_info()
        ]
        if any(pool["num_threads"] > limit for pool in pools):
            raise RuntimeError("a numerical thread pool exceeds the configured CPU limit")
        yield pools


def _layer_result(
    text_features: np.ndarray,
    vision_features: np.ndarray,
    layer_id: Any,
    k_values: tuple[int, ...],
    permutations: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    n = text_features.shape[0]
    max_k = max(k_values)
    text_normalized = normalize_rows_l2(text_features)
    vision_normalized = normalize_rows_l2(vision_features)
    knn_text = _cosine_knn_from_normalized(text_normalized, max_k)
    knn_vision = _cosine_knn_from_normalized(vision_normalized, max_k)

    gram_text = centered_linear_gram(text_normalized)
    gram_vision = centered_linear_gram(vision_normalized)
    observed_cka = biased_linear_cka_from_centered_grams(gram_text, gram_vision)
    cka_null_mean = exact_biased_cka_null_mean(gram_text, gram_vision)
    cka_null_samples: list[float] = []

    permutation_samples: dict[int, dict[str, list[float]]] = {
        k: {
            "cycle_text_to_vision": [],
            "cycle_vision_to_text": [],
            "delta_text_minus_vision": [],
        }
        for k in k_values
    }
    for _ in range(permutations):
        order = rng.permutation(n)
        permuted_vision_graph = _permute_neighbor_graph_unchecked(knn_vision, order)
        permuted_gram_vision = gram_vision[np.ix_(order, order)]
        cka_null_samples.append(
            biased_linear_cka_from_centered_grams(gram_text, permuted_gram_vision)
        )
        for k in k_values:
            text_to_vision = _cycle_knn_score_unchecked(knn_text, permuted_vision_graph, k)
            vision_to_text = _cycle_knn_score_unchecked(permuted_vision_graph, knn_text, k)
            permutation_samples[k]["cycle_text_to_vision"].append(text_to_vision)
            permutation_samples[k]["cycle_vision_to_text"].append(vision_to_text)
            permutation_samples[k]["delta_text_minus_vision"].append(text_to_vision - vision_to_text)

    per_k: list[dict[str, Any]] = []
    for k in k_values:
        observed_text_to_vision = _cycle_knn_score_unchecked(knn_text, knn_vision, k)
        observed_vision_to_text = _cycle_knn_score_unchecked(knn_vision, knn_text, k)
        null_text_to_vision = exact_cycle_null_mean(knn_text, k)
        null_vision_to_text = exact_cycle_null_mean(knn_vision, k)
        observed_delta = observed_text_to_vision - observed_vision_to_text
        null_delta = null_text_to_vision - null_vision_to_text
        samples = permutation_samples[k]
        text_to_vision_stats = _safe_mean_sd(samples["cycle_text_to_vision"])
        vision_to_text_stats = _safe_mean_sd(samples["cycle_vision_to_text"])
        delta_stats = _safe_mean_sd(samples["delta_text_minus_vision"])
        per_k.append(
            {
                "k": k,
                "role": "primary" if k == PRIMARY_K else "sensitivity",
                "cycle_text_to_vision": {
                    "first_hop_space": "vision",
                    "return_space": "text",
                    "observed": observed_text_to_vision,
                    "exact_unrestricted_permutation_null_mean": null_text_to_vision,
                    "observed_minus_null_mean": observed_text_to_vision - null_text_to_vision,
                    "permutation_sample_summary": text_to_vision_stats,
                    "two_sided_add_one_p_about_exact_null_mean": _two_sided_add_one_p(
                        observed_text_to_vision,
                        null_text_to_vision,
                        samples["cycle_text_to_vision"],
                    ),
                },
                "cycle_vision_to_text": {
                    "first_hop_space": "text",
                    "return_space": "vision",
                    "observed": observed_vision_to_text,
                    "exact_unrestricted_permutation_null_mean": null_vision_to_text,
                    "observed_minus_null_mean": observed_vision_to_text - null_vision_to_text,
                    "permutation_sample_summary": vision_to_text_stats,
                    "two_sided_add_one_p_about_exact_null_mean": _two_sided_add_one_p(
                        observed_vision_to_text,
                        null_vision_to_text,
                        samples["cycle_vision_to_text"],
                    ),
                },
                "delta_text_minus_vision": {
                    "definition": "cycle_text_to_vision - cycle_vision_to_text",
                    "observed": observed_delta,
                    "exact_unrestricted_permutation_null_mean": null_delta,
                    "observed_minus_null_mean": observed_delta - null_delta,
                    "permutation_sample_summary": delta_stats,
                    "two_sided_add_one_p_about_exact_null_mean": _two_sided_add_one_p(
                        observed_delta,
                        null_delta,
                        samples["delta_text_minus_vision"],
                    ),
                    "p_value_method": (
                        "(1 + count(|sample - exact_null_mean| >= "
                        "|observed - exact_null_mean|)) / (R + 1)"
                        if permutations
                        else None
                    ),
                },
            }
        )

    cka_stats = _safe_mean_sd(cka_null_samples)
    cka_mean = cka_stats["mean"]
    cka_standard_error = cka_stats["monte_carlo_se"]
    cka_result = {
        "kernel": "linear",
        "estimator": "centered_biased_hsic_cka",
        "observed": observed_cka,
        "analytic_unrestricted_permutation_null_mean": cka_null_mean,
        "observed_minus_analytic_null_mean": observed_cka - cka_null_mean,
        "denominator_epsilon": CKA_DENOMINATOR_EPSILON,
        "permutation_sample_summary": cka_stats,
        "permutation_sample_mean_minus_analytic_mean": (
            None if cka_mean is None else float(cka_mean - cka_null_mean)
        ),
        "analytic_mean_deviation_in_monte_carlo_se": (
            None
            if cka_mean is None or cka_standard_error in (None, 0.0)
            else float((cka_mean - cka_null_mean) / cka_standard_error)
        ),
        "null_check": "empirical row-relabel mean compared with exact trace-formula mean",
    }
    return {
        "layer_id": _json_safe(layer_id),
        "k_results": per_k,
        "biased_linear_cka": cka_result,
    }


def analyze_arrays(
    text: Any,
    vision: Any,
    ids: Any,
    *,
    layer_ids: Any = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    permutations: int = 200,
    seed: int = 0,
    cache_provenance: dict[str, Any] | None = None,
    cache_uncompressed_bytes: int | None = None,
    max_memory_bytes: int = MAX_MEMORY_BYTES,
) -> dict[str, Any]:
    """Analyze row-aligned [N,L,D] arrays without importing project/model code."""
    text_array = np.asarray(text)
    vision_array = np.asarray(vision)
    if text_array.ndim != 3 or vision_array.ndim != 3:
        raise ValueError("text and vision arrays must each have shape [N, L, D]")
    if text_array.shape[:2] != vision_array.shape[:2]:
        raise ValueError("text and vision arrays must have the same N and L")
    if any(dimension < 1 for dimension in text_array.shape + vision_array.shape):
        raise ValueError("feature dimensions must be positive")
    if text_array.dtype.kind not in "fiu" or vision_array.dtype.kind not in "fiu":
        raise ValueError("text and vision features must be real numeric arrays")
    if not _all_finite_in_chunks(text_array) or not _all_finite_in_chunks(vision_array):
        raise ValueError("text and vision features must be finite")
    n, layers, text_dim = text_array.shape
    vision_dim = vision_array.shape[2]
    if n < 2:
        raise ValueError("at least two paired samples are required")
    id_strings, ids_sha256 = _validate_ids(ids, n)
    if layer_ids is None:
        normalized_layer_ids: list[Any] = list(range(layers))
        layer_id_source = "implicit zero-based layer indices"
    else:
        layer_array = np.asarray(layer_ids)
        if layer_array.ndim != 1 or layer_array.shape[0] != layers:
            raise ValueError("layer_ids must be a one-dimensional array of length L")
        normalized_layer_ids = [_json_safe(item) for item in layer_array]
        if len({str(item) for item in normalized_layer_ids}) != layers:
            raise ValueError("layer_ids must be unique")
        layer_id_source = "cache layer_ids array"

    ks = tuple(sorted({_positive_int(k, "k") for k in k_values}))
    if not ks or max(ks) >= n:
        raise ValueError("k_values must be non-empty and every k must be smaller than N")
    permutation_count = _nonnegative_int(permutations, "permutations")
    random_seed = _nonnegative_int(seed, "seed")
    memory_limit = _positive_int(max_memory_bytes, "max_memory_bytes")
    if memory_limit > MAX_MEMORY_BYTES:
        raise ValueError("max_memory_bytes cannot exceed the 1 GiB task limit")
    uncompressed_bytes = (
        int(cache_uncompressed_bytes)
        if cache_uncompressed_bytes is not None
        else int(text_array.nbytes + vision_array.nbytes + np.asarray(ids).nbytes)
    )
    if uncompressed_bytes < 0:
        raise ValueError("cache_uncompressed_bytes must be non-negative")
    estimated_peak = _estimate_peak_bytes(uncompressed_bytes, n, max(text_dim, vision_dim), max(ks))
    if estimated_peak > memory_limit:
        raise MemoryError(
            f"estimated peak {estimated_peak} bytes exceeds the {memory_limit}-byte memory budget"
        )

    started_wall = time.perf_counter()
    started_cpu = time.process_time()
    rng = np.random.default_rng(random_seed)
    with _limit_blas_threads(CPU_THREAD_LIMIT) as thread_pools:
        layer_results = [
            _layer_result(
                text_array[:, layer, :],
                vision_array[:, layer, :],
                normalized_layer_ids[layer],
                ks,
                permutation_count,
                rng,
            )
            for layer in range(layers)
        ]
    wall_seconds = time.perf_counter() - started_wall
    cpu_seconds = time.process_time() - started_cpu
    try:
        peak_rss_bytes = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    except (AttributeError, OSError):
        peak_rss_bytes = None

    return {
        "schema_version": 1,
        "status": "descriptive cache diagnostic; no causal or attractor inference",
        "input": {
            **(cache_provenance or {}),
            "n_samples": n,
            "n_layers": layers,
            "text_feature_dim": text_dim,
            "vision_feature_dim": vision_dim,
            "paired_ids_verified": True,
            "unique_id_count": len(id_strings),
            "ids_sha256": ids_sha256,
            "layer_id_source": layer_id_source,
        },
        "configuration": {
            "k_values": list(ks),
            "primary_k": PRIMARY_K if PRIMARY_K in ks else None,
            "k_roles": {str(k): ("primary" if k == PRIMARY_K else "sensitivity") for k in ks},
            "permutations": permutation_count,
            "seed": random_seed,
            "permutation_scheme": "unrestricted whole-row relabeling of vision features",
            "permutation_p_values": (
                "exploratory add-one two-sided calibration around each exact conditional null mean"
                if permutation_count
                else "not computed because permutations=0"
            ),
            "direction_convention": (
                "cycle(A,B) first hops through B and returns through A; "
                "delta is cycle(text,vision) - cycle(vision,text)"
            ),
            "feature_preprocessing": "per-layer row L2 normalization; cosine kNN",
            "nearest_neighbor_search": "exhaustive float32 cosine similarities; stable sort for ties",
            "cka": "linear centered biased CKA; float64 Gram accumulation; production denominator epsilon retained",
        },
        "null_formulas": {
            "cycle": (
                "for return-graph indegrees d_i: mean_i [1 - product_{r=0}^{k-1} "
                "(1 - d_i/(N-1-r))], evaluated as log1p/expm1"
            ),
            "biased_linear_cka": (
                "tr(Kc) tr(Lc) / ((N-1) * (sqrt(||Kc||_F^2 ||Lc||_F^2) + 1e-6))"
            ),
            "scope": "unrestricted exchangeability of individual paired rows",
        },
        "resources": {
            "cpu_thread_limit": CPU_THREAD_LIMIT,
            "thread_pools_during_run": thread_pools,
            "memory_budget_bytes": memory_limit,
            "estimated_peak_bytes": estimated_peak,
            "observed_process_peak_rss_bytes": peak_rss_bytes,
            "wall_seconds": wall_seconds,
            "process_cpu_seconds": cpu_seconds,
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
        "layers": layer_results,
    }


def analyze_npz(
    cache_path: str | Path,
    *,
    text_key: str | None = None,
    vision_key: str | None = None,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    permutations: int = 200,
    seed: int = 0,
    max_memory_bytes: int = MAX_MEMORY_BYTES,
) -> dict[str, Any]:
    """Load a paired .npz cache and return strict-JSON-friendly diagnostics."""
    memory_limit = _positive_int(max_memory_bytes, "max_memory_bytes")
    if memory_limit > MAX_MEMORY_BYTES:
        raise ValueError("max_memory_bytes cannot exceed the 1 GiB task limit")
    permutation_count = _nonnegative_int(permutations, "permutations")
    if not k_values:
        raise ValueError("k_values must be non-empty")
    validated_k = tuple(sorted({_positive_int(k, "k") for k in k_values}))
    path = Path(cache_path).expanduser().resolve(strict=True)
    if path.suffix.lower() != ".npz":
        raise ValueError("cache path must end in .npz")
    uncompressed_bytes = _npz_uncompressed_bytes(path)
    if uncompressed_bytes > memory_limit * 0.70:
        raise MemoryError("uncompressed .npz members exceed 70% of the configured memory budget")
    sha256 = _hash_file(path)
    with np.load(path, allow_pickle=False) as archive:
        keys = set(archive.files)
        if text_key is None:
            text_key = next((key for key in ("text_features", "text") if key in keys), None)
        if vision_key is None:
            vision_key = next((key for key in ("vision_features", "vision") if key in keys), None)
        if text_key is None or text_key not in keys or vision_key is None or vision_key not in keys:
            raise ValueError(
                "cache must contain text_features/vision_features or text/vision keys, "
                "or provide explicit existing feature keys"
            )
        text = archive[text_key]
        vision = archive[vision_key]
        shared_id_keys = [key for key in ("ids", "sample_ids") if key in keys]
        if shared_id_keys:
            ids = archive[shared_id_keys[0]]
            for key in shared_id_keys[1:]:
                if not np.array_equal(ids, archive[key]):
                    raise ValueError(f"{shared_id_keys[0]} and {key} differ")
            if "text_ids" in keys and not np.array_equal(ids, archive["text_ids"]):
                raise ValueError(f"{shared_id_keys[0]} and text_ids differ")
            if "vision_ids" in keys and not np.array_equal(ids, archive["vision_ids"]):
                raise ValueError(f"{shared_id_keys[0]} and vision_ids differ")
        elif "text_ids" in keys and "vision_ids" in keys:
            text_ids = archive["text_ids"]
            vision_ids = archive["vision_ids"]
            if not np.array_equal(text_ids, vision_ids):
                raise ValueError("text_ids and vision_ids differ; paired row order is not established")
            ids = text_ids
        else:
            raise ValueError(
                "cache must contain shared ids/sample_ids or matching text_ids and vision_ids"
            )
        layer_ids = archive["layer_ids"] if "layer_ids" in keys else None
        id_strings, ids_sha256 = _validate_ids(ids, text.shape[0] if text.ndim == 3 else -1)
        if text.ndim != 3 or vision.ndim != 3 or text.shape[:2] != vision.shape[:2]:
            raise ValueError("feature arrays must have matching [N, L, D] leading dimensions")
        estimated_peak = _estimate_peak_bytes(
            uncompressed_bytes,
            text.shape[0],
            max(text.shape[2], vision.shape[2]),
            max(validated_k),
        )
        if estimated_peak > memory_limit:
            raise MemoryError(
                f"estimated peak {estimated_peak} bytes exceeds the {memory_limit}-byte memory budget"
            )
        provenance = {
            "cache_path": str(path),
            "cache_sha256": sha256,
            "cache_size_bytes": path.stat().st_size,
            "cache_uncompressed_bytes": uncompressed_bytes,
            "text_key": text_key,
            "vision_key": vision_key,
            "id_key": shared_id_keys[0] if shared_id_keys else "text_ids == vision_ids",
            "cache_keys": sorted(keys),
            "ids_sha256": ids_sha256,
        }
        return analyze_arrays(
            text,
            vision,
            ids,
            layer_ids=layer_ids,
            k_values=k_values,
            permutations=permutation_count,
            seed=seed,
            cache_provenance=provenance,
            cache_uncompressed_bytes=uncompressed_bytes,
            max_memory_bytes=memory_limit,
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="paired .npz cache with [N,L,D] text/vision arrays and IDs")
    parser.add_argument("--output", type=Path, help="JSON output path (default: next to the cache)")
    parser.add_argument("--text-key", help="feature key (default: text_features, then text)")
    parser.add_argument("--vision-key", help="feature key (default: vision_features, then vision)")
    parser.add_argument("--k", nargs="+", type=int, default=list(DEFAULT_K_VALUES))
    parser.add_argument("--permutations", type=int, default=200, help="whole-row permutations; use 0 to omit p-values")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-memory-gb", type=float, default=1.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if not math.isfinite(args.max_memory_gb) or not 0 < args.max_memory_gb <= 1.0:
        raise ValueError("--max-memory-gb must be greater than 0 and at most 1.0")
    output_path = args.output or args.cache.with_name(f"{args.cache.stem}_cycle_controls.json")
    result = analyze_npz(
        args.cache,
        text_key=args.text_key,
        vision_key=args.vision_key,
        k_values=args.k,
        permutations=args.permutations,
        seed=args.seed,
        max_memory_bytes=int(args.max_memory_gb * (1 << 30)),
    )
    result["source_code"] = {
        "path": str(Path(__file__).resolve()),
        "sha256": _hash_file(Path(__file__).resolve()),
    }
    result["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(result), stream, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"output": str(output_path), "cache_sha256": result["input"]["cache_sha256"], "layers": len(result["layers"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

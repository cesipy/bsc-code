#!/usr/bin/env python3
"""CPU-only spectrum and paired-geometry controls for cached features.

This is intentionally a direct-file CLI: it does not import the ``analyses``
package, model code, datasets, torch, or the frozen evaluator.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import platform
import resource
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# This tool runs in its own process. Pin common BLAS pools before NumPy loads;
# threadpoolctl below also applies the bound if NumPy was already imported.
CPU_THREAD_LIMIT = 4
for _thread_env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_thread_env] = str(CPU_THREAD_LIMIT)

import numpy as np

try:
    from threadpoolctl import threadpool_limits
except ImportError:  # pragma: no cover - environment variables cap the CLI fallback
    threadpool_limits = None


MAX_MEMORY_BYTES = 1 << 30
MAX_CACHE_UNCOMPRESSED_BYTES = 256 << 20
MAX_SAMPLES = 8192
MAX_LAYERS = 128
MAX_FEATURE_DIM = 8192
MAX_REPORTED_SINGULAR_VALUES = 2_000_000
FLOAT_DTYPES = (np.dtype(np.float32), np.dtype(np.float64))
SCHEMA_VERSION = "autoresearch-spectral-controls-v1"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _archive_uncompressed_bytes(path: Path) -> int:
    if not zipfile.is_zipfile(path):
        raise ValueError("input must be a valid .npz archive")
    with zipfile.ZipFile(path, "r") as archive:
        return sum(member.file_size for member in archive.infolist())


def _canonical_id(value: Any) -> str:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="strict")
    if isinstance(scalar, (float, np.floating)) and not math.isfinite(float(scalar)):
        raise ValueError("sample_ids cannot contain non-finite numeric IDs")
    return str(scalar)


def _json_scalar(value: Any) -> str | int | float | bool:
    scalar = value.item() if isinstance(value, np.generic) else value
    if isinstance(scalar, bytes):
        return scalar.decode("utf-8", errors="strict")
    if isinstance(scalar, (str, int, bool)):
        return scalar
    if isinstance(scalar, float) and math.isfinite(scalar):
        return scalar
    raise ValueError("layer_ids must contain finite JSON scalar values")


def _validate_ids(values: Any, n_samples: int) -> tuple[str, list[str]]:
    ids = np.asarray(values)
    if (
        ids.ndim != 1
        or ids.shape[0] != n_samples
        or ids.dtype.kind not in "biufSU"
    ):
        raise ValueError("sample_ids must be a one-dimensional, non-object ID array of length N")
    canonical = [_canonical_id(item) for item in ids]
    if any(item == "" for item in canonical):
        raise ValueError("sample_ids cannot contain empty IDs")
    if len(set(canonical)) != n_samples:
        raise ValueError("sample_ids must be unique")
    payload = json.dumps(
        canonical, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest(), canonical


def _metadata_hash(archive: Any) -> str | None:
    if "metadata" not in archive.files:
        return None
    values = np.asarray(archive["metadata"])
    if values.ndim != 0 or values.dtype.kind not in "SU":
        raise ValueError("metadata must be a scalar non-object string when present")
    raw = _canonical_id(values.item()).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _load_cache(path: Path) -> dict[str, Any]:
    if path.stat().st_size > MAX_CACHE_UNCOMPRESSED_BYTES:
        raise ValueError("compressed feature cache exceeds the 256 MiB input limit")
    uncompressed_bytes = _archive_uncompressed_bytes(path)
    if uncompressed_bytes > MAX_CACHE_UNCOMPRESSED_BYTES:
        raise ValueError("uncompressed feature cache exceeds the 256 MiB input limit")

    try:
        archive = np.load(path, allow_pickle=False)
    except (OSError, ValueError) as exc:
        raise ValueError(f"could not load feature cache: {exc}") from exc
    with archive:
        keys = set(archive.files)
        text_key = "text_features" if "text_features" in keys else "text"
        vision_key = "vision_features" if "vision_features" in keys else "vision"
        if text_key not in keys or vision_key not in keys:
            raise ValueError("cache must contain text_features and vision_features arrays")
        if "sample_ids" in keys:
            ids_values = archive["sample_ids"]
        elif "ids" in keys:
            ids_values = archive["ids"]
        elif "text_ids" in keys and "vision_ids" in keys:
            text_ids = np.asarray(archive["text_ids"])
            vision_ids = np.asarray(archive["vision_ids"])
            if text_ids.shape != vision_ids.shape or not np.array_equal(text_ids, vision_ids):
                raise ValueError("text_ids and vision_ids must match in order")
            ids_values = text_ids
        else:
            raise ValueError("cache must contain sample_ids (or matching ids/text_ids/vision_ids)")

        text = np.asarray(archive[text_key])
        vision = np.asarray(archive[vision_key])
        if text.ndim != 3 or vision.ndim != 3 or text.shape != vision.shape:
            raise ValueError("text and vision features must have the same [N, L, D] shape")
        if any(size < 1 for size in text.shape):
            raise ValueError("feature arrays must have non-empty sample, layer, and feature axes")
        if text.dtype not in FLOAT_DTYPES or vision.dtype not in FLOAT_DTYPES:
            raise ValueError("text and vision feature arrays must use float32 or float64")
        n_samples, n_layers, feature_dim = map(int, text.shape)
        if n_samples < 2:
            raise ValueError("at least two samples are required")
        if n_samples > MAX_SAMPLES or n_layers > MAX_LAYERS or feature_dim > MAX_FEATURE_DIM:
            raise ValueError(
                f"cache exceeds finite shape limits N<={MAX_SAMPLES}, "
                f"L<={MAX_LAYERS}, D<={MAX_FEATURE_DIM}"
            )
        if not np.isfinite(text).all() or not np.isfinite(vision).all():
            raise ValueError("feature arrays must contain only finite values")

        ids_hash, ids = _validate_ids(ids_values, n_samples)
        if "layer_ids" in keys:
            layer_values = np.asarray(archive["layer_ids"])
            if (
                layer_values.ndim != 1
                or layer_values.shape[0] != n_layers
                or layer_values.dtype.kind not in "biufSU"
            ):
                raise ValueError("layer_ids must be a one-dimensional, non-object array of length L")
            layer_ids = [_json_scalar(item) for item in layer_values]
        else:
            layer_ids = list(range(n_layers))

        return {
            "text": text,
            "vision": vision,
            "text_key": text_key,
            "vision_key": vision_key,
            "sample_ids_hash": ids_hash,
            "n_samples": n_samples,
            "n_layers": n_layers,
            "feature_dim": feature_dim,
            "layer_ids": layer_ids,
            "metadata_hash": _metadata_hash(archive),
            "uncompressed_bytes": int(uncompressed_bytes),
            "array_bytes": int(text.nbytes + vision.nbytes),
        }


def _center_rows(rows: Any, normalize: bool) -> tuple[np.ndarray, int]:
    values = np.asarray(rows, dtype=np.float64).copy()
    if not np.isfinite(values).all():
        raise ValueError("feature values became non-finite during float64 conversion")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(norms).all():
        raise ValueError("feature row norm overflowed float64")
    zero_norm_count = int(np.count_nonzero(norms == 0.0))
    if normalize:
        np.divide(
            values,
            norms[:, None],
            out=values,
            where=norms[:, None] > 0.0,
        )
    values -= values.mean(axis=0, keepdims=True)
    return values, zero_norm_count


def _numerical_rank(singular_values: np.ndarray, shape: tuple[int, int], dtype: np.dtype) -> tuple[int, float, float]:
    largest = float(singular_values[0]) if singular_values.size else 0.0
    relative = float(max(shape) * np.finfo(dtype).eps)
    absolute = largest * relative
    return int(np.count_nonzero(singular_values > absolute)), relative, absolute


def _spectrum_summary(
    centered: np.ndarray,
    *,
    left_vectors: bool = False,
) -> tuple[dict[str, Any], np.ndarray | None, np.ndarray]:
    if left_vectors:
        left, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    else:
        left = None
        singular_values = np.linalg.svd(centered, compute_uv=False)
    singular_values = np.asarray(singular_values, dtype=np.float64)
    if not np.isfinite(singular_values).all():
        raise ValueError("SVD returned non-finite singular values")

    n_samples, feature_dim = centered.shape
    upper_bound = max(0, min(n_samples - 1, feature_dim))
    rank32, rel32, abs32 = _numerical_rank(
        singular_values, centered.shape, np.dtype(np.float32)
    )
    rank64, rel64, abs64 = _numerical_rank(
        singular_values, centered.shape, np.dtype(np.float64)
    )

    if singular_values.size == 0 or singular_values[0] == 0.0:
        summary = {
            "status": "zero_variance",
            "participation_ratio_rank": None,
            "effective_entropy_rank": None,
            "stable_rank": None,
            "pca95_components": 0,
            "pca99_components": 0,
            "singular_values": singular_values.tolist(),
            "numerical_rank_float32": rank32,
            "numerical_rank_float32_relative_threshold": rel32,
            "numerical_rank_float32_absolute_threshold": abs32,
            "numerical_rank_float64": rank64,
            "numerical_rank_float64_relative_threshold": rel64,
            "numerical_rank_float64_absolute_threshold": abs64,
            "algebraic_rank_upper_bound": upper_bound,
        }
        return summary, left, singular_values

    scaled_eigenvalues = np.square(singular_values / singular_values[0])
    eigenvalue_sum = float(np.sum(scaled_eigenvalues, dtype=np.float64))
    eigenvalue_square_sum = float(
        np.sum(np.square(scaled_eigenvalues), dtype=np.float64)
    )
    probabilities = scaled_eigenvalues / eigenvalue_sum
    positive = probabilities > 0.0
    entropy = -math.fsum(
        float(probability * math.log(float(probability)))
        for probability in probabilities[positive]
    )
    cumulative = np.cumsum(probabilities, dtype=np.float64)
    pca95 = int(np.searchsorted(cumulative, 0.95, side="left") + 1)
    pca99 = int(np.searchsorted(cumulative, 0.99, side="left") + 1)
    summary = {
        "status": "ok",
        "participation_ratio_rank": float(eigenvalue_sum**2 / eigenvalue_square_sum),
        "effective_entropy_rank": float(math.exp(entropy)),
        "stable_rank": eigenvalue_sum,
        "pca95_components": pca95,
        "pca99_components": pca99,
        "singular_values": singular_values.tolist(),
        "numerical_rank_float32": rank32,
        "numerical_rank_float32_relative_threshold": rel32,
        "numerical_rank_float32_absolute_threshold": abs32,
        "numerical_rank_float64": rank64,
        "numerical_rank_float64_relative_threshold": rel64,
        "numerical_rank_float64_absolute_threshold": abs64,
        "algebraic_rank_upper_bound": upper_bound,
    }
    return summary, left, singular_values


def _svcca95(
    left_text: np.ndarray | None,
    summary_text: dict[str, Any],
    left_vision: np.ndarray | None,
    summary_vision: dict[str, Any],
) -> dict[str, Any]:
    if left_text is None or left_vision is None:
        raise ValueError("SVCCA requires left singular vectors")
    rank_text = min(
        int(summary_text["pca95_components"]),
        int(summary_text["numerical_rank_float32"]),
    )
    rank_vision = min(
        int(summary_vision["pca95_components"]),
        int(summary_vision["numerical_rank_float32"]),
    )
    if rank_text == 0 or rank_vision == 0:
        return {
            "value": None,
            "status": "zero_variance_after_float32_stable_rank_truncation",
            "retained_text_components": rank_text,
            "retained_vision_components": rank_vision,
            "canonical_correlations": [],
        }

    cross_subspace = left_text[:, :rank_text].T @ left_vision[:, :rank_vision]
    correlations = np.linalg.svd(cross_subspace, compute_uv=False)
    correlations = np.clip(correlations, 0.0, 1.0)
    return {
        "value": float(np.mean(correlations, dtype=np.float64)),
        "status": "ok",
        "retained_text_components": rank_text,
        "retained_vision_components": rank_vision,
        "canonical_correlations": correlations.tolist(),
    }


def _procrustes_similarity(centered_text: np.ndarray, centered_vision: np.ndarray) -> dict[str, Any]:
    norm_text = float(np.linalg.norm(centered_text, ord="fro"))
    norm_vision = float(np.linalg.norm(centered_vision, ord="fro"))
    denominator = norm_text * norm_vision
    if denominator == 0.0:
        return {
            "value": None,
            "status": "zero_variance",
            "formula": "nuclear_norm(Xc.T @ Yc) / (frobenius_norm(Xc) * frobenius_norm(Yc))",
        }
    cross_covariance = centered_text.T @ centered_vision
    singular_values = np.linalg.svd(cross_covariance, compute_uv=False)
    value = float(np.sum(singular_values, dtype=np.float64) / denominator)
    if not math.isfinite(value):
        raise ValueError("Procrustes similarity became non-finite")
    # The ratio is bounded by one by the nuclear/Frobenius norm inequality;
    # clipping only removes floating-point overshoot at the bound.
    value = float(np.clip(value, 0.0, 1.0))
    return {
        "value": value,
        "status": "ok",
        "formula": "nuclear_norm(Xc.T @ Yc) / (frobenius_norm(Xc) * frobenius_norm(Yc))",
    }


def _estimate_peak_bytes(
    uncompressed_cache_bytes: int,
    feature_array_bytes: int,
    n_samples: int,
    feature_dim: int,
    paired_metrics: bool,
) -> int:
    # Covers loaded feature arrays, centered float64 layer copies, thin SVD
    # vectors/workspace, the optional D-by-D cross-covariance, and interpreter.
    d = feature_dim
    m = min(n_samples, d)
    workspace = 8 * (6 * n_samples * d + 4 * m * m)
    if paired_metrics:
        workspace += 8 * (4 * d * d + 4 * n_samples * d + 4 * m * m)
    # NumPy/BLAS and Python's resident baseline can dominate these small
    # feature matrices. Keep a conservative process reserve and room for the
    # in-memory strict-JSON report/serialization as well as array workspaces.
    runtime_reserve = 512 << 20
    report_reserve = 128 << 20
    return int(
        uncompressed_cache_bytes
        + feature_array_bytes
        + workspace
        + runtime_reserve
        + report_reserve
    )


def analyze_cache(
    cache_path: str | Path,
    *,
    normalization: str = "both",
    paired_metrics: bool = True,
    max_memory_bytes: int = MAX_MEMORY_BYTES,
) -> dict[str, Any]:
    """Analyze a paired ``[N,L,D]`` NPZ feature cache without model imports."""
    if normalization not in ("raw", "l2", "both"):
        raise ValueError("normalization must be 'raw', 'l2', or 'both'")
    if isinstance(max_memory_bytes, bool) or not isinstance(max_memory_bytes, int):
        raise ValueError("max_memory_bytes must be an integer")
    if not 1 <= max_memory_bytes <= MAX_MEMORY_BYTES:
        raise ValueError("max_memory_bytes must be between 1 byte and 1 GiB")

    path = Path(cache_path).expanduser().resolve(strict=True)
    if not path.is_file():
        raise ValueError("cache_path must name a regular file")
    loaded = _load_cache(path)
    n_samples = loaded["n_samples"]
    n_layers = loaded["n_layers"]
    feature_dim = loaded["feature_dim"]
    view_names = ("raw", "l2_normalized") if normalization == "both" else (
        "raw" if normalization == "raw" else "l2_normalized",
    )
    reported_values = (
        len(view_names) * 2 * n_layers * min(n_samples, feature_dim)
    )
    if reported_values > MAX_REPORTED_SINGULAR_VALUES:
        raise ValueError(
            "requested singular-spectrum report exceeds the finite 2,000,000-value limit"
        )

    estimated_peak = _estimate_peak_bytes(
        loaded["uncompressed_bytes"],
        loaded["array_bytes"],
        n_samples,
        feature_dim,
        paired_metrics,
    )
    if estimated_peak > max_memory_bytes:
        raise ValueError(
            f"estimated peak memory {estimated_peak} exceeds limit {max_memory_bytes} bytes"
        )

    started = time.perf_counter()
    view_results: dict[str, Any] = {}
    for view_name in view_names:
        normalize = view_name == "l2_normalized"
        stream_records: dict[str, list[dict[str, Any]]] = {"text": [], "vision": []}
        final: dict[str, Any] | None = None
        for layer_index, layer_id in enumerate(loaded["layer_ids"]):
            text_centered, text_zero_count = _center_rows(
                loaded["text"][:, layer_index, :], normalize
            )
            vision_centered, vision_zero_count = _center_rows(
                loaded["vision"][:, layer_index, :], normalize
            )
            is_final_for_metrics = paired_metrics and layer_index == n_layers - 1
            text_summary, text_left, text_singulars = _spectrum_summary(
                text_centered, left_vectors=is_final_for_metrics
            )
            vision_summary, vision_left, vision_singulars = _spectrum_summary(
                vision_centered, left_vectors=is_final_for_metrics
            )
            text_record = {
                "layer_id": layer_id,
                "n_samples": n_samples,
                "feature_dim": feature_dim,
                "zero_norm_sample_count_before_view": text_zero_count,
                **text_summary,
            }
            vision_record = {
                "layer_id": layer_id,
                "n_samples": n_samples,
                "feature_dim": feature_dim,
                "zero_norm_sample_count_before_view": vision_zero_count,
                **vision_summary,
            }
            stream_records["text"].append(text_record)
            stream_records["vision"].append(vision_record)
            if is_final_for_metrics:
                final = {
                    "text_centered": text_centered,
                    "vision_centered": vision_centered,
                    "text_left": text_left,
                    "vision_left": vision_left,
                    "text_singulars": text_singulars,
                    "vision_singulars": vision_singulars,
                    "text_summary": text_summary,
                    "vision_summary": vision_summary,
                }

        paired: dict[str, Any] | None = None
        if paired_metrics:
            if final is None:
                raise RuntimeError("final-layer features were not retained for paired metrics")
            paired = {
                "layer_id": loaded["layer_ids"][-1],
                "svcca95": _svcca95(
                    final["text_left"],
                    final["text_summary"],
                    final["vision_left"],
                    final["vision_summary"],
                ),
                "centered_normalized_orthogonal_procrustes_similarity": _procrustes_similarity(
                    final["text_centered"], final["vision_centered"]
                ),
            }
        view_results[view_name] = {
            "preprocessing": (
                "center columns across samples in float64; no row scaling"
                if not normalize
                else "L2-normalize each sample row per layer in float64 (zero rows stay zero), "
                "then center columns across samples in float64"
            ),
            "text": stream_records["text"],
            "vision": stream_records["vision"],
            "paired_final_layer_metrics": paired,
        }

    elapsed = time.perf_counter() - started
    input_hash = _sha256_file(path)
    source_path = Path(__file__).resolve()
    source_hash = _sha256_file(source_path)
    return {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input": {
            "cache_path": str(path),
            "cache_sha256": input_hash,
            "cache_uncompressed_bytes": loaded["uncompressed_bytes"],
            "feature_keys": {
                "text": loaded["text_key"],
                "vision": loaded["vision_key"],
            },
            "feature_shape": [n_samples, n_layers, feature_dim],
            "feature_dtypes": {
                "text": str(loaded["text"].dtype),
                "vision": str(loaded["vision"].dtype),
            },
            "layer_ids": loaded["layer_ids"],
            "sample_id_count": n_samples,
            "sample_ids_sha256": loaded["sample_ids_hash"],
            "metadata_sha256": loaded["metadata_hash"],
        },
        "method": {
            "centering": "subtract per-feature sample mean",
            "working_precision": "float64 SVD and covariance calculations",
            "covariance_eigenvalues": "singular_values**2 / (n_samples - 1)",
            "participation_ratio_rank": "(sum(lambda))**2 / sum(lambda**2)",
            "effective_entropy_rank": "exp(-sum(p*log(p))) for normalized covariance eigenvalues p",
            "stable_rank": "sum(singular_values**2) / largest_singular_value**2",
            "pca_components": "smallest component count reaching 95% or 99% centered variance",
            "numerical_rank": (
                "count singular values > s_max * max(n_samples, feature_dim) * eps(dtype); "
                "the same float64 SVD spectrum is thresholded with float32 and float64 eps"
            ),
            "svcca95": (
                "exact SVD-based CCA of whitened left singular subspaces after per-stream PCA95; "
                "retained rank is capped by float32-threshold numerical rank; report mean canonical correlation"
            ),
            "procrustes_similarity": (
                "nuclear_norm(Xc.T @ Yc) / (frobenius_norm(Xc) * frobenius_norm(Yc)); "
                "centered features, bounded [0,1], zero-variance result is null"
            ),
        },
        "configuration": {
            "normalization_views": list(view_names),
            "paired_metrics_enabled": bool(paired_metrics),
            "cpu_thread_limit": CPU_THREAD_LIMIT,
            "max_memory_bytes": int(max_memory_bytes),
            "estimated_peak_memory_bytes": estimated_peak,
            "runtime_seconds": float(elapsed),
        },
        "results": view_results,
        "provenance": {
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "platform": platform.platform(),
        },
    }


def _peak_rss_bytes() -> int | None:
    try:
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except (AttributeError, ValueError):
        return None
    # Linux reports KiB; macOS reports bytes. This tool runs on Linux, but the
    # branch keeps the output unit correct on a developer workstation too.
    return int(rss * 1024 if sys.platform.startswith("linux") else rss)


def _write_immutable_json(path: str | Path, report: dict[str, Any]) -> Path:
    output = Path(path).expanduser().resolve()
    payload = json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(output, flags, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(payload)
    return output


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cache", type=Path, help="offline paired-feature .npz cache")
    parser.add_argument("--output", type=Path, required=True, help="create-only JSON output path")
    parser.add_argument(
        "--normalization", choices=("raw", "l2", "both"), default="both",
        help="analyze raw features, per-row L2-normalized features, or both",
    )
    parser.add_argument(
        "--paired-metrics", action=argparse.BooleanOptionalAction, default=True,
        help="include exact final-layer SVCCA95 and normalized Procrustes similarity",
    )
    parser.add_argument(
        "--max-memory-mib", type=int, default=1024,
        help="estimated peak-memory ceiling (1–1024 MiB; hard maximum 1 GiB)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not 1 <= args.max_memory_mib <= 1024:
        parser.error("--max-memory-mib must be between 1 and 1024")
    try:
        input_path = args.cache.expanduser().resolve(strict=True)
        output_path = args.output.expanduser().resolve()
        if input_path == output_path:
            raise ValueError("output must not overwrite the input cache")
        start = time.perf_counter()
        limiter = (
            threadpool_limits(limits=CPU_THREAD_LIMIT)
            if threadpool_limits is not None
            else contextlib.nullcontext()
        )
        with limiter:
            report = analyze_cache(
                input_path,
                normalization=args.normalization,
                paired_metrics=args.paired_metrics,
                max_memory_bytes=args.max_memory_mib << 20,
            )
        report["configuration"]["wall_time_seconds"] = float(time.perf_counter() - start)
        report["configuration"]["peak_rss_bytes"] = _peak_rss_bytes()
        written = _write_immutable_json(output_path, report)
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        parser.error(str(exc))
    print(json.dumps({
        "status": "completed",
        "output": str(written),
        "cache_sha256": report["input"]["cache_sha256"],
        "estimated_peak_memory_bytes": report["configuration"]["estimated_peak_memory_bytes"],
        "peak_rss_bytes": report["configuration"]["peak_rss_bytes"],
        "wall_time_seconds": report["configuration"]["wall_time_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

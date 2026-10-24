#!/usr/bin/env python3
"""CPU-only layer-wise dimensionality analysis of cached CLS features.

Direct-file CLI. It reads the immutable ``[N, L, D]`` feature caches written by
the evaluator, never imports torch, model code or datasets, and writes new files
only below ``--output-dir`` (create-only).

For every cache, stream, layer and view it reports participation-ratio rank (PR),
entropy rank, stable rank, PCA component counts (90/95/99% of variance), the
variance share of the first principal component and of the largest single
channel. Two views are computed: ``raw`` (columns centered, as in the thesis
PCA@95%) and ``standardized`` (each column also divided by its standard deviation,
which removes differences in channel scale). Metrics are also recomputed on
paired random subsets of the rows to show sample-size dependence and noise, and
step changes at configured cross-attention layers are tabulated against the step
changes at the other layers of the same stream.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

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

SCHEMA_VERSION = "autoresearch-collapse-analysis-v1"
VIEWS = ("raw", "standardized")
STREAMS = ("text", "vision")
METRICS = (
    "participation_ratio",
    "entropy_rank",
    "stable_rank",
    "pca90",
    "pca95",
    "pca99",
    "top1_pc_share",
)
DEFAULT_TRIALS = (
    "robustness-no-ca-2025-repair1",
    "robustness-early-2025",
    "early1570",
    "early1571",
    "df1",
    "robustness-df3-2026",
    "df2",
    "robustness-rev-2026",
    "late1570",
    "late1571-repeat",
    "robustness-late-2025",
    "task-mm-df1",
    "task-mm-df2",
    "task-mm-rev",
    "task-upmc-df1",
    "task-upmc-df2",
    "task-upmc-rev",
)
MAX_CACHE_BYTES = 256 << 20


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def family_of(text_ids: list[int], vision_ids: list[int]) -> str:
    """Name the fusion schedule from the configured zero-based layer ids."""
    t, v = sorted(text_ids), sorted(vision_ids)
    if not t and not v:
        return "no-cross-attention"
    if t == v == [3, 4, 5]:
        return "early"
    if t == v == [9, 10, 11]:
        return "late"
    if t == [3, 4, 5]:
        return "delayed-vision"
    if t == [9, 10, 11] and v == [3, 4, 5]:
        return "reverse"
    return "other"


def load_cache(path: Path, name: str) -> dict[str, Any]:
    """Load one evaluator cache and the fusion schedule stored in its metadata."""
    if path.stat().st_size > MAX_CACHE_BYTES:
        raise ValueError(f"{path} exceeds the {MAX_CACHE_BYTES} byte input bound")
    with np.load(path, allow_pickle=False) as archive:
        text = np.asarray(archive["text_features"], dtype=np.float64)
        vision = np.asarray(archive["vision_features"], dtype=np.float64)
        layer_ids = [int(x) for x in archive["layer_ids"]]
        sample_ids = [str(x) for x in archive["sample_ids"]]
        metadata = json.loads(str(archive["metadata"]))
    if text.shape != vision.shape or text.ndim != 3 or text.shape[0] != len(sample_ids):
        raise ValueError(f"{path}: inconsistent feature shapes")
    if not (np.isfinite(text).all() and np.isfinite(vision).all()):
        raise ValueError(f"{path}: non-finite feature values")
    if layer_ids != list(range(text.shape[1])):
        raise ValueError(f"{path}: layer ids are not 0..L-1")
    config = metadata["checkpoint_architecture"]["saved_checkpoint_config"]
    text_ids = [int(x) for x in config["text_cross_attention_layers"]]
    vision_ids = [int(x) for x in config["vision_cross_attention_layers"]]
    return {
        "name": name,
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "task": str(metadata["task"]),
        "text": text,
        "vision": vision,
        "sample_ids": sample_ids,
        "text_ids": text_ids,
        "vision_ids": vision_ids,
        "family": family_of(text_ids, vision_ids),
        "checkpoint_sha256": metadata.get("cache_identity", {}).get("checkpoint_sha256"),
        "config_seed": config.get("seed"),
    }


def center(matrix: np.ndarray) -> np.ndarray:
    return matrix - matrix.mean(axis=0, keepdims=True)


def standardize(centered: np.ndarray) -> tuple[np.ndarray, int]:
    """Divide centered columns by their standard deviation; drop constant columns."""
    scale = centered.std(axis=0)
    keep = scale > 1e-12 * max(float(scale.max()), 1e-300)
    return centered[:, keep] / scale[keep], int(keep.sum())


def spectrum_metrics(centered: np.ndarray) -> dict[str, float]:
    """Rank summaries of the covariance spectrum of an already centered matrix."""
    values = np.linalg.svd(centered, compute_uv=False)
    eigenvalues = values * values
    total = float(eigenvalues.sum())
    if not math.isfinite(total) or total <= 0.0:
        return {name: float("nan") for name in METRICS}
    share = eigenvalues / total
    cumulative = np.cumsum(eigenvalues) / total
    positive = share[share > 0.0]
    return {
        "participation_ratio": total * total / float((eigenvalues * eigenvalues).sum()),
        "entropy_rank": float(np.exp(-(positive * np.log(positive)).sum())),
        "stable_rank": total / float(eigenvalues[0]),
        "pca90": float(np.searchsorted(cumulative, 0.90, side="left") + 1),
        "pca95": float(np.searchsorted(cumulative, 0.95, side="left") + 1),
        "pca99": float(np.searchsorted(cumulative, 0.99, side="left") + 1),
        "top1_pc_share": float(share[0]),
    }


def view_metrics(matrix: np.ndarray, view: str) -> dict[str, float]:
    centered = center(matrix)
    if view == "raw":
        out = spectrum_metrics(centered)
        variance = centered.var(axis=0)
        total = float(variance.sum())
        out["top_channel_share"] = float(variance.max() / total) if total > 0 else float("nan")
        out["n_features_used"] = float(matrix.shape[1])
        return out
    if view == "standardized":
        scaled, used = standardize(centered)
        out = spectrum_metrics(scaled) if used else {name: float("nan") for name in METRICS}
        out["top_channel_share"] = float("nan")
        out["n_features_used"] = float(used)
        return out
    raise ValueError(f"unknown view {view!r}")


def draw_subsets(n_samples: int, sizes: tuple[int, ...], reps: int, seed: int) -> dict[int, np.ndarray]:
    """Row subsets without replacement; identical for every cache with the same rows."""
    subsets: dict[int, np.ndarray] = {}
    for size in sizes:
        if not 2 < size < n_samples:
            raise ValueError(f"subset size {size} must lie in (2, {n_samples})")
        rng = np.random.default_rng([seed, size])
        subsets[size] = np.stack([np.sort(rng.choice(n_samples, size, replace=False)) for _ in range(reps)])
    return subsets


def analyze_cache(
    cache: dict[str, Any],
    sizes: tuple[int, ...],
    subsets: dict[int, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Return full-sample rows and subsample summary rows for one cache."""
    full_rows: list[dict[str, Any]] = []
    sub_rows: list[dict[str, Any]] = []
    n_layers = cache["text"].shape[1]
    for stream in STREAMS:
        features = cache[stream]
        for layer in range(n_layers):
            matrix = features[:, layer, :]
            for view in VIEWS:
                base = {
                    "checkpoint": cache["name"],
                    "family": cache["family"],
                    "task": cache["task"],
                    "stream": stream,
                    "layer": layer,
                    "view": view,
                }
                full_rows.append({**base, "n": matrix.shape[0], **view_metrics(matrix, view)})
                for size in sizes:
                    draws = [view_metrics(matrix[index], view) for index in subsets[size]]
                    for metric in METRICS:
                        series = np.array([d[metric] for d in draws], dtype=np.float64)
                        sub_rows.append(
                            {
                                **base,
                                "n": size,
                                "metric": metric,
                                "mean": float(np.nanmean(series)),
                                "sd": float(np.nanstd(series, ddof=1)),
                                "reps": len(draws),
                            }
                        )
    return full_rows, sub_rows


def step_table(cache: dict[str, Any], full_rows: list[dict[str, Any]], metrics: tuple[str, ...]) -> list[dict[str, Any]]:
    """Log-ratio of each metric between a layer and the previous layer.

    Layer ``k`` is the output of block ``k`` and a configured cross-attention
    index ``k`` acts before block ``k``, so the step from ``k-1`` to ``k`` is the
    first change that can contain that cross-attention operation.
    """
    table: list[dict[str, Any]] = []
    for stream, ids in (("text", cache["text_ids"]), ("vision", cache["vision_ids"])):
        for view in VIEWS:
            by_layer = {r["layer"]: r for r in full_rows if r["stream"] == stream and r["view"] == view}
            for metric in metrics:
                for layer in range(1, len(by_layer)):
                    prev, cur = by_layer[layer - 1][metric], by_layer[layer][metric]
                    if not (prev > 0 and cur > 0):
                        step = float("nan")
                    else:
                        step = math.log(cur / prev)
                    table.append(
                        {
                            "checkpoint": cache["name"],
                            "family": cache["family"],
                            "task": cache["task"],
                            "stream": stream,
                            "view": view,
                            "metric": metric,
                            "layer": layer,
                            "is_cross_attention_step": layer in ids,
                            "is_first_cross_attention_step": bool(ids) and layer == min(ids),
                            "log_ratio": step,
                        }
                    )
    return table


def validate_against_spectrum(caches: list[dict[str, Any]], full_rows: list[dict[str, Any]], directory: Path) -> list[dict[str, Any]]:
    """Compare raw-view metrics with the earlier spectrum-control reports, matched by cache hash."""
    checks: list[dict[str, Any]] = []
    reports = {}
    for path in sorted(directory.glob("*.json")):
        report = json.loads(path.read_text())
        reports[report["input"]["cache_sha256"]] = (path.name, report)
    keys = {
        "participation_ratio": "participation_ratio_rank",
        "entropy_rank": "effective_entropy_rank",
        "stable_rank": "stable_rank",
        "pca95": "pca95_components",
    }
    for cache in caches:
        if cache["file_sha256"] not in reports:
            continue
        name, report = reports[cache["file_sha256"]]
        worst = {metric: 0.0 for metric in keys}
        for stream in STREAMS:
            for entry in report["results"]["raw"][stream]:
                row = next(
                    r for r in full_rows
                    if r["checkpoint"] == cache["name"] and r["stream"] == stream
                    and r["layer"] == int(entry["layer_id"]) and r["view"] == "raw"
                )
                for metric, key in keys.items():
                    worst[metric] = max(worst[metric], abs(row[metric] - float(entry[key])))
        checks.append({"checkpoint": cache["name"], "reference": name, "max_abs_difference": worst})
    return checks


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = list(rows[0].keys())
    with path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if isinstance(v, float) and math.isnan(v) else v) for k, v in row.items()})


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    return value


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=False)
    sizes = tuple(int(s) for s in args.sizes)
    caches: list[dict[str, Any]] = []
    for trial in args.trial:
        found = sorted((args.trials_root / trial / "features").glob("features-*.npz"))
        if len(found) != 1:
            raise FileNotFoundError(f"{trial}: expected exactly one feature cache, found {len(found)}")
        caches.append(load_cache(found[0], trial))
    reference_ids: dict[str, list[str]] = {}
    for cache in caches:
        reference_ids.setdefault(cache["task"], cache["sample_ids"])
        if cache["sample_ids"] != reference_ids[cache["task"]]:
            raise ValueError(f"{cache['name']}: sample ids differ from the other {cache['task']} caches")
    subsets_by_task = {
        task: draw_subsets(len(ids), sizes, args.reps, args.seed) for task, ids in reference_ids.items()
    }

    all_full: list[dict[str, Any]] = []
    all_sub: list[dict[str, Any]] = []
    all_steps: list[dict[str, Any]] = []
    for cache in caches:
        full_rows, sub_rows = analyze_cache(cache, sizes, subsets_by_task[cache["task"]])
        all_full += full_rows
        all_sub += sub_rows
        all_steps += step_table(cache, full_rows, ("participation_ratio", "pca95", "entropy_rank"))
        print(f"{cache['name']}: done ({time.time() - started:.0f}s)", flush=True)

    _write_csv(output_dir / "collapse_metrics_full.csv", all_full)
    _write_csv(output_dir / "collapse_metrics_subsample.csv", all_sub)
    _write_csv(output_dir / "collapse_steps.csv", all_steps)
    validation = (
        validate_against_spectrum(caches, all_full, args.validate_spectrum_dir)
        if args.validate_spectrum_dir
        else []
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": _sha256_file(Path(__file__)),
        "parameters": {"sizes": list(sizes), "reps": args.reps, "seed": args.seed, "views": list(VIEWS)},
        "definitions": {
            "centering": "subtract per-feature sample mean (per subset for subsamples)",
            "participation_ratio": "(sum lambda)^2 / sum lambda^2 over covariance eigenvalues",
            "entropy_rank": "exp(Shannon entropy of lambda / sum lambda)",
            "stable_rank": "sum lambda / lambda_max",
            "pca_k": "smallest component count whose cumulative variance share reaches k%",
            "standardized": "centered columns divided by their sample standard deviation; constant columns dropped",
            "top_channel_share": "largest single-column variance divided by total variance (raw view only)",
            "step": "ln(metric at layer k / metric at layer k-1); layer k is the output of block k",
        },
        "inputs": [
            {
                "checkpoint": c["name"],
                "task": c["task"],
                "family": c["family"],
                "text_cross_attention_ids": c["text_ids"],
                "vision_cross_attention_ids": c["vision_ids"],
                "config_seed": c["config_seed"],
                "cache_path": c["path"],
                "cache_sha256": c["file_sha256"],
                "checkpoint_sha256": c["checkpoint_sha256"],
                "n_samples": len(c["sample_ids"]),
            }
            for c in caches
        ],
        "validation_against_spectrum_reports": validation,
        "runtime_seconds": time.time() - started,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "cpu_thread_limit": CPU_THREAD_LIMIT,
    }
    with (output_dir / "collapse_summary.json").open("x") as stream:
        json.dump(_json_safe(summary), stream, indent=2, allow_nan=False)
        stream.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials-root", type=Path, default=Path("autoresearch/trials"))
    parser.add_argument("--trial", action="append", help="trial directory name; default is the 17-cache cohort")
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; must not exist")
    parser.add_argument("--sizes", type=int, nargs="+", default=[125, 250, 375])
    parser.add_argument("--reps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20261023)
    parser.add_argument("--validate-spectrum-dir", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.trial:
        args.trial = list(DEFAULT_TRIALS)
    run(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

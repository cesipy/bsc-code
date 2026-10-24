#!/usr/bin/env python3
"""CPU-only label-based retention probe on cached CLS features.

Direct-file CLI. For each cache, stream and layer it asks how much of a task label a
regularised linear probe can recover from (a) all standardized features and (b) only
the top-k principal directions. Standardization, PCA and the inner choice of the
regularisation strength are fitted on the training part of each cross-validation fold
only. Labels are read from the dataset files named in the cache metadata (read-only);
the caches themselves stay label-free.

Reported: pooled out-of-fold ROC-AUC per target with a row-bootstrap interval, and the
fold-to-fold spread. Binary targets use ROC-AUC; multi-label targets (MM-IMDb genres)
are summarised per genre. Nothing here is a significance test.
"""
from __future__ import annotations

import argparse
import ast
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
for _thread_env in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_thread_env] = str(CPU_THREAD_LIMIT)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

SCHEMA_VERSION = "autoresearch-retention-probe-v1"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0)
INNER_FOLDS = 3
MAX_CACHE_BYTES = 1 << 30


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_cache(path: Path, name: str) -> dict[str, Any]:
    if path.stat().st_size > MAX_CACHE_BYTES:
        raise ValueError(f"{path} exceeds the {MAX_CACHE_BYTES} byte input bound")
    with np.load(path, allow_pickle=False) as archive:
        text = np.asarray(archive["text_features"], dtype=np.float64)
        vision = np.asarray(archive["vision_features"], dtype=np.float64)
        sample_ids = [str(x) for x in archive["sample_ids"]]
        metadata = json.loads(str(archive["metadata"]))
    if text.shape != vision.shape or text.shape[0] != len(sample_ids):
        raise ValueError(f"{path}: inconsistent shapes")
    if not (np.isfinite(text).all() and np.isfinite(vision).all()):
        raise ValueError(f"{path}: non-finite features")
    config = metadata["checkpoint_architecture"]["saved_checkpoint_config"]
    return {
        "name": name, "path": str(path), "file_sha256": _sha256_file(path), "task": str(metadata["task"]),
        "data_root": metadata.get("data_root"), "text": text, "vision": vision, "sample_ids": sample_ids,
        "text_ids": [int(x) for x in config["text_cross_attention_layers"]],
        "vision_ids": [int(x) for x in config["vision_cross_attention_layers"]],
    }


def _key_from_id(task: str, sample_id: str) -> str:
    parts = sample_id.split(":")
    if task == "hateful_memes":
        return str(int(parts[1]))
    if task in ("mm_imdb", "upmc_food"):
        return parts[1]
    raise ValueError(f"unsupported task {task!r}")


def load_labels(task: str, data_root: Path, sample_ids: list[str]) -> dict[str, Any]:
    """Read labels for the cached rows from the dataset files (read-only)."""
    keys = [_key_from_id(task, s) for s in sample_ids]
    if task == "hateful_memes":
        table = {}
        with (data_root / "dev_seen.jsonl").open() as stream:
            for line in stream:
                row = json.loads(line)
                table[str(int(row["id"]))] = int(row["label"])
        missing = [k for k in keys if k not in table]
        if missing:
            raise KeyError(f"{len(missing)} cached rows have no label, e.g. {missing[:3]}")
        return {"kind": "binary", "targets": {"hateful": np.array([table[k] for k in keys], dtype=int)}}
    if task == "mm_imdb":
        table = {}
        with (data_root / "mmimdb_trainval.csv").open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                table[str(int(row["img_index"]))] = ast.literal_eval(row["genre"])
        missing = [k for k in keys if k not in table]
        if missing:
            raise KeyError(f"{len(missing)} cached rows have no label, e.g. {missing[:3]}")
        matrix = np.array([table[k] for k in keys], dtype=int)
        return {"kind": "multilabel", "targets": {f"genre{j:02d}": matrix[:, j] for j in range(matrix.shape[1])}}
    if task == "upmc_food":
        table = {}
        with (data_root / "upmcfood_trainval.csv").open(newline="", encoding="utf-8-sig") as stream:
            for row in csv.DictReader(stream):
                table.setdefault(row["img_file"], row["class"])
        missing = [k for k in keys if k not in table]
        if missing:
            raise KeyError(f"{len(missing)} cached rows have no label, e.g. {missing[:3]}")
        return {"kind": "multiclass", "classes": [table[k] for k in keys]}
    raise ValueError(f"unsupported task {task!r}")


def select_targets(labels: dict[str, Any], min_each: int) -> dict[str, np.ndarray]:
    """Binary targets with at least ``min_each`` positives and negatives."""
    if labels["kind"] == "multiclass":
        names = np.array(labels["classes"])
        out = {}
        for cls in sorted(set(names)):
            y = (names == cls).astype(int)
            if y.sum() >= min_each and (1 - y).sum() >= min_each:
                out[f"class:{cls}"] = y
        return out
    return {k: v for k, v in labels["targets"].items() if v.sum() >= min_each and (1 - v).sum() >= min_each}


def make_folds(y: np.ndarray, n_folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return [(tr, te) for tr, te in splitter.split(np.zeros(len(y)), y)]


SVD_FALLBACKS = 0


def _svd(a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Singular values and right vectors; the slower, more robust gesvd driver if gesdd fails to converge."""
    global SVD_FALLBACKS
    try:
        _, s, vt = np.linalg.svd(a, full_matrices=False)
    except np.linalg.LinAlgError:
        from scipy.linalg import svd as scipy_svd

        SVD_FALLBACKS += 1
        _, s, vt = scipy_svd(a, full_matrices=False, lapack_driver="gesvd")
    return s, vt


def _fit_transform(train: np.ndarray, test: np.ndarray, k: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Standardize (and optionally whiten the top-k PCs) using training rows only."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    keep = scale > 1e-12 * max(float(scale.max()), 1e-300)
    a = (train[:, keep] - mean[keep]) / scale[keep]
    b = (test[:, keep] - mean[keep]) / scale[keep]
    if k is None:
        return a, b
    s, vt = _svd(a)
    k = min(k, int((s > 1e-10 * s[0]).sum()))
    comp = vt[:k]
    sd = s[:k] / math.sqrt(max(a.shape[0] - 1, 1))
    return (a @ comp.T) / sd, (b @ comp.T) / sd


def _fit_predict(train_x: np.ndarray, train_y: np.ndarray, test_x: np.ndarray, seed: int) -> np.ndarray:
    """Logistic probe; C chosen on the training rows only by inner stratified CV (ROC-AUC)."""
    inner = StratifiedKFold(n_splits=INNER_FOLDS, shuffle=True, random_state=seed)
    scores = np.zeros(len(C_GRID))
    for tr, va in inner.split(train_x, train_y):
        if len(set(train_y[tr])) < 2 or len(set(train_y[va])) < 2:
            continue
        for i, c in enumerate(C_GRID):
            model = LogisticRegression(C=c, max_iter=300).fit(train_x[tr], train_y[tr])
            scores[i] += roc_auc_score(train_y[va], model.decision_function(train_x[va]))
    best = C_GRID[int(np.argmax(scores))]
    return LogisticRegression(C=best, max_iter=300).fit(train_x, train_y).decision_function(test_x)


def probe_auc(features: np.ndarray, y: np.ndarray, folds, k: int | None, seed: int) -> dict[str, Any]:
    """Pooled out-of-fold ROC-AUC and fold spread for one feature matrix and target."""
    oof = np.zeros(len(y))
    fold_auc = []
    for tr, te in folds:
        a, b = _fit_transform(features[tr], features[te], k)
        oof[te] = _fit_predict(a, y[tr], b, seed)
        fold_auc.append(roc_auc_score(y[te], oof[te]))
    return {"oof": oof, "auroc": float(roc_auc_score(y, oof)), "fold_mean": float(np.mean(fold_auc)),
            "fold_sd": float(np.std(fold_auc, ddof=1))}


def bootstrap_interval(y: np.ndarray, oof: np.ndarray, reps: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(reps):
        idx = rng.integers(0, len(y), len(y))
        if len(set(y[idx])) == 2:
            draws.append(roc_auc_score(y[idx], oof[idx]))
    return float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=False)
    ks: list[int | None] = [None if k == 0 else k for k in args.ks]
    layers = args.layers
    rows: list[dict[str, Any]] = []
    inputs = []
    written = 0
    csv_stream =(out_dir / "retention_probe.csv").open("x", newline="")
    writer: csv.DictWriter | None = None
    for trial in args.trial:
        found = sorted((args.trials_root / trial / "features").glob("features-*.npz"))
        if len(found) != 1:
            raise FileNotFoundError(f"{trial}: expected exactly one feature cache, found {len(found)}")
        cache = load_cache(found[0], trial)
        root = Path(args.data_root or cache["data_root"])
        labels = load_labels(cache["task"], root, cache["sample_ids"])
        targets = select_targets(labels, args.min_each)
        if not targets:
            raise ValueError(f"{trial}: no target has {args.min_each} positives and negatives")
        folds = {name: make_folds(y, args.folds, args.seed) for name, y in targets.items()}
        inputs.append({"checkpoint": trial, "task": cache["task"], "cache_sha256": cache["file_sha256"],
                       "n": len(cache["sample_ids"]), "targets": {n: int(y.sum()) for n, y in targets.items()},
                       "label_root": str(root), "vision_cross_attention_ids": cache["vision_ids"]})
        for stream in args.streams:
            for layer in layers:
                matrix = cache[stream][:, layer, :]
                for k in ks:
                    for name, y in targets.items():
                        res = probe_auc(matrix, y, folds[name], k, args.seed)
                        lo, hi = bootstrap_interval(y, res["oof"], args.bootstrap, args.seed)
                        rows.append({"checkpoint": trial, "task": cache["task"], "stream": stream, "layer": layer,
                                     "k": 0 if k is None else k, "target": name, "n": len(y), "n_pos": int(y.sum()),
                                     "auroc_oof": res["auroc"], "ci_lo": lo, "ci_hi": hi,
                                     "fold_mean": res["fold_mean"], "fold_sd": res["fold_sd"]})
        if writer is None:
            writer = csv.DictWriter(csv_stream, fieldnames=list(rows[0].keys()))
            writer.writeheader()
        writer.writerows(rows[written:])  # one checkpoint at a time, so a crash keeps the finished ones
        written = len(rows)
        csv_stream.flush()
        print(f"{trial}: done ({time.time() - started:.0f}s)", flush=True)
    csv_stream.close()
    summary = {
        "schema_version": SCHEMA_VERSION, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "script_sha256": _sha256_file(Path(__file__)),
        "parameters": {"folds": args.folds, "seed": args.seed, "ks_0_means_all_features": list(args.ks), "layers": list(layers),
                       "streams": list(args.streams), "min_each": args.min_each, "bootstrap": args.bootstrap,
                       "c_grid": list(C_GRID), "inner_folds": INNER_FOLDS},
        "definitions": {
            "probe": "L2 logistic regression on features standardized with training-fold statistics; C chosen by inner stratified CV on the training fold",
            "top_k": "training-fold PCA of the standardized features, top-k components whitened",
            "auroc_oof": "ROC-AUC of the pooled out-of-fold decision scores",
            "ci": "2.5-97.5 percentile of row-bootstrap resamples of the out-of-fold scores (does not re-fit the probe)",
        },
        "inputs": inputs, "runtime_seconds": time.time() - started, "svd_gesvd_fallbacks": SVD_FALLBACKS,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform()},
        "cpu_thread_limit": CPU_THREAD_LIMIT,
    }
    with (out_dir / "retention_summary.json").open("x") as stream:
        json.dump(summary, stream, indent=2)
        stream.write("\n")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials-root", type=Path, default=Path("autoresearch/trials"))
    parser.add_argument("--trial", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True, help="new directory; must not exist")
    parser.add_argument("--data-root", type=Path, default=None, help="override the data root stored in the cache metadata")
    parser.add_argument("--streams", nargs="+", default=["vision"], choices=["text", "vision"])
    parser.add_argument("--layers", type=int, nargs="+", default=list(range(12)))
    parser.add_argument("--ks", type=int, nargs="+", default=[0, 20, 5], help="0 means all features")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--min-each", type=int, default=50, help="minimum positives and negatives for a target")
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--seed", type=int, default=20261023)
    return parser


def main(argv: list[str] | None = None) -> int:
    run(build_parser().parse_args(argv))
    return 0


if __name__ == "__main__":
    sys.exit(main())

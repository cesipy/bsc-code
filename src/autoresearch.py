"""Checkpoint-only validation evaluator for the ViLBERT autoresearch pilot.

The CLI has two operations: freeze a deterministic, ordered validation-ID
manifest, then extract representations from an existing checkpoint and score
those cached features. It never imports the training, test-loader, or
performance-collector code paths. Runtime dependencies are imported only after
the requested command has passed its path and manifest checks.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path
from typing import Any


EVALUATOR_VERSION = "checkpoint-validation-evaluator-v1"
EXTRACTION_VERSION = "vilbert-cls-layer-pair-v1"
MAX_BATCH_SIZE = 64
MAX_PERMUTATIONS = 10000
READ_ONLY_ROOTS = (
    Path("/home/cedric.sillaber/coding/bsc-code"),
    Path("/home/cedric.sillaber/coding/bsc-code-research-material"),
)


class EvaluatorError(ValueError):
    """Raised when a requested evaluation violates the frozen protocol."""


@contextmanager
def _working_directory(path: Path):
    """Keep legacy relative log/cache side effects inside the output tree."""
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    from analyses.autoresearch_data import sha256_file

    return sha256_file(path)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    enum_value = getattr(value, "value", None)
    if enum_value is not None and enum_value is not value:
        return _jsonable(enum_value)
    scalar_value = getattr(value, "item", None)
    if callable(scalar_value):
        try:
            return _jsonable(scalar_value())
        except (TypeError, ValueError):
            pass
    return str(value)


def authorized_output_root(project_root: str | Path | None = None) -> Path:
    root = Path(project_root) if project_root is not None else Path(__file__).resolve().parents[1]
    root = root.resolve(strict=True)
    output_root = root / "autoresearch"
    if output_root.is_symlink():
        raise EvaluatorError("the project autoresearch output root must not be a symlink")
    try:
        resolved = output_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvaluatorError(
            f"authorized output root does not exist: {output_root}"
        ) from exc
    if not resolved.is_dir():
        raise EvaluatorError(f"authorized output root is not a directory: {resolved}")
    for protected in READ_ONLY_ROOTS:
        protected_resolved = protected.resolve(strict=False)
        if resolved.is_relative_to(protected_resolved) or protected_resolved.is_relative_to(resolved):
            raise EvaluatorError("autoresearch output root overlaps a read-only workspace")
    return resolved


def safe_output_path(
    path: str | Path,
    *,
    authorized_root: str | Path,
    data_root: str | Path,
    require_directory: bool = False,
) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise EvaluatorError("output paths must be absolute")
    root_path = Path(authorized_root)
    if root_path.is_symlink():
        raise EvaluatorError("authorized output root must not be a symlink")
    root = root_path.resolve(strict=True)
    resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(root):
        raise EvaluatorError(f"output must stay below {root}")
    data = Path(data_root).resolve(strict=True)
    if resolved.is_relative_to(data) or data.is_relative_to(resolved):
        raise EvaluatorError("output path must not overlap the read-only data root")
    for protected in READ_ONLY_ROOTS:
        protected_resolved = protected.resolve(strict=False)
        if resolved.is_relative_to(protected_resolved):
            raise EvaluatorError(f"output path is inside a read-only workspace: {protected}")
    if target.exists() and target.is_symlink():
        raise EvaluatorError("output path must not be a symlink")
    if require_directory and target.exists() and not target.is_dir():
        raise EvaluatorError(f"output path is not a directory: {target}")
    return resolved


def _prepare_legacy_log_directory(
    working_directory: Path, *, authorized_root: Path, data_root: Path
) -> None:
    """Make the existing project's relative Logger output safe before imports."""
    log_directory = working_directory / "logs"
    safe_output_path(
        log_directory,
        authorized_root=authorized_root,
        data_root=data_root,
        require_directory=True,
    )
    log_directory.mkdir(parents=True, exist_ok=True)
    safe_output_path(
        log_directory,
        authorized_root=authorized_root,
        data_root=data_root,
        require_directory=True,
    )


def _safe_input_file(path: str | Path, description: str) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise EvaluatorError(f"{description} path must be absolute")
    try:
        resolved = target.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvaluatorError(f"{description} does not exist: {target}") from exc
    if not resolved.is_file():
        raise EvaluatorError(f"{description} must be a file: {resolved}")
    return resolved


def _data_root_without_output_overlap(
    data_root: str | Path, output_path: str | Path, authorized_root: Path
) -> Path:
    root = Path(data_root)
    if not root.is_absolute():
        raise EvaluatorError("--data-root must be an absolute path")
    try:
        resolved_data = root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EvaluatorError(f"data root does not exist: {root}") from exc
    if not resolved_data.is_dir():
        raise EvaluatorError(f"data root is not a directory: {resolved_data}")
    safe_output_path(
        output_path,
        authorized_root=authorized_root,
        data_root=resolved_data,
        require_directory=True,
    )
    return resolved_data


def _code_hashes(project_root: Path) -> dict[str, str]:
    paths = (
        "src/autoresearch.py",
        "src/analyses/autoresearch_data.py",
        "src/analyses/null_calibration.py",
        "src/vilbert.py",
        "src/config.py",
        "src/metrics.py",
        "src/datasets/dataset_utils.py",
        "src/augments_transforms.py",
    )
    return {
        name: _sha256_file(project_root / name)
        for name in paths
        if (project_root / name).is_file()
    }


def _dataset_code_hashes(project_root: Path, task: str) -> dict[str, str]:
    name = {
        "hateful_memes": "src/datasets/dataset_hateful_memes.py",
        "mm_imdb": "src/datasets/dataset_mm_imdb.py",
        "upmc_food": "src/datasets/dataset_upmc.py",
    }.get(task)
    return {name: _sha256_file(project_root / name)} if name else {}


def _check_manifest_location(
    path: str | Path, *, authorized_root: Path, data_root: Path
) -> Path:
    manifest_path = _safe_input_file(path, "sample manifest")
    safe_output_path(
        manifest_path,
        authorized_root=authorized_root,
        data_root=data_root,
    )
    return manifest_path


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze saved ViLBERT checkpoints on an explicit validation split. "
            "No training or test evaluation is performed."
        )
    )
    commands = parser.add_subparsers(dest="command", required=True)

    manifest = commands.add_parser(
        "manifest", help="create an immutable ordered validation-ID manifest"
    )
    manifest.add_argument("--task", required=True)
    manifest.add_argument("--split", required=True)
    manifest.add_argument("--data-root", required=True)
    manifest.add_argument("--manifest-output", required=True)
    manifest.add_argument("--n", type=int, required=True)
    manifest.add_argument("--k", type=int, required=True)
    manifest.add_argument("--seed", type=int, required=True)

    evaluate = commands.add_parser(
        "evaluate", help="extract/cache CLS features and score one saved checkpoint"
    )
    evaluate.add_argument("--checkpoint", required=True)
    evaluate.add_argument("--task", required=True)
    evaluate.add_argument("--split", required=True)
    evaluate.add_argument("--data-root", required=True)
    evaluate.add_argument("--manifest", required=True)
    evaluate.add_argument("--output-dir", required=True)
    evaluate.add_argument("--n", type=int, required=True)
    evaluate.add_argument("--k", type=int, required=True)
    evaluate.add_argument("--seed", type=int, required=True)
    evaluate.add_argument("--batch-size", type=int, required=True)
    evaluate.add_argument("--device", required=True, help="explicit device: cpu, cuda, or cuda:N")
    evaluate.add_argument("--tokenizer-path", default="google-bert/bert-base-uncased")
    evaluate.add_argument("--n-permutations", type=int, default=200)
    return parser


def _validate_sample_parameters(n: int, k: int, seed: int) -> None:
    from analyses.autoresearch_data import MAX_SAMPLES

    if not 2 <= n <= MAX_SAMPLES:
        raise EvaluatorError(f"--n must be between 2 and {MAX_SAMPLES}")
    if not 1 <= k < n:
        raise EvaluatorError("--k must be between 1 and n-1")
    if seed < 0:
        raise EvaluatorError("--seed must be non-negative")


def _run_manifest(args: argparse.Namespace, project_root: Path) -> int:
    from analyses.autoresearch_data import (
        close_validation_index,
        create_ordered_manifest,
        open_validation_index,
        write_manifest_once,
    )

    output_root = authorized_output_root(project_root)
    data_root = Path(args.data_root)
    if not data_root.is_absolute():
        raise EvaluatorError("--data-root must be an absolute path")
    resolved_data = _data_root_without_output_overlap(
        data_root, args.manifest_output, output_root
    )
    manifest_output = safe_output_path(
        args.manifest_output,
        authorized_root=output_root,
        data_root=resolved_data,
    )
    _prepare_legacy_log_directory(
        output_root, authorized_root=output_root, data_root=resolved_data
    )
    _validate_sample_parameters(args.n, args.k, args.seed)
    with _working_directory(output_root):
        index = open_validation_index(args.task, args.split, resolved_data)
        try:
            manifest = create_ordered_manifest(
                index, n_samples=args.n, seed=args.seed
            )
            path = write_manifest_once(
                manifest,
                manifest_output,
                authorized_output_root=output_root,
                data_root=resolved_data,
            )
        finally:
            close_validation_index(index)
    print(
        json.dumps(
            {
                "manifest": str(path),
                "manifest_sha256": _sha256_bytes(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
                ),
                "task": manifest["task"],
                "split": manifest["split"],
                "validation_role": manifest["validation_role"],
                "n": args.n,
                "k": args.k,
                "seed": args.seed,
                "duplicate_image_records_removed": manifest["selection"][
                    "duplicate_image_records_removed"
                ],
            },
            indent=2,
        )
    )
    return 0


def _metric_shape_status(x: Any, y: Any) -> tuple[bool, str]:
    import numpy as np

    x_norms = np.linalg.norm(x, axis=1)
    y_norms = np.linalg.norm(y, axis=1)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        return False, "non_finite_features"
    if np.any(x_norms == 0) or np.any(y_norms == 0):
        return False, "one_or_more_zero_norm_sample_vectors"
    return True, "ok"


def _linear_cka(x: Any, y: Any) -> tuple[float | None, str]:
    import numpy as np

    x64 = np.asarray(x, dtype=np.float64)
    y64 = np.asarray(y, dtype=np.float64)
    if x64.ndim != 2 or y64.ndim != 2 or x64.shape[0] != y64.shape[0]:
        return None, "incompatible_feature_shapes"
    if not np.isfinite(x64).all() or not np.isfinite(y64).all():
        return None, "non_finite_features"
    x_centered = x64 - x64.mean(axis=0, keepdims=True)
    y_centered = y64 - y64.mean(axis=0, keepdims=True)
    gram_x = x_centered @ x_centered.T
    gram_y = y_centered @ y_centered.T
    norm_x = float(np.sum(gram_x * gram_x))
    norm_y = float(np.sum(gram_y * gram_y))
    denominator = (norm_x * norm_y) ** 0.5
    if denominator <= np.finfo(np.float64).tiny:
        return None, "zero_variance_or_zero_rank"
    value = float(np.sum(gram_x * gram_y) / denominator)
    return value, "ok"


def _rank_diagnostics(features: Any) -> dict[str, Any]:
    import numpy as np

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        return {
            "participation_ratio_rank": None,
            "numerical_rank": None,
            "numerical_rank_threshold": None,
            "status": "invalid_or_non_finite_features",
            "definition": "not_computed",
        }
    centered = values - values.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, compute_uv=False)
    rank_upper_bound = max(0, min(centered.shape[0] - 1, centered.shape[1]))
    if singular_values.size == 0 or singular_values[0] == 0:
        return {
            "participation_ratio_rank": None,
            "numerical_rank": 0,
            "algebraic_rank_upper_bound": rank_upper_bound,
            "numerical_rank_threshold": 0.0,
            "status": "zero_variance",
            "definition": (
                "participation ratio of covariance eigenvalues: "
                "(sum(lambda))^2 / sum(lambda^2)"
            ),
        }
    threshold = float(
        singular_values[0]
        * max(centered.shape)
        * np.finfo(np.float64).eps
    )
    numerical_rank = int(np.count_nonzero(singular_values > threshold))
    eigenvalues = singular_values * singular_values
    sum_eigenvalues = float(np.sum(eigenvalues))
    sum_squared_eigenvalues = float(np.sum(eigenvalues * eigenvalues))
    if sum_squared_eigenvalues <= np.finfo(np.float64).tiny:
        participation_ratio = None
        status = "zero_variance"
    else:
        participation_ratio = float(
            (sum_eigenvalues * sum_eigenvalues) / sum_squared_eigenvalues
        )
        status = "full_numerical_rank" if numerical_rank == rank_upper_bound else "rank_deficient"
    return {
        "participation_ratio_rank": participation_ratio,
        "numerical_rank": numerical_rank,
        "algebraic_rank_upper_bound": rank_upper_bound,
        "numerical_rank_threshold": threshold,
        "status": status,
        "definition": (
            "participation ratio of centered covariance eigenvalues: "
            "(sum(lambda))^2 / sum(lambda^2); numerical rank counts singular "
            "values above max(singular_value) * max(n_samples,n_features) * eps; "
            "its algebraic upper bound is min(n_samples-1,n_features)"
        ),
    }


def _torch_normalize(features: Any) -> Any:
    import torch.nn.functional as functional

    return functional.normalize(features, p=2, dim=1)


def _layer_metrics(text: Any, vision: Any, *, k: int) -> dict[str, Any]:
    import numpy as np

    raw_cka, raw_status = _linear_cka(text, vision)
    text_rank = _rank_diagnostics(text)
    vision_rank = _rank_diagnostics(vision)
    vectors_ok, vector_status = _metric_shape_status(text, vision)
    result: dict[str, Any] = {
        "linear_cka_raw": raw_cka,
        "linear_cka_raw_status": raw_status,
        "linear_cka_l2_normalized": None,
        "linear_cka_l2_normalized_status": vector_status,
        "mknn": None,
        "mknn_status": vector_status,
        "cycle_knn_call_A_text_B_vision": None,
        "cycle_knn_call_A_vision_B_text": None,
        "delta_cycle_knn_text_call_minus_vision_call": None,
        "cycle_knn_status": vector_status,
        "text_rank": text_rank,
        "vision_rank": vision_rank,
    }
    if not vectors_ok:
        return result

    import torch
    import metrics

    text_tensor = torch.as_tensor(np.asarray(text, dtype=np.float32))
    vision_tensor = torch.as_tensor(np.asarray(vision, dtype=np.float32))
    text_normalized = _torch_normalize(text_tensor)
    vision_normalized = _torch_normalize(vision_tensor)
    normalized_cka, normalized_status = _linear_cka(
        text_normalized.numpy(), vision_normalized.numpy()
    )
    result["linear_cka_l2_normalized"] = normalized_cka
    result["linear_cka_l2_normalized_status"] = normalized_status
    try:
        result["mknn"] = metrics.AlignmentMetrics.measure(
            "mutual_knn", text_normalized, vision_normalized, topk=k
        )
        result["cycle_knn_call_A_text_B_vision"] = metrics.AlignmentMetrics.cycle_knn(
            feats_A=text_normalized, feats_B=vision_normalized, topk=k
        )
        result["cycle_knn_call_A_vision_B_text"] = metrics.AlignmentMetrics.cycle_knn(
            feats_A=vision_normalized, feats_B=text_normalized, topk=k
        )
        result["delta_cycle_knn_text_call_minus_vision_call"] = (
            result["cycle_knn_call_A_text_B_vision"]
            - result["cycle_knn_call_A_vision_B_text"]
        )
        result["mknn_status"] = "ok"
        result["cycle_knn_status"] = "ok"
    except Exception as exc:
        result["mknn_status"] = f"metric_error:{type(exc).__name__}:{exc}"
        result["cycle_knn_status"] = result["mknn_status"]
    return result


def _cached_cka_metric(text: Any, vision: Any) -> tuple[Any, Any, Any]:
    """Prepare raw linear CKA null scoring with cached sample Gram matrices."""
    import numpy as np

    text64 = np.asarray(text, dtype=np.float64)
    vision64 = np.asarray(vision, dtype=np.float64)
    text_centered = text64 - text64.mean(axis=0, keepdims=True)
    vision_centered = vision64 - vision64.mean(axis=0, keepdims=True)
    gram_text = text_centered @ text_centered.T
    gram_vision = vision_centered @ vision_centered.T
    text_norm = float(np.sum(gram_text * gram_text))
    vision_norm = float(np.sum(gram_vision * gram_vision))
    denominator = (text_norm * vision_norm) ** 0.5
    if denominator <= np.finfo(np.float64).tiny:
        raise EvaluatorError("final-layer CKA is undefined for zero-variance features")
    n_samples = text64.shape[0]

    def metric(_text_features: Any, indexed_vision_features: Any) -> float:
        ids = np.rint(indexed_vision_features[:, -1]).astype(np.int64)
        if (
            ids.shape != (n_samples,)
            or np.any(ids < 0)
            or np.any(ids >= n_samples)
            or np.unique(ids).size != n_samples
        ):
            raise EvaluatorError("calibration sample-ID sentinel was corrupted")
        permuted_gram = gram_vision[np.ix_(ids, ids)]
        return float(np.sum(gram_text * permuted_gram) / denominator)

    indexed_vision = np.column_stack(
        (vision64, np.arange(n_samples, dtype=np.float64))
    )
    return text64, indexed_vision, metric


def score_cached_features(
    text_features: Any,
    vision_features: Any,
    *,
    layer_ids: list[int],
    cross_attention_layers: list[int],
    sample_ids: list[str],
    k: int,
    seed: int,
    n_permutations: int,
) -> dict[str, Any]:
    import numpy as np
    from analyses.null_calibration import calibrate_permutation_null

    text_values = np.asarray(text_features)
    vision_values = np.asarray(vision_features)
    if text_values.ndim != 3 or vision_values.ndim != 3:
        raise EvaluatorError("cached features must have shape [samples,layers,width]")
    if text_values.shape[:2] != vision_values.shape[:2]:
        raise EvaluatorError("text/vision sample and layer axes do not match")
    if text_values.shape[0] != len(sample_ids) or len(layer_ids) != text_values.shape[1]:
        raise EvaluatorError("feature rows/layers do not match the ordered manifest")
    if not np.isfinite(text_values).all() or not np.isfinite(vision_values).all():
        raise EvaluatorError("cached features contain non-finite values")
    if not 1 <= k < text_values.shape[0]:
        raise EvaluatorError("k must be between 1 and n-1")
    if n_permutations < 1 or n_permutations > MAX_PERMUTATIONS:
        raise EvaluatorError(
            f"n_permutations must be between 1 and {MAX_PERMUTATIONS}"
        )

    per_layer = []
    for layer_index, layer_id in enumerate(layer_ids):
        metrics_for_layer = _layer_metrics(
            text_values[:, layer_index, :], vision_values[:, layer_index, :], k=k
        )
        metrics_for_layer["layer_id_zero_based"] = int(layer_id)
        metrics_for_layer["is_cross_attention"] = int(layer_id) in cross_attention_layers
        per_layer.append(metrics_for_layer)

    final_text = text_values[:, -1, :]
    final_vision = vision_values[:, -1, :]
    final_cka = per_layer[-1]["linear_cka_raw"]
    calibration: dict[str, Any]
    if final_cka is None:
        calibration = {
            "status": "not_run_degenerate_final_layer",
            "reason": per_layer[-1]["linear_cka_raw_status"],
        }
    else:
        cka_x, indexed_y, metric = _cached_cka_metric(final_text, final_vision)
        calibration = calibrate_permutation_null(
            cka_x,
            indexed_y,
            metric,
            n_permutations=n_permutations,
            seed=seed,
            alternative="greater",
            quantile=0.95,
            groups=None,
            maximum=1.0,
        )
        if not np.isclose(calibration["observed"], final_cka, rtol=1e-9, atol=1e-10):
            raise EvaluatorError(
                "cached-Gram calibration metric disagrees with final-layer raw CKA"
            )
        calibration["status"] = "completed"
        calibration["scope"] = "predeclared final layer only; no best-layer selection"
        calibration["metric"] = "raw centered linear CKA on unnormalized CLS features"
        calibration["permutation_unit"] = "paired rows of the vision representation"
        calibration["feature_inference_repeated"] = False
    return {
        "metric_semantics": {
            "linear_cka_raw": (
                "uncalibrated biased linear CKA on centered, unnormalized CLS features"
            ),
            "linear_cka_l2_normalized": (
                "same CKA after per-example L2 normalization, matching existing "
                "alignment-analysis preprocessing"
            ),
            "mknn": "existing AlignmentMetrics.mutual_knn on per-example L2-normalized CLS",
            "cycle_knn_call_A_text_B_vision": (
                "existing cycle_knn(feats_A=text, feats_B=vision); call order preserved"
            ),
            "cycle_knn_call_A_vision_B_text": (
                "existing cycle_knn(feats_A=vision, feats_B=text); reverse call order"
            ),
            "delta_cycle_knn_text_call_minus_vision_call": (
                "first existing cycle_knn call minus reverse call, matching the "
                "repository's text-minus-vision delta sign"
            ),
        },
        "n_samples": len(sample_ids),
        "k": k,
        "layer_ids_zero_based": [int(layer) for layer in layer_ids],
        "cross_attention_layer_ids_zero_based": [int(layer) for layer in cross_attention_layers],
        "ordered_sample_ids": list(sample_ids),
        "layers": per_layer,
        "calibration": calibration,
        "claim_scope": (
            "descriptive validation-discovery checkpoint analysis; no causal or "
            "downstream-performance claim"
        ),
    }


def _feature_cache_identity(
    *,
    checkpoint_hash: str,
    manifest_hash: str,
    code_hashes: dict[str, str],
    tokenizer_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "code_sha256": code_hashes,
        "extraction_version": EXTRACTION_VERSION,
        "tokenizer_identity": tokenizer_identity,
    }


def _cache_key(identity: dict[str, Any]) -> str:
    serialized = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return _sha256_bytes(serialized.encode("utf-8"))


def _load_cached_features(path: Path, expected_identity: dict[str, Any]) -> dict[str, Any] | None:
    if not path.exists():
        return None
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata"].item()))
            identity = metadata.get("cache_identity")
            if identity != expected_identity:
                raise EvaluatorError(
                    "feature cache exists but its checkpoint/data/code identity differs"
                )
            return {
                "text_features": archive["text_features"].copy(),
                "vision_features": archive["vision_features"].copy(),
                "layer_ids": [int(value) for value in archive["layer_ids"].tolist()],
                "cross_attention_layers": [
                    int(value) for value in archive["cross_attention_layers"].tolist()
                ],
                "sample_ids": [str(value) for value in archive["sample_ids"].tolist()],
                "metadata": metadata,
            }
    except EvaluatorError:
        raise
    except Exception as exc:
        raise EvaluatorError(f"feature cache is unreadable: {path}: {exc}") from exc


def _write_feature_cache_once(
    path: Path,
    *,
    text_features: Any,
    vision_features: Any,
    layer_ids: list[int],
    cross_attention_layers: list[int],
    sample_ids: list[str],
    metadata: dict[str, Any],
) -> None:
    import numpy as np

    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise EvaluatorError(f"refusing to replace existing feature cache: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(fd, "wb") as file:
            np.savez_compressed(
                file,
                text_features=np.asarray(text_features, dtype=np.float32),
                vision_features=np.asarray(vision_features, dtype=np.float32),
                layer_ids=np.asarray(layer_ids, dtype=np.int64),
                cross_attention_layers=np.asarray(cross_attention_layers, dtype=np.int64),
                sample_ids=np.asarray(sample_ids, dtype=np.str_),
                metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            )
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_name, path)
    except FileExistsError as exc:
        raise EvaluatorError(f"refusing to replace existing feature cache: {path}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _load_model_without_pretrained_downloads(
    checkpoint: Path, device_name: str, *, working_directory: Path
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    with _working_directory(working_directory):
        return _load_model_without_pretrained_downloads_in_directory(
            checkpoint, device_name
        )


def _load_model_without_pretrained_downloads_in_directory(
    checkpoint: Path, device_name: str
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Use ViLBERT.load_model while disabling its default base-weight fetches."""
    import inspect
    import torch
    import timm
    from transformers import BertConfig, BertModel

    from vilbert import ViLBERT

    if device_name == "cpu":
        device = torch.device("cpu")
    elif device_name == "cuda" or device_name.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise EvaluatorError(f"requested {device_name}, but CUDA is unavailable")
        device = torch.device(device_name)
    else:
        raise EvaluatorError("--device must be cpu, cuda, or cuda:N")

    original_bert_loader = inspect.getattr_static(BertModel, "from_pretrained")
    original_timm_factory = timm.create_model
    original_torch_load = torch.load
    captured_payload: list[Any] = []

    def offline_bert_loader(cls: Any, *_args: Any, **_kwargs: Any) -> Any:
        # BERT-base defaults define the architecture; all learned tensors are
        # immediately replaced by the complete state dict from the checkpoint.
        return cls(BertConfig())

    def offline_timm_factory(name: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return original_timm_factory(name, *args, **kwargs)

    def capture_checkpoint(*args: Any, **kwargs: Any) -> Any:
        value = original_torch_load(*args, **kwargs)
        if not captured_payload:
            captured_payload.append(value)
        return value

    setattr(BertModel, "from_pretrained", classmethod(offline_bert_loader))
    timm.create_model = offline_timm_factory
    torch.load = capture_checkpoint
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    try:
        model = ViLBERT.load_model(str(checkpoint), device=str(device))
    finally:
        setattr(BertModel, "from_pretrained", original_bert_loader)
        timm.create_model = original_timm_factory
        torch.load = original_torch_load
    model.to(device)
    model.eval()
    payload = captured_payload[0] if captured_payload else {}
    embedded = {}
    if isinstance(payload, dict):
        for key in (
            "metadata",
            "provenance",
            "epoch",
            "seed",
            "training_seed",
            "task",
            "run_id",
            "checkpoint_selection",
        ):
            if key in payload and key != "model_state_dict":
                embedded[key] = _jsonable(payload[key])
    config = _jsonable(vars(model.config))
    from config import VIT_MODEL_NAME

    architecture = {
        "saved_checkpoint_config": config,
        "model_depth": int(model.depth),
        "text_cross_attention_layers_zero_based": _jsonable(
            model.t_biattention_ids
        ),
        "vision_cross_attention_layers_zero_based": _jsonable(
            model.v_biattention_ids
        ),
        "text_backbone_architecture": "BERT-base-uncased (ViLBERT constructor default)",
        "vision_backbone_architecture": str(VIT_MODEL_NAME),
        "backbone_initialization_for_load": (
            "random architecture shells; complete checkpoint state dict loaded "
            "strictly by ViLBERT.load_model"
        ),
    }
    provenance = {
        "checkpoint_payload_fields": sorted(payload.keys()) if isinstance(payload, dict) else [],
        "embedded_metadata": embedded,
        "config_seed": config.get("seed") if isinstance(config, dict) else None,
        "run_selection_rule": "unknown unless present in embedded checkpoint metadata",
        "task_metric_used_for_selection": "unknown unless present in embedded checkpoint metadata",
    }
    return model, architecture, provenance


def _extract_features(
    *,
    model: Any,
    index: Any,
    manifest: dict[str, Any],
    tokenizer: Any,
    tokenizer_identity: dict[str, Any],
    batch_size: int,
    device_name: str,
) -> dict[str, Any]:
    import numpy as np
    import torch
    from analyses.autoresearch_data import (
        load_existing_dataset_item,
        load_hateful_memes_item,
    )

    if index.dataset is not None:
        index.dataset.tokenizer = tokenizer

    records_by_id = index.by_id()
    ordered_records = [records_by_id[sample_id] for sample_id in manifest["ordered_sample_ids"]]
    features_text: dict[int, list[Any]] = {}
    features_vision: dict[int, list[Any]] = {}
    cross_layers: set[int] = set()
    from config import TOKENIZER_MAX_LEN

    max_length = int(TOKENIZER_MAX_LEN)
    if device_name == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device(device_name)

    from datasets.dataset_utils import process_image

    for offset in range(0, len(ordered_records), batch_size):
        chunk = ordered_records[offset : offset + batch_size]
        texts = []
        images = []
        already_tokenized = []
        for record in chunk:
            if index.task == "hateful_memes":
                text, image_path = load_hateful_memes_item(record)
                processed = process_image(image_path, transform=None)
                if processed is None or processed.get("pixel_values") is None:
                    raise EvaluatorError(
                        f"image preprocessing failed for validation ID {record.sample_id}"
                    )
                images.append(processed["pixel_values"].squeeze(0))
                texts.append(text)
            else:
                text_features, image_features = load_existing_dataset_item(index, record)
                already_tokenized.append(text_features)
                image_values = image_features.get("pixel_values")
                if image_values is None:
                    raise EvaluatorError(
                        f"image dataset returned no pixel_values for {record.sample_id}"
                    )
                images.append(image_values.squeeze(0))

        if already_tokenized:
            text_keys = set(already_tokenized[0])
            if any(set(item) != text_keys for item in already_tokenized):
                raise EvaluatorError("tokenized validation rows have different fields")
            text_batch = {
                key: torch.cat([item[key] for item in already_tokenized], dim=0).to(device)
                for key in text_keys
            }
        else:
            text_batch = tokenizer(
                texts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=max_length,
            )
            text_batch = {key: value.to(device) for key, value in text_batch.items()}
        image_batch = torch.stack(images, dim=0).to(device)
        with torch.inference_mode():
            _, _, intermediate = model(
                text_input_ids=text_batch["input_ids"],
                text_attention_mask=text_batch.get("attention_mask"),
                text_token_type_ids=text_batch.get("token_type_ids"),
                image_pixel_values=image_batch,
                image_attention_mask=None,
                save_intermediate_representations=True,
            )
        if not intermediate:
            raise EvaluatorError("ViLBERT returned no layer representations")
        for item in intermediate:
            layer_id = int(item["layer"])
            text_cls = item["text_embedding"][:, 0, :].detach().to("cpu").numpy()
            vision_cls = item["vision_embedding"][:, 0, :].detach().to("cpu").numpy()
            if text_cls.shape[0] != len(chunk) or vision_cls.shape[0] != len(chunk):
                raise EvaluatorError("model output rows lost validation sample order")
            features_text.setdefault(layer_id, []).append(text_cls.astype(np.float32, copy=False))
            features_vision.setdefault(layer_id, []).append(
                vision_cls.astype(np.float32, copy=False)
            )
            if item.get("is_cross_attention"):
                cross_layers.add(layer_id)
        del intermediate, text_batch, image_batch

    layer_ids = sorted(features_text)
    expected_layers = list(range(int(model.depth)))
    if layer_ids != expected_layers or sorted(features_vision) != expected_layers:
        raise EvaluatorError(
            f"checkpoint layer outputs {layer_ids} do not match model depth {expected_layers}"
        )
    text_array = np.stack(
        [np.concatenate(features_text[layer], axis=0) for layer in layer_ids], axis=1
    )
    vision_array = np.stack(
        [np.concatenate(features_vision[layer], axis=0) for layer in layer_ids], axis=1
    )
    expected_ids = manifest["ordered_sample_ids"]
    if text_array.shape[0] != len(expected_ids) or vision_array.shape[0] != len(expected_ids):
        raise EvaluatorError("extracted feature count differs from ordered manifest")
    return {
        "text_features": text_array,
        "vision_features": vision_array,
        "layer_ids": layer_ids,
        "cross_attention_layers": sorted(cross_layers),
        "sample_ids": expected_ids,
        "tokenizer_identity": tokenizer_identity,
        "image_preprocessing": (
            "existing dataset preprocessing: project minimal ViT normalization; "
            "validation transforms are deterministic"
        ),
        "text_max_length": max_length,
    }


def _load_local_tokenizer(
    tokenizer_path: str,
    output_dir: Path,
    *,
    authorized_root: Path,
    data_root: Path,
) -> tuple[Any, dict[str, Any]]:
    from transformers import BertTokenizerFast

    hf_home = output_dir / "hf-home"
    hf_cache = output_dir / "hf-hub-cache"
    for target in (hf_home, hf_cache):
        safe_output_path(
            target,
            authorized_root=authorized_root,
            data_root=data_root,
            require_directory=True,
        )
        target.mkdir(parents=True, exist_ok=True)
        safe_output_path(
            target,
            authorized_root=authorized_root,
            data_root=data_root,
            require_directory=True,
        )
    os.environ["HF_HOME"] = str(hf_home)
    os.environ["HF_HUB_CACHE"] = str(hf_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(hf_cache)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    tokenizer = BertTokenizerFast.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        cache_dir=str(hf_cache),
    )
    vocab = tokenizer.get_vocab()
    vocab_payload = json.dumps(
        sorted((str(token), int(index)) for token, index in vocab.items()),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    init_kwargs = _jsonable(getattr(tokenizer, "init_kwargs", {}))
    init_payload = json.dumps(
        init_kwargs, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    backend_payload = tokenizer.backend_tokenizer.to_str().encode("utf-8")
    tokenizer_source = Path(tokenizer_path)
    source_files: dict[str, str] = {}
    if tokenizer_source.is_dir():
        for filename in (
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.txt",
            "added_tokens.json",
            "config.json",
        ):
            candidate = tokenizer_source / filename
            if candidate.is_file():
                source_files[filename] = _sha256_file(candidate.resolve(strict=True))
    tokenizer_identity = {
        "requested_path": tokenizer_path,
        "vocabulary_sha256": _sha256_bytes(vocab_payload),
        "vocab_size": int(len(vocab)),
        "special_tokens_map": _jsonable(tokenizer.special_tokens_map),
        "model_max_length": int(tokenizer.model_max_length),
        "do_lower_case": _jsonable(init_kwargs.get("do_lower_case")),
        "init_config_sha256": _sha256_bytes(init_payload),
        "normalizer_backend_sha256": _sha256_bytes(backend_payload),
        "source_files_sha256": source_files,
    }
    return tokenizer, tokenizer_identity


def _write_result_once(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if path.exists():
        try:
            existing = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise EvaluatorError(f"could not inspect existing result {path}: {exc}") from exc
        if existing == serialized:
            return
        raise EvaluatorError(f"refusing to replace an existing different result: {path}")
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_name, path)
    except FileExistsError as exc:
        raise EvaluatorError(f"refusing to replace an existing result: {path}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _run_evaluation(args: argparse.Namespace, project_root: Path) -> int:
    from analyses.autoresearch_data import (
        close_validation_index,
        manifest_sha256,
        open_validation_index,
        read_manifest,
        validate_manifest_against_index,
    )

    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise EvaluatorError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if args.n_permutations < 1 or args.n_permutations > MAX_PERMUTATIONS:
        raise EvaluatorError(
            f"--n-permutations must be between 1 and {MAX_PERMUTATIONS}"
        )
    _validate_sample_parameters(args.n, args.k, args.seed)
    output_root = authorized_output_root(project_root)
    data_root = Path(args.data_root)
    output_dir = Path(args.output_dir)
    resolved_data = _data_root_without_output_overlap(data_root, output_dir, output_root)
    manifest_path = _check_manifest_location(
        args.manifest, authorized_root=output_root, data_root=resolved_data
    )
    checkpoint = _safe_input_file(args.checkpoint, "checkpoint")
    if args.device not in ("cpu", "cuda") and not re.fullmatch(
        r"cuda:[0-9]+", args.device
    ):
        raise EvaluatorError("--device must be cpu, cuda, or cuda:N")
    if args.n < 2:
        raise EvaluatorError("--n must be at least 2")

    manifest = read_manifest(manifest_path)
    if manifest.get("task") != args.task or manifest.get("split") != args.split:
        raise EvaluatorError("requested task/split does not match the immutable manifest")
    if len(manifest["ordered_sample_ids"]) != args.n:
        raise EvaluatorError("--n does not match the immutable manifest sample count")
    selection = manifest["selection"]
    if selection.get("seed") != args.seed:
        raise EvaluatorError("--seed does not match the immutable manifest selection seed")
    if selection.get("n_samples") != args.n:
        raise EvaluatorError("manifest selection n does not match --n")

    output_dir_resolved = safe_output_path(
        output_dir,
        authorized_root=output_root,
        data_root=resolved_data,
        require_directory=True,
    )
    output_dir_resolved.mkdir(parents=True, exist_ok=True)
    # Resolve again after mkdir so symlink traversal cannot redirect writes.
    output_dir_resolved = safe_output_path(
        output_dir_resolved,
        authorized_root=output_root,
        data_root=resolved_data,
        require_directory=True,
    )
    features_dir = output_dir_resolved / "features"
    results_dir = output_dir_resolved / "results"
    for target in (features_dir, results_dir):
        safe_output_path(target, authorized_root=output_root, data_root=resolved_data)
        target.mkdir(parents=True, exist_ok=True)
        safe_output_path(target, authorized_root=output_root, data_root=resolved_data)

    _prepare_legacy_log_directory(
        output_dir_resolved,
        authorized_root=output_root,
        data_root=resolved_data,
    )

    with _working_directory(output_dir_resolved):
        index = open_validation_index(args.task, args.split, resolved_data)
    try:
        validate_manifest_against_index(manifest, index)
        checkpoint_hash = _sha256_file(checkpoint)
        manifest_hash = manifest_sha256(manifest)
        code_hashes = _code_hashes(project_root)
        code_hashes.update(_dataset_code_hashes(project_root, args.task))
        tokenizer, tokenizer_identity = _load_local_tokenizer(
            args.tokenizer_path,
            output_dir_resolved,
            authorized_root=output_root,
            data_root=resolved_data,
        )
        identity = _feature_cache_identity(
            checkpoint_hash=checkpoint_hash,
            manifest_hash=manifest_hash,
            code_hashes=code_hashes,
            tokenizer_identity=tokenizer_identity,
        )
        cache_identity = _cache_key(identity)
        cache_path = features_dir / f"features-{cache_identity[:20]}.npz"
        safe_output_path(cache_path, authorized_root=output_root, data_root=resolved_data)
        cached = _load_cached_features(cache_path, identity)
        feature_cache_hit = cached is not None
        if cached is None:
            # All network-capable model construction is blocked. ViLBERT's
            # existing loader still reconstructs and loads the full checkpoint.
            os.environ["HF_HUB_OFFLINE"] = "1"
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            model, checkpoint_architecture, checkpoint_provenance = (
                _load_model_without_pretrained_downloads(
                    checkpoint,
                    args.device,
                    working_directory=output_dir_resolved,
                )
            )
            extracted = _extract_features(
                model=model,
                index=index,
                manifest=manifest,
                tokenizer=tokenizer,
                tokenizer_identity=tokenizer_identity,
                batch_size=args.batch_size,
                device_name=args.device,
            )
            feature_metadata = {
                "cache_identity": identity,
                "cache_identity_sha256": cache_identity,
                "checkpoint_path": str(checkpoint),
                "checkpoint_architecture": checkpoint_architecture,
                "checkpoint_provenance": checkpoint_provenance,
                "task": args.task,
                "split": args.split,
                "validation_role": manifest.get("validation_role"),
                "data_root": str(resolved_data),
                "source_data_sha256": manifest.get("source_data_sha256"),
                "data_manifest_sha256": manifest_hash,
                "code_sha256": code_hashes,
                "extraction_version": EXTRACTION_VERSION,
                "image_preprocessing": extracted["image_preprocessing"],
                "tokenizer_identity": extracted["tokenizer_identity"],
                "text_max_length": extracted["text_max_length"],
                "layer_ids_zero_based": extracted["layer_ids"],
                "cross_attention_layer_ids_zero_based": extracted[
                    "cross_attention_layers"
                ],
                "ordered_sample_ids": extracted["sample_ids"],
                "sample_pair_hashes": {
                    item["id"]: item["pair_sha256"] for item in manifest["samples"]
                },
            }
            _write_feature_cache_once(
                cache_path,
                text_features=extracted["text_features"],
                vision_features=extracted["vision_features"],
                layer_ids=extracted["layer_ids"],
                cross_attention_layers=extracted["cross_attention_layers"],
                sample_ids=extracted["sample_ids"],
                metadata=feature_metadata,
            )
            cached = _load_cached_features(cache_path, identity)
            assert cached is not None
            del model
        else:
            checkpoint_architecture = cached["metadata"].get(
                "checkpoint_architecture", {}
            )
            checkpoint_provenance = cached["metadata"].get("checkpoint_provenance", {})

        if cached["sample_ids"] != manifest["ordered_sample_ids"]:
            raise EvaluatorError("cached feature rows do not match immutable manifest order")
        scored = score_cached_features(
            cached["text_features"],
            cached["vision_features"],
            layer_ids=cached["layer_ids"],
            cross_attention_layers=cached["cross_attention_layers"],
            sample_ids=cached["sample_ids"],
            k=args.k,
            seed=args.seed,
            n_permutations=args.n_permutations,
        )
        analysis_config = {
            "n": args.n,
            "k": args.k,
            "seed": args.seed,
            "n_permutations": args.n_permutations,
            "batch_size": args.batch_size,
            "device": args.device,
        }
        result_identity = {
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest_hash,
            "tokenizer_identity": tokenizer_identity,
            "analysis_config": analysis_config,
            "code_sha256": code_hashes,
            "evaluator_version": EVALUATOR_VERSION,
        }
        result_id = _cache_key(result_identity)
        result = {
            "format": "autoresearch-checkpoint-validation-result",
            "evaluator_version": EVALUATOR_VERSION,
            "analysis_id": result_id,
            "analysis_identity": result_identity,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_hash,
                "architecture_from_saved_config": checkpoint_architecture,
                "provenance_available_in_checkpoint": checkpoint_provenance,
            },
            "tokenizer": tokenizer_identity,
            "data": {
                "task": args.task,
                "split": args.split,
                "validation_role": manifest.get("validation_role"),
                "ordered_sample_ids": manifest["ordered_sample_ids"],
                "sample_manifest_path": str(manifest_path),
                "sample_manifest_sha256": manifest_hash,
                "source_data_sha256": manifest.get("source_data_sha256"),
                "duplicate_image_records_removed": selection.get(
                    "duplicate_image_records_removed"
                ),
                "data_root": str(resolved_data),
            },
            "features": {
                "path": str(cache_path),
                "sha256": _sha256_file(cache_path),
                "persisted_before_scoring": True,
                "calibration_reused_cached_features": True,
                "extraction_version": EXTRACTION_VERSION,
            },
            "analysis_config": analysis_config,
            **scored,
        }
        result_path = results_dir / f"result-{result_id[:20]}.json"
        safe_output_path(result_path, authorized_root=output_root, data_root=resolved_data)
        _write_result_once(result_path, result)
    finally:
        with _working_directory(output_dir_resolved):
            close_validation_index(index)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "analysis_id": result_id,
                "cache": str(cache_path),
                "feature_cache_preexisting_for_this_invocation": feature_cache_hit,
                "task": args.task,
                "split": args.split,
                "n": args.n,
                "k": args.k,
                "n_permutations": args.n_permutations,
                "final_layer_linear_cka_raw": scored["layers"][-1]["linear_cka_raw"],
                "calibration_status": scored["calibration"].get("status", "completed"),
            },
            indent=2,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    source_root = str((project_root / "src").resolve(strict=True))
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    try:
        if args.command == "manifest":
            return _run_manifest(args, project_root)
        if args.command == "evaluate":
            return _run_evaluation(args, project_root)
        parser.error(f"unknown command {args.command}")
    except (EvaluatorError, ValueError) as exc:
        print(f"autoresearch: error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Offline cross-layer CKA analysis for saved autoresearch feature caches.

This module reads the ``.npz`` files written by ``src/autoresearch.py``. It
does not load a checkpoint, dataset, tokenizer, or model. Every grid cell is
the raw centered linear CKA between one cached text layer and one cached
vision layer, with samples paired by their saved row order.
"""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from typing import Any, Iterator


MAX_SAMPLES = 1024
MAX_PERMUTATIONS = 10_000
MAX_WORKING_BYTES = 900 * 1024 * 1024
CPU_THREAD_LIMIT = 4
COMPATIBILITY_FIELDS = (
    "task",
    "split",
    "validation_role",
    "source_data_sha256",
    "data_manifest_sha256",
    "image_preprocessing",
    "tokenizer_identity",
    "text_max_length",
    "extraction_version",
    "code_sha256",
)


class LayerGridError(ValueError):
    """Raised when a cache cannot support a valid paired layer comparison."""


@contextmanager
def _cpu_thread_limit() -> Iterator[None]:
    """Cap BLAS/OpenMP work at four threads without adding a dependency."""
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[name] = str(CPU_THREAD_LIMIT)
    try:
        from threadpoolctl import threadpool_limits
    except ImportError:
        yield
    else:
        with threadpool_limits(limits=CPU_THREAD_LIMIT):
            yield


def _canonical(value: Any, name: str) -> str:
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise LayerGridError(f"cache metadata field {name!r} is not finite JSON data") from exc


def _metadata_layer_ids(metadata: dict[str, Any], field: str) -> list[int] | None:
    architecture = metadata.get("checkpoint_architecture")
    if not isinstance(architecture, dict):
        return None
    values = architecture.get(field)
    if values is None:
        saved_config = architecture.get("saved_checkpoint_config")
        if isinstance(saved_config, dict):
            config_field = (
                "text_cross_attention_layers"
                if field.startswith("text_")
                else "vision_cross_attention_layers"
            )
            values = saved_config.get(config_field)
    if values is None:
        return None
    if not isinstance(values, (list, tuple)):
        raise LayerGridError(f"saved checkpoint metadata field {field!r} must be a list")
    parsed: list[int] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, int):
            raise LayerGridError(f"saved checkpoint metadata field {field!r} must contain integers")
        parsed.append(int(value))
    if any(value < 0 for value in parsed) or len(set(parsed)) != len(parsed):
        raise LayerGridError(f"saved checkpoint metadata field {field!r} has invalid layer IDs")
    return parsed


def _load_cache(path: str | Path) -> dict[str, Any]:
    import numpy as np

    target = Path(path)
    if not target.is_absolute():
        target = Path.cwd() / target
    try:
        target = target.resolve(strict=True)
    except FileNotFoundError as exc:
        raise LayerGridError(f"feature cache does not exist: {path}") from exc
    if not target.is_file():
        raise LayerGridError(f"feature cache is not a file: {target}")

    required = {
        "text_features",
        "vision_features",
        "layer_ids",
        "cross_attention_layers",
        "sample_ids",
        "metadata",
    }
    try:
        with np.load(target, allow_pickle=False) as archive:
            missing = required.difference(archive.files)
            if missing:
                raise LayerGridError(
                    f"feature cache is missing arrays: {', '.join(sorted(missing))}"
                )
            metadata_value = archive["metadata"].item()
            metadata = json.loads(str(metadata_value))
            text_features = archive["text_features"].copy()
            vision_features = archive["vision_features"].copy()
            layer_ids_raw = archive["layer_ids"].tolist()
            cross_attention_raw = archive["cross_attention_layers"].tolist()
            sample_ids_raw = archive["sample_ids"].tolist()
    except LayerGridError:
        raise
    except Exception as exc:
        raise LayerGridError(f"could not read feature cache {target}: {exc}") from exc

    if not isinstance(metadata, dict):
        raise LayerGridError(f"feature cache metadata must be a JSON object: {target}")
    for name, values in (("layer_ids", layer_ids_raw), ("cross_attention_layers", cross_attention_raw)):
        if any(isinstance(value, bool) or not isinstance(value, (int, np.integer)) for value in values):
            raise LayerGridError(f"{name} must contain integer IDs in {target}")
    layer_ids = [int(value) for value in layer_ids_raw]
    cross_attention_layers = [int(value) for value in cross_attention_raw]
    if not layer_ids or layer_ids != sorted(set(layer_ids)) or layer_ids[0] < 0:
        raise LayerGridError(f"layer_ids must be unique, non-negative, and increasing in {target}")
    if cross_attention_layers != sorted(set(cross_attention_layers)) or any(
        value not in layer_ids for value in cross_attention_layers
    ):
        raise LayerGridError(f"cross_attention_layers must be unique cached layer IDs in {target}")

    sample_ids = [str(value) for value in sample_ids_raw]
    if any(not sample_id for sample_id in sample_ids) or len(set(sample_ids)) != len(sample_ids):
        raise LayerGridError(f"sample_ids must be non-empty and unique in {target}")
    if metadata.get("ordered_sample_ids") != sample_ids:
        raise LayerGridError(f"cached sample_ids do not match metadata row order in {target}")

    expected_layers = metadata.get("layer_ids_zero_based")
    if expected_layers is not None and expected_layers != layer_ids:
        raise LayerGridError(f"cached layer_ids do not match metadata in {target}")
    expected_cross = metadata.get("cross_attention_layer_ids_zero_based")
    if expected_cross is not None and expected_cross != cross_attention_layers:
        raise LayerGridError(f"cached cross_attention_layers do not match metadata in {target}")

    if text_features.ndim != 3 or vision_features.ndim != 3:
        raise LayerGridError("cached features must have shape [samples,layers,width]")
    if text_features.shape[:2] != vision_features.shape[:2]:
        raise LayerGridError("text and vision feature sample/layer axes do not match")
    n_samples, n_layers, text_width = text_features.shape
    vision_width = vision_features.shape[2]
    if n_samples != len(sample_ids) or n_layers != len(layer_ids):
        raise LayerGridError("cached feature rows/layers do not match sample_ids/layer_ids")
    if n_samples < 2 or n_samples > MAX_SAMPLES:
        raise LayerGridError(f"cache must contain between 2 and {MAX_SAMPLES} samples")
    if text_width < 1 or vision_width < 1:
        raise LayerGridError("cached feature widths must be non-empty")
    for name, values in (("text_features", text_features), ("vision_features", vision_features)):
        if values.dtype.kind not in "biuf":
            raise LayerGridError(f"{name} must contain real numeric values")
        if not np.isfinite(values).all():
            raise LayerGridError(f"{name} contains non-finite values")

    metadata_layer_ids = _metadata_layer_ids(
        metadata, "text_cross_attention_layers_zero_based"
    )
    metadata_vision_layer_ids = _metadata_layer_ids(
        metadata, "vision_cross_attention_layers_zero_based"
    )
    if metadata_layer_ids is not None and metadata_vision_layer_ids is not None:
        if len(metadata_layer_ids) != len(metadata_vision_layer_ids):
            raise LayerGridError("saved text and vision cross-attention lists differ in length")

    identity = metadata.get("cache_identity")
    if not isinstance(identity, dict):
        raise LayerGridError(f"cache metadata has no cache_identity object: {target}")
    if identity.get("manifest_sha256") != metadata.get("data_manifest_sha256"):
        raise LayerGridError(f"cache identity manifest hash disagrees with metadata: {target}")

    return {
        "path": str(target),
        "text_features": text_features,
        "vision_features": vision_features,
        "layer_ids": layer_ids,
        "cross_attention_layers": cross_attention_layers,
        "sample_ids": sample_ids,
        "metadata": metadata,
    }


def _protocol_signature(cache: dict[str, Any]) -> dict[str, str]:
    metadata = cache["metadata"]
    missing = [field for field in COMPATIBILITY_FIELDS if field not in metadata]
    if missing:
        raise LayerGridError(
            f"cache metadata is missing comparison identity fields: {', '.join(missing)}"
        )
    signature = {
        field: _canonical(metadata[field], field) for field in COMPATIBILITY_FIELDS
    }
    pair_hashes = metadata.get("sample_pair_hashes")
    if pair_hashes is not None:
        if not isinstance(pair_hashes, dict) or set(pair_hashes) != set(cache["sample_ids"]):
            raise LayerGridError("sample_pair_hashes must cover exactly the cached sample IDs")
    signature["sample_pair_hashes"] = _canonical(pair_hashes, "sample_pair_hashes")
    signature["ordered_sample_ids"] = _canonical(cache["sample_ids"], "ordered_sample_ids")
    return signature


def _check_compatible(reference: dict[str, str], candidate: dict[str, str], path: str) -> None:
    mismatches = [field for field in reference if reference[field] != candidate[field]]
    if mismatches:
        raise LayerGridError(
            f"cache is not paired with the first input ({path}); mismatched "
            f"cohort, row order, or transform fields: {', '.join(mismatches)}"
        )


def _check_memory_budget(text_features: Any, vision_features: Any) -> None:
    n_samples, n_layers, text_width = text_features.shape
    vision_width = vision_features.shape[2]
    gram_bytes = 2 * n_layers * n_samples * n_samples * 8
    one_gram = n_samples * n_samples * 8
    centered_features = 3 * n_samples * max(text_width, vision_width) * 8
    estimate = (
        text_features.nbytes
        + vision_features.nbytes
        + gram_bytes
        + 3 * one_gram
        + centered_features
    )
    if estimate > MAX_WORKING_BYTES:
        raise LayerGridError(
            "estimated peak memory exceeds the 900 MiB analysis bound "
            f"({estimate / (1024 ** 2):.1f} MiB)"
        )


def _centered_grams(features: Any) -> tuple[Any, Any]:
    import numpy as np

    n_samples, n_layers, _ = features.shape
    grams = np.empty((n_layers, n_samples, n_samples), dtype=np.float64)
    norms = np.empty(n_layers, dtype=np.float64)
    for layer_index in range(n_layers):
        values = np.asarray(features[:, layer_index, :], dtype=np.float64)
        centered = values - values.mean(axis=0, keepdims=True)
        gram = centered @ centered.T
        if not np.isfinite(centered).all() or not np.isfinite(gram).all():
            raise LayerGridError("float64 centering or Gram computation produced non-finite values")
        norm_squared = float(np.sum(gram * gram, dtype=np.float64))
        if not np.isfinite(norm_squared):
            raise LayerGridError("centered Gram norm is non-finite")
        grams[layer_index] = gram
        norms[layer_index] = norm_squared
    return grams, norms


def _score_grid(text_grams: Any, vision_grams: Any, text_norms: Any, vision_norms: Any) -> dict[str, Any]:
    import numpy as np

    n_text = len(text_norms)
    n_vision = len(vision_norms)
    values: list[list[float | None]] = []
    statuses: list[list[str]] = []
    valid_values: list[float] = []
    tiny = np.finfo(np.float64).tiny
    for text_index in range(n_text):
        value_row: list[float | None] = []
        status_row: list[str] = []
        for vision_index in range(n_vision):
            denominator = float(np.sqrt(text_norms[text_index]) * np.sqrt(vision_norms[vision_index]))
            if denominator <= tiny:
                value_row.append(None)
                status_row.append("zero_variance_or_zero_rank")
                continue
            numerator = float(
                np.sum(text_grams[text_index] * vision_grams[vision_index], dtype=np.float64)
            )
            if not np.isfinite(numerator) or not np.isfinite(denominator):
                raise LayerGridError("CKA score computation produced a non-finite value")
            value = numerator / denominator
            if not np.isfinite(value):
                raise LayerGridError("CKA score computation produced a non-finite value")
            value = float(value)
            value_row.append(value)
            status_row.append("ok")
            valid_values.append(value)
        values.append(value_row)
        statuses.append(status_row)
    return {
        "values": values,
        "statuses": statuses,
        "max_grid_raw": max(valid_values) if valid_values else None,
        "n_valid_cells": len(valid_values),
    }


def _calibrate_max_grid(
    text_grams: Any,
    vision_grams: Any,
    text_norms: Any,
    vision_norms: Any,
    observed_max: float,
    *,
    n_permutations: int,
    seed: int,
) -> dict[str, Any]:
    import numpy as np

    if not 1 <= n_permutations <= MAX_PERMUTATIONS:
        raise LayerGridError(f"n_permutations must be between 1 and {MAX_PERMUTATIONS}")
    if seed < 0:
        raise LayerGridError("seed must be non-negative")

    calibration_path = Path(__file__).with_name("null_calibration.py")
    try:
        spec = importlib.util.spec_from_file_location(
            "_autoresearch_layer_grid_null_calibration", calibration_path
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"no loader for {calibration_path}")
        calibration_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(calibration_module)
        calibrate_permutation_null = calibration_module.calibrate_permutation_null
    except (ImportError, OSError, AttributeError) as exc:
        raise LayerGridError(f"could not load the existing null calibrator: {exc}") from exc

    n_samples = text_grams.shape[1]
    tiny = np.finfo(np.float64).tiny

    def maximum_for_index_order(indexed_y: Any) -> float:
        ids = np.rint(indexed_y[:, 0]).astype(np.int64)
        if (
            ids.shape != (n_samples,)
            or np.any(ids < 0)
            or np.any(ids >= n_samples)
            or np.unique(ids).size != n_samples
        ):
            raise LayerGridError("calibration sample-index sentinel was corrupted")
        maximum = -np.inf
        for vision_index, gram in enumerate(vision_grams):
            # Conjugating each cached Gram by the same row order is equivalent
            # to permuting that layer's cached sample features. Only one
            # permuted n-by-n matrix is resident at a time.
            permuted_gram = gram[np.ix_(ids, ids)]
            for text_index, text_gram in enumerate(text_grams):
                denominator = float(
                    np.sqrt(text_norms[text_index])
                    * np.sqrt(vision_norms[vision_index])
                )
                if denominator <= tiny:
                    continue
                numerator = float(np.sum(text_gram * permuted_gram, dtype=np.float64))
                score = numerator / denominator
                if not np.isfinite(score):
                    raise LayerGridError("permuted max-grid CKA is non-finite")
                maximum = max(maximum, score)
        if not np.isfinite(maximum):
            raise LayerGridError("maximum grid is unavailable because every cell is degenerate")
        return float(maximum)

    dummy_x = np.zeros((n_samples, 1), dtype=np.float64)
    row_ids = np.arange(n_samples, dtype=np.float64).reshape(-1, 1)
    calibration = calibrate_permutation_null(
        dummy_x,
        row_ids,
        lambda _x, indexed_y: maximum_for_index_order(indexed_y),
        n_permutations=n_permutations,
        seed=seed,
        alternative="greater",
        quantile=0.95,
        groups=None,
        maximum=1.0,
    )
    if not np.isclose(calibration["observed"], observed_max, rtol=1e-9, atol=1e-10):
        raise LayerGridError("aggregate-aware calibration observed maximum disagrees with the grid")
    calibration["scope"] = "maximum over all valid text-layer x vision-layer cells"
    calibration["metric"] = "raw centered linear CKA on cached, unnormalized CLS features"
    calibration["permutation_unit"] = "paired sample rows; one Y-row permutation shared by every vision layer"
    calibration["exchangeability_caveat"] = (
        "uses unrestricted individual-row exchangeability; duplicate or grouped examples "
        "require a separately justified group-aware null"
    )
    calibration["feature_inference_repeated"] = False
    calibration["pointwise_after_selection_guarantee"] = False
    return calibration


def _configured_pairs(cache: dict[str, Any], grid: dict[str, Any]) -> list[dict[str, Any]]:
    metadata = cache["metadata"]
    text_configured = _metadata_layer_ids(
        metadata, "text_cross_attention_layers_zero_based"
    )
    vision_configured = _metadata_layer_ids(
        metadata, "vision_cross_attention_layers_zero_based"
    )
    if text_configured is None or vision_configured is None:
        return []
    layer_ids = cache["layer_ids"]
    layer_positions = {layer_id: index for index, layer_id in enumerate(layer_ids)}
    pairs: list[dict[str, Any]] = []
    for ordinal, (text_layer_id, vision_layer_id) in enumerate(
        zip(text_configured, vision_configured, strict=True), start=1
    ):
        entry: dict[str, Any] = {
            "ordinal": ordinal,
            "configured_text_layer_id_zero_based": text_layer_id,
            "configured_vision_layer_id_zero_based": vision_layer_id,
            "pairing_source": "saved checkpoint configuration metadata",
            "interpretation": (
                "selects the configured ordinal pair of cached output layer IDs; "
                "it does not assert that either saved output is the immediate causal "
                "result of that cross-attention operation"
            ),
        }
        if text_layer_id not in layer_positions or vision_layer_id not in layer_positions:
            entry.update({"linear_cka_raw": None, "status": "configured_layer_missing_from_cache"})
        else:
            text_index = layer_positions[text_layer_id]
            vision_index = layer_positions[vision_layer_id]
            entry.update(
                {
                    "linear_cka_raw": grid["values"][text_index][vision_index],
                    "status": grid["statuses"][text_index][vision_index],
                }
            )
        pairs.append(entry)
    return pairs


def _checkpoint_summary(cache: dict[str, Any]) -> dict[str, Any]:
    metadata = cache["metadata"]
    identity = metadata["cache_identity"]
    provenance = metadata.get("checkpoint_provenance")
    embedded = provenance.get("embedded_metadata", {}) if isinstance(provenance, dict) else {}
    if not isinstance(embedded, dict):
        embedded = {}
    available_seeds = {
        name: value
        for name, value in (
            ("seed", embedded.get("seed")),
            ("training_seed", embedded.get("training_seed")),
            ("config_seed", provenance.get("config_seed") if isinstance(provenance, dict) else None),
        )
        if isinstance(value, int) and not isinstance(value, bool)
    }
    return {
        "checkpoint_path": metadata.get("checkpoint_path"),
        "checkpoint_sha256": identity.get("checkpoint_sha256"),
        "available_seed_fields": available_seeds,
        "text_cross_attention_layers_zero_based": _metadata_layer_ids(
            metadata, "text_cross_attention_layers_zero_based"
        ),
        "vision_cross_attention_layers_zero_based": _metadata_layer_ids(
            metadata, "vision_cross_attention_layers_zero_based"
        ),
    }


def _analyze_cache(
    cache: dict[str, Any], *, calibrate_max_grid: bool, n_permutations: int, seed: int
) -> dict[str, Any]:
    import numpy as np

    text_features = cache["text_features"]
    vision_features = cache["vision_features"]
    _check_memory_budget(text_features, vision_features)
    with _cpu_thread_limit():
        text_grams, text_norms = _centered_grams(text_features)
        vision_grams, vision_norms = _centered_grams(vision_features)
        scores = _score_grid(text_grams, vision_grams, text_norms, vision_norms)
        grid = {
            "text_layer_ids_zero_based": cache["layer_ids"],
            "vision_layer_ids_zero_based": cache["layer_ids"],
            "values": scores["values"],
            "statuses": scores["statuses"],
            "score": "uncalibrated biased linear CKA on centered, unnormalized CLS features",
        }
        positions = {layer_id: index for index, layer_id in enumerate(cache["layer_ids"])}
        same_layer = [
            {
                "layer_id_zero_based": layer_id,
                "linear_cka_raw": scores["values"][index][index],
                "status": scores["statuses"][index][index],
            }
            for index, layer_id in enumerate(cache["layer_ids"])
        ]
        configured_pairs = _configured_pairs(cache, grid)
        calibration: dict[str, Any]
        if not calibrate_max_grid:
            calibration = {"status": "not_requested"}
        elif scores["max_grid_raw"] is None:
            calibration = {
                "status": "not_run_degenerate_grid",
                "reason": "every text/vision layer pair has a zero-norm centered Gram",
            }
        else:
            calibration = _calibrate_max_grid(
                text_grams,
                vision_grams,
                text_norms,
                vision_norms,
                scores["max_grid_raw"],
                n_permutations=n_permutations,
                seed=seed,
            )
    del text_features, vision_features, text_grams, vision_grams, text_norms, vision_norms

    metadata = cache["metadata"]
    sample_manifest = {
        "task": metadata.get("task"),
        "split": metadata.get("split"),
        "validation_role": metadata.get("validation_role"),
        "data_manifest_sha256": metadata.get("data_manifest_sha256"),
        "ordered_sample_ids_sha256": hashlib.sha256(
            _canonical(cache["sample_ids"], "ordered_sample_ids").encode("utf-8")
        ).hexdigest(),
        "n_samples": len(cache["sample_ids"]),
    }
    return {
        "feature_cache": cache["path"],
        "checkpoint": _checkpoint_summary(cache),
        "sample_manifest": sample_manifest,
        "layer_ids_zero_based": cache["layer_ids"],
        "cross_attention_output_layer_ids_zero_based": cache["cross_attention_layers"],
        "grid": grid,
        "same_layer_pairs": same_layer,
        "configured_ordinal_cross_attention_pairs": configured_pairs,
        "max_grid_raw": scores["max_grid_raw"],
        "n_valid_grid_cells": scores["n_valid_cells"],
        "max_grid_calibration": calibration,
    }


def _seed_match(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    left = first["checkpoint"]["available_seed_fields"]
    right = second["checkpoint"]["available_seed_fields"]
    common = sorted(set(left).intersection(right))
    matches = [field for field in common if left[field] == right[field]]
    differs = [field for field in common if left[field] != right[field]]
    if differs:
        status = "mismatch"
    elif matches:
        status = "match_on_available_fields"
    else:
        status = "unknown"
    return {"status": status, "matching_fields": matches, "mismatching_fields": differs}


def _pairwise_grid_comparison(first: dict[str, Any], second: dict[str, Any]) -> dict[str, Any]:
    first_grid = first["grid"]
    second_grid = second["grid"]
    first_text = {value: index for index, value in enumerate(first_grid["text_layer_ids_zero_based"])}
    first_vision = {value: index for index, value in enumerate(first_grid["vision_layer_ids_zero_based"])}
    second_text = {value: index for index, value in enumerate(second_grid["text_layer_ids_zero_based"])}
    second_vision = {value: index for index, value in enumerate(second_grid["vision_layer_ids_zero_based"])}
    text_ids = sorted(set(first_text).intersection(second_text))
    vision_ids = sorted(set(first_vision).intersection(second_vision))
    cells: list[dict[str, Any]] = []
    for text_id in text_ids:
        for vision_id in vision_ids:
            row_a = first_text[text_id]
            col_a = first_vision[vision_id]
            row_b = second_text[text_id]
            col_b = second_vision[vision_id]
            score_a = first_grid["values"][row_a][col_a]
            score_b = second_grid["values"][row_b][col_b]
            cells.append(
                {
                    "text_layer_id_zero_based": text_id,
                    "vision_layer_id_zero_based": vision_id,
                    "first_cache_linear_cka_raw": score_a,
                    "second_cache_linear_cka_raw": score_b,
                    "second_minus_first": (
                        float(score_b - score_a)
                        if score_a is not None and score_b is not None
                        else None
                    ),
                }
            )
    return {
        "first_cache": first["feature_cache"],
        "second_cache": second["feature_cache"],
        "available_pretraining_seed_comparison": _seed_match(first, second),
        "common_layer_grid_cells": cells,
        "comparison_scope": "descriptive paired-cache differences; not a causal or across-seed estimate",
    }


def analyze_caches(
    cache_paths: list[str | Path],
    *,
    calibrate_max_grid: bool = False,
    n_permutations: int = 200,
    seed: int = 0,
) -> dict[str, Any]:
    """Analyze one or more paired cached feature files without model loading."""
    if not cache_paths:
        raise LayerGridError("provide at least one explicit feature-cache path")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise LayerGridError("seed must be a non-negative integer")
    if isinstance(n_permutations, bool) or not isinstance(n_permutations, int):
        raise LayerGridError("n_permutations must be an integer")
    if not 1 <= n_permutations <= MAX_PERMUTATIONS:
        raise LayerGridError(f"n_permutations must be between 1 and {MAX_PERMUTATIONS}")

    results: list[dict[str, Any]] = []
    reference_signature: dict[str, str] | None = None
    reference_path = ""
    for path in cache_paths:
        cache = _load_cache(path)
        signature = _protocol_signature(cache)
        if reference_signature is None:
            reference_signature = signature
            reference_path = cache["path"]
        else:
            _check_compatible(reference_signature, signature, cache["path"])
        results.append(
            _analyze_cache(
                cache,
                calibrate_max_grid=calibrate_max_grid,
                n_permutations=n_permutations,
                seed=seed,
            )
        )
        del cache

    comparisons = [
        _pairwise_grid_comparison(results[first], results[second])
        for first in range(len(results))
        for second in range(first + 1, len(results))
    ]
    return {
        "format": "autoresearch-cross-layer-grid-v1",
        "analysis": "offline cached-feature cross-layer centered linear CKA",
        "feature_cache_count": len(results),
        "runs": results,
        "pairwise_comparisons": comparisons,
        "claim_scope": (
            "All grid cells and cross-cache differences are descriptive. The raw grid maximum "
            "is not inferentially calibrated unless --calibrate-max-grid is requested. "
            "Aggregate calibration applies to the maximum over the whole valid grid under its "
            "row-exchangeability null; it supplies no pointwise after-selection guarantee. "
            "Matched checkpoint contrasts do not establish causality or model-seed replication."
        ),
        "resource_limits": {
            "device": "CPU only; no model, checkpoint, tokenizer, or dataset loading",
            "maximum_samples": MAX_SAMPLES,
            "maximum_estimated_working_memory_mib": MAX_WORKING_BYTES // (1024 * 1024),
            "maximum_cpu_threads": CPU_THREAD_LIMIT,
        },
        "comparison_protocol_identity_source": reference_path,
    }


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute offline text-layer x vision-layer centered linear CKA grids "
            "from explicit autoresearch .npz feature caches."
        )
    )
    parser.add_argument("caches", nargs="+", help="one or more explicit feature-cache .npz files")
    parser.add_argument("--output", help="write JSON to this new file; prints to stdout when omitted")
    parser.add_argument(
        "--calibrate-max-grid",
        action="store_true",
        help="run the existing permutation-null calibrator on the maximum over all valid grid cells",
    )
    parser.add_argument(
        "--n-permutations",
        type=int,
        default=200,
        help=f"fixed Monte Carlo draws for --calibrate-max-grid (1..{MAX_PERMUTATIONS}; default 200)",
    )
    parser.add_argument("--seed", type=int, default=0, help="fixed permutation seed (default 0)")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)
    try:
        result = analyze_caches(
            args.caches,
            calibrate_max_grid=args.calibrate_max_grid,
            n_permutations=args.n_permutations,
            seed=args.seed,
        )
        output = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        if args.output:
            target = Path(args.output)
            if not target.is_absolute():
                target = Path.cwd() / target
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                with target.open("x", encoding="utf-8") as handle:
                    handle.write(output)
            except FileExistsError as exc:
                raise LayerGridError(f"refusing to replace existing output: {target}") from exc
        else:
            print(output, end="")
    except (LayerGridError, OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

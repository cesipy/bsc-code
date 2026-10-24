"""Checkpoint-only Hateful Memes input-shuffle diagnostic.

The diagnostic compares matched validation inputs with deterministic image-only
and text-only derangements. It reports sensitivity of each fixed trained model
on one historical discovery cohort; it does not estimate a causal effect of a
training architecture or modality information outside this cohort.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import re
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable, Sequence

from autoresearch import (
    EvaluatorError,
    _load_local_tokenizer,
    _load_model_without_pretrained_downloads,
    _safe_input_file,
    authorized_output_root,
    safe_output_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = PROJECT_ROOT / "autoresearch/manifests/hm-devseen-n500-seed42.json"
DEFAULT_SEEDS = (101, 202, 303)
MAX_SAMPLES = 500
MAX_BATCH_SIZE = 2
RESULT_FILENAME = "input-shuffle-diagnostic.json"


class RelianceError(ValueError):
    """Raised when an input-shuffle request violates the frozen protocol."""


def _record_value(record: Any, key: str) -> Any:
    if isinstance(record, dict):
        return record[key]
    return getattr(record, key)


def _record_payload(record: Any) -> dict[str, Any]:
    if isinstance(record, dict):
        return record["payload"]
    return record.payload


def make_derangement(n: int, seed: int) -> tuple[list[int], int]:
    """Return a seeded no-fixed-point map from each row to its donor row."""
    if isinstance(n, bool) or not isinstance(n, int) or n < 2 or n > MAX_SAMPLES:
        raise RelianceError(f"sample count must be between 2 and {MAX_SAMPLES}")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise RelianceError("shuffle seed must be a non-negative integer")
    rng = random.Random(seed)
    attempts = 0
    while True:
        donor_for_row = list(range(n))
        rng.shuffle(donor_for_row)
        attempts += 1
        if all(row != donor for row, donor in enumerate(donor_for_row)):
            return donor_for_row, attempts


def _validated_seeds(seeds: Sequence[int]) -> tuple[int, ...]:
    selected = tuple(seeds)
    if len(selected) != 3 or len(set(selected)) != 3:
        raise RelianceError("exactly three distinct shuffle seeds are required")
    if any(isinstance(seed, bool) or not isinstance(seed, int) or seed < 0 for seed in selected):
        raise RelianceError("shuffle seeds must be non-negative integers")
    return selected


def validate_hm_manifest_protocol(
    manifest: dict[str, Any], *, requested_n: int
) -> list[str]:
    """Check the discovery-only HM split and return its immutable row IDs."""
    if isinstance(requested_n, bool) or not isinstance(requested_n, int):
        raise RelianceError("--n must be an integer")
    if not 2 <= requested_n <= MAX_SAMPLES:
        raise RelianceError(f"--n must be between 2 and {MAX_SAMPLES}")
    if manifest.get("task") != "hateful_memes":
        raise RelianceError("the input-shuffle diagnostic only accepts Hateful Memes")
    if manifest.get("split") != "dev_seen":
        raise RelianceError("only the explicit Hateful Memes dev_seen split is allowed")
    if manifest.get("validation_role") != "validation_discovery_seen_historical":
        raise RelianceError("the manifest must be marked as historical discovery data")
    ids = manifest.get("ordered_sample_ids")
    selection = manifest.get("selection")
    if not isinstance(ids, list) or len(ids) != requested_n:
        raise RelianceError("--n must match the manifest's ordered sample count")
    if not isinstance(selection, dict) or selection.get("n_samples") != requested_n:
        raise RelianceError("manifest selection count does not match --n")
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise RelianceError("manifest sample IDs must be unique non-empty strings")
    return ids


def validate_output_destination(
    output_dir: str | Path,
    *,
    authorized_root: str | Path,
    data_root: str | Path,
    input_paths: Sequence[str | Path] = (),
) -> Path:
    """Validate a new output directory or the runner's stdout-only directory."""
    try:
        resolved = safe_output_path(
            output_dir,
            authorized_root=authorized_root,
            data_root=data_root,
            require_directory=True,
        )
    except EvaluatorError as exc:
        raise RelianceError(str(exc)) from exc
    for value in input_paths:
        source = Path(value).resolve(strict=True)
        if (
            resolved == source
            or resolved.is_relative_to(source)
            or source.is_relative_to(resolved)
        ):
            raise RelianceError(f"output directory overlaps an input path: {source}")
    target = Path(output_dir)
    if target.is_symlink():
        raise RelianceError("output directory must not be a symlink")
    if target.exists():
        if not target.is_dir():
            raise RelianceError(f"output path is not a directory: {target}")
        entries = list(target.iterdir())
        if (
            len(entries) != 1
            or entries[0].name != "stdout.log"
            or entries[0].is_symlink()
            or not entries[0].is_file()
        ):
            raise RelianceError(
                "precreated output directory may contain only the runner's regular stdout.log"
            )
    return resolved


def fuse_cls_embeddings(text_embedding: Any, image_embedding: Any, fusion_method: str) -> Any:
    """Apply the current HM trainer's final CLS fusion formula."""
    if text_embedding.ndim != 2 or image_embedding.ndim != 2:
        raise RelianceError("final CLS embeddings must both have shape [batch, width]")
    if text_embedding.shape != image_embedding.shape:
        raise RelianceError("text and image CLS embedding shapes must match")
    if fusion_method == "concat":
        import torch

        return torch.cat((text_embedding, image_embedding), dim=1)
    if fusion_method == "sum":
        return text_embedding + image_embedding
    if fusion_method == "hadamard":
        return text_embedding * image_embedding
    raise RelianceError(f"unsupported CLS fusion method: {fusion_method!r}")


def _linear_layers(head: Any) -> list[Any]:
    modules = head.modules() if callable(getattr(head, "modules", None)) else ()
    return [
        module
        for module in modules
        if hasattr(module, "in_features") and hasattr(module, "out_features")
    ]


def validate_head_dimensions(head: Any, fusion_method: str, embedding_dim: int) -> dict[str, int]:
    if fusion_method not in ("concat", "sum", "hadamard"):
        raise RelianceError(f"unsupported CLS fusion method: {fusion_method!r}")
    layers = _linear_layers(head)
    if not layers:
        raise RelianceError("model.fc has no inspectable linear layer")
    expected_input = embedding_dim * (2 if fusion_method == "concat" else 1)
    if int(layers[0].in_features) != expected_input:
        raise RelianceError(
            "model.fc input width does not match the configured CLS fusion "
            f"({layers[0].in_features} != {expected_input})"
        )
    if int(layers[-1].out_features) != 1:
        raise RelianceError("the Hateful Memes model.fc head must emit one logit")
    return {
        "input_features": int(layers[0].in_features),
        "output_features": int(layers[-1].out_features),
        "first_linear_output_features": int(layers[0].out_features),
    }


def _run_condition_logits(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    text_loader: Callable[[Any], str],
    image_loader: Callable[[Any], Any],
    text_donors: Sequence[int],
    image_donors: Sequence[int],
    batch_size: int,
    device: str,
    max_length: int,
    fusion_method: str,
    embedding_dim: int,
) -> list[float]:
    import torch

    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise RelianceError(f"batch size must be between 1 and {MAX_BATCH_SIZE}")
    if len(text_donors) != len(records) or len(image_donors) != len(records):
        raise RelianceError("each row needs one text donor and one image donor")
    if getattr(model, "training", False):
        raise RelianceError("model must be in evaluation mode before inference")
    head = getattr(model, "fc", None)
    if head is None:
        raise RelianceError("loaded checkpoint has no Hateful Memes model.fc head")
    validate_head_dimensions(head, fusion_method, embedding_dim)

    logits: list[float] = []
    for offset in range(0, len(records), batch_size):
        target_rows = list(range(offset, min(offset + batch_size, len(records))))
        text_rows = [text_donors[row] for row in target_rows]
        image_rows = [image_donors[row] for row in target_rows]
        texts = [text_loader(records[row]) for row in text_rows]
        token_batch = tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )
        token_batch = {
            key: value.to(device)
            for key, value in token_batch.items()
            if key in ("input_ids", "attention_mask", "token_type_ids")
        }
        if "input_ids" not in token_batch:
            raise RelianceError("local tokenizer returned no input_ids")
        images = []
        for row in image_rows:
            image = image_loader(records[row])
            if image.ndim == 4 and image.shape[0] == 1:
                image = image.squeeze(0)
            images.append(image)
        image_batch = torch.stack(images, dim=0).to(device)
        with torch.inference_mode():
            text_cls, image_cls = model(
                text_input_ids=token_batch["input_ids"],
                text_attention_mask=token_batch.get("attention_mask"),
                text_token_type_ids=token_batch.get("token_type_ids"),
                image_pixel_values=image_batch,
                image_attention_mask=None,
            )
            fused = fuse_cls_embeddings(text_cls, image_cls, fusion_method)
            batch_logits = head(fused)
        if batch_logits.numel() != len(target_rows):
            raise RelianceError("model.fc must return one logit per validation row")
        logits.extend(float(value) for value in batch_logits.detach().to("cpu").reshape(-1))
    if len(logits) != len(records) or any(not math.isfinite(value) for value in logits):
        raise RelianceError("model produced missing or non-finite validation logits")
    return logits


def _sigmoid(logit: float) -> float:
    if logit >= 0:
        return 1.0 / (1.0 + math.exp(-logit))
    exp_value = math.exp(logit)
    return exp_value / (1.0 + exp_value)


def _normalized_binary_labels(labels: Sequence[Any]) -> list[int]:
    if not labels:
        raise RelianceError("Hateful Memes metrics need non-empty row labels")
    normalized_labels: list[int] = []
    for label in labels:
        if isinstance(label, bool):
            normalized = int(label)
        else:
            try:
                numeric = float(label)
            except (TypeError, ValueError) as exc:
                raise RelianceError("Hateful Memes row labels must be binary") from exc
            if numeric not in (0.0, 1.0):
                raise RelianceError("Hateful Memes row labels must be binary")
            normalized = int(numeric)
        normalized_labels.append(normalized)
    if not any(normalized_labels) or all(normalized_labels):
        raise RelianceError("AUROC requires both classes in the fixed cohort")
    return normalized_labels


def binary_metrics(labels: Sequence[Any], logits: Sequence[float]) -> dict[str, float | int]:
    if len(labels) != len(logits) or not labels:
        raise RelianceError("metrics need equally sized non-empty labels and logits")
    normalized_labels = _normalized_binary_labels(labels)
    numeric_logits = [float(value) for value in logits]
    if any(not math.isfinite(value) for value in numeric_logits):
        raise RelianceError("metrics cannot use non-finite logits")
    positives = sum(normalized_labels)
    negatives = len(normalized_labels) - positives

    ordered = sorted(range(len(numeric_logits)), key=numeric_logits.__getitem__)
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        score = numeric_logits[ordered[start]]
        while end < len(ordered) and numeric_logits[ordered[end]] == score:
            end += 1
        average_rank = ((start + 1) + end) / 2.0
        positive_rank_sum += sum(normalized_labels[ordered[i]] for i in range(start, end)) * average_rank
        start = end
    auroc = (
        positive_rank_sum - positives * (positives + 1) / 2.0
    ) / (positives * negatives)
    bce = sum(
        max(logit, 0.0) - logit * label + math.log1p(math.exp(-abs(logit)))
        for label, logit in zip(normalized_labels, numeric_logits)
    ) / len(normalized_labels)
    accuracy = sum(
        (logit >= 0.0) == bool(label)
        for label, logit in zip(normalized_labels, numeric_logits)
    ) / len(normalized_labels)
    return {
        "auroc": float(auroc),
        "bce": float(bce),
        "accuracy_at_0_5": float(accuracy),
        "n": len(normalized_labels),
    }


def _prediction_change(
    baseline_logits: Sequence[float], changed_logits: Sequence[float]
) -> dict[str, float | int]:
    if len(baseline_logits) != len(changed_logits) or not baseline_logits:
        raise RelianceError("prediction-change inputs must have matching non-empty rows")
    changed_labels = sum(
        (_sigmoid(base) >= 0.5) != (_sigmoid(changed) >= 0.5)
        for base, changed in zip(baseline_logits, changed_logits)
    )
    n = len(baseline_logits)
    return {
        "threshold_0_5_changed_count": int(changed_labels),
        "threshold_0_5_changed_rate": changed_labels / n,
        "mean_absolute_probability_change": sum(
            abs(_sigmoid(base) - _sigmoid(changed))
            for base, changed in zip(baseline_logits, changed_logits)
        ) / n,
        "mean_absolute_logit_change": sum(
            abs(float(base) - float(changed))
            for base, changed in zip(baseline_logits, changed_logits)
        ) / n,
    }


def _metric_range(rows: Sequence[dict[str, Any]], section: str, keys: Sequence[str]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for key in keys:
        values = [float(row[section][key]) for row in rows]
        summary[key] = {
            "mean": sum(values) / len(values),
            "range": [min(values), max(values)],
        }
    return summary


def _condition_record(
    *,
    sample_ids: Sequence[str],
    labels: Sequence[Any],
    image_donors: Sequence[int],
    text_donors: Sequence[int],
    logits: Sequence[float],
    seed: int | None,
) -> dict[str, Any]:
    return {
        "seed": seed,
        "sample_ids": list(sample_ids),
        "row_labels": list(labels),
        "image_partner_ids": [sample_ids[index] for index in image_donors],
        "text_partner_ids": [sample_ids[index] for index in text_donors],
        "logits": [float(value) for value in logits],
        "probabilities": [_sigmoid(float(value)) for value in logits],
    }


def evaluate_input_shuffles(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    sample_ids: Sequence[str],
    row_labels: Sequence[Any],
    text_loader: Callable[[Any], str],
    image_loader: Callable[[Any], Any],
    batch_size: int,
    device: str,
    max_length: int,
    fusion_method: str,
    embedding_dim: int,
    seeds: Sequence[int] = DEFAULT_SEEDS,
) -> dict[str, Any]:
    """Score matched, modality-shuffled, and CPU-derived joint-shuffle rows."""
    n = len(records)
    if not 2 <= n <= MAX_SAMPLES:
        raise RelianceError(f"record count must be between 2 and {MAX_SAMPLES}")
    if len(sample_ids) != n or len(row_labels) != n:
        raise RelianceError("IDs and original row labels must match the record count")
    selected_seeds = _validated_seeds(seeds)
    if tuple(sample_ids) != tuple(_record_value(record, "sample_id") for record in records):
        raise RelianceError("record order must match the frozen manifest IDs")
    if hasattr(model, "eval"):
        model.eval()
    if getattr(model, "training", False):
        raise RelianceError("model could not be placed in evaluation mode")
    derangements = {
        seed: make_derangement(n, seed)
        for seed in selected_seeds
    }
    identity = list(range(n))
    baseline_logits = _run_condition_logits(
        model=model,
        tokenizer=tokenizer,
        records=records,
        text_loader=text_loader,
        image_loader=image_loader,
        text_donors=identity,
        image_donors=identity,
        batch_size=batch_size,
        device=device,
        max_length=max_length,
        fusion_method=fusion_method,
        embedding_dim=embedding_dim,
    )
    predictions: dict[str, dict[str, Any]] = {
        "matched_input": _condition_record(
            sample_ids=sample_ids,
            labels=row_labels,
            image_donors=identity,
            text_donors=identity,
            logits=baseline_logits,
            seed=None,
        )
    }
    baseline_metrics = binary_metrics(row_labels, baseline_logits)
    grouped_metrics: dict[str, list[dict[str, Any]]] = {
        "image_shuffled": [],
        "text_shuffled": [],
        "joint_pair_shuffle": [],
    }
    for seed in selected_seeds:
        donor_rows, attempts = derangements[seed]
        image_logits = _run_condition_logits(
            model=model,
            tokenizer=tokenizer,
            records=records,
            text_loader=text_loader,
            image_loader=image_loader,
            text_donors=identity,
            image_donors=donor_rows,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
            fusion_method=fusion_method,
            embedding_dim=embedding_dim,
        )
        text_logits = _run_condition_logits(
            model=model,
            tokenizer=tokenizer,
            records=records,
            text_loader=text_loader,
            image_loader=image_loader,
            text_donors=donor_rows,
            image_donors=identity,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
            fusion_method=fusion_method,
            embedding_dim=embedding_dim,
        )
        joint_logits = [baseline_logits[donor] for donor in donor_rows]
        for condition, logits, image_map, text_map in (
            ("image_shuffled", image_logits, donor_rows, identity),
            ("text_shuffled", text_logits, identity, donor_rows),
            ("joint_pair_shuffle", joint_logits, donor_rows, donor_rows),
        ):
            record = _condition_record(
                sample_ids=sample_ids,
                labels=row_labels,
                image_donors=image_map,
                text_donors=text_map,
                logits=logits,
                seed=seed,
            )
            key = f"{condition}_seed_{seed}"
            record["derangement_shuffle_attempts"] = attempts
            predictions[key] = record
            grouped_metrics[condition].append(
                {
                    "seed": seed,
                    "metrics": binary_metrics(row_labels, logits),
                    "prediction_change_vs_matched": _prediction_change(
                        baseline_logits, logits
                    ),
                }
            )

    metric_keys = ("auroc", "bce", "accuracy_at_0_5")
    change_keys = (
        "threshold_0_5_changed_rate",
        "mean_absolute_probability_change",
        "mean_absolute_logit_change",
    )
    condition_metrics: dict[str, Any] = {
        "matched_input": {"metrics": baseline_metrics},
    }
    for condition, rows in grouped_metrics.items():
        condition_metrics[condition] = {
            "per_seed": rows,
            "across_seeds": {
                "metrics": _metric_range(rows, "metrics", metric_keys),
                "prediction_change_vs_matched": _metric_range(
                    rows, "prediction_change_vs_matched", change_keys
                ),
            },
        }
    return {
        "predictions": predictions,
        "condition_metrics": condition_metrics,
        "derangements": {
            str(seed): {
                "seed": seed,
                "donor_row_indices": derangements[seed][0],
                "shuffle_attempts_until_no_fixed_points": derangements[seed][1],
                "donor_sample_ids": [sample_ids[index] for index in derangements[seed][0]],
            }
            for seed in selected_seeds
        },
        "joint_shuffle_method": (
            "CPU-only paired-input/label-decoupling diagnostic: for each target row i, "
            "reuse matched-input logits from the donor row selected by the same seed"
        ),
    }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_code_hashes(project_root: Path) -> dict[str, str]:
    relative_paths = (
        Path(__file__).resolve().relative_to(project_root),
        Path("src/autoresearch.py"),
        Path("src/analyses/autoresearch_data.py"),
        Path("src/config.py"),
        Path("src/datasets/dataset_utils.py"),
        Path("src/augments_transforms.py"),
        Path("src/trainer/base_trainer.py"),
        Path("src/trainer/hm_trainer.py"),
        Path("src/vilbert.py"),
        Path("src/attention.py"),
        Path("src/logger.py"),
        Path("autoresearch/task_reliance_protocol.md"),
    )
    result = {}
    for relative in relative_paths:
        path = project_root / relative
        result[str(relative)] = _sha256_file(path.resolve(strict=True))
    return result


def _write_json_create_once(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n"
    if path.exists() or path.is_symlink():
        raise RelianceError(f"refusing to replace existing result: {path}")
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
        raise RelianceError(f"refusing to replace existing result: {path}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure checkpoint sensitivity to Hateful Memes input pairing."
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tokenizer-path", required=True)
    parser.add_argument("--device", required=True, help="explicitly choose cpu or cuda[:N]")
    parser.add_argument("--batch-size", required=True, type=int)
    parser.add_argument("--n", type=int, default=500)
    parser.add_argument(
        "--seeds", nargs=3, type=int, default=list(DEFAULT_SEEDS), metavar=("S1", "S2", "S3")
    )
    return parser


def _load_hm_data_helpers_in_output_directory(
    output_dir: Path,
) -> tuple[Callable[..., Any], ...]:
    with __import__("autoresearch")._working_directory(output_dir):
        from analyses.autoresearch_data import (
            close_validation_index,
            load_hateful_memes_item,
            manifest_sha256,
            open_validation_index,
            read_manifest,
            validate_manifest_against_index,
        )
    return (
        close_validation_index,
        load_hateful_memes_item,
        manifest_sha256,
        open_validation_index,
        read_manifest,
        validate_manifest_against_index,
    )


def run(args: argparse.Namespace) -> Path:
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise RelianceError(f"--batch-size must be between 1 and {MAX_BATCH_SIZE}")
    if args.device not in ("cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", args.device):
        raise RelianceError("--device must be cpu, cuda, or cuda:N")
    seeds = _validated_seeds(args.seeds)
    if seeds != DEFAULT_SEEDS:
        raise RelianceError("the predeclared seeds are fixed at 101, 202, and 303")
    output_root = authorized_output_root(PROJECT_ROOT)
    data_root = Path(args.data_root)
    try:
        resolved_data = data_root.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RelianceError(f"data root does not exist: {data_root}") from exc
    if not resolved_data.is_dir():
        raise RelianceError(f"data root must be a directory: {resolved_data}")

    checkpoint = _safe_input_file(args.checkpoint, "checkpoint")
    if "finetuned_hateful_memes" not in checkpoint.name:
        raise RelianceError("--checkpoint must be a saved fine-tuned Hateful Memes model")
    manifest_path = _safe_input_file(args.manifest, "sample manifest")
    frozen_manifest = FROZEN_MANIFEST.resolve(strict=True)
    if manifest_path != frozen_manifest:
        raise RelianceError(f"--manifest must be the frozen cohort at {frozen_manifest}")
    tokenizer_path = Path(args.tokenizer_path)
    try:
        tokenizer_source = tokenizer_path.resolve(strict=True)
    except FileNotFoundError as exc:
        raise RelianceError(f"tokenizer path does not exist: {tokenizer_path}") from exc
    if not tokenizer_source.is_dir():
        raise RelianceError("--tokenizer-path must name a local tokenizer directory")

    owned_output_root = PROJECT_ROOT / "autoresearch/task-reliance"
    requested_output = Path(os.path.abspath(args.output_dir))
    if requested_output.parent != owned_output_root:
        raise RelianceError(
            f"--output-dir must be one new model directory directly under {owned_output_root}"
        )
    if owned_output_root.is_symlink() or requested_output.is_symlink():
        raise RelianceError("the task-reliance output directory must not use symlinks")
    output_dir = validate_output_destination(
        requested_output,
        authorized_root=output_root,
        data_root=resolved_data,
        input_paths=(checkpoint, manifest_path, tokenizer_source),
    )
    result_path = output_dir / RESULT_FILENAME
    try:
        safe_output_path(
            result_path,
            authorized_root=output_root,
            data_root=resolved_data,
        )
    except EvaluatorError as exc:
        raise RelianceError(str(exc)) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = safe_output_path(
        output_dir,
        authorized_root=output_root,
        data_root=resolved_data,
        require_directory=True,
    )
    (
        close_validation_index,
        load_hateful_memes_item,
        manifest_sha256,
        open_validation_index,
        read_manifest,
        validate_manifest_against_index,
    ) = _load_hm_data_helpers_in_output_directory(output_dir)
    manifest = read_manifest(manifest_path)
    sample_ids = validate_hm_manifest_protocol(manifest, requested_n=args.n)
    index = open_validation_index("hateful_memes", "dev_seen", resolved_data)
    try:
        validate_manifest_against_index(manifest, index)
        records_by_id = index.by_id()
        records = [records_by_id[sample_id] for sample_id in sample_ids]
        row_labels = [_record_payload(record)["label"] for record in records]
        _normalized_binary_labels(row_labels)
        checkpoint_sha256 = _sha256_file(checkpoint)
        manifest_hash = manifest_sha256(manifest)
        code_hashes = _source_code_hashes(PROJECT_ROOT)

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with __import__("autoresearch")._working_directory(output_dir):
            tokenizer, tokenizer_identity = _load_local_tokenizer(
                str(tokenizer_source),
                output_dir,
                authorized_root=output_root,
                data_root=resolved_data,
            )
            model, architecture, provenance = _load_model_without_pretrained_downloads(
                checkpoint,
                args.device,
                working_directory=output_dir,
            )
            from config import CLS_FUSION_METHOD, TOKENIZER_MAX_LEN
            from datasets.dataset_utils import process_image

            embedding_dim = int(model.config.embedding_dim)
            head_dimensions = validate_head_dimensions(
                model.fc, CLS_FUSION_METHOD, embedding_dim
            )

            def text_loader(record: Any) -> str:
                return load_hateful_memes_item(record)[0]

            def image_loader(record: Any) -> Any:
                processed = process_image(
                    load_hateful_memes_item(record)[1], transform=None
                )
                if processed is None or processed.get("pixel_values") is None:
                    raise RelianceError(
                        f"image preprocessing failed for {_record_value(record, 'sample_id')}"
                    )
                return processed["pixel_values"].squeeze(0)

            evaluated = evaluate_input_shuffles(
                model=model,
                tokenizer=tokenizer,
                records=records,
                sample_ids=sample_ids,
                row_labels=row_labels,
                text_loader=text_loader,
                image_loader=image_loader,
                batch_size=args.batch_size,
                device=args.device,
                max_length=int(TOKENIZER_MAX_LEN),
                fusion_method=CLS_FUSION_METHOD,
                embedding_dim=embedding_dim,
                seeds=seeds,
            )
        source_samples = {item["id"]: item for item in manifest["samples"]}
        result = {
            "format": "autoresearch-hateful-memes-input-shuffle-v1",
            "created_by": "src/autoresearch_task_reliance.py",
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
                "architecture": architecture,
                "provenance": provenance,
            },
            "cohort": {
                "task": "hateful_memes",
                "split": "dev_seen",
                "validation_role": manifest["validation_role"],
                "interpretation": "historical discovery cohort, not independent confirmation",
                "n": len(sample_ids),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "ordered_sample_ids": sample_ids,
                "row_labels": list(row_labels),
                "source_file_sha256": dict(manifest["source_data_sha256"]),
                "selected_image_sha256": {
                    sample_id: source_samples[sample_id]["image_sha256"]
                    for sample_id in sample_ids
                },
            },
            "protocol": {
                "predeclared_seeds": list(seeds),
                "derangement_rule": (
                    "Python random.Random(seed).shuffle(range(n)); redraw with the same "
                    "seeded generator until every target row has a different donor"
                ),
                "label_reference": "original target row for every condition",
                "baseline": "matched image and text from the original row",
                "image_shuffle": "original text with image from the seed's donor row",
                "text_shuffle": "original image with text from the seed's donor row",
                "joint_shuffle": evaluated["joint_shuffle_method"],
                "primary_metric": "AUROC",
                "secondary_metrics": ["binary cross entropy", "accuracy at fixed probability 0.5"],
                "prediction_change": (
                    "per row versus matched input: threshold-0.5 label-change rate and "
                    "mean absolute probability and logit changes"
                ),
                "seed_summary": "per-seed values plus mean and min/max range over the three fixed derangements",
                "limits": [
                    "functional sensitivity on this selected validation cohort only",
                    "does not identify a causal training architecture effect",
                    "does not estimate unimodal information or general reliability",
                    "derangement spread is conditional input-pairing variation, not independent-model uncertainty",
                ],
            },
            "preprocessing": {
                "image": "datasets.dataset_utils.process_image(transform=None), project minimal ViT normalization",
                "text": "local BERT tokenizer; max_length from config.TOKENIZER_MAX_LEN; fixed-length padding and truncation",
                "tokenizer_identity": tokenizer_identity,
            },
            "head_and_fusion": {
                "source_formula": "src/trainer/base_trainer.py:get_final_representation",
                "fusion_method": CLS_FUSION_METHOD,
                "formula": {
                    "concat": "torch.cat((text_cls, image_cls), dim=1)",
                    "sum": "text_cls + image_cls",
                    "hadamard": "text_cls * image_cls",
                }[CLS_FUSION_METHOD],
                "head": "model.fc applied to final text/image CLS fusion; output is one logit",
                "embedding_dim": embedding_dim,
                "linear_dimensions": head_dimensions,
            },
            "code_sha256": code_hashes,
            "analysis": {
                "predictions": evaluated["predictions"],
                "condition_metrics": evaluated["condition_metrics"],
                "derangements": evaluated["derangements"],
            },
        }
        _write_json_create_once(result_path, result)
        return result_path
    finally:
        close_validation_index(index)


def main(argv: Sequence[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        result = run(args)
    except (EvaluatorError, RelianceError, OSError, RuntimeError, ValueError) as exc:
        print(f"input-shuffle diagnostic failed: {exc}", file=sys.stderr)
        return 2
    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

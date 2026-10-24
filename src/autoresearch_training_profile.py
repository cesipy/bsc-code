#!/usr/bin/env python3
"""Profile one real ViLBERT AP+MLM+MIM optimizer update variant.

Each process validates the same offline 20-example CC training manifest and
checkpoint, times the real zero-worker data pipeline separately, then reuses
its in-memory batches for two warmup and eight measured joint updates. Data
wait and cached full-model timings are separate and must not be summed.
This is a bounded engineering profile, not a pretraining reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import resource
import sys
import time
from typing import Any, Callable, Mapping, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = ROOT / "autoresearch/manifests/profile-cc-train20-v1.json"
DEFAULT_CHECKPOINT = Path(
    "/home/cedric.sillaber/coding/bsc-code/res/checkpoints/pretrains/"
    "20261012-122011_pretrained_123.pt"
)
MANIFEST_SHA256 = "1739c83baab879dc3a4079276f911ca94f299e8c7519114b7f84df1c435ac385"
CHECKPOINT_SHA256 = "ac2da984bebae43e22165ec4e46ddae569bd0e0880464108c80b1c5748407e5e"
VERSION = "vilbert-three-objective-update-profile-v1"
VARIANTS = {
    "eager_dense": ("eager", False),
    "eager_selected": ("eager", True),
    "sdpa_selected": ("sdpa", True),
}
PROFILE_BATCH = 2
ACCUMULATION = 1
WARMUP = 2
MEASURED = 8
UPDATES = WARMUP + MEASURED
MAX_SECONDS = 600.0
MAX_RSS = 8 * 1024**3
MAX_ARTIFACT = 100 * 1024**2
THREADS = 4
FUSION_TEXT = [3, 4, 5]
FUSION_VISION = [9, 10, 11]
PROTECTED_ROOTS = (
    Path("/home/cedric.sillaber/coding/bsc-code"),
    Path("/home/cedric.sillaber/coding/bsc-code-research-material"),
)
SOURCE_PATHS = (
    "src/autoresearch_training_profile.py",
    "src/config.py",
    "src/vilbert.py",
    "src/attention.py",
    "src/trainer/trainers.py",
    "src/trainer/mlm_loss.py",
    "src/datasets/dataset_pretrain.py",
    "src/datasets/dataset_utils.py",
    "src/augments_transforms.py",
)


class ProfileError(RuntimeError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> str:
    actual = sha256_file(path)
    if actual != expected:
        raise ProfileError(f"{label} SHA-256 mismatch: expected {expected}, got {actual}")
    return actual


def _rows_at(path: Path, indices: set[int]) -> dict[int, dict[str, str]]:
    found: dict[int, dict[str, str]] = {}
    with path.open("r", encoding="utf-8", newline="") as stream:
        for index, row in enumerate(csv.DictReader(stream)):
            if index in indices:
                found[index] = row
    if set(found) != indices:
        raise ProfileError(f"manifest references missing data rows in {path}")
    return found


def validate_manifest(
    manifest_path: str | Path,
    *,
    expected_sha256: str | None = MANIFEST_SHA256,
) -> tuple[dict[str, Any], str]:
    path = Path(manifest_path).resolve(strict=True)
    manifest_sha = sha256_file(path)
    if expected_sha256 is not None and manifest_sha != expected_sha256:
        raise ProfileError(f"manifest SHA-256 mismatch: expected {expected_sha256}, got {manifest_sha}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    samples = manifest.get("samples", [])
    if manifest.get("schema_version") != "profile-cc-train20-v1":
        raise ProfileError("unsupported manifest schema")
    if manifest.get("split") != "training-only" or manifest.get("sample_count") != 20 or len(samples) != 20:
        raise ProfileError("manifest must contain exactly 20 training-only samples")
    if manifest.get("train_csv_row_indexing") != "1-based physical CSV line number including the header line":
        raise ProfileError("unexpected frozen CSV row-index convention")

    source_csv = Path(manifest["source_csv_path"]).resolve(strict=True)
    selection_csv = Path(manifest["selection_csv_path"]).resolve(strict=True)
    _verify_hash(source_csv, manifest["source_csv_sha256"], "train.csv")
    _verify_hash(selection_csv, manifest["selection_csv_sha256"], "train-local.csv")
    image_root = Path(manifest["image_root"]).resolve(strict=True)
    ids, image_hashes, caption_hashes = set(), set(), set()
    source_indices, selection_indices = set(), set()
    for sample in samples:
        sid = sample["sample_id"]
        caption = sample["caption"]
        relpath = sample["image_path_rel"]
        image_path = Path(sample["image_path_abs"]).resolve(strict=True)
        if sample["image_path_absolute"] != str(image_path):
            raise ProfileError(f"absolute image-path aliases differ for {sid}")
        if sample["image_path_relative_to_manual_root"] != relpath:
            raise ProfileError(f"relative image-path aliases differ for {sid}")
        if not image_path.is_relative_to(image_root):
            raise ProfileError(f"image path escapes image root for {sid}")
        if hashlib.sha256(caption.encode("utf-8")).hexdigest() != sample["caption_sha256"]:
            raise ProfileError(f"caption hash mismatch for {sid}")
        if image_path.stat().st_size != sample["image_size_bytes"] or sha256_file(image_path) != sample["image_sha256"]:
            raise ProfileError(f"image size/hash mismatch for {sid}")
        if sample.get("image_dimensions") != [224, 224]:
            raise ProfileError(f"profile image is not 224x224 for {sid}")
        ids.add(sid)
        image_hashes.add(sample["image_sha256"])
        caption_hashes.add(sample["caption_sha256"])
        source_indices.add(int(sample["train_source_row_index_zero_based"]))
        selection_indices.add(int(sample["train_local_row_index_zero_based"]))
    if any(len(values) != 20 for values in (ids, image_hashes, caption_hashes, source_indices, selection_indices)):
        raise ProfileError("sample IDs, content hashes, and source row indices must be unique")

    source_rows = _rows_at(source_csv, source_indices)
    selection_rows = _rows_at(selection_csv, selection_indices)
    for sample in samples:
        for rows, index_key in (
            (source_rows, "train_source_row_index_zero_based"),
            (selection_rows, "train_local_row_index_zero_based"),
        ):
            row = rows[int(sample[index_key])]
            if row.get("text") != sample["caption"] or row.get("filepath") != sample["image_path_rel"]:
                raise ProfileError(f"CSV membership mismatch for {sample['sample_id']}")

    tokenizer_path = Path(manifest["tokenizer_path"]).resolve(strict=True)
    tokenizer = manifest["tokenizer"]
    tokenizer_hashes = manifest["tokenizer_source_files_sha256"]
    if tokenizer.get("local_path") != str(tokenizer_path) or tokenizer.get("max_length") != 197:
        raise ProfileError("tokenizer path/length does not match the frozen profile")
    if tokenizer.get("source_file_sha256") != tokenizer_hashes:
        raise ProfileError("tokenizer hash maps disagree")
    for name, expected in tokenizer_hashes.items():
        _verify_hash(tokenizer_path / name, expected, f"tokenizer file {name}")
    bert_config = Path(manifest["bert_model_config_path"]).resolve(strict=True)
    if not bert_config.is_file():
        raise ProfileError("local offline BERT architecture config is unavailable")

    manifest["_identity"] = {
        "manifest_sha256": manifest_sha,
        "source_csv_sha256": manifest["source_csv_sha256"],
        "selection_csv_sha256": manifest["selection_csv_sha256"],
        "sample_ids": [sample["sample_id"] for sample in samples],
        "tokenizer_source_files_sha256": tokenizer_hashes,
        "bert_config_sha256": sha256_file(bert_config),
        "vision_architecture": "timm vit_base_patch16_224 constructed with pretrained=False; strict checkpoint supplies weights",
    }
    return manifest, manifest_sha


def variant_settings(variant: str) -> dict[str, Any]:
    if variant not in VARIANTS:
        raise ProfileError(f"unknown variant {variant!r}")
    backend, selected = VARIANTS[variant]
    return {"cross_attention_backend": backend, "mlm_selected_tokens_only": selected}


def validate_fusion_config(config: Mapping[str, Any]) -> None:
    if list(config.get("text_cross_attention_layers", [])) != FUSION_TEXT:
        raise ProfileError("checkpoint text fusion indices must be [3, 4, 5]")
    if list(config.get("vision_cross_attention_layers", [])) != FUSION_VISION:
        raise ProfileError("checkpoint vision fusion indices must be [9, 10, 11]")


def _configure_offline() -> None:
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = str(THREADS)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    sys.dont_write_bytecode = True


def _seed(seed: int, torch: Any, np: Any, *, device_type: str = "cpu") -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    # Avoid torch.manual_seed's all-device CUDA seeding on the CPU validation path.
    torch.random.default_generator.manual_seed(seed)
    if device_type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _hash_tree(value: Any, digest: Any, torch: Any) -> None:
    if isinstance(value, torch.Tensor):
        tensor = value.detach().contiguous().cpu()
        digest.update(b"T" + str(tensor.dtype).encode() + b"\\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode("utf-8") + b"\\0")
            _hash_tree(value[key], digest, torch)
    elif isinstance(value, (list, tuple)):
        digest.update(b"L")
        for item in value:
            _hash_tree(item, digest, torch)
    elif value is None or isinstance(value, (str, int, float, bool)):
        digest.update(json.dumps(value, separators=(",", ":")).encode() + b"\\0")
    else:
        digest.update(repr(value).encode() + b"\\0")


def hash_tensor_tree(value: Any, torch: Any) -> str:
    digest = hashlib.sha256()
    _hash_tree(value, digest, torch)
    return digest.hexdigest()


def _hash_state(state: Mapping[str, Any], torch: Any) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update(name.encode() + b"\\0" + str(tensor.dtype).encode() + b"\\0")
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode() + b"\\0")
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _silent_logger() -> None:
    import types

    class SilentLogger:
        def info(self, *_: Any, **__: Any) -> None:
            return None

        warn = info
        error = info

    module = types.ModuleType("logger")
    module.Logger = SilentLogger
    sys.modules["logger"] = module


def _make_loaders(manifest: Mapping[str, Any], tokenizer: Any, torch: Any, seed: int) -> tuple[dict[str, Any], dict[str, list[str]]]:
    _silent_logger()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from torch.utils.data import DataLoader
    from datasets.dataset_pretrain import PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
    import augments_transforms

    data = [(row["image_path_abs"], row["caption"]) for row in manifest["samples"]]
    specs = (
        ("ap", PretrainDatasetAP, {"is_train": True}),
        ("mlm", PretrainDatasetMLM, {}),
        ("mim", PretrainDatasetMIM, {
            "transforms_weak": augments_transforms.get_transform_unmasked(),
            "transforms_strong": augments_transforms.get_transform_masked(),
        }),
    )
    datasets = {}
    for name, dataset_type, extra in specs:
        random.seed(seed)
        datasets[name] = dataset_type(data, tokenizer, None, **extra)

    id_for_path = {row["image_path_abs"]: row["sample_id"] for row in manifest["samples"]}
    order = {}
    first_order = None
    for name, dataset in datasets.items():
        current = [id_for_path[row[1]] for row in dataset.data]
        if first_order is not None and current != first_order:
            raise ProfileError("real task datasets did not preserve the same sample order")
        first_order = current
        order[name] = current
    loaders = {
        name: DataLoader(dataset, batch_size=PROFILE_BATCH, shuffle=False, num_workers=0, pin_memory=False)
        for name, dataset in datasets.items()
    }
    if any(len(loader) != UPDATES for loader in loaders.values()):
        raise ProfileError("each real loader must yield ten fixed batch-size-two batches")
    return loaders, order


def _validate_batch(batch: Mapping[str, Any], task: str, torch: Any) -> dict[str, int]:
    if task == "mim":
        mask = batch["masked_patches_idxs"]
        if tuple(mask.shape) != (2, 196):
            raise ProfileError("MIM mask must have shape [2,196]")
        masked = mask.ne(0).sum(dim=1)
        unmasked = mask.eq(0).sum(dim=1)
        if torch.any(masked == 0) or torch.any(unmasked == 0):
            raise ProfileError("each MIM sample must contain masked and unmasked patches")
        return {"masked_patches": int(masked.sum().item()), "unmasked_patches": int(unmasked.sum().item())}
    if task == "mlm":
        supervised = int(batch["label"].ne(-100).sum().item())
        if supervised == 0:
            raise ProfileError("fixed MLM batch has no supervised tokens")
        return {"supervised_tokens": supervised}
    if "label" not in batch or "img" not in batch or "text" not in batch:
        raise ProfileError("AP batch is missing required project fields")
    return {"samples": PROFILE_BATCH}


def materialize_batches(manifest: Mapping[str, Any], tokenizer: Any, torch: Any, np: Any) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _seed(int(manifest["rng_seed"]), torch, np)
    loaders, sample_order = _make_loaders(manifest, tokenizer, torch, int(manifest["rng_seed"]))
    iterator = {name: iter(loader) for name, loader in loaders.items()}
    waits = {name: [] for name in ("ap", "mlm", "mim")}
    counts = {name: [] for name in waits}
    batches = []
    for _ in range(UPDATES):
        triad = {}
        for task in ("ap", "mlm", "mim"):
            started = time.perf_counter()
            batch = next(iterator[task])
            waits[task].append(time.perf_counter() - started)
            counts[task].append(_validate_batch(batch, task, torch))
            triad[task] = batch
        batches.append(triad)
    timings = {
        task: {"per_batch_seconds": values, "total_seconds": sum(values), "mean_seconds": sum(values) / UPDATES}
        for task, values in waits.items()
    }
    timings["triad_total_seconds"] = sum(sum(waits[task][i] for task in waits) for i in range(UPDATES))
    timings["scope"] = "real Dataset/DataLoader next() with zero workers; decode, transform, and tokenization; separate from cached model steps"
    return batches, {
        "input_batch_sha256": hash_tensor_tree(batches, torch),
        "sample_order_ids": sample_order,
        "objective_counts_by_update": counts,
        "data_wait": timings,
    }


def _load_model(checkpoint_path: Path, manifest: Mapping[str, Any], variant: str, torch: Any) -> tuple[Any, Any, dict[str, Any]]:
    _verify_hash(checkpoint_path, CHECKPOINT_SHA256, "checkpoint")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    saved = payload.get("config")
    if hasattr(saved, "to_dict"):
        saved = saved.to_dict()
    elif not isinstance(saved, Mapping):
        saved = vars(saved) if hasattr(saved, "__dict__") else None
    if not isinstance(saved, Mapping) or "model_state_dict" not in payload:
        raise ProfileError("checkpoint must contain config and model_state_dict")
    saved = dict(saved)
    validate_fusion_config(saved)
    if saved.get("use_contrastive_loss", False):
        raise ProfileError("checkpoint enables contrastive AP loss; profile contract is BCE")

    _silent_logger()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from config import ViLBERTConfig
    from transformers import BertConfig, BertModel
    import timm
    from vilbert import ViLBERT

    config = ViLBERTConfig.from_dict(saved)
    backend, selected = VARIANTS[variant]
    config.cross_attention_backend = backend
    config.mlm_selected_tokens_only = selected
    config.pretrain_batch_size = PROFILE_BATCH
    config.gradient_accumulation = ACCUMULATION
    config.learning_rate = float(saved.get("learning_rate", 1e-4))
    if not math.isfinite(config.learning_rate) or config.learning_rate <= 0:
        raise ProfileError("checkpoint learning rate is invalid")
    bert_config = BertConfig.from_pretrained(manifest["bert_model_config_path"], local_files_only=True)
    create_model = timm.create_model
    import inspect
    bert_from_pretrained = inspect.getattr_static(BertModel, "from_pretrained")

    def no_pretrained_vision(name: str, *args: Any, **kwargs: Any) -> Any:
        kwargs["pretrained"] = False
        return create_model(name, *args, **kwargs)

    def no_pretrained_bert(cls: Any, *args: Any, **kwargs: Any) -> Any:
        return cls(bert_config)

    BertModel.from_pretrained = classmethod(no_pretrained_bert)
    timm.create_model = no_pretrained_vision
    try:
        model = ViLBERT(config)
    finally:
        BertModel.from_pretrained = bert_from_pretrained
        timm.create_model = create_model
    model.load_state_dict(payload["model_state_dict"], strict=True)
    initial_hash = _hash_state(model.state_dict(), torch)
    del payload
    loaded = {
        "initial_model_state_sha256": initial_hash,
        "text_cross_attention_layers": list(model.config.text_cross_attention_layers),
        "vision_cross_attention_layers": list(model.config.vision_cross_attention_layers),
        "cross_attention_backend": backend,
        "mlm_selected_tokens_only": selected,
        "learning_rate": config.learning_rate,
    }
    if loaded["text_cross_attention_layers"] != FUSION_TEXT or loaded["vision_cross_attention_layers"] != FUSION_VISION:
        raise ProfileError("strict-loaded model fusion placement differs from saved config")
    return model, config, loaded


def _make_trainer(model: Any, config: Any, device_type: str, torch: Any) -> Any:
    _silent_logger()
    source = str(ROOT / "src")
    if source not in sys.path:
        sys.path.insert(0, source)
    from trainer.trainers import PretrainingTrainer
    from info_nce import InfoNCE
    from torch import nn

    trainer = PretrainingTrainer.__new__(PretrainingTrainer)
    trainer.lr = config.learning_rate
    trainer.optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    trainer.device = device_type
    trainer.model = model
    trainer.use_contrastive_loss = False
    trainer.loss_fn_alignment = nn.BCEWithLogitsLoss()
    trainer.loss_fn_mlm = nn.CrossEntropyLoss()
    trainer.loss_fn_mim = InfoNCE()
    trainer.loss_info_nce = InfoNCE()
    trainer.scaler = torch.amp.grad_scaler.GradScaler(device=device_type)
    trainer.config = config
    trainer.gradient_accumulation = ACCUMULATION
    trainer.scheduler = None
    trainer.input_buffer = []
    return trainer


def run_joint_update(
    trainer: Any,
    triad: Mapping[str, Any],
    *,
    synchronize: Callable[[], None],
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, Any]:
    synchronize()
    started = clock()
    losses, task_seconds = {}, {}
    methods = (
        ("ap", "train_alignment_prediction_batch"),
        ("mlm", "train_mlm_batch"),
        ("mim", "train_mim_batch"),
    )
    for task, method in methods:
        task_started = clock()
        value = float(getattr(trainer, method)(triad[task], flag_optimizer=False))
        task_seconds[task] = clock() - task_started
        if not math.isfinite(value):
            raise ProfileError(f"nonfinite {task} loss: {value}")
        losses[task] = value
    tracked = next(parameter for parameter in trainer.model.parameters() if parameter.requires_grad)
    if tracked.grad is None:
        raise ProfileError("joint objectives produced no gradient for the tracked parameter")
    before = trainer.optimizer.state.get(tracked, {}).get("step", 0)
    before = before.item() if hasattr(before, "item") else before
    update_started = clock()
    trainer.scaler.step(trainer.optimizer)
    trainer.scaler.update()
    trainer.optimizer.zero_grad()
    synchronize()
    update_seconds = clock() - update_started
    after = trainer.optimizer.state.get(tracked, {}).get("step", 0)
    after = after.item() if hasattr(after, "item") else after
    return {
        "task_losses": losses,
        "task_seconds": task_seconds,
        "optimizer_update_seconds": update_seconds,
        "cached_full_update_seconds": clock() - started,
        "actual_optimizer_step": float(after) > float(before),
        "tracked_parameter_optimizer_step": int(after),
    }


def _rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)


def _check_budget(started: float, phase: str) -> None:
    if time.monotonic() - started > MAX_SECONDS:
        raise ProfileError(f"600-second job limit exceeded during {phase}")
    if _rss_bytes() > MAX_RSS:
        raise ProfileError(f"8-GiB RSS limit exceeded during {phase}")


def _source_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in SOURCE_PATHS if (ROOT / path).is_file()}


def run_profile(manifest_path: str | Path, checkpoint_path: str | Path, variant: str, device_name: str) -> dict[str, Any]:
    started = time.monotonic()
    settings = variant_settings(variant)
    manifest, manifest_sha = validate_manifest(manifest_path)
    checkpoint_path = Path(checkpoint_path).resolve(strict=True)
    if checkpoint_path != DEFAULT_CHECKPOINT.resolve(strict=True):
        raise ProfileError("only the frozen parity-replay checkpoint is allowed")
    checkpoint_sha = _verify_hash(checkpoint_path, CHECKPOINT_SHA256, "checkpoint")
    _configure_offline()
    import numpy as np
    import torch
    from transformers import BertTokenizerFast

    torch.set_num_threads(THREADS)
    try:
        torch.set_num_interop_threads(THREADS)
    except RuntimeError:
        pass
    if device_name == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        device = torch.device("cpu")
        device_type = "cpu"
        synchronize = lambda: None
        allocator_fraction = None
    else:
        if not device_name.startswith("cuda:") or not torch.cuda.is_available():
            raise ProfileError("device must be cpu or an available explicit cuda:N")
        device = torch.device(device_name)
        if device.index is None or device.index >= torch.cuda.device_count():
            raise ProfileError(f"invalid CUDA device {device_name}")
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(0.30, device=device)
        device_type = "cuda"
        synchronize = lambda: torch.cuda.synchronize(device)
        allocator_fraction = 0.30
    _check_budget(started, "runtime setup")

    tokenizer = BertTokenizerFast.from_pretrained(manifest["tokenizer_path"], local_files_only=True)
    if tokenizer.vocab_size != 30522 or tokenizer.mask_token_id is None:
        raise ProfileError("offline tokenizer does not match the frozen BERT vocabulary")
    data_started = time.monotonic()
    batches, data_report = materialize_batches(manifest, tokenizer, torch, np)
    data_report["materialization_wall_seconds"] = time.monotonic() - data_started
    _check_budget(started, "real data pipeline")

    model, config, checkpoint_report = _load_model(checkpoint_path, manifest, variant, torch)
    model.to(device)
    model.train()
    if device_type == "cuda":
        synchronize()
        torch.cuda.reset_peak_memory_stats(device)
    trainer = _make_trainer(model, config, device_type, torch)
    tracked_name = next(name for name, p in model.named_parameters() if p.requires_grad)
    training_seed = int(manifest["rng_seed"]) + 1
    _seed(training_seed, torch, np, device_type=device_type)
    batch_hash = data_report["input_batch_sha256"]
    step_results = []
    for index, triad in enumerate(batches):
        _check_budget(started, f"update {index + 1} preflight")
        if device_type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        result = run_joint_update(trainer, triad, synchronize=synchronize)
        result.update({
            "update_index": index + 1,
            "phase": "warmup" if index < WARMUP else "measured",
            "data_wait_seconds": {
                task: data_report["data_wait"][task]["per_batch_seconds"][index]
                for task in ("ap", "mlm", "mim")
            },
            "processed_samples_per_objective": PROFILE_BATCH,
            "rss_peak_bytes": _rss_bytes(),
        })
        if device_type == "cuda":
            result["gpu_peak_allocated_bytes"] = int(torch.cuda.max_memory_allocated(device))
            result["gpu_peak_reserved_bytes"] = int(torch.cuda.max_memory_reserved(device))
        else:
            result["gpu_peak_allocated_bytes"] = None
            result["gpu_peak_reserved_bytes"] = None
        result["optimizer_step_status"] = "applied" if result["actual_optimizer_step"] else "skipped_by_grad_scaler"
        step_results.append(result)
        _check_budget(started, f"update {index + 1}")

    if hash_tensor_tree(batches, torch) != batch_hash:
        raise ProfileError("fixed cached input batches mutated during the run")
    final_hash = _hash_state(model.state_dict(), torch)
    actual_steps = sum(bool(step["actual_optimizer_step"]) for step in step_results)
    measured = [step for step in step_results if step["phase"] == "measured"]
    if len(measured) != MEASURED:
        raise ProfileError("internal measured-step count mismatch")
    gpu_peak = None
    if device_type == "cuda":
        gpu_peak = {
            "allocated_bytes": max(step["gpu_peak_allocated_bytes"] for step in step_results),
            "reserved_bytes": max(step["gpu_peak_reserved_bytes"] for step in step_results),
        }
    durations = sorted(step["cached_full_update_seconds"] for step in measured)
    return {
        "schema_version": VERSION,
        "status": "completed",
        "variant": variant,
        "variant_settings": settings,
        "inputs": {
            "manifest_path": str(Path(manifest_path).resolve()),
            **manifest["_identity"],
            "input_batch_sha256": batch_hash,
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": checkpoint_sha,
            **checkpoint_report,
            "tracked_parameter_for_optimizer_step_count": tracked_name,
            "data_rng_seed": int(manifest["rng_seed"]),
            "training_rng_seed": training_seed,
            "model_source_sha256": _source_hashes(),
        },
        "profile": {
            "is_engineering_profile_not_recipe_replication": True,
            "objectives": ["AP", "MLM", "MIM"],
            "joint_update": "AP backward + MLM backward + both masked and unmasked MIM forward/backward paths + one AdamW update",
            "batch_size_per_objective": PROFILE_BATCH,
            "gradient_accumulation": ACCUMULATION,
            "warmup_joint_updates": WARMUP,
            "measured_joint_updates": MEASURED,
            "total_joint_updates": UPDATES,
            "processed_samples_per_objective": PROFILE_BATCH * UPDATES,
            "optimizer": "project AdamW, weight_decay=0.01, one GradScaler.step/update/zero_grad after each triad",
            "learning_rate": float(config.learning_rate),
            "learning_rate_source": "checkpoint config; constant during profile, long-run scheduler omitted",
            "amp": "same autocast and GradScaler construction/device type as the actual PretrainingTrainer",
            "grad_scaler_enabled": bool(trainer.scaler.is_enabled()),
            "grad_scaler_scale_after_run": float(trainer.scaler.get_scale()),
            "data_wait_and_cached_update_are_separate": True,
            "checkpoint_or_optimizer_state_written": False,
            "test_loader_or_mlflow_used": False,
        },
        "environment": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "autocast_device_type": device_type,
            "cpu_threads": torch.get_num_threads(),
            "cpu_interop_threads": torch.get_num_interop_threads(),
            "cuda_allocator_fraction": allocator_fraction,
        },
        "data_pipeline": data_report,
        "timings": {
            "mean_measured_cached_full_update_seconds": sum(durations) / len(durations),
            "median_measured_cached_full_update_seconds": (durations[3] + durations[4]) / 2,
            "mean_measured_optimizer_update_seconds": sum(s["optimizer_update_seconds"] for s in measured) / MEASURED,
            "max_measured_gpu_peak": gpu_peak,
            "elapsed_job_seconds": time.monotonic() - started,
            "rss_peak_bytes": _rss_bytes(),
            "do_not_sum_data_wait_and_cached_step": True,
        },
        "optimizer_accounting": {
            "actual_steps": actual_steps,
            "skipped_steps": len(step_results) - actual_steps,
            "all_expected_steps_applied": actual_steps == UPDATES,
            "tracked_parameter_final_step": step_results[-1]["tracked_parameter_optimizer_step"],
        },
        "weights": {
            "initial_state_sha256": checkpoint_report["initial_model_state_sha256"],
            "final_state_sha256": final_hash,
            "changed": final_hash != checkpoint_report["initial_model_state_sha256"],
        },
        "steps": step_results,
    }


def authorized_output_path(path: str | Path) -> Path:
    target = Path(path)
    if not target.is_absolute():
        raise ProfileError("--output must be absolute")
    output_root = (ROOT / "autoresearch").resolve(strict=True)
    resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(output_root):
        raise ProfileError(f"output must stay under {output_root}")
    for protected in PROTECTED_ROOTS:
        if resolved.is_relative_to(protected.resolve(strict=False)):
            raise ProfileError("output overlaps a read-only workspace")
    if target.exists() or target.is_symlink():
        raise ProfileError(f"refusing to overwrite existing output: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    resolved = target.resolve(strict=False)
    if not resolved.is_relative_to(output_root):
        raise ProfileError("output parent resolved outside autoresearch")
    return resolved


def write_result_once(path: str | Path, result: Mapping[str, Any]) -> None:
    target = Path(path)
    encoded = (json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(encoded) > MAX_ARTIFACT:
        raise ProfileError("JSON artifact exceeds 100 MiB")
    temporary = target.with_name(f".{target.name}.{os.getpid()}.tmp")
    fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, target)
        temporary.unlink()
    except Exception:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        raise


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=str(DEFAULT_MANIFEST))
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--variant", choices=tuple(VARIANTS), required=True)
    parser.add_argument("--device", required=True, help="cpu for validation or explicit cuda:N for operator jobs")
    parser.add_argument("--output", required=True, help="new absolute JSON path under autoresearch/")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = make_parser().parse_args(argv)
    output = None
    try:
        output = authorized_output_path(args.output)
        result = run_profile(args.manifest, args.checkpoint, args.variant, args.device)
        write_result_once(output, result)
    except Exception as exc:
        if output is not None and not output.exists():
            try:
                write_result_once(output, {
                    "schema_version": VERSION,
                    "status": "failed",
                    "variant": args.variant,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                })
            except Exception:
                pass
        print(f"training profile failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "status": result["status"],
        "variant": result["variant"],
        "output": str(output),
        "mean_measured_cached_full_update_seconds": result["timings"]["mean_measured_cached_full_update_seconds"],
        "data_loader_next_total_seconds": result["data_pipeline"]["data_wait"]["triad_total_seconds"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

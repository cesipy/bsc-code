"""Capture full cross-attention-block event states for frozen HM checkpoints.

Hooks copy only text and vision CLS vectors at the inputs and outputs of each
``model.c_layers[j]`` block. A matching prior checkpoint result/cache is
required as a control for both the event inputs and final backbone CLS states.
No hook changes a module's inputs or outputs.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
import platform
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
    _working_directory,
    authorized_output_root,
    safe_output_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FROZEN_MANIFEST = PROJECT_ROOT / "autoresearch/manifests/hm-devseen-n500-seed42.json"
OUTPUT_RELATIVE_ROOT = Path("event-analysis")
MAX_SAMPLES = 500
MAX_BATCH_SIZE = 2
EVENT_COUNT = 3
CONTROL_ATOL = 1e-4
CONTROL_RTOL = 1e-4
EVENT_NPZ_NAME = "events.npz"
EVENT_RESULT_NAME = "event-extraction-result.json"
IMAGE_PREPROCESSING_ID = (
    "existing dataset preprocessing: project minimal ViT normalization; "
    "validation transforms are deterministic"
)
REFERENCE_CODE_FILES = (
    "src/analyses/autoresearch_data.py",
    "src/analyses/null_calibration.py",
    "src/augments_transforms.py",
    "src/autoresearch.py",
    "src/config.py",
    "src/datasets/dataset_hateful_memes.py",
    "src/datasets/dataset_utils.py",
    "src/metrics.py",
    "src/vilbert.py",
)
EVENT_CODE_FILES = (
    *REFERENCE_CODE_FILES,
    "src/attention.py",
    "src/autoresearch_events.py",
    "autoresearch/event_protocol.md",
)


class EventExtractionError(ValueError):
    """Raised when event capture inputs or outputs violate the frozen protocol."""


@dataclass(frozen=True)
class EventSpec:
    """A c_layers ordinal and its configured backbone insertion IDs."""

    ordinal_id: int
    text_insertion_layer_id: int
    vision_insertion_layer_id: int


def event_specs_from_architecture(
    architecture: dict[str, Any], *, cross_attention_block_count: int
) -> tuple[EventSpec, ...]:
    text_ids = architecture.get("text_cross_attention_layers_zero_based")
    vision_ids = architecture.get("vision_cross_attention_layers_zero_based")
    if not isinstance(text_ids, list) or not isinstance(vision_ids, list):
        raise EventExtractionError("saved checkpoint config lacks cross-attention insertion IDs")
    if len(text_ids) != EVENT_COUNT or len(vision_ids) != EVENT_COUNT:
        raise EventExtractionError("the frozen DF1/DF2 protocol requires exactly three events")
    if cross_attention_block_count != EVENT_COUNT:
        raise EventExtractionError("model.c_layers must contain exactly three event blocks")
    if any(type(layer_id) is not int or not 0 <= layer_id < 12 for layer_id in (*text_ids, *vision_ids)):
        raise EventExtractionError("cross-attention insertion IDs must be backbone IDs in [0, 11]")
    if any(left >= right for left, right in zip(text_ids, text_ids[1:])):
        raise EventExtractionError("text insertion IDs must be strictly increasing")
    if any(left >= right for left, right in zip(vision_ids, vision_ids[1:])):
        raise EventExtractionError("vision insertion IDs must be strictly increasing")
    return tuple(
        EventSpec(index, text_id, vision_id)
        for index, (text_id, vision_id) in enumerate(zip(text_ids, vision_ids))
    )


class CrossAttentionEventCapture:
    """Temporarily observe each c_layers block once per validation batch.

    For event ``j``, pre-hooks snapshot positional inputs ``(text, vision)``
    immediately before the full block and forward hooks snapshot its tuple
    outputs immediately after the full block. CLS vectors are copied to CPU
    float32 before any later model operation can mutate or alias them.
    """

    def __init__(
        self,
        modules: Sequence[Any],
        event_specs: Sequence[EventSpec],
        *,
        embedding_dim: int,
    ) -> None:
        self.modules = tuple(modules)
        self.event_specs = tuple(event_specs)
        self.embedding_dim = int(embedding_dim)
        if len(self.modules) != len(self.event_specs) or not self.modules:
            raise EventExtractionError("event specs must map one-to-one to c_layers modules")
        if tuple(spec.ordinal_id for spec in self.event_specs) != tuple(range(len(self.modules))):
            raise EventExtractionError("event ordinals must follow c_layers order from zero")
        if self.embedding_dim <= 0:
            raise EventExtractionError("CLS embedding width must be positive")
        self._handles: list[Any] = []
        self._entered = False
        self._batch_size: int | None = None
        self._expected_tokens = tuple(
            token
            for ordinal in range(len(self.modules))
            for token in (("pre", ordinal), ("post", ordinal))
        )
        self._tokens: list[tuple[str, int]] = []
        self._active_ordinal: int | None = None
        self._batch_pre: dict[int, Any] = {}
        self._batch_post: dict[int, Any] = {}
        self.pre_call_counts = [0] * len(self.modules)
        self.post_call_counts = [0] * len(self.modules)
        self.completed_batches = 0

    def __enter__(self) -> "CrossAttentionEventCapture":
        if self._entered:
            raise EventExtractionError("event capture context cannot be entered twice")
        self._entered = True
        try:
            for ordinal, module in enumerate(self.modules):
                self._handles.append(
                    module.register_forward_pre_hook(self._make_pre_hook(ordinal))
                )
                self._handles.append(
                    module.register_forward_hook(self._make_post_hook(ordinal))
                )
        except BaseException:
            self.remove_hooks()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.abort_batch()
        self.remove_hooks()
        return False

    def remove_hooks(self) -> None:
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except Exception:
                pass
        self._handles.clear()
        self._entered = False

    def begin_batch(self, batch_size: int) -> None:
        if not self._entered:
            raise EventExtractionError("event hooks must be installed before beginning a batch")
        if self._batch_size is not None:
            raise EventExtractionError("the prior event batch was not completed")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise EventExtractionError("event batch size must be a positive integer")
        self._batch_size = batch_size
        self._tokens = []
        self._active_ordinal = None
        self._batch_pre = {}
        self._batch_post = {}

    def _next_token(self, phase: str, ordinal: int) -> None:
        position = len(self._tokens)
        if position >= len(self._expected_tokens):
            raise EventExtractionError("an event block was called more than once in this batch")
        expected = self._expected_tokens[position]
        if expected != (phase, ordinal):
            raise EventExtractionError(
                "event hook call order changed: "
                f"expected {expected[0]} c_layers[{expected[1]}], "
                f"observed {phase} c_layers[{ordinal}]"
            )
        self._tokens.append((phase, ordinal))

    def _copy_pair(self, text: Any, vision: Any, *, phase: str, ordinal: int) -> Any:
        import torch

        if not torch.is_tensor(text) or not torch.is_tensor(vision):
            raise EventExtractionError(f"{phase} c_layers[{ordinal}] must expose tensor states")
        for name, tensor in (("text", text), ("vision", vision)):
            if tensor.ndim != 3:
                raise EventExtractionError(
                    f"{phase} {name} state for c_layers[{ordinal}] must have shape [batch, tokens, width]"
                )
            if tensor.shape[0] != self._batch_size:
                raise EventExtractionError(
                    f"{phase} {name} state batch size differs from the current input batch"
                )
            if tensor.shape[1] < 1 or tensor.shape[2] != self.embedding_dim:
                raise EventExtractionError(
                    f"{phase} {name} CLS width differs from configured embedding_dim"
                )
        text_cls = text.detach()[:, 0, :].to(device="cpu", dtype=torch.float32).clone()
        vision_cls = vision.detach()[:, 0, :].to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(text_cls).all() or not torch.isfinite(vision_cls).all():
            raise EventExtractionError(f"{phase} CLS state contains non-finite values")
        return torch.stack((text_cls, vision_cls), dim=1)

    def _make_pre_hook(self, ordinal: int) -> Callable[..., None]:
        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            if self._batch_size is None:
                raise EventExtractionError("c_layers pre-hook fired outside a validation batch")
            if self._active_ordinal is not None:
                raise EventExtractionError("nested or overlapping c_layers calls are unsupported")
            self._next_token("pre", ordinal)
            if len(inputs) != 2:
                raise EventExtractionError(
                    f"c_layers[{ordinal}] must receive positional (text, vision) inputs"
                )
            self._batch_pre[ordinal] = self._copy_pair(
                inputs[0], inputs[1], phase="pre", ordinal=ordinal
            )
            self.pre_call_counts[ordinal] += 1
            self._active_ordinal = ordinal
            return None

        return pre_hook

    def _make_post_hook(self, ordinal: int) -> Callable[..., None]:
        def post_hook(_module: Any, _inputs: tuple[Any, ...], outputs: Any) -> None:
            if self._batch_size is None:
                raise EventExtractionError("c_layers forward hook fired outside a validation batch")
            if self._active_ordinal != ordinal:
                raise EventExtractionError("c_layers forward output does not match its active pre-hook")
            self._next_token("post", ordinal)
            if not isinstance(outputs, tuple) or len(outputs) != 2:
                raise EventExtractionError(
                    f"c_layers[{ordinal}] must return a (text, vision) tensor tuple"
                )
            self._batch_post[ordinal] = self._copy_pair(
                outputs[0], outputs[1], phase="post", ordinal=ordinal
            )
            self.post_call_counts[ordinal] += 1
            self._active_ordinal = None
            return None

        return post_hook

    def finish_batch(self) -> tuple[Any, Any]:
        import torch

        if self._batch_size is None:
            raise EventExtractionError("no event batch is active")
        if self._active_ordinal is not None or tuple(self._tokens) != self._expected_tokens:
            raise EventExtractionError("each c_layers event must fire once in c_layers order per batch")
        ordinals = tuple(range(len(self.modules)))
        if tuple(self._batch_pre) != ordinals or tuple(self._batch_post) != ordinals:
            raise EventExtractionError("event pre/post snapshots are missing or out of order")
        pre = torch.stack([self._batch_pre[ordinal] for ordinal in ordinals], dim=1)
        post = torch.stack([self._batch_post[ordinal] for ordinal in ordinals], dim=1)
        expected_shape = (self._batch_size, len(ordinals), 2, self.embedding_dim)
        if tuple(pre.shape) != expected_shape or tuple(post.shape) != expected_shape:
            raise EventExtractionError("captured event arrays have an unexpected batch shape")
        self._batch_size = None
        self._tokens = []
        self._active_ordinal = None
        self._batch_pre = {}
        self._batch_post = {}
        self.completed_batches += 1
        return pre, post

    def abort_batch(self) -> None:
        self._batch_size = None
        self._tokens = []
        self._active_ordinal = None
        self._batch_pre = {}
        self._batch_post = {}

    def validate_total_calls(self, expected_batches: int) -> None:
        expected = [expected_batches] * len(self.modules)
        if self.pre_call_counts != expected or self.post_call_counts != expected:
            raise EventExtractionError("each c_layers event must fire once for every batch")


def validate_hm_manifest_protocol(manifest: dict[str, Any], *, requested_n: int = MAX_SAMPLES) -> list[str]:
    if manifest.get("task") != "hateful_memes" or manifest.get("split") != "dev_seen":
        raise EventExtractionError("only the explicit Hateful Memes dev_seen split is allowed")
    if manifest.get("validation_role") != "validation_discovery_seen_historical":
        raise EventExtractionError("manifest must be the historical discovery dev_seen cohort")
    ids = manifest.get("ordered_sample_ids")
    selection = manifest.get("selection")
    samples = manifest.get("samples")
    if requested_n != MAX_SAMPLES or not isinstance(ids, list) or len(ids) != MAX_SAMPLES:
        raise EventExtractionError("event capture requires the frozen 500-row manifest")
    if not isinstance(selection, dict) or selection.get("n_samples") != MAX_SAMPLES:
        raise EventExtractionError("manifest selection must contain exactly 500 rows")
    if not isinstance(samples, list) or [row.get("id") for row in samples] != ids:
        raise EventExtractionError("manifest sample records must match the frozen ID order")
    if len(ids) != len(set(ids)) or any(not isinstance(value, str) or not value for value in ids):
        raise EventExtractionError("manifest IDs must be unique non-empty strings")
    return ids


def validate_output_destination(
    output_dir: str | Path,
    *,
    authorized_root: str | Path,
    data_root: str | Path,
    input_paths: Sequence[str | Path] = (),
) -> Path:
    """Permit a new output directory or a runner directory with stdout.log only."""
    try:
        resolved = safe_output_path(
            output_dir,
            authorized_root=authorized_root,
            data_root=data_root,
            require_directory=True,
        )
    except EvaluatorError as exc:
        raise EventExtractionError(str(exc)) from exc
    target = Path(output_dir)
    if target.is_symlink():
        raise EventExtractionError("event output directory must not be a symlink")
    for value in input_paths:
        source = Path(value).resolve(strict=True)
        if resolved == source or resolved.is_relative_to(source) or source.is_relative_to(resolved):
            raise EventExtractionError(f"output directory overlaps an input: {source}")
    if target.exists():
        if not target.is_dir():
            raise EventExtractionError("event output path must be a directory")
        entries = list(target.iterdir())
        if entries and (
            len(entries) != 1
            or entries[0].name != "stdout.log"
            or entries[0].is_symlink()
            or not entries[0].is_file()
        ):
            raise EventExtractionError(
                "precreated output directory may contain only the runner's regular stdout.log"
            )
    return resolved


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EventExtractionError(f"could not read JSON input {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise EventExtractionError(f"JSON input must be an object: {path}")
    return value


def _event_code_hashes(project_root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in EVENT_CODE_FILES:
        path = project_root / relative
        try:
            resolved = path.resolve(strict=True)
        except FileNotFoundError as exc:
            raise EventExtractionError(f"required source file is missing: {relative}") from exc
        if not resolved.is_relative_to(project_root.resolve(strict=True)):
            raise EventExtractionError(f"source path escapes project root: {relative}")
        result[relative] = _sha256_file(resolved)
    return result


def _runtime_metadata(device: str) -> dict[str, Any]:
    import numpy as np
    import timm
    import torch
    import transformers

    runtime = {
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "numpy_version": np.__version__,
        "torch_version": str(torch.__version__),
        "torch_cuda_build": torch.version.cuda,
        "transformers_version": transformers.__version__,
        "timm_version": timm.__version__,
        "requested_device": device,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device_name": (
            torch.cuda.get_device_name(torch.device(device))
            if device.startswith("cuda") and torch.cuda.is_available()
            else None
        ),
    }
    runtime["runtime_sha256"] = _sha256_bytes(_canonical_json(runtime))
    return runtime


def _safe_reference_path(
    path: str | Path,
    *,
    authorized_root: Path,
    data_root: Path,
    description: str,
) -> Path:
    resolved = _safe_input_file(path, description)
    if Path(path).is_symlink():
        raise EventExtractionError(f"{description} must not be a symlink")
    try:
        safe_output_path(resolved, authorized_root=authorized_root, data_root=data_root)
    except EvaluatorError as exc:
        raise EventExtractionError(f"{description} must be inside autoresearch: {exc}") from exc
    return resolved


def _relative_overlap(output_dir: Path, source: Path) -> bool:
    return output_dir == source or output_dir.is_relative_to(source) or source.is_relative_to(output_dir)


def validate_reference_bundle(
    reference_result_path: str | Path,
    *,
    project_root: str | Path,
    authorized_root: str | Path,
    data_root: str | Path,
    expected_model: str,
    checkpoint_path: str | Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    source_data_sha256: dict[str, str],
    ordered_sample_ids: Sequence[str],
    sample_pair_hashes: dict[str, str],
    tokenizer_identity: dict[str, Any],
    image_preprocessing: str,
    text_max_length: int,
    current_code_sha256: dict[str, str],
) -> dict[str, Any]:
    """Validate a prior checkpoint result/cache and return copied reference arrays."""
    root = Path(project_root).resolve(strict=True)
    auto_root = Path(authorized_root).resolve(strict=True)
    resolved_data = Path(data_root).resolve(strict=True)
    result_path = _safe_reference_path(
        reference_result_path,
        authorized_root=auto_root,
        data_root=resolved_data,
        description="reference result",
    )
    try:
        result_rel = result_path.relative_to(auto_root)
    except ValueError as exc:
        raise EventExtractionError("reference result must be under the autoresearch root") from exc
    if len(result_rel.parts) < 4 or result_rel.parts[0:3] != (
        "trials", expected_model, "results"
    ):
        raise EventExtractionError(
            f"{expected_model} output requires its matching prior trials/{expected_model} result"
        )
    result = _read_json(result_path)
    if result.get("format") != "autoresearch-checkpoint-validation-result":
        raise EventExtractionError("reference result is not a checkpoint-validation result")

    checkpoint = result.get("checkpoint", {})
    requested_checkpoint = Path(checkpoint_path).resolve(strict=True)
    reference_checkpoint = Path(checkpoint.get("path", "")).resolve(strict=True)
    if requested_checkpoint != reference_checkpoint:
        raise EventExtractionError("reference result belongs to a different checkpoint path")
    if checkpoint.get("sha256") != checkpoint_sha256:
        raise EventExtractionError("reference result checkpoint hash differs from requested checkpoint")

    data = result.get("data", {})
    if data.get("task") != "hateful_memes" or data.get("split") != "dev_seen":
        raise EventExtractionError("reference result must be for Hateful Memes dev_seen")
    if data.get("validation_role") != "validation_discovery_seen_historical":
        raise EventExtractionError("reference result has the wrong validation role")
    if Path(data.get("data_root", "")).resolve(strict=True) != resolved_data:
        raise EventExtractionError("reference result used a different data root")
    if data.get("sample_manifest_sha256") != manifest_sha256:
        raise EventExtractionError("reference result manifest hash differs")
    if data.get("source_data_sha256") != source_data_sha256:
        raise EventExtractionError("reference result source data hashes differ")
    if result.get("ordered_sample_ids") != list(ordered_sample_ids):
        raise EventExtractionError("reference result ordered IDs differ from the current manifest")
    identity = result.get("analysis_identity", {})
    if identity.get("checkpoint_sha256") != checkpoint_sha256:
        raise EventExtractionError("reference analysis identity checkpoint hash differs")
    if identity.get("manifest_sha256") != manifest_sha256:
        raise EventExtractionError("reference analysis identity manifest hash differs")
    if result.get("tokenizer") != tokenizer_identity or identity.get("tokenizer_identity") != tokenizer_identity:
        raise EventExtractionError("reference tokenizer identity differs")

    features = result.get("features", {})
    cache_path = _safe_reference_path(
        features.get("path", ""),
        authorized_root=auto_root,
        data_root=resolved_data,
        description="reference feature cache",
    )
    if cache_path.suffix != ".npz" or _sha256_file(cache_path) != features.get("sha256"):
        raise EventExtractionError("reference NPZ hash differs from its saved result")
    if cache_path.parent.name != "features" or cache_path.parent.parent.name != expected_model:
        raise EventExtractionError("reference feature cache is outside its matching trial directory")

    import numpy as np

    try:
        with np.load(cache_path, allow_pickle=False) as archive:
            required = {
                "text_features", "vision_features", "layer_ids", "sample_ids", "metadata"
            }
            if not required.issubset(archive.files):
                raise EventExtractionError("reference NPZ is missing required feature arrays")
            reference_text = np.array(archive["text_features"], copy=True)
            reference_vision = np.array(archive["vision_features"], copy=True)
            layer_ids = np.array(archive["layer_ids"], copy=True)
            cache_ids = [str(value) for value in np.array(archive["sample_ids"], copy=True).tolist()]
            cache_metadata = json.loads(str(archive["metadata"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, EventExtractionError):
            raise
        raise EventExtractionError(f"could not read reference feature NPZ: {exc}") from exc

    expected_shape = (len(ordered_sample_ids), 12, 768)
    if reference_text.shape != expected_shape or reference_vision.shape != expected_shape:
        raise EventExtractionError("reference final features have an unexpected row/layer/width shape")
    if reference_text.dtype != np.float32 or reference_vision.dtype != np.float32:
        raise EventExtractionError("reference feature arrays must be float32")
    if not np.isfinite(reference_text).all() or not np.isfinite(reference_vision).all():
        raise EventExtractionError("reference feature arrays contain non-finite values")
    if layer_ids.tolist() != list(range(12)):
        raise EventExtractionError("reference feature layers are not ordered 0 through 11")
    if cache_ids != list(ordered_sample_ids):
        raise EventExtractionError("reference NPZ row IDs differ from the current manifest")
    if cache_metadata.get("ordered_sample_ids") != list(ordered_sample_ids):
        raise EventExtractionError("reference cache metadata row IDs differ")
    if cache_metadata.get("sample_pair_hashes") != sample_pair_hashes:
        raise EventExtractionError("reference cache sample pairing hashes differ")
    cache_identity = cache_metadata.get("cache_identity", {})
    cache_identity_hash = _sha256_bytes(_canonical_json(cache_identity))
    if cache_metadata.get("cache_identity_sha256") != cache_identity_hash:
        raise EventExtractionError("reference cache identity hash does not match its canonical metadata")
    if features.get("cache_identity_sha256") not in (None, cache_identity_hash):
        raise EventExtractionError("reference result and NPZ cache identity hashes differ")
    if cache_identity.get("checkpoint_sha256") != checkpoint_sha256:
        raise EventExtractionError("reference cache is bound to another checkpoint")
    if cache_identity.get("manifest_sha256") != manifest_sha256:
        raise EventExtractionError("reference cache is bound to another manifest")
    if cache_identity.get("tokenizer_identity") != tokenizer_identity or cache_metadata.get("tokenizer_identity") != tokenizer_identity:
        raise EventExtractionError("reference cache tokenizer identity differs")
    if cache_metadata.get("data_manifest_sha256") != manifest_sha256:
        raise EventExtractionError("reference cache manifest hash differs")
    if cache_metadata.get("source_data_sha256") != source_data_sha256:
        raise EventExtractionError("reference cache source data hashes differ")
    if cache_metadata.get("data_root") and Path(cache_metadata["data_root"]).resolve(strict=True) != resolved_data:
        raise EventExtractionError("reference cache used a different data root")
    if cache_metadata.get("task") != "hateful_memes" or cache_metadata.get("split") != "dev_seen":
        raise EventExtractionError("reference cache is not for Hateful Memes dev_seen")
    if cache_metadata.get("validation_role") != "validation_discovery_seen_historical":
        raise EventExtractionError("reference cache has the wrong validation role")
    if cache_metadata.get("image_preprocessing") != image_preprocessing:
        raise EventExtractionError("reference image preprocessing identity differs")
    if cache_metadata.get("text_max_length") != text_max_length:
        raise EventExtractionError("reference text max length differs")

    reference_codes = cache_identity.get("code_sha256", {})
    if not isinstance(reference_codes, dict):
        raise EventExtractionError("reference cache code identity is missing")
    for relative in REFERENCE_CODE_FILES:
        if reference_codes.get(relative) != current_code_sha256.get(relative):
            raise EventExtractionError(f"reference source-code identity differs for {relative}")
    if cache_metadata.get("code_sha256") != reference_codes:
        raise EventExtractionError("reference cache code-hash record is inconsistent")
    if identity.get("code_sha256") != reference_codes:
        raise EventExtractionError("reference result and NPZ code-hash records differ")
    if identity.get("tokenizer_identity") != tokenizer_identity:
        raise EventExtractionError("reference result tokenizer identity differs")
    if cache_metadata.get("extraction_version") != "vilbert-cls-layer-pair-v1":
        raise EventExtractionError("reference cache extraction version is not the expected baseline")

    return {
        "result": result,
        "result_path": result_path,
        "result_sha256": _sha256_file(result_path),
        "cache_path": cache_path,
        "cache_sha256": features["sha256"],
        "cache_identity_sha256": cache_metadata["cache_identity_sha256"],
        "cache_metadata": cache_metadata,
        "text_features": reference_text,
        "vision_features": reference_vision,
    }


def _compare_arrays(observed: Any, expected: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(observed, dtype=np.float32)
    right = np.asarray(expected, dtype=np.float32)
    if left.shape != right.shape:
        return {
            "allclose": False,
            "observed_shape": list(left.shape),
            "reference_shape": list(right.shape),
            "atol": atol,
            "rtol": rtol,
        }
    absolute = np.abs(left - right)
    denominator = np.maximum(np.abs(right), np.finfo(np.float32).tiny)
    relative = absolute / denominator
    return {
        "allclose": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        "max_abs_error": float(absolute.max(initial=0.0)),
        "max_relative_error": float(relative.max(initial=0.0)),
        "atol": atol,
        "rtol": rtol,
    }


def validate_capture_controls(
    *,
    event_pre_cls: Any,
    final_text_cls: Any,
    final_vision_cls: Any,
    reference_text_features: Any,
    reference_vision_features: Any,
    event_specs: Sequence[EventSpec],
    atol: float = CONTROL_ATOL,
    rtol: float = CONTROL_RTOL,
) -> dict[str, Any]:
    """Check event inputs and final outputs against the prior feature cache."""
    import numpy as np

    pre = np.asarray(event_pre_cls, dtype=np.float32)
    if pre.ndim != 4 or pre.shape[1:3] != (len(event_specs), 2):
        raise EventExtractionError("pre-event CLS arrays must have [row,event,modality,width] axes")
    text_reference = np.asarray(reference_text_features, dtype=np.float32)
    vision_reference = np.asarray(reference_vision_features, dtype=np.float32)
    text_final = np.asarray(final_text_cls, dtype=np.float32)
    vision_final = np.asarray(final_vision_cls, dtype=np.float32)
    if text_reference.ndim != 3 or vision_reference.shape != text_reference.shape:
        raise EventExtractionError("reference backbone features must share [row,layer,width] shape")
    if pre.shape[0] != text_reference.shape[0] or pre.shape[-1] != text_reference.shape[-1]:
        raise EventExtractionError("captured event inputs do not match the reference cache shape")

    stage_controls: dict[str, Any] = {}
    all_pass = True
    for spec in event_specs:
        event_record: dict[str, Any] = {}
        for modality_index, (modality, layer_id, reference) in enumerate(
            (
                ("text", spec.text_insertion_layer_id - 1, text_reference),
                ("vision", spec.vision_insertion_layer_id - 1, vision_reference),
            )
        ):
            if not 0 <= layer_id < reference.shape[1]:
                raise EventExtractionError("event insertion ID has no preceding backbone cache layer")
            comparison = _compare_arrays(
                pre[:, spec.ordinal_id, modality_index, :],
                reference[:, layer_id, :],
                atol=atol,
                rtol=rtol,
            )
            comparison["reference_backbone_layer_id"] = layer_id
            event_record[modality] = comparison
            all_pass = all_pass and comparison["allclose"]
        stage_controls[str(spec.ordinal_id)] = event_record

    final_controls = {
        "text": _compare_arrays(
            text_final, text_reference[:, -1, :], atol=atol, rtol=rtol
        ),
        "vision": _compare_arrays(
            vision_final, vision_reference[:, -1, :], atol=atol, rtol=rtol
        ),
    }
    final_controls["text"]["reference_backbone_layer_id"] = int(text_reference.shape[1] - 1)
    final_controls["vision"]["reference_backbone_layer_id"] = int(vision_reference.shape[1] - 1)
    all_pass = all_pass and all(item["allclose"] for item in final_controls.values())
    if not all_pass:
        raise EventExtractionError("instrumented forward did not match reference CLS controls")
    return {
        "tolerance": {"atol": atol, "rtol": rtol},
        "pre_event_inputs_vs_backbone_cache": stage_controls,
        "final_backbone_cls": final_controls,
        "all_controls_passed": True,
    }


def extract_event_features(
    *,
    model: Any,
    tokenizer: Any,
    records: Sequence[Any],
    sample_ids: Sequence[str],
    text_loader: Callable[[Any], str],
    image_loader: Callable[[Any], Any],
    batch_size: int,
    device: str,
    max_length: int,
    event_specs: Sequence[EventSpec],
    embedding_dim: int,
) -> dict[str, Any]:
    """Run the matched frozen cohort and return event plus final CLS arrays."""
    import numpy as np
    import torch

    n = len(records)
    if n != MAX_SAMPLES or len(sample_ids) != n:
        raise EventExtractionError("event capture requires 500 records in manifest order")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise EventExtractionError("batch size must be between 1 and 2")
    if getattr(model, "training", False):
        raise EventExtractionError("model must remain in evaluation mode")
    layers = getattr(model, "c_layers", None)
    if layers is None or len(layers) != len(event_specs):
        raise EventExtractionError("model.c_layers does not match the saved event mapping")

    pre_batches: list[np.ndarray] = []
    post_batches: list[np.ndarray] = []
    final_text_batches: list[np.ndarray] = []
    final_vision_batches: list[np.ndarray] = []
    expected_batches = math.ceil(n / batch_size)
    with CrossAttentionEventCapture(
        layers, event_specs, embedding_dim=embedding_dim
    ) as capture:
        for offset in range(0, n, batch_size):
            rows = list(range(offset, min(offset + batch_size, n)))
            chunk_records = [records[row] for row in rows]
            texts = [text_loader(record) for record in chunk_records]
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
                raise EventExtractionError("local tokenizer returned no input_ids")
            images = [image_loader(record) for record in chunk_records]
            image_batch = torch.stack(images, dim=0).to(device)
            capture.begin_batch(len(rows))
            try:
                with torch.inference_mode():
                    output = model(
                        text_input_ids=token_batch["input_ids"],
                        text_attention_mask=token_batch.get("attention_mask"),
                        text_token_type_ids=token_batch.get("token_type_ids"),
                        image_pixel_values=image_batch,
                        image_attention_mask=None,
                        save_intermediate_representations=True,
                    )
                if not isinstance(output, tuple) or len(output) != 3:
                    raise EventExtractionError(
                        "ViLBERT must return final text CLS, vision CLS, and intermediate states"
                    )
                text_final, vision_final, intermediate = output
                if not isinstance(intermediate, list) or not intermediate:
                    raise EventExtractionError("saving intermediate states returned no backbone layers")
                if not torch.is_tensor(text_final) or not torch.is_tensor(vision_final):
                    raise EventExtractionError("final CLS outputs must be tensors")
                for name, tensor in (("text", text_final), ("vision", vision_final)):
                    if tensor.ndim != 2 or tuple(tensor.shape) != (len(rows), embedding_dim):
                        raise EventExtractionError(f"final {name} CLS tensor shape is unexpected")
                pre_batch, post_batch = capture.finish_batch()
                final_text_batches.append(
                    text_final.detach().to(device="cpu", dtype=torch.float32).clone().numpy()
                )
                final_vision_batches.append(
                    vision_final.detach().to(device="cpu", dtype=torch.float32).clone().numpy()
                )
                pre_batches.append(pre_batch.numpy())
                post_batches.append(post_batch.numpy())
            except BaseException:
                capture.abort_batch()
                raise
        capture.validate_total_calls(expected_batches)
        counts = {
            str(spec.ordinal_id): {
                "pre_hook_calls": capture.pre_call_counts[spec.ordinal_id],
                "post_hook_calls": capture.post_call_counts[spec.ordinal_id],
                "expected_calls": expected_batches,
            }
            for spec in event_specs
        }

    arrays = {
        "pre_cls": np.concatenate(pre_batches, axis=0).astype(np.float32, copy=False),
        "post_cls": np.concatenate(post_batches, axis=0).astype(np.float32, copy=False),
        "final_text_cls": np.concatenate(final_text_batches, axis=0).astype(np.float32, copy=False),
        "final_vision_cls": np.concatenate(final_vision_batches, axis=0).astype(np.float32, copy=False),
        "hook_call_counts": counts,
        "batch_count": expected_batches,
    }
    expected_event_shape = (n, len(event_specs), 2, embedding_dim)
    if arrays["pre_cls"].shape != expected_event_shape or arrays["post_cls"].shape != expected_event_shape:
        raise EventExtractionError("concatenated event capture does not preserve all validation rows")
    if arrays["final_text_cls"].shape != (n, embedding_dim) or arrays["final_vision_cls"].shape != (n, embedding_dim):
        raise EventExtractionError("concatenated final CLS arrays do not preserve all validation rows")
    return arrays


def _load_hm_helpers_in_output_directory(output_dir: Path) -> tuple[Callable[..., Any], ...]:
    with _working_directory(output_dir):
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


def _write_npz_once(path: Path, arrays: dict[str, Any]) -> None:
    import numpy as np

    if path.exists() or path.is_symlink():
        raise EventExtractionError(f"refusing to replace existing event NPZ: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            np.savez(file, **arrays)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_name, path)
    except FileExistsError as exc:
        raise EventExtractionError(f"refusing to replace existing event NPZ: {path}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _write_json_once(path: Path, value: dict[str, Any]) -> None:
    serialized = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"
    if path.exists() or path.is_symlink():
        raise EventExtractionError(f"refusing to replace existing event result: {path}")
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            file.write(serialized)
            file.flush()
            os.fsync(file.fileno())
        os.link(temporary_name, path)
    except FileExistsError as exc:
        raise EventExtractionError(f"refusing to replace existing event result: {path}") from exc
    finally:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass


def _build_result_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Capture full-block text/vision CLS states at the three saved Hateful Memes "
            "cross-attention events. The matching checkpoint/cache control is mandatory."
        )
    )
    parser.add_argument("--checkpoint", required=True, help="absolute local checkpoint path")
    parser.add_argument("--data-root", required=True, help="absolute Hateful Memes data root")
    parser.add_argument("--manifest", required=True, help="frozen 500-row HM dev_seen manifest")
    parser.add_argument("--output-dir", required=True, help="autoresearch/event-analysis/{df1,df2}")
    parser.add_argument("--tokenizer-path", required=True, help="absolute local BERT tokenizer path")
    parser.add_argument("--reference-result", required=True, help="matching prior HM checkpoint result JSON")
    parser.add_argument("--device", required=True, help="explicitly choose cpu or cuda[:N]")
    parser.add_argument("--batch-size", required=True, type=int, help="must be 1 or 2")
    return parser


def build_parser() -> argparse.ArgumentParser:
    """Return the public CLI parser for runner and operator validation."""
    return _build_result_parser()


def run(args: argparse.Namespace) -> Path:
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise EventExtractionError("--batch-size must be between 1 and 2")
    if args.device not in ("cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", args.device):
        raise EventExtractionError("--device must be cpu, cuda, or cuda:N")

    output_root = authorized_output_root(PROJECT_ROOT)
    data_arg = Path(args.data_root)
    if not data_arg.is_absolute():
        raise EventExtractionError("--data-root must be an absolute path")
    try:
        data_root = data_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EventExtractionError(f"data root does not exist: {data_arg}") from exc
    if not data_root.is_dir():
        raise EventExtractionError("--data-root must resolve to a directory")

    checkpoint = _safe_input_file(args.checkpoint, "checkpoint")
    tokenizer_arg = Path(args.tokenizer_path)
    if not tokenizer_arg.is_absolute():
        raise EventExtractionError("--tokenizer-path must be absolute")
    try:
        tokenizer_source = tokenizer_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise EventExtractionError(f"tokenizer path does not exist: {tokenizer_arg}") from exc
    if not tokenizer_source.is_dir():
        raise EventExtractionError("--tokenizer-path must name a local tokenizer directory")

    manifest_path = _safe_input_file(args.manifest, "manifest")
    try:
        safe_output_path(manifest_path, authorized_root=output_root, data_root=data_root)
    except EvaluatorError as exc:
        raise EventExtractionError(f"manifest must be under autoresearch: {exc}") from exc
    if manifest_path.resolve(strict=True) != FROZEN_MANIFEST.resolve(strict=True):
        raise EventExtractionError("only the frozen HM dev_seen 500-row manifest is allowed")

    output_arg = Path(os.path.abspath(args.output_dir))
    model_id = output_arg.name
    if model_id not in ("df1", "df2"):
        raise EventExtractionError("--output-dir must end in df1 or df2")
    expected_output = output_root / OUTPUT_RELATIVE_ROOT / model_id
    if output_arg != expected_output:
        raise EventExtractionError(
            f"--output-dir must be exactly {expected_output}"
        )
    reference_path = _safe_reference_path(
        args.reference_result,
        authorized_root=output_root,
        data_root=data_root,
        description="reference result",
    )
    initial_inputs = (checkpoint, manifest_path, tokenizer_source, reference_path, data_root)
    output_dir = validate_output_destination(
        output_arg,
        authorized_root=output_root,
        data_root=data_root,
        input_paths=initial_inputs,
    )
    result_path = output_dir / EVENT_RESULT_NAME
    archive_path = output_dir / EVENT_NPZ_NAME
    for target in (result_path, archive_path):
        try:
            safe_output_path(target, authorized_root=output_root, data_root=data_root)
        except EvaluatorError as exc:
            raise EventExtractionError(str(exc)) from exc
        if target.exists() or target.is_symlink():
            raise EventExtractionError(f"refusing to overwrite existing event output: {target}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = validate_output_destination(
        output_dir,
        authorized_root=output_root,
        data_root=data_root,
        input_paths=initial_inputs,
    )
    if output_dir != expected_output.resolve(strict=True):
        raise EventExtractionError("resolved event output directory changed during creation")

    (
        close_validation_index,
        load_hateful_memes_item,
        manifest_sha256_fn,
        open_validation_index,
        read_manifest,
        validate_manifest_against_index,
    ) = _load_hm_helpers_in_output_directory(output_dir)
    manifest = read_manifest(manifest_path)
    sample_ids = validate_hm_manifest_protocol(manifest)
    index = open_validation_index("hateful_memes", "dev_seen", data_root)
    archive_written = False
    try:
        validate_manifest_against_index(manifest, index)
        manifest_hash = manifest_sha256_fn(manifest)
        if manifest_hash != "444fa2e8169670d6f5a5f81c40aa90b94241f97f0891ad1eb34fedd1bb723d59":
            raise EventExtractionError("frozen manifest content hash differs from the declared cohort")
        records_by_id = index.by_id()
        records = [records_by_id[sample_id] for sample_id in sample_ids]
        row_labels = [record.payload["label"] for record in records]
        if any(label not in (0, 1, 0.0, 1.0, True, False) for label in row_labels):
            raise EventExtractionError("Hateful Memes row labels must be binary")
        source_data_sha256 = dict(sorted(index.source_files.items()))
        sample_pair_hashes = {sample["id"]: sample["pair_sha256"] for sample in manifest["samples"]}
        image_group_ids = [sample["group_id"] for sample in manifest["samples"]]
        text_group_ids = [sample["caption_group_id"] for sample in manifest["samples"]]
        checkpoint_sha256 = _sha256_file(checkpoint)
        code_sha256 = _event_code_hashes(PROJECT_ROOT)
        reference_cache_hint = _read_json(reference_path).get("features", {}).get("path")
        preliminary_inputs = (*initial_inputs, reference_cache_hint) if reference_cache_hint else initial_inputs
        if any(_relative_overlap(output_dir, Path(path).resolve(strict=True)) for path in preliminary_inputs):
            raise EventExtractionError("event output directory overlaps a reference/checkpoint input")

        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with _working_directory(output_dir):
            tokenizer, tokenizer_identity = _load_local_tokenizer(
                str(tokenizer_source),
                output_dir,
                authorized_root=output_root,
                data_root=data_root,
            )
            from config import TOKENIZER_MAX_LEN
            from datasets.dataset_utils import process_image

            max_length = int(TOKENIZER_MAX_LEN)
            if max_length <= 0:
                raise EventExtractionError("config.TOKENIZER_MAX_LEN must be positive")
            reference = validate_reference_bundle(
                reference_path,
                project_root=PROJECT_ROOT,
                authorized_root=output_root,
                data_root=data_root,
                expected_model=model_id,
                checkpoint_path=checkpoint,
                checkpoint_sha256=checkpoint_sha256,
                manifest_sha256=manifest_hash,
                source_data_sha256=source_data_sha256,
                ordered_sample_ids=sample_ids,
                sample_pair_hashes=sample_pair_hashes,
                tokenizer_identity=tokenizer_identity,
                image_preprocessing=IMAGE_PREPROCESSING_ID,
                text_max_length=max_length,
                current_code_sha256=code_sha256,
            )

            model, architecture, provenance = _load_model_without_pretrained_downloads(
                checkpoint,
                args.device,
                working_directory=output_dir,
            )
            if getattr(model, "training", True):
                raise EventExtractionError("checkpoint loader did not leave model in evaluation mode")
            specs = event_specs_from_architecture(
                architecture, cross_attention_block_count=len(model.c_layers)
            )
            if tuple(model.t_biattention_ids) != tuple(
                architecture["text_cross_attention_layers_zero_based"]
            ) or tuple(model.v_biattention_ids) != tuple(
                architecture["vision_cross_attention_layers_zero_based"]
            ):
                raise EventExtractionError("model insertion IDs differ from its saved checkpoint config")
            embedding_dim = int(model.config.embedding_dim)
            if embedding_dim != 768:
                raise EventExtractionError("the reference HM checkpoint must have 768-wide CLS states")

            def text_loader(record: Any) -> str:
                return load_hateful_memes_item(record)[0]

            def image_loader(record: Any) -> Any:
                image_path = load_hateful_memes_item(record)[1]
                processed = process_image(image_path, transform=None)
                if processed is None or processed.get("pixel_values") is None:
                    raise EventExtractionError(
                        f"image preprocessing failed for {_record_sample_id(record)}"
                    )
                return processed["pixel_values"].squeeze(0)

            captured = extract_event_features(
                model=model,
                tokenizer=tokenizer,
                records=records,
                sample_ids=sample_ids,
                text_loader=text_loader,
                image_loader=image_loader,
                batch_size=args.batch_size,
                device=args.device,
                max_length=max_length,
                event_specs=specs,
                embedding_dim=embedding_dim,
            )

        controls = validate_capture_controls(
            event_pre_cls=captured["pre_cls"],
            final_text_cls=captured["final_text_cls"],
            final_vision_cls=captured["final_vision_cls"],
            reference_text_features=reference["text_features"],
            reference_vision_features=reference["vision_features"],
            event_specs=specs,
        )
        event_records = [
            {
                **asdict(spec),
                "module_name": f"c_layers[{spec.ordinal_id}]",
                "pre_hook": (
                    "positional inputs (text, vision) immediately before the full "
                    "CrossAttentionBlock; CLS token index 0"
                ),
                "post_hook": (
                    "tuple outputs (text, vision) immediately after the full "
                    "CrossAttentionBlock, including attention, residual, and feed-forward "
                    "updates; CLS token index 0"
                ),
            }
            for spec in specs
        ]
        runtime = _runtime_metadata(args.device)
        metadata = {
            "format": "autoresearch-hm-event-features-v1",
            "axis_order": ["sample", "event_ordinal", "modality_text_then_vision", "feature"],
            "final_cls_axis_order": ["sample", "feature"],
            "array_shapes": {
                "pre_cls": list(captured["pre_cls"].shape),
                "post_cls": list(captured["post_cls"].shape),
                "final_text_cls": list(captured["final_text_cls"].shape),
                "final_vision_cls": list(captured["final_vision_cls"].shape),
            },
            "event_ordinal_ids": [spec.ordinal_id for spec in specs],
            "modalities": ["text", "vision"],
            "feature_semantics": "CLS vector at token position zero",
            "ordered_sample_ids": sample_ids,
            "row_labels": [int(label) for label in row_labels],
            "image_group_ids": image_group_ids,
            "text_group_ids": text_group_ids,
            "event_mapping": event_records,
            "checkpoint_sha256": checkpoint_sha256,
            "manifest_sha256": manifest_hash,
            "source_data_sha256": source_data_sha256,
            "sample_pair_hashes": sample_pair_hashes,
            "tokenizer_identity": tokenizer_identity,
            "image_preprocessing": IMAGE_PREPROCESSING_ID,
            "text_max_length": max_length,
            "code_sha256": code_sha256,
            "runtime_sha256": runtime["runtime_sha256"],
        }
        import numpy as np

        npz_arrays = {
            "pre_cls": captured["pre_cls"],
            "post_cls": captured["post_cls"],
            "final_text_cls": captured["final_text_cls"],
            "final_vision_cls": captured["final_vision_cls"],
            "ordered_sample_ids": np.asarray(sample_ids, dtype="U64"),
            "row_labels": np.asarray(row_labels, dtype=np.int64),
            "image_group_ids": np.asarray(image_group_ids, dtype="U80"),
            "text_group_ids": np.asarray(text_group_ids, dtype="U80"),
            "event_ordinal_ids": np.asarray([spec.ordinal_id for spec in specs], dtype=np.int64),
            "text_insertion_layer_ids": np.asarray(
                [spec.text_insertion_layer_id for spec in specs], dtype=np.int64
            ),
            "vision_insertion_layer_ids": np.asarray(
                [spec.vision_insertion_layer_id for spec in specs], dtype=np.int64
            ),
            "metadata_json": np.asarray(json.dumps(
                metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False
            )),
        }
        _write_npz_once(archive_path, npz_arrays)
        archive_written = True
        result = {
            "format": "autoresearch-hateful-memes-event-extraction-v1",
            "status": "complete",
            "model_id": model_id,
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
                "n": len(sample_ids),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "source_data_sha256": source_data_sha256,
                "ordered_sample_ids": sample_ids,
                "row_labels": [int(label) for label in row_labels],
                "image_group_ids": image_group_ids,
                "text_group_ids": text_group_ids,
            },
            "reference": {
                "result_path": str(reference["result_path"]),
                "result_sha256": reference["result_sha256"],
                "feature_cache_path": str(reference["cache_path"]),
                "feature_cache_sha256": reference["cache_sha256"],
                "cache_identity_sha256": reference["cache_identity_sha256"],
            },
            "preprocessing": {
                "image": IMAGE_PREPROCESSING_ID,
                "text": "local BERT tokenizer, fixed-length padding/truncation",
                "text_max_length": max_length,
                "tokenizer_identity": tokenizer_identity,
            },
            "event_semantics": (
                "Event outputs are after the full CrossAttentionBlock, including attention, "
                "residual, and feed-forward operations; they are not isolated attention math."
            ),
            "events": event_records,
            "capture_verification": {
                "mode": "eval + torch.inference_mode",
                "batch_size": args.batch_size,
                "batch_count": captured["batch_count"],
                "hook_call_counts": captured["hook_call_counts"],
                "hook_effect": "observation-only hooks return None and do not alter inputs, outputs, or parameters",
            },
            "control": controls,
            "code_sha256": code_sha256,
            "runtime": runtime,
            "npz": {
                "path": str(archive_path),
                "sha256": _sha256_file(archive_path),
                "pre_cls_shape": list(captured["pre_cls"].shape),
                "post_cls_shape": list(captured["post_cls"].shape),
            "final_text_cls_shape": list(captured["final_text_cls"].shape),
            "final_vision_cls_shape": list(captured["final_vision_cls"].shape),
                "dtype": "float32",
            },
        }
        _write_json_once(result_path, result)
        return result_path
    except BaseException:
        if archive_written and not result_path.exists():
            try:
                archive_path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        close_validation_index(index)


def _record_sample_id(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("sample_id", "unknown"))
    return str(getattr(record, "sample_id", "unknown"))


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result_path = run(args)
    except (EvaluatorError, EventExtractionError, OSError, RuntimeError, ValueError) as exc:
        print(f"event extraction failed: {exc}", file=sys.stderr)
        return 2
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

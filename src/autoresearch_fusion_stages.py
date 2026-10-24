"""Observe the internal stages of each saved ViLBERT cross-attention block.

This is an offline, checkpoint-only diagnostic on the frozen historical
Hateful Memes dev_seen cohort. Hooks return None and only copy detached CLS
vectors plus small per-example update summaries. It never trains or writes
model weights.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import resource
import sys
import time
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
from autoresearch_events import (
    EVENT_CODE_FILES,
    EVENT_COUNT,
    IMAGE_PREPROCESSING_ID,
    REFERENCE_CODE_FILES,
    EventExtractionError,
    EventSpec,
    _event_code_hashes,
    _read_json,
    _runtime_metadata,
    _write_json_once,
    _write_npz_once,
    event_specs_from_architecture,
    validate_hm_manifest_protocol,
    validate_output_destination,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTO_ROOT = PROJECT_ROOT / "autoresearch"
FROZEN_MANIFEST = AUTO_ROOT / "manifests/hm-devseen-n500-seed42.json"
FROZEN_MANIFEST_SHA256 = "444fa2e8169670d6f5a5f81c40aa90b94241f97f0891ad1eb34fedd1bb723d59"
OUTPUT_PARENT = AUTO_ROOT / "fusion-stage-analysis"
MAX_SAMPLES = 500
MAX_BATCH_SIZE = 2
MAX_JOB_WALL_SECONDS = 900
MAX_HOST_RSS_BYTES = 8 * 1024**3
MAX_ARTIFACT_BYTES = 512 * 1024**2
CONTROL_ATOL = 1e-4
CONTROL_RTOL = 1e-4
RESULT_NAME = "fusion-stage-result.json"
NPZ_NAME = "fusion-stages.npz"

STAGE_ARRAY_NAMES = (
    "pre_cls",
    "projected_update_cls",
    "pre_ln_cls",
    "post_ln_cls",
    "post_cls",
)
STAGE_DESCRIPTIONS = {
    "pre_cls": (
        "CLS immediately before c_layers[ordinal], the saved text/vision states "
        "after backbone block insertion_id - 1"
    ),
    "projected_update_cls": (
        "CLS of the attention output after the stream-specific output projection "
        "and before adding the residual"
    ),
    "pre_ln_cls": (
        "CLS after projected attention has been added to the residual and before "
        "the output LayerNorm"
    ),
    "post_ln_cls": "CLS after the output LayerNorm and before the feed-forward block",
    "post_cls": (
        "CLS after the full CrossAttentionBlock, including output LayerNorm and "
        "feed-forward residual"
    ),
}
STREAMS = ("text", "vision")
MODEL_INSERTION_IDS = {
    "df1": ((3, 4, 5), (5, 6, 7)),
    "df2": ((3, 4, 5), (9, 10, 11)),
    "df3": ((3, 4, 5), (7, 8, 9)),
    "rev": ((9, 10, 11), (3, 4, 5)),
}
FUSION_STAGE_CODE_FILES = (
    "src/autoresearch_fusion_stages.py",
    "src/autoresearch_task_reliance.py",
    "autoresearch/fusion_stages_protocol.md",
)
_INNER_HOOK_ORDER = (
    ("projected_update", "text"),
    ("projected_update", "vision"),
    ("pre_ln", "text"),
    ("post_ln", "text"),
    ("pre_ln", "vision"),
    ("post_ln", "vision"),
    ("post_block", "pair"),
)


class FusionStageError(ValueError):
    """Raised when a staged capture violates its frozen input or hook contract."""


class CrossAttentionStageCapture:
    """Observation-only hooks for all five states of each CrossAttentionBlock."""

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
        if not self.modules or len(self.modules) != len(self.event_specs):
            raise FusionStageError("event specs must map one-to-one to c_layers modules")
        if tuple(spec.ordinal_id for spec in self.event_specs) != tuple(
            range(len(self.modules))
        ):
            raise FusionStageError("event ordinals must follow c_layers order from zero")
        if self.embedding_dim < 1:
            raise FusionStageError("embedding_dim must be positive")
        for ordinal, block in enumerate(self.modules):
            output = getattr(block, "output", None)
            if output is None:
                raise FusionStageError(f"c_layers[{ordinal}] has no output module")
            required = (
                "projection_text",
                "projection_vision",
                "norm_text",
                "norm_vision",
            )
            if any(not hasattr(output, name) for name in required):
                raise FusionStageError(
                    f"c_layers[{ordinal}].output does not expose the expected "
                    "projection and normalization submodules"
                )
        self._handles: list[Any] = []
        self._entered = False
        self._batch_size: int | None = None
        self._active_ordinal: int | None = None
        self._next_event = 0
        self._inner_cursor = 0
        self._batch_events: list[dict[str, Any]] = []
        self._residual_cls: dict[str, Any] = {}
        self._residual_rms: dict[str, Any] = {}
        self.pre_call_counts = [0] * len(self.modules)
        self.post_call_counts = [0] * len(self.modules)
        self.stage_call_counts = [
            {
                "projected_update_text": 0,
                "projected_update_vision": 0,
                "pre_ln_text": 0,
                "pre_ln_vision": 0,
                "post_ln_text": 0,
                "post_ln_vision": 0,
            }
            for _ in self.modules
        ]
        self.completed_batches = 0
        self.registered_handle_count = 0
        self.removed_handle_count = 0
        self.hook_removal_errors: list[str] = []

    def __enter__(self) -> "CrossAttentionStageCapture":
        if self._entered:
            raise FusionStageError("capture context cannot be entered twice")
        self._entered = True
        try:
            for ordinal, block in enumerate(self.modules):
                self._handles.append(
                    block.register_forward_pre_hook(self._make_block_pre_hook(ordinal))
                )
                self._handles.append(
                    block.register_forward_hook(self._make_block_post_hook(ordinal))
                )
                output = block.output
                for stream, projection in (
                    ("text", output.projection_text),
                    ("vision", output.projection_vision),
                ):
                    self._handles.append(
                        projection.register_forward_hook(
                            self._make_projection_hook(ordinal, stream)
                        )
                    )
                for stream, norm in (
                    ("text", output.norm_text),
                    ("vision", output.norm_vision),
                ):
                    self._handles.append(
                        norm.register_forward_pre_hook(
                            self._make_norm_pre_hook(ordinal, stream)
                        )
                    )
                    self._handles.append(
                        norm.register_forward_hook(
                            self._make_norm_post_hook(ordinal, stream)
                        )
                    )
            self.registered_handle_count = len(self._handles)
        except BaseException:
            self.remove_hooks()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self.abort_batch()
        self.remove_hooks()
        return False

    def remove_hooks(self) -> None:
        removal_errors: list[str] = []
        for handle in reversed(self._handles):
            try:
                handle.remove()
            except BaseException as exc:
                removal_errors.append(f"{type(exc).__name__}: {exc}")
            else:
                self.removed_handle_count += 1
        self._handles.clear()
        self._entered = False
        self.hook_removal_errors = removal_errors

    def begin_batch(self, batch_size: int) -> None:
        if not self._entered:
            raise FusionStageError("hooks must be installed before beginning a batch")
        if self._batch_size is not None:
            raise FusionStageError("the prior staged-capture batch is unfinished")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise FusionStageError("batch_size must be a positive integer")
        self._batch_size = batch_size
        self._active_ordinal = None
        self._next_event = 0
        self._inner_cursor = 0
        self._batch_events = []
        self._residual_cls = {}
        self._residual_rms = {}

    def _require_tensor(self, value: Any, *, stage: str, ordinal: int) -> Any:
        import torch

        if not torch.is_tensor(value) or value.ndim != 3:
            raise FusionStageError(
                f"{stage} c_layers[{ordinal}] must expose [batch,tokens,width] tensors"
            )
        if self._batch_size is None or value.shape[0] != self._batch_size:
            raise FusionStageError(f"{stage} c_layers[{ordinal}] batch size changed")
        if value.shape[1] < 1 or value.shape[2] != self.embedding_dim:
            raise FusionStageError(f"{stage} c_layers[{ordinal}] has unexpected token/width shape")
        return value

    def _copy_cls(self, value: Any, *, stage: str, ordinal: int) -> Any:
        import torch

        tensor = self._require_tensor(value, stage=stage, ordinal=ordinal)
        copied = tensor.detach()[:, 0, :].to(device="cpu", dtype=torch.float32).clone()
        if not torch.isfinite(copied).all():
            raise FusionStageError(f"{stage} c_layers[{ordinal}] CLS contains non-finite values")
        return copied

    def _copy_pair(self, text: Any, vision: Any, *, stage: str, ordinal: int) -> Any:
        import torch

        return torch.stack(
            (
                self._copy_cls(text, stage=f"{stage} text", ordinal=ordinal),
                self._copy_cls(vision, stage=f"{stage} vision", ordinal=ordinal),
            ),
            dim=1,
        )

    def _advance_inner(self, ordinal: int, phase: str, stream: str) -> None:
        if self._active_ordinal != ordinal:
            raise FusionStageError(
                f"{phase} hook for c_layers[{ordinal}] fired outside its active block"
            )
        if self._inner_cursor >= len(_INNER_HOOK_ORDER):
            raise FusionStageError(f"extra {phase} hook fired for c_layers[{ordinal}]")
        expected = _INNER_HOOK_ORDER[self._inner_cursor]
        if expected != (phase, stream):
            raise FusionStageError(
                f"unexpected hook order in c_layers[{ordinal}]: expected "
                f"{expected}, observed {(phase, stream)}"
            )
        self._inner_cursor += 1

    def _make_block_pre_hook(self, ordinal: int) -> Callable[..., None]:
        def pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            import torch

            if self._batch_size is None:
                raise FusionStageError("c_layers pre-hook fired outside an active batch")
            if self._active_ordinal is not None:
                raise FusionStageError("nested or overlapping c_layers calls are unsupported")
            if ordinal != self._next_event:
                raise FusionStageError(
                    f"out-of-order c_layers call: expected {self._next_event}, observed {ordinal}"
                )
            if len(inputs) != 2:
                raise FusionStageError(
                    f"c_layers[{ordinal}] must receive positional text and vision tensors"
                )
            text = self._require_tensor(inputs[0], stage="pre", ordinal=ordinal)
            vision = self._require_tensor(inputs[1], stage="pre", ordinal=ordinal)
            self._active_ordinal = ordinal
            self._inner_cursor = 0
            event = {
                "stages": {
                    "pre_cls": self._copy_pair(text, vision, stage="pre", ordinal=ordinal)
                },
                "summary": {"per_stream": {}},
            }
            self._batch_events.append(event)
            self._residual_cls = {
                "text": text.detach().float()[:, 0, :].clone(),
                "vision": vision.detach().float()[:, 0, :].clone(),
            }
            self._residual_rms = {
                "text": text.detach().float().square().mean(dim=(1, 2)).sqrt(),
                "vision": vision.detach().float().square().mean(dim=(1, 2)).sqrt(),
            }
            if not all(
                bool(torch.isfinite(value).all())
                for value in self._residual_rms.values()
            ):
                raise FusionStageError("pre-event token RMS contains non-finite values")
            self.pre_call_counts[ordinal] += 1
            return None

        return pre_hook

    def _make_projection_hook(self, ordinal: int, stream: str) -> Callable[..., None]:
        def projection_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            import torch
            import torch.nn.functional as functional

            self._advance_inner(ordinal, "projected_update", stream)
            update = self._require_tensor(
                output, stage=f"projected update {stream}", ordinal=ordinal
            )
            residual_cls = self._residual_cls[stream]
            residual_rms = self._residual_rms[stream]
            update_float = update.detach().float()
            update_rms = update_float.square().mean(dim=(1, 2)).sqrt()
            update_cls = update_float[:, 0, :]
            if not all(
                bool(torch.isfinite(value).all())
                for value in (update_rms, update_cls)
            ):
                raise FusionStageError("projected attention update contains non-finite values")
            event = self._batch_events[ordinal]
            event["stages"].setdefault("projected_update_cls", {})[stream] = (
                update_cls.detach().to(device="cpu", dtype=torch.float32).clone()
            )
            ratio = update_rms / residual_rms.clamp_min(torch.finfo(torch.float32).tiny)
            cls_ratio = torch.linalg.vector_norm(update_cls, dim=1) / torch.linalg.vector_norm(
                residual_cls, dim=1
            ).clamp_min(torch.finfo(torch.float32).tiny)
            cls_cosine = functional.cosine_similarity(
                update_cls, residual_cls, dim=1, eps=1e-8
            )
            summary = event["summary"]["per_stream"].setdefault(stream, {})
            summary["projected_update_full_token_rms"] = (
                update_rms.detach().to("cpu", dtype=torch.float32).clone()
            )
            summary["pre_event_full_token_rms"] = (
                residual_rms.detach().to("cpu", dtype=torch.float32).clone()
            )
            summary["projected_update_to_pre_event_norm_ratio"] = (
                ratio.detach().to("cpu", dtype=torch.float32).clone()
            )
            summary["projected_update_cls_to_pre_event_cls_norm_ratio"] = (
                cls_ratio.detach().to("cpu", dtype=torch.float32).clone()
            )
            summary["projected_update_cls_cosine_to_pre_event_cls"] = (
                cls_cosine.detach().to("cpu", dtype=torch.float32).clone()
            )
            self.stage_call_counts[ordinal][f"projected_update_{stream}"] += 1
            return None

        return projection_hook

    def _make_norm_pre_hook(self, ordinal: int, stream: str) -> Callable[..., None]:
        def norm_pre_hook(_module: Any, inputs: tuple[Any, ...]) -> None:
            self._advance_inner(ordinal, "pre_ln", stream)
            if len(inputs) != 1:
                raise FusionStageError(
                    f"output LayerNorm for {stream} in c_layers[{ordinal}] must take one tensor"
                )
            self._batch_events[ordinal]["stages"].setdefault("pre_ln_cls", {})[stream] = (
                self._copy_cls(inputs[0], stage="pre-LN residual sum", ordinal=ordinal)
            )
            self.stage_call_counts[ordinal][f"pre_ln_{stream}"] += 1
            return None

        return norm_pre_hook

    def _make_norm_post_hook(self, ordinal: int, stream: str) -> Callable[..., None]:
        def norm_post_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            self._advance_inner(ordinal, "post_ln", stream)
            self._batch_events[ordinal]["stages"].setdefault("post_ln_cls", {})[stream] = (
                self._copy_cls(output, stage="post-LN", ordinal=ordinal)
            )
            self.stage_call_counts[ordinal][f"post_ln_{stream}"] += 1
            return None

        return norm_post_hook

    def _make_block_post_hook(self, ordinal: int) -> Callable[..., None]:
        def block_post_hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if self._batch_size is None:
                raise FusionStageError("c_layers forward hook fired outside an active batch")
            self._advance_inner(ordinal, "post_block", "pair")
            if not isinstance(output, tuple) or len(output) != 2:
                raise FusionStageError(
                    f"c_layers[{ordinal}] must return a (text, vision) tuple"
                )
            self._batch_events[ordinal]["stages"]["post_cls"] = self._copy_pair(
                output[0], output[1], stage="post-block", ordinal=ordinal
            )
            self.post_call_counts[ordinal] += 1
            self._active_ordinal = None
            self._next_event += 1
            self._inner_cursor = 0
            self._residual_cls = {}
            self._residual_rms = {}
            return None

        return block_post_hook

    def finish_batch(self) -> dict[str, Any]:
        import numpy as np
        import torch

        if self._batch_size is None:
            raise FusionStageError("no capture batch is active")
        if self._active_ordinal is not None or self._next_event != len(self.modules):
            raise FusionStageError("not every c_layers stage fired exactly once in order")
        expected_summary = (
            "projected_update_full_token_rms",
            "pre_event_full_token_rms",
            "projected_update_to_pre_event_norm_ratio",
            "projected_update_cls_to_pre_event_cls_norm_ratio",
            "projected_update_cls_cosine_to_pre_event_cls",
        )
        stage_arrays: dict[str, Any] = {}
        for stage_name in STAGE_ARRAY_NAMES:
            per_event = []
            for ordinal, event in enumerate(self._batch_events):
                stage = event["stages"].get(stage_name)
                if stage_name in ("pre_cls", "post_cls"):
                    if stage is None:
                        raise FusionStageError(
                            f"missing {stage_name} for c_layers[{ordinal}]"
                        )
                    per_event.append(stage)
                else:
                    if not isinstance(stage, dict) or set(stage) != set(STREAMS):
                        raise FusionStageError(
                            f"missing {stage_name} stream for c_layers[{ordinal}]"
                        )
                    per_event.append(torch.stack((stage["text"], stage["vision"]), dim=1))
            stage_arrays[stage_name] = (
                torch.stack(per_event, dim=1).numpy().astype(np.float32, copy=False)
            )

        summary_arrays: dict[str, Any] = {}
        for summary_name in expected_summary:
            per_event = []
            for ordinal, event in enumerate(self._batch_events):
                per_stream = event["summary"]["per_stream"]
                if set(per_stream) != set(STREAMS) or any(
                    summary_name not in per_stream[stream] for stream in STREAMS
                ):
                    raise FusionStageError(
                        f"missing {summary_name} for c_layers[{ordinal}]"
                    )
                per_event.append(
                    torch.stack(
                        (
                            per_stream["text"][summary_name],
                            per_stream["vision"][summary_name],
                        ),
                        dim=1,
                    )
                )
            summary_arrays[summary_name] = (
                torch.stack(per_event, dim=1).numpy().astype(np.float32, copy=False)
            )
        expected_shape = (self._batch_size, len(self.modules), 2, self.embedding_dim)
        for name, array in stage_arrays.items():
            if array.shape != expected_shape or not np.isfinite(array).all():
                raise FusionStageError(f"captured stage {name} has invalid shape or values")
        if any(not np.isfinite(array).all() for array in summary_arrays.values()):
            raise FusionStageError("per-example update summaries contain non-finite values")
        self._batch_size = None
        self._batch_events = []
        self._next_event = 0
        self._inner_cursor = 0
        self.completed_batches += 1
        return {**stage_arrays, **summary_arrays}

    def abort_batch(self) -> None:
        self._batch_size = None
        self._active_ordinal = None
        self._next_event = 0
        self._inner_cursor = 0
        self._batch_events = []
        self._residual_cls = {}
        self._residual_rms = {}

    def validate_total_calls(self, expected_batches: int) -> None:
        if self.hook_removal_errors:
            raise FusionStageError(
                "one or more forward hooks could not be removed: "
                + "; ".join(self.hook_removal_errors)
            )
        if self.completed_batches != expected_batches:
            raise FusionStageError("staged capture batch count differs from expected")
        if self.pre_call_counts != [expected_batches] * len(self.modules):
            raise FusionStageError("each c_layers pre-hook must fire once per batch")
        if self.post_call_counts != [expected_batches] * len(self.modules):
            raise FusionStageError("each c_layers post-hook must fire once per batch")
        for counts in self.stage_call_counts:
            if any(value != expected_batches for value in counts.values()):
                raise FusionStageError("each projection and normalization hook must fire once per batch")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _safe_record_id(record: Any) -> str:
    if isinstance(record, dict):
        return str(record.get("sample_id", record.get("id", "unknown")))
    return str(getattr(record, "sample_id", "unknown"))


def _record_payload(record: Any) -> dict[str, Any]:
    return record["payload"] if isinstance(record, dict) else record.payload


def _load_hm_helpers(output_dir: Path) -> tuple[Callable[..., Any], ...]:
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


def _copy_final_cls(output: Any, *, batch_size: int, embedding_dim: int, label: str) -> Any:
    import torch

    if not torch.is_tensor(output) or tuple(output.shape) != (batch_size, embedding_dim):
        raise FusionStageError(f"model {label} CLS output must have shape [batch, embedding_dim]")
    result = output.detach().to(device="cpu", dtype=torch.float32).clone()
    if not torch.isfinite(result).all():
        raise FusionStageError(f"model {label} CLS output contains non-finite values")
    return result


def _hm_logits(
    model: Any,
    text_cls: Any,
    vision_cls: Any,
    *,
    enabled: bool,
    fusion_method: str,
) -> Any | None:
    import torch

    if not enabled:
        return None
    from autoresearch_task_reliance import fuse_cls_embeddings, validate_head_dimensions

    head = getattr(model, "fc", None)
    if head is None:
        raise FusionStageError("HM fine-tuned checkpoint has no model.fc head")
    validate_head_dimensions(head, fusion_method, int(text_cls.shape[1]))
    logits = head(fuse_cls_embeddings(text_cls, vision_cls, fusion_method))
    if logits.ndim != 2 or logits.shape != (text_cls.shape[0], 1):
        raise FusionStageError("HM classifier must return one logit per row")
    if not torch.isfinite(logits).all():
        raise FusionStageError("HM logits contain non-finite values")
    return logits


def _forward_final(
    model: Any,
    batch: dict[str, Any],
    *,
    embedding_dim: int,
    mode: str,
    fusion_method: str,
) -> tuple[Any, Any, Any | None]:
    import torch

    output = model(
        text_input_ids=batch["input_ids"],
        text_attention_mask=batch.get("attention_mask"),
        text_token_type_ids=batch.get("token_type_ids"),
        image_pixel_values=batch["image_pixel_values"],
        image_attention_mask=None,
    )
    if not isinstance(output, tuple) or len(output) != 2:
        raise FusionStageError("ViLBERT forward must return final text and vision CLS tensors")
    batch_size = int(batch["input_ids"].shape[0])
    text = _copy_final_cls(output[0], batch_size=batch_size, embedding_dim=embedding_dim, label="text")
    vision = _copy_final_cls(
        output[1], batch_size=batch_size, embedding_dim=embedding_dim, label="vision"
    )
    logits = _hm_logits(
        model,
        output[0],
        output[1],
        enabled=(mode == "hm-finetuned"),
        fusion_method=fusion_method,
    )
    if logits is not None:
        logits = logits.detach().to(device="cpu", dtype=torch.float32).clone()
    return text, vision, logits


def _compare_arrays(observed: Any, expected: Any, *, atol: float, rtol: float) -> dict[str, Any]:
    import numpy as np

    left = np.asarray(observed, dtype=np.float32)
    right = np.asarray(expected, dtype=np.float32)
    if left.shape != right.shape:
        return {
            "allclose": False,
            "observed_shape": list(left.shape),
            "expected_shape": list(right.shape),
            "atol": atol,
            "rtol": rtol,
        }
    abs_diff = np.abs(left - right)
    return {
        "allclose": bool(np.allclose(left, right, atol=atol, rtol=rtol)),
        "max_abs_error": float(abs_diff.max(initial=0.0)),
        "atol": atol,
        "rtol": rtol,
    }


def _prepare_batch(
    records: Sequence[Any],
    *,
    tokenizer: Any,
    text_loader: Callable[[Any], str],
    image_loader: Callable[[Any], Any],
    max_length: int,
    device: str,
) -> dict[str, Any]:
    import torch

    texts = [text_loader(record) for record in records]
    tokens = tokenizer(
        texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    token_batch = {
        key: value.to(device)
        for key, value in tokens.items()
        if key in ("input_ids", "attention_mask", "token_type_ids")
    }
    if "input_ids" not in token_batch:
        raise FusionStageError("local tokenizer returned no input_ids")
    images = [image_loader(record) for record in records]
    image_batch = torch.stack(images, dim=0).to(device)
    if image_batch.ndim != 4:
        raise FusionStageError("processed image batch must have shape [batch,channels,height,width]")
    return {**token_batch, "image_pixel_values": image_batch}


def extract_staged_features(
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
    mode: str,
    fusion_method: str,
    atol: float = CONTROL_ATOL,
    rtol: float = CONTROL_RTOL,
    resource_check: Callable[[], None] | None = None,
) -> dict[str, Any]:
    """Extract exact pre/update/pre-LN/post-LN/post-block CLS states."""
    import numpy as np
    import torch

    if len(records) != MAX_SAMPLES or len(sample_ids) != MAX_SAMPLES:
        raise FusionStageError("staged extraction requires the exact frozen 500-row cohort")
    if not 1 <= batch_size <= MAX_BATCH_SIZE:
        raise FusionStageError("batch_size must be 1 or 2")
    if mode not in ("pretrain", "hm-finetuned"):
        raise FusionStageError("mode must be pretrain or hm-finetuned")
    if getattr(model, "training", True):
        raise FusionStageError("model must be in eval mode for observation-only capture")
    layers = getattr(model, "c_layers", None)
    if layers is None or len(layers) != len(event_specs):
        raise FusionStageError("model.c_layers and event_specs must have the same length")

    def text_loader_checked(record: Any) -> str:
        value = text_loader(record)
        if not isinstance(value, str):
            raise FusionStageError(f"text loader returned a non-string for {_safe_record_id(record)}")
        return value

    expected_batches = math.ceil(MAX_SAMPLES / batch_size)
    stage_parts = {name: [] for name in STAGE_ARRAY_NAMES}
    summary_names = (
        "projected_update_full_token_rms",
        "pre_event_full_token_rms",
        "projected_update_to_pre_event_norm_ratio",
        "projected_update_cls_to_pre_event_cls_norm_ratio",
        "projected_update_cls_cosine_to_pre_event_cls",
    )
    summary_parts = {name: [] for name in summary_names}
    final_text_parts: list[Any] = []
    final_vision_parts: list[Any] = []
    logits_parts: list[Any] = []
    baseline_features: dict[str, Any] | None = None
    capture = CrossAttentionStageCapture(
        layers, event_specs, embedding_dim=embedding_dim
    )
    first_rows = list(range(0, min(batch_size, MAX_SAMPLES)))
    first_batch = _prepare_batch(
        [records[row] for row in first_rows],
        tokenizer=tokenizer,
        text_loader=text_loader_checked,
        image_loader=image_loader,
        max_length=max_length,
        device=device,
    )
    # Run the uninstrumented control before registering any hooks, then reuse
    # the exact token/image tensors in the first hooked forward.
    with torch.inference_mode():
        raw_text, raw_vision, raw_logits = _forward_final(
            model,
            first_batch,
            embedding_dim=embedding_dim,
            mode=mode,
            fusion_method=fusion_method,
        )
    baseline_features = {
        "text_cls": raw_text.numpy().copy(),
        "vision_cls": raw_vision.numpy().copy(),
        "hm_logits": None if raw_logits is None else raw_logits.numpy().copy(),
    }

    with capture:
        for offset in range(0, MAX_SAMPLES, batch_size):
            rows = list(range(offset, min(offset + batch_size, MAX_SAMPLES)))
            if offset == 0:
                batch = first_batch
            else:
                batch = _prepare_batch(
                    [records[row] for row in rows],
                    tokenizer=tokenizer,
                    text_loader=text_loader_checked,
                    image_loader=image_loader,
                    max_length=max_length,
                    device=device,
                )
            capture.begin_batch(len(rows))
            try:
                with torch.inference_mode():
                    hook_output = model(
                        text_input_ids=batch["input_ids"],
                        text_attention_mask=batch.get("attention_mask"),
                        text_token_type_ids=batch.get("token_type_ids"),
                        image_pixel_values=batch["image_pixel_values"],
                        image_attention_mask=None,
                    )
                if not isinstance(hook_output, tuple) or len(hook_output) != 2:
                    raise FusionStageError(
                        "hooked ViLBERT forward must return final text and vision CLS tensors"
                    )
                final_text = _copy_final_cls(
                    hook_output[0],
                    batch_size=len(rows),
                    embedding_dim=embedding_dim,
                    label="text",
                )
                final_vision = _copy_final_cls(
                    hook_output[1],
                    batch_size=len(rows),
                    embedding_dim=embedding_dim,
                    label="vision",
                )
                logits = _hm_logits(
                    model,
                    hook_output[0],
                    hook_output[1],
                    enabled=(mode == "hm-finetuned"),
                    fusion_method=fusion_method,
                )
                captured = capture.finish_batch()
            except BaseException:
                capture.abort_batch()
                raise
            if offset == 0:
                assert baseline_features is not None
                parity = {
                    "text_final_cls": _compare_arrays(
                        final_text.numpy(), baseline_features["text_cls"], atol=atol, rtol=rtol
                    ),
                    "vision_final_cls": _compare_arrays(
                        final_vision.numpy(),
                        baseline_features["vision_cls"],
                        atol=atol,
                        rtol=rtol,
                    ),
                    "hm_logits": (
                        None
                        if logits is None
                        else _compare_arrays(
                            logits.detach().to("cpu", dtype=torch.float32).numpy(),
                            baseline_features["hm_logits"],
                            atol=atol,
                            rtol=rtol,
                        )
                    ),
                    "same_first_batch_reused": True,
                    "baseline_forward_had_capture_hooks": False,
                }
                if not parity["text_final_cls"]["allclose"] or not parity[
                    "vision_final_cls"
                ]["allclose"] or (
                    parity["hm_logits"] is not None
                    and not parity["hm_logits"]["allclose"]
                ):
                    raise FusionStageError(
                        "instrumented first-batch features or HM logits differ from the no-hook pass"
                    )
            if logits is not None:
                logits_parts.append(logits.detach().to(device="cpu", dtype=torch.float32).clone().numpy())
            final_text_parts.append(final_text.numpy())
            final_vision_parts.append(final_vision.numpy())
            for name in STAGE_ARRAY_NAMES:
                stage_parts[name].append(captured[name])
            for name in summary_names:
                summary_parts[name].append(captured[name])
            if resource_check is not None:
                resource_check()

    capture.validate_total_calls(expected_batches)
    if capture.removed_handle_count != capture.registered_handle_count:
        raise FusionStageError("not all staged capture hooks were removed")
    arrays = {
        name: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
        for name, parts in stage_parts.items()
    }
    arrays.update(
        {
            name: np.concatenate(parts, axis=0).astype(np.float32, copy=False)
            for name, parts in summary_parts.items()
        }
    )
    arrays["final_text_cls"] = np.concatenate(final_text_parts, axis=0).astype(
        np.float32, copy=False
    )
    arrays["final_vision_cls"] = np.concatenate(final_vision_parts, axis=0).astype(
        np.float32, copy=False
    )
    if logits_parts:
        arrays["hm_logits"] = np.concatenate(logits_parts, axis=0).astype(np.float32, copy=False)
    else:
        arrays["hm_logits"] = None
    expected_stage_shape = (MAX_SAMPLES, len(event_specs), 2, embedding_dim)
    for name in STAGE_ARRAY_NAMES:
        if arrays[name].shape != expected_stage_shape or not np.isfinite(arrays[name]).all():
            raise FusionStageError(f"full-cohort array {name} failed shape/finiteness validation")
    expected_summary_shape = (MAX_SAMPLES, len(event_specs), 2)
    for name in summary_names:
        if arrays[name].shape != expected_summary_shape or not np.isfinite(arrays[name]).all():
            raise FusionStageError(f"full-cohort summary {name} failed validation")
    if arrays["final_text_cls"].shape != (MAX_SAMPLES, embedding_dim):
        raise FusionStageError("final text CLS array has unexpected shape")
    if arrays["final_vision_cls"].shape != (MAX_SAMPLES, embedding_dim):
        raise FusionStageError("final vision CLS array has unexpected shape")
    arrays["hook_call_counts"] = {
        str(spec.ordinal_id): {
            "block_pre": capture.pre_call_counts[spec.ordinal_id],
            "block_post": capture.post_call_counts[spec.ordinal_id],
            **capture.stage_call_counts[spec.ordinal_id],
            "expected_batches": expected_batches,
        }
        for spec in event_specs
    }
    arrays["first_batch_parity"] = parity
    arrays["batch_count"] = expected_batches
    arrays["registered_hook_count"] = capture.registered_handle_count
    arrays["removed_hook_count"] = capture.removed_handle_count
    return arrays


def _validate_reference_event_result(
    reference_path: Path,
    *,
    model_id: str,
    mode: str,
    checkpoint: Path,
    checkpoint_sha256: str,
    manifest_sha256: str,
    sample_ids: Sequence[str],
    row_labels: Sequence[int],
    image_group_ids: Sequence[str],
    text_group_ids: Sequence[str],
    source_data_sha256: dict[str, str],
    tokenizer_identity: dict[str, Any],
    text_max_length: int,
    current_code_hashes: dict[str, str],
    event_specs: Sequence[EventSpec],
) -> dict[str, Any]:
    import numpy as np

    if mode != "pretrain" or model_id not in ("df1", "df2"):
        raise FusionStageError("legacy event-cache references are only valid for DF1/DF2 pretrain mode")
    if reference_path.is_symlink():
        raise FusionStageError("reference event result must not be a symlink")
    auto_root = AUTO_ROOT.resolve(strict=True)
    relative = reference_path.resolve(strict=True).relative_to(auto_root)
    if relative.parts != ("event-analysis", model_id, "event-extraction-result.json"):
        raise FusionStageError("reference result must be the frozen matching DF1/DF2 event artifact")
    result = _read_json(reference_path)
    if (
        result.get("format") != "autoresearch-hateful-memes-event-extraction-v1"
        or result.get("status") != "complete"
        or result.get("model_id") != model_id
    ):
        raise FusionStageError("reference event result has the wrong model, format, or status")
    checkpoint_record = result.get("checkpoint", {})
    if (
        Path(checkpoint_record.get("path", "")).resolve(strict=True) != checkpoint.resolve(strict=True)
        or checkpoint_record.get("sha256") != checkpoint_sha256
    ):
        raise FusionStageError("reference event result belongs to a different checkpoint")
    cohort = result.get("cohort", {})
    if (
        cohort.get("task") != "hateful_memes"
        or cohort.get("split") != "dev_seen"
        or cohort.get("validation_role") != "validation_discovery_seen_historical"
        or cohort.get("manifest_sha256") != manifest_sha256
        or cohort.get("ordered_sample_ids") != list(sample_ids)
        or cohort.get("row_labels") != list(row_labels)
        or cohort.get("image_group_ids") != list(image_group_ids)
        or cohort.get("text_group_ids") != list(text_group_ids)
        or cohort.get("source_data_sha256") != source_data_sha256
    ):
        raise FusionStageError("reference event result cohort differs from frozen inputs")
    preprocessing = result.get("preprocessing", {})
    if (
        preprocessing.get("image") != IMAGE_PREPROCESSING_ID
        or preprocessing.get("text_max_length") != text_max_length
        or preprocessing.get("tokenizer_identity") != tokenizer_identity
    ):
        raise FusionStageError("reference event result preprocessing identity differs")
    reference_codes = result.get("code_sha256", {})
    if not isinstance(reference_codes, dict) or any(
        not isinstance(reference_codes.get(relative_source), str)
        or not re.fullmatch(r"[0-9a-f]{64}", reference_codes[relative_source])
        for relative_source in EVENT_CODE_FILES
    ):
        raise FusionStageError("reference event result lacks complete source hash provenance")
    source_hash_mismatches = [
        relative_source
        for relative_source in EVENT_CODE_FILES
        if reference_codes.get(relative_source) != current_code_hashes.get(relative_source)
    ]
    event_map = result.get("events", [])
    expected_map = [
        (spec.ordinal_id, spec.text_insertion_layer_id, spec.vision_insertion_layer_id)
        for spec in event_specs
    ]
    observed_map = [
        (
            item.get("ordinal_id"),
            item.get("text_insertion_layer_id"),
            item.get("vision_insertion_layer_id"),
        )
        for item in event_map
    ]
    if observed_map != expected_map:
        raise FusionStageError("reference event mapping differs from saved checkpoint config")
    npz_record = result.get("npz", {})
    npz_path = Path(npz_record.get("path", "")).resolve(strict=True)
    expected_npz = reference_path.parent / "events.npz"
    if npz_path != expected_npz.resolve(strict=True) or _sha256_file(npz_path) != npz_record.get(
        "sha256"
    ):
        raise FusionStageError("reference event NPZ path or hash differs from its result")
    try:
        with np.load(npz_path, allow_pickle=False) as archive:
            required = {
                "pre_cls",
                "post_cls",
                "final_text_cls",
                "final_vision_cls",
                "ordered_sample_ids",
                "row_labels",
                "image_group_ids",
                "text_group_ids",
                "event_ordinal_ids",
                "text_insertion_layer_ids",
                "vision_insertion_layer_ids",
                "metadata_json",
            }
            if not required.issubset(archive.files):
                raise FusionStageError("reference event NPZ lacks required identities")
            arrays = {
                key: np.array(archive[key], copy=True)
                for key in required
                if key != "metadata_json"
            }
            metadata = json.loads(str(archive["metadata_json"].item()))
    except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
        if isinstance(exc, FusionStageError):
            raise
        raise FusionStageError(f"could not read reference event NPZ: {exc}") from exc
    expected_shape = (MAX_SAMPLES, EVENT_COUNT, 2, 768)
    for key in ("pre_cls", "post_cls"):
        if arrays[key].dtype != np.float32 or arrays[key].shape != expected_shape:
            raise FusionStageError(f"reference {key} has unexpected dtype or shape")
        if not np.isfinite(arrays[key]).all():
            raise FusionStageError(f"reference {key} contains non-finite values")
    for key in ("final_text_cls", "final_vision_cls"):
        if arrays[key].dtype != np.float32 or arrays[key].shape != (MAX_SAMPLES, 768):
            raise FusionStageError(f"reference {key} has unexpected dtype or shape")
        if not np.isfinite(arrays[key]).all():
            raise FusionStageError(f"reference {key} contains non-finite values")
    if [str(value) for value in arrays["ordered_sample_ids"].tolist()] != list(sample_ids):
        raise FusionStageError("reference NPZ ordered sample IDs differ")
    if arrays["row_labels"].tolist() != list(row_labels):
        raise FusionStageError("reference NPZ labels differ")
    if arrays["image_group_ids"].tolist() != list(image_group_ids):
        raise FusionStageError("reference NPZ image groups differ")
    if arrays["text_group_ids"].tolist() != list(text_group_ids):
        raise FusionStageError("reference NPZ text groups differ")
    if arrays["event_ordinal_ids"].tolist() != [spec.ordinal_id for spec in event_specs]:
        raise FusionStageError("reference NPZ ordinals differ")
    if arrays["text_insertion_layer_ids"].tolist() != [
        spec.text_insertion_layer_id for spec in event_specs
    ] or arrays["vision_insertion_layer_ids"].tolist() != [
        spec.vision_insertion_layer_id for spec in event_specs
    ]:
        raise FusionStageError("reference NPZ insertion IDs differ")
    if metadata.get("ordered_sample_ids") != list(sample_ids):
        raise FusionStageError("reference NPZ metadata row IDs differ")
    return {
        "result_path": str(reference_path.resolve(strict=True)),
        "result_sha256": _sha256_file(reference_path),
        "npz_path": str(npz_path),
        "npz_sha256": npz_record["sha256"],
        "pre_cls": arrays["pre_cls"],
        "post_cls": arrays["post_cls"],
        "final_text_cls": arrays["final_text_cls"],
        "final_vision_cls": arrays["final_vision_cls"],
        "code_sha256": reference_codes,
        "source_hash_mismatches": source_hash_mismatches,
    }


def _compare_reference_events(
    captured: dict[str, Any],
    reference: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if reference is None:
        return None
    pre = _compare_arrays(
        captured["pre_cls"], reference["pre_cls"], atol=CONTROL_ATOL, rtol=CONTROL_RTOL
    )
    post = _compare_arrays(
        captured["post_cls"], reference["post_cls"], atol=CONTROL_ATOL, rtol=CONTROL_RTOL
    )
    final_text = _compare_arrays(
        captured["final_text_cls"], reference["final_text_cls"],
        atol=CONTROL_ATOL, rtol=CONTROL_RTOL
    )
    final_vision = _compare_arrays(
        captured["final_vision_cls"], reference["final_vision_cls"],
        atol=CONTROL_ATOL, rtol=CONTROL_RTOL
    )
    if not all(item["allclose"] for item in (pre, post, final_text, final_vision)):
        raise FusionStageError(
            "staged PRE/full-block/final outputs failed old DF1/DF2 feature replay"
        )
    return {
        "pre_cls": pre,
        "post_cls": post,
        "final_text_cls": final_text,
        "final_vision_cls": final_vision,
        "reference_code_sha256": reference["code_sha256"],
        "current_vs_reference_source_hash_mismatches": reference["source_hash_mismatches"],
        "source_hash_mismatch_interpretation": (
            "reference arrays were captured under earlier code; numeric replay is required, "
            "and mismatched source hashes are reported rather than treated as identical code"
        ),
    }


def _relative_overlap(left: Path, right: Path) -> bool:
    left_resolved = left.resolve(strict=False)
    right_resolved = right.resolve(strict=False)
    return left_resolved == right_resolved or left_resolved in right_resolved.parents or right_resolved in left_resolved.parents


def _resource_snapshot(device: str) -> dict[str, Any]:
    import torch

    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    rss_bytes = int(usage if sys.platform == "darwin" else usage * 1024)
    gpu = None
    if device.startswith("cuda") and torch.cuda.is_available():
        device_obj = torch.device(device)
        gpu = {
            "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device_obj)),
            "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device_obj)),
        }
    return {"host_rss_peak_bytes": rss_bytes, "gpu": gpu}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Extract PRE, projected update, pre-LN residual sum, post-LN, and "
            "post-block ViLBERT cross-attention CLS states on the frozen HM dev_seen cohort."
        )
    )
    parser.add_argument("--model-id", required=True, choices=("df1", "df2", "df3", "rev"))
    parser.add_argument(
        "--checkpoint-mode",
        required=True,
        choices=("pretrain", "hm-finetuned"),
        help="pretrain captures do not emit HM logits; HM-finetuned also saves model.fc logits",
    )
    parser.add_argument("--checkpoint", required=True, help="absolute local checkpoint path")
    parser.add_argument("--data-root", required=True, help="absolute local HM dataset root")
    parser.add_argument("--manifest", required=True, help="must be the frozen 500-row dev_seen manifest")
    parser.add_argument("--tokenizer-path", required=True, help="absolute local tokenizer directory")
    parser.add_argument("--output-dir", required=True, help="new directory under autoresearch/fusion-stage-analysis")
    parser.add_argument("--run-id", required=True, help="stable output label using letters, digits, dot, underscore, dash")
    parser.add_argument("--device", required=True, help="explicit device: cpu, cuda, or cuda:N")
    parser.add_argument("--batch-size", required=True, type=int, help="must be 1 or 2")
    parser.add_argument(
        "--reference-event-result",
        help=(
            "matching existing event-extraction-result.json; required for DF1/DF2 "
            "pretrain captures to verify full-cohort replay"
        ),
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return _make_parser()


def run(args: argparse.Namespace) -> Path:
    import numpy as np
    import torch

    started = time.monotonic()
    if not 1 <= args.batch_size <= MAX_BATCH_SIZE:
        raise FusionStageError("--batch-size must be 1 or 2")
    if args.device not in ("cpu", "cuda") and not re.fullmatch(r"cuda:[0-9]+", args.device):
        raise FusionStageError("--device must be cpu, cuda, or cuda:N")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", args.run_id):
        raise FusionStageError("--run-id contains unsupported characters")
    output_root = authorized_output_root(PROJECT_ROOT)
    requested_output = Path(os.path.abspath(args.output_dir))
    if requested_output.parent.resolve(strict=False) != OUTPUT_PARENT.resolve(strict=False):
        raise FusionStageError(f"--output-dir must be a new direct child of {OUTPUT_PARENT}")
    if requested_output.name != args.run_id:
        raise FusionStageError("--output-dir basename must equal --run-id")
    data_arg = Path(args.data_root)
    if not data_arg.is_absolute():
        raise FusionStageError("--data-root must be absolute")
    try:
        data_root = data_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FusionStageError(f"data root does not exist: {data_arg}") from exc
    if not data_root.is_dir():
        raise FusionStageError("--data-root must be a directory")
    checkpoint = _safe_input_file(args.checkpoint, "checkpoint")
    if Path(args.checkpoint).is_symlink():
        raise FusionStageError("--checkpoint must not be a symlink")
    tokenizer_arg = Path(args.tokenizer_path)
    if not tokenizer_arg.is_absolute():
        raise FusionStageError("--tokenizer-path must be absolute")
    try:
        tokenizer_source = tokenizer_arg.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FusionStageError(f"tokenizer directory does not exist: {tokenizer_arg}") from exc
    if not tokenizer_source.is_dir():
        raise FusionStageError("--tokenizer-path must name a local tokenizer directory")
    manifest_path = _safe_input_file(args.manifest, "manifest")
    if manifest_path.resolve(strict=True) != FROZEN_MANIFEST.resolve(strict=True):
        raise FusionStageError("only the frozen HM dev_seen 500-row manifest is allowed")
    try:
        safe_output_path(manifest_path, authorized_root=output_root, data_root=data_root)
    except EvaluatorError as exc:
        raise FusionStageError(f"manifest must be under autoresearch: {exc}") from exc
    initial_inputs: tuple[Path, ...] = (
        checkpoint,
        manifest_path,
        tokenizer_source,
        data_root,
    )
    output_dir = validate_output_destination(
        requested_output,
        authorized_root=output_root,
        data_root=data_root,
        input_paths=initial_inputs,
    )
    result_path = output_dir / RESULT_NAME
    archive_path = output_dir / NPZ_NAME
    for target in (result_path, archive_path):
        safe_output_path(target, authorized_root=output_root, data_root=data_root)
        if target.exists() or target.is_symlink():
            raise FusionStageError(f"refusing to overwrite existing stage output: {target}")

    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir = validate_output_destination(
        output_dir,
        authorized_root=output_root,
        data_root=data_root,
        input_paths=initial_inputs,
    )
    if output_dir.resolve(strict=True) != requested_output.resolve(strict=True):
        raise FusionStageError("resolved output directory changed during creation")
    (
        close_validation_index,
        load_hateful_memes_item,
        manifest_sha256,
        open_validation_index,
        read_manifest,
        validate_manifest_against_index,
    ) = _load_hm_helpers(output_dir)
    manifest = read_manifest(manifest_path)
    sample_ids = validate_hm_manifest_protocol(manifest, requested_n=MAX_SAMPLES)
    index = open_validation_index("hateful_memes", "dev_seen", data_root)
    archive_written = False
    try:
        validate_manifest_against_index(manifest, index)
        manifest_hash = manifest_sha256(manifest)
        if manifest_hash != FROZEN_MANIFEST_SHA256:
            raise FusionStageError("frozen HM dev_seen manifest content hash differs")
        records_by_id = index.by_id()
        records = [records_by_id[sample_id] for sample_id in sample_ids]
        row_labels = [int(_record_payload(record)["label"]) for record in records]
        if any(label not in (0, 1) for label in row_labels):
            raise FusionStageError("frozen HM labels must be binary")
        source_data_sha256 = dict(sorted(index.source_files.items()))
        source_samples = {item["id"]: item for item in manifest["samples"]}
        image_group_ids = [source_samples[sample_id]["group_id"] for sample_id in sample_ids]
        text_group_ids = [
            source_samples[sample_id]["caption_group_id"] for sample_id in sample_ids
        ]
        sample_pair_hashes = {
            item["id"]: item["pair_sha256"] for item in manifest["samples"]
        }
        checkpoint_sha256 = _sha256_file(checkpoint)
        if args.device.startswith("cuda"):
            if not torch.cuda.is_available():
                raise FusionStageError(f"requested {args.device}, but CUDA is unavailable")
            device_index = (
                int(args.device.split(":", 1)[1]) if ":" in args.device else 0
            )
            torch.cuda.set_per_process_memory_fraction(0.30, device_index)
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        torch.set_num_threads(4)
        try:
            torch.set_num_interop_threads(2)
        except RuntimeError:
            pass
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        with _working_directory(output_dir):
            tokenizer, tokenizer_identity = _load_local_tokenizer(
                str(tokenizer_source),
                output_dir,
                authorized_root=output_root,
                data_root=data_root,
            )
            from config import CLS_FUSION_METHOD, TOKENIZER_MAX_LEN
            from datasets.dataset_utils import process_image

            text_max_length = int(TOKENIZER_MAX_LEN)
            if text_max_length < 1:
                raise FusionStageError("TOKENIZER_MAX_LEN must be positive")
            model, architecture, provenance = _load_model_without_pretrained_downloads(
                checkpoint,
                args.device,
                working_directory=output_dir,
            )
            if getattr(model, "training", True):
                raise FusionStageError("offline checkpoint loader did not set eval mode")
            specs = event_specs_from_architecture(
                architecture, cross_attention_block_count=len(model.c_layers)
            )
            expected_text_ids, expected_vision_ids = MODEL_INSERTION_IDS[args.model_id]
            if (
                tuple(architecture["text_cross_attention_layers_zero_based"])
                != expected_text_ids
                or tuple(architecture["vision_cross_attention_layers_zero_based"])
                != expected_vision_ids
            ):
                raise FusionStageError(
                    f"--model-id {args.model_id} does not match the checkpoint's "
                    "saved text/vision insertion IDs"
                )
            if tuple(model.t_biattention_ids) != tuple(
                architecture["text_cross_attention_layers_zero_based"]
            ) or tuple(model.v_biattention_ids) != tuple(
                architecture["vision_cross_attention_layers_zero_based"]
            ):
                raise FusionStageError("loaded model insertion IDs differ from saved config")
            embedding_dim = int(model.config.embedding_dim)
            if embedding_dim != 768:
                raise FusionStageError("frozen HM event cohort requires 768-wide CLS states")
            if args.checkpoint_mode == "hm-finetuned":
                from autoresearch_task_reliance import validate_head_dimensions

                validate_head_dimensions(model.fc, CLS_FUSION_METHOD, embedding_dim)

            code_hashes = _event_code_hashes(PROJECT_ROOT)
            code_hashes.update(
                {
                    relative: _sha256_file(PROJECT_ROOT / relative)
                    for relative in FUSION_STAGE_CODE_FILES
                }
            )
            reference: dict[str, Any] | None = None
            reference_path: Path | None = None
            if args.reference_event_result:
                reference_path = _safe_input_file(
                    args.reference_event_result, "reference event result"
                )
                if _relative_overlap(output_dir, reference_path.resolve(strict=True)):
                    raise FusionStageError("output directory overlaps reference event result")
                reference = _validate_reference_event_result(
                    reference_path,
                    model_id=args.model_id,
                    mode=args.checkpoint_mode,
                    checkpoint=checkpoint,
                    checkpoint_sha256=checkpoint_sha256,
                    manifest_sha256=manifest_hash,
                    sample_ids=sample_ids,
                    row_labels=row_labels,
                    image_group_ids=image_group_ids,
                    text_group_ids=text_group_ids,
                    source_data_sha256=source_data_sha256,
                    tokenizer_identity=tokenizer_identity,
                    text_max_length=text_max_length,
                    current_code_hashes=code_hashes,
                    event_specs=specs,
                )
            elif args.model_id in ("df1", "df2") and args.checkpoint_mode == "pretrain":
                raise FusionStageError(
                    "DF1/DF2 pretrain captures require --reference-event-result for "
                    "full-cohort historical feature replay"
                )

            def text_loader(record: Any) -> str:
                return load_hateful_memes_item(record)[0]

            def image_loader(record: Any) -> Any:
                image_path = load_hateful_memes_item(record)[1]
                processed = process_image(image_path, transform=None)
                if processed is None or processed.get("pixel_values") is None:
                    raise FusionStageError(
                        f"image preprocessing failed for {_safe_record_id(record)}"
                    )
                return processed["pixel_values"].squeeze(0)

            def resource_check() -> None:
                elapsed = time.monotonic() - started
                snapshot = _resource_snapshot(args.device)
                if elapsed > MAX_JOB_WALL_SECONDS:
                    raise FusionStageError(
                        f"job exceeded {MAX_JOB_WALL_SECONDS}s wall limit"
                    )
                if snapshot["host_rss_peak_bytes"] > MAX_HOST_RSS_BYTES:
                    raise FusionStageError("job exceeded 8 GiB host RSS limit")

            captured = extract_staged_features(
                model=model,
                tokenizer=tokenizer,
                records=records,
                sample_ids=sample_ids,
                text_loader=text_loader,
                image_loader=image_loader,
                batch_size=args.batch_size,
                device=args.device,
                max_length=text_max_length,
                event_specs=specs,
                embedding_dim=embedding_dim,
                mode=args.checkpoint_mode,
                fusion_method=CLS_FUSION_METHOD,
                resource_check=resource_check,
            )
        reference_comparison = _compare_reference_events(captured, reference)
        for target in (archive_path, result_path):
            safe_output_path(target, authorized_root=output_root, data_root=data_root)
        if time.monotonic() - started > MAX_JOB_WALL_SECONDS:
            raise FusionStageError(f"job exceeded {MAX_JOB_WALL_SECONDS}s wall limit")
        resources = _resource_snapshot(args.device)
        if resources["host_rss_peak_bytes"] > MAX_HOST_RSS_BYTES:
            raise FusionStageError("job exceeded 8 GiB host RSS limit")
        if resources["gpu"] is not None and (
            resources["gpu"]["peak_reserved_bytes"] > int(0.30 * torch.cuda.get_device_properties(
                torch.device(args.device)
            ).total_memory)
        ):
            raise FusionStageError("job exceeded its 30% CUDA allocator reservation")

        event_records = []
        for spec in specs:
            event_records.append(
                {
                    "ordinal_id": spec.ordinal_id,
                    "text_insertion_layer_id": spec.text_insertion_layer_id,
                    "vision_insertion_layer_id": spec.vision_insertion_layer_id,
                    "text_pre_event_backbone_layer_id": spec.text_insertion_layer_id - 1,
                    "vision_pre_event_backbone_layer_id": spec.vision_insertion_layer_id - 1,
                    "text_update_recipient": "text queries attend to vision keys/values",
                    "vision_update_recipient": "vision queries attend to text keys/values",
                    "depth_comparison_note": (
                        "event ordinals pair calls in c_layers order; compare insertion IDs "
                        "explicitly because text and vision may enter at different depths"
                    ),
                }
            )
        runtime = _runtime_metadata(args.device)
        data_arrays = {
            **{name: captured[name] for name in STAGE_ARRAY_NAMES},
            **{
                name: captured[name]
                for name in (
                    "projected_update_full_token_rms",
                    "pre_event_full_token_rms",
                    "projected_update_to_pre_event_norm_ratio",
                    "projected_update_cls_to_pre_event_cls_norm_ratio",
                    "projected_update_cls_cosine_to_pre_event_cls",
                )
            },
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
            "event_stage_names": np.asarray(STAGE_ARRAY_NAMES, dtype="U32"),
            "stream_names": np.asarray(STREAMS, dtype="U8"),
        }
        if captured["hm_logits"] is not None:
            data_arrays["hm_logits"] = captured["hm_logits"]
        metadata = {
            "format": "autoresearch-hm-fusion-stages-v1",
            "axis_order": ["sample", "event_ordinal", "stream_text_then_vision", "feature"],
            "summary_axis_order": ["sample", "event_ordinal", "stream_text_then_vision"],
            "event_stages": {
                name: STAGE_DESCRIPTIONS[name] for name in STAGE_ARRAY_NAMES
            },
            "event_mapping": event_records,
            "ordered_sample_ids": sample_ids,
            "row_labels": row_labels,
            "image_group_ids": image_group_ids,
            "text_group_ids": text_group_ids,
            "checkpoint_sha256": checkpoint_sha256,
            "checkpoint_mode": args.checkpoint_mode,
            "model_id": args.model_id,
            "manifest_sha256": manifest_hash,
            "source_data_sha256": source_data_sha256,
            "sample_pair_hashes": sample_pair_hashes,
            "tokenizer_identity": tokenizer_identity,
            "text_max_length": text_max_length,
        }
        data_arrays["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        _write_npz_once(archive_path, data_arrays)
        archive_written = True
        artifact_bytes = archive_path.stat().st_size
        if artifact_bytes > MAX_ARTIFACT_BYTES:
            raise FusionStageError("stage NPZ exceeded the 512 MiB artifact limit")
        result = {
            "format": "autoresearch-hateful-memes-fusion-stages-v1",
            "status": "complete",
            "model_id": args.model_id,
            "checkpoint_mode": args.checkpoint_mode,
            "checkpoint": {
                "path": str(checkpoint),
                "sha256": checkpoint_sha256,
                "architecture": architecture,
                "provenance": provenance,
            },
            "cohort": {
                "task": "hateful_memes",
                "split": "dev_seen",
                "validation_role": "validation_discovery_seen_historical",
                "interpretation": (
                    "historical exposed discovery cohort; not independent confirmation "
                    "and not a causal architecture estimate"
                ),
                "n": MAX_SAMPLES,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_hash,
                "ordered_sample_ids": sample_ids,
                "row_labels": row_labels,
                "source_data_sha256": source_data_sha256,
                "image_group_ids": image_group_ids,
                "text_group_ids": text_group_ids,
            },
            "event_mapping": event_records,
            "stage_semantics": {
                name: STAGE_DESCRIPTIONS[name] for name in STAGE_ARRAY_NAMES
            },
            "update_summaries": {
                "projected_update_full_token_rms": (
                    "per example/stream sqrt(mean(update^2)) over every token and feature"
                ),
                "pre_event_full_token_rms": (
                    "per example/stream sqrt(mean(residual^2)) over every token and feature"
                ),
                "projected_update_to_pre_event_norm_ratio": (
                    "full-token Frobenius norm of projected update divided by residual norm"
                ),
                "projected_update_cls_to_pre_event_cls_norm_ratio": (
                    "CLS-vector L2 norm of projected update divided by residual CLS L2 norm"
                ),
                "projected_update_cls_cosine_to_pre_event_cls": (
                    "cosine similarity of projected update CLS and residual CLS"
                ),
            },
            "capture_verification": {
                "mode": "eval + torch.inference_mode",
                "hook_effect": "all hooks return None and only observe/copy values",
                "first_batch_no_hook_parity": captured["first_batch_parity"],
                "expected_batch_size": args.batch_size,
                "batch_count": captured["batch_count"],
                "hook_call_counts": captured["hook_call_counts"],
                "hook_handles_registered": captured["registered_hook_count"],
                "hook_handles_removed": captured["removed_hook_count"],
                "checkpoint_reference_comparison": reference_comparison,
            },
            "head_and_fusion": (
                None
                if args.checkpoint_mode == "pretrain"
                else {
                    "fusion_method": CLS_FUSION_METHOD,
                    "formula_source": "autoresearch_task_reliance.fuse_cls_embeddings",
                    "head": "loaded model.fc; logits are saved in hm_logits",
                }
            ),
            "preprocessing": {
                "image": IMAGE_PREPROCESSING_ID,
                "text": "local BERT tokenizer, fixed-length padding/truncation",
                "text_max_length": text_max_length,
                "tokenizer_identity": tokenizer_identity,
            },
            "limits": {
                "batch_size_max": MAX_BATCH_SIZE,
                "cpu_threads": 4,
                "cuda_allocator_fraction": 0.30,
                "host_rss_bytes_max": MAX_HOST_RSS_BYTES,
                "wall_seconds_max": MAX_JOB_WALL_SECONDS,
                "artifact_bytes_max": MAX_ARTIFACT_BYTES,
                "checkpoint_or_optimizer_state_written": False,
            },
            "resource_observed": {
                **resources,
                "elapsed_wall_seconds": time.monotonic() - started,
            },
            "code_sha256": code_hashes,
            "runtime": runtime,
            "npz": {
                "path": str(archive_path),
                "sha256": _sha256_file(archive_path),
                "bytes": artifact_bytes,
                "array_shapes": {
                    name: list(value.shape)
                    for name, value in data_arrays.items()
                    if hasattr(value, "shape")
                },
                "dtype": "float32 for feature/summary arrays",
            },
        }
        _write_json_once(result_path, result)
        archive_written = False
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


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result_path = run(args)
    except (
        EvaluatorError,
        EventExtractionError,
        FusionStageError,
        OSError,
        RuntimeError,
        ValueError,
    ) as exc:
        print(f"fusion stage extraction failed: {exc}", file=sys.stderr)
        return 2
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Run one DF1 local input-state counterfactual through frozen ViT block 6.

This thin adapter reuses the frozen event driver's manifest, preprocessing,
checkpoint, cache, output, and instrumentation controls. Its additional hooks
compare block 6 on the actual post-cross-attention input with the same block on
the full vision-token state captured immediately before DF1 c_layers[1].
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

import autoresearch_events as base


PROJECT_ROOT = base.PROJECT_ROOT
OUTPUT_RELATIVE_ROOT = Path("local-counterfactual")
RESULT_FORMAT = "autoresearch-hateful-memes-local-counterfactual-v1"
NPZ_FORMAT = "autoresearch-hm-local-counterfactual-features-v1"
EXPECTED_TEXT_INSERTION_IDS = (3, 4, 5)
EXPECTED_VISION_INSERTION_IDS = (5, 6, 7)
TARGET_VISION_BLOCK_ID = 6
EXPECTED_TOKENS = 197
EXPECTED_WIDTH = 768
COUNTERFACTUAL_ARRAY_KEYS = (
    "cf_input_cls",
    "actual_input_cls",
    "actual_output_cls",
    "cf_output_cls",
)


class LocalCounterfactualError(base.EventExtractionError):
    """Raised when the DF1 local counterfactual violates its frozen contract."""


class LocalCounterfactualCapture:
    """Capture four CLS arrays while rerunning ViT block 6 on pre-CA1 tokens.

    The only retained full-token tensor is a batch-local clone from the
    ``c_layers[1]`` pre-hook. It is passed once to the same block-6 module and
    released immediately; persistent results contain CLS vectors only.
    """

    def __init__(
        self,
        ca1_module: Any,
        vision_block: Any,
        *,
        token_count: int = EXPECTED_TOKENS,
        embedding_dim: int = EXPECTED_WIDTH,
    ) -> None:
        self.ca1_module = ca1_module
        self.vision_block = vision_block
        self.token_count = int(token_count)
        self.embedding_dim = int(embedding_dim)
        if self.token_count <= 0 or self.embedding_dim <= 0:
            raise LocalCounterfactualError("token count and embedding width must be positive")
        self._handles: list[Any] = []
        self._entered = False
        self._reentry = False
        self._current: dict[str, Any] | None = None
        self._arrays: dict[str, list[Any]] = {key: [] for key in COUNTERFACTUAL_ARRAY_KEYS}
        self.call_counts = {
            "ca1_pre_hook": 0,
            "ca1_post_hook": 0,
            "block6_actual_pre_hook": 0,
            "block6_actual_post_hook": 0,
            "block6_counterfactual_calls": 0,
        }

    def __enter__(self) -> "LocalCounterfactualCapture":
        if self._entered:
            raise LocalCounterfactualError("counterfactual hooks cannot be entered twice")
        self._entered = True
        try:
            self._handles.append(
                self.ca1_module.register_forward_pre_hook(self._ca1_pre_hook)
            )
            self._handles.append(
                self.ca1_module.register_forward_hook(self._ca1_post_hook)
            )
            self._handles.append(
                self.vision_block.register_forward_pre_hook(self._block_pre_hook)
            )
            self._handles.append(
                self.vision_block.register_forward_hook(self._block_post_hook)
            )
        except BaseException:
            self.remove_hooks()
            raise
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> bool:
        self._current = None
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

    def _validate_full_tokens(self, value: Any, *, label: str) -> Any:
        import torch

        if not torch.is_tensor(value) or value.ndim != 3:
            raise LocalCounterfactualError(f"{label} must be a [batch,tokens,width] tensor")
        if tuple(value.shape[1:]) != (self.token_count, self.embedding_dim):
            raise LocalCounterfactualError(
                f"{label} shape must end in [{self.token_count},{self.embedding_dim}]"
            )
        if not torch.isfinite(value).all():
            raise LocalCounterfactualError(f"{label} contains non-finite values")
        return value

    def _copy_cls(self, value: Any, *, label: str) -> Any:
        import torch

        tensor = self._validate_full_tokens(value, label=label)
        return tensor.detach()[:, 0, :].to(device="cpu", dtype=torch.float32).clone()

    def _commit_batch(self) -> None:
        if self._current is None:
            return
        if self._current.get("stage") != "complete":
            raise LocalCounterfactualError(
                "a new c_layers[1] call arrived before block 6 counterfactual capture completed"
            )
        for key in COUNTERFACTUAL_ARRAY_KEYS:
            value = self._current.get(key)
            if value is None:
                raise LocalCounterfactualError(f"counterfactual capture is missing {key}")
            self._arrays[key].append(value)
        self._current = None

    def _ca1_pre_hook(self, _module: Any, inputs: tuple[Any, ...]) -> None:
        if self._reentry:
            return None
        if self._current is not None:
            self._commit_batch()
        if len(inputs) != 2:
            raise LocalCounterfactualError("DF1 c_layers[1] must receive (text, vision) inputs")
        vision = self._validate_full_tokens(inputs[1], label="pre-CA1 vision input")
        self._current = {
            "stage": "before_ca1",
            # Keep this transient clone on the model's existing device/dtype.
            "pre_ca1_vision": vision.detach().clone(),
            "cf_input_cls": self._copy_cls(vision, label="pre-CA1 vision input"),
        }
        self.call_counts["ca1_pre_hook"] += 1
        return None

    def _ca1_post_hook(self, _module: Any, _inputs: tuple[Any, ...], outputs: Any) -> None:
        if self._reentry:
            return None
        if self._current is None or self._current.get("stage") != "before_ca1":
            raise LocalCounterfactualError("DF1 c_layers[1] output has no matching pre-hook")
        if not isinstance(outputs, tuple) or len(outputs) != 2:
            raise LocalCounterfactualError("DF1 c_layers[1] must return (text, vision)")
        pre_ca1 = self._current["pre_ca1_vision"]
        actual_vision = self._validate_full_tokens(outputs[1], label="post-CA1 vision output")
        if actual_vision.shape != pre_ca1.shape:
            raise LocalCounterfactualError("CA1 changed the vision token tensor shape")
        if actual_vision.device != pre_ca1.device or actual_vision.dtype != pre_ca1.dtype:
            raise LocalCounterfactualError("CA1 changed vision token device or dtype")
        self._current["stage"] = "after_ca1"
        self.call_counts["ca1_post_hook"] += 1
        return None

    def _block_pre_hook(self, module: Any, inputs: tuple[Any, ...]) -> None:
        if self._reentry:
            return None
        if self._current is None or self._current.get("stage") != "after_ca1":
            raise LocalCounterfactualError(
                "ViT block 6 must run exactly after DF1 c_layers[1] in this forward"
            )
        if len(inputs) != 1:
            raise LocalCounterfactualError("ViT block 6 must receive one positional token tensor")
        actual_input = self._validate_full_tokens(inputs[0], label="actual block-6 input")
        counterfactual_input = self._current["pre_ca1_vision"]
        if actual_input.shape != counterfactual_input.shape:
            raise LocalCounterfactualError("actual and counterfactual block-6 token shapes differ")
        if actual_input.device != counterfactual_input.device or actual_input.dtype != counterfactual_input.dtype:
            raise LocalCounterfactualError("actual and counterfactual block-6 device/dtype differ")
        self._current["actual_input_cls"] = self._copy_cls(
            actual_input, label="actual block-6 input"
        )
        self.call_counts["block6_actual_pre_hook"] += 1

        import torch

        self._reentry = True
        try:
            with torch.inference_mode():
                counterfactual_output = module(counterfactual_input)
        finally:
            self._reentry = False
        self._current["cf_output_cls"] = self._copy_cls(
            counterfactual_output, label="counterfactual block-6 output"
        )
        self._current["stage"] = "block6_actual_running"
        self.call_counts["block6_counterfactual_calls"] += 1
        return None

    def _block_post_hook(self, _module: Any, _inputs: tuple[Any, ...], outputs: Any) -> None:
        if self._reentry:
            return None
        if self._current is None or self._current.get("stage") != "block6_actual_running":
            raise LocalCounterfactualError("ViT block 6 output has no matching actual pre-hook")
        self._current["actual_output_cls"] = self._copy_cls(
            outputs, label="actual block-6 output"
        )
        self._current["stage"] = "complete"
        self.call_counts["block6_actual_post_hook"] += 1
        # The full-token input is no longer needed after the counterfactual call.
        self._current.pop("pre_ca1_vision", None)
        return None

    def finish(self, *, expected_batches: int) -> dict[str, Any]:
        import numpy as np

        if not self._entered:
            raise LocalCounterfactualError("counterfactual hooks must be active at finish")
        self._commit_batch()
        if expected_batches < 1:
            raise LocalCounterfactualError("expected batch count must be positive")
        if any(value != expected_batches for value in self.call_counts.values()):
            raise LocalCounterfactualError(
                f"counterfactual hook counts differ from expected {expected_batches} batches"
            )
        arrays = {
            key: np.concatenate(values, axis=0).astype(np.float32, copy=False)
            for key, values in self._arrays.items()
        }
        if any(value.ndim != 2 or value.shape[1] != self.embedding_dim for value in arrays.values()):
            raise LocalCounterfactualError("counterfactual CLS arrays have an unexpected shape")
        row_count = next(iter(arrays.values())).shape[0]
        if any(value.shape != (row_count, self.embedding_dim) for value in arrays.values()):
            raise LocalCounterfactualError("counterfactual CLS array shapes do not agree")
        return arrays


def validate_df1_architecture(model: Any, *, max_length: int) -> None:
    if tuple(getattr(model, "t_biattention_ids", ())) != EXPECTED_TEXT_INSERTION_IDS:
        raise LocalCounterfactualError("local counterfactual is restricted to DF1 text insertion IDs")
    if tuple(getattr(model, "v_biattention_ids", ())) != EXPECTED_VISION_INSERTION_IDS:
        raise LocalCounterfactualError("local counterfactual is restricted to DF1 vision insertion IDs")
    if max_length != EXPECTED_TOKENS:
        raise LocalCounterfactualError("local counterfactual requires the frozen 197-token text input")
    if len(getattr(model, "c_layers", ())) != 3:
        raise LocalCounterfactualError("DF1 checkpoint must contain exactly three cross-attention blocks")
    vit = getattr(model, "vit", None)
    blocks = getattr(vit, "blocks", None)
    if blocks is None or len(blocks) <= TARGET_VISION_BLOCK_ID:
        raise LocalCounterfactualError("DF1 model has no ViT block 6")
    if int(getattr(getattr(model, "config", None), "embedding_dim", -1)) != EXPECTED_WIDTH:
        raise LocalCounterfactualError("local counterfactual requires 768-wide ViT CLS states")
    if getattr(model, "training", True):
        raise LocalCounterfactualError("local counterfactual requires evaluation mode")


def validate_counterfactual_alignment(
    arrays: dict[str, Any],
    *,
    event_pre_cls: Any,
    event_post_cls: Any,
) -> dict[str, Any]:
    """Require exact equality with adjacent frozen event hook CLS snapshots."""
    import numpy as np

    pre = np.asarray(event_pre_cls)
    post = np.asarray(event_post_cls)
    expected = {
        "cf_input_cls": pre[:, 1, 1, :],
        "actual_input_cls": post[:, 1, 1, :],
        "actual_output_cls": pre[:, 2, 1, :],
    }
    checks: dict[str, Any] = {}
    for key, reference in expected.items():
        observed = np.asarray(arrays[key])
        if observed.shape != reference.shape:
            raise LocalCounterfactualError(f"{key} shape differs from its adjacent event state")
        exact = bool(np.array_equal(observed, reference))
        max_abs = float(np.max(np.abs(observed.astype(np.float64) - reference.astype(np.float64)), initial=0.0))
        checks[key] = {"matches_event_state_exactly": exact, "max_abs_error": max_abs}
        if not exact:
            raise LocalCounterfactualError(f"{key} does not exactly match its adjacent event state")
    return checks


def _version_signature(model: Any) -> dict[str, tuple[int, ...]]:
    return {
        "parameters": tuple(int(parameter._version) for parameter in model.parameters()),
        "buffers": tuple(int(buffer._version) for buffer in model.buffers()),
    }


def _rng_signature(device: str) -> dict[str, Any]:
    import torch

    result: dict[str, Any] = {"cpu": torch.random.get_rng_state().clone()}
    if device.startswith("cuda") and torch.cuda.is_available():
        result["cuda"] = torch.cuda.get_rng_state(torch.device(device)).clone()
    else:
        result["cuda"] = None
    return result


def _state_integrity(before_rng: dict[str, Any], before_versions: dict[str, tuple[int, ...]], model: Any, device: str) -> dict[str, bool]:
    import torch

    after_rng = _rng_signature(device)
    after_versions = _version_signature(model)
    checks = {
        "cpu_rng_unchanged": bool(torch.equal(before_rng["cpu"], after_rng["cpu"])),
        "cuda_rng_unchanged": (
            True
            if before_rng["cuda"] is None and after_rng["cuda"] is None
            else bool(
                before_rng["cuda"] is not None
                and after_rng["cuda"] is not None
                and torch.equal(before_rng["cuda"], after_rng["cuda"])
            )
        ),
        "parameter_versions_unchanged": before_versions["parameters"] == after_versions["parameters"],
        "buffer_versions_unchanged": before_versions["buffers"] == after_versions["buffers"],
    }
    if not all(checks.values()):
        raise LocalCounterfactualError("counterfactual call changed RNG, parameter, or buffer state")
    return checks


def _array_hashes(arrays: dict[str, Any]) -> dict[str, str]:
    import numpy as np

    return {
        key: hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
        for key, value in arrays.items()
    }


def _counterfactual_report(
    *,
    arrays: dict[str, Any],
    alignment_checks: dict[str, Any],
    capture: LocalCounterfactualCapture,
    batch_count: int,
    state_integrity: dict[str, bool],
) -> dict[str, Any]:
    return {
        "question": (
            "Does ViT block 6 produce a different output when it receives the full vision-token "
            "state immediately before DF1 c_layers[1], instead of the actual post-c_layers[1] state?"
        ),
        "target": "model.vit.blocks[6]",
        "conditioning_state": (
            "full vision token tensor at the pre-hook input to model.c_layers[1], before its "
            "CrossAttentionBlock update; passed to the same frozen ViT block 6"
        ),
        "semantics": (
            "acute within-checkpoint input-state intervention; not a training-architecture "
            "causal effect, mediation estimate, or task-information/reliability claim"
        ),
        "array_axis_order": ["sample", "feature"],
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "array_sha256": _array_hashes(arrays),
        "batch_count": batch_count,
        "call_counts": dict(capture.call_counts),
        "alignment_checks": alignment_checks,
        "state_integrity": state_integrity,
    }


@contextmanager
def _patched_base_for_run(state: dict[str, Any]):
    """Temporarily adapt the frozen base runner, restoring every global on error."""
    original_extract = base.extract_event_features
    original_npz_writer = base._write_npz_once
    original_json_writer = base._write_json_once
    original_output_root = base.OUTPUT_RELATIVE_ROOT
    original_code_files = base.EVENT_CODE_FILES

    def wrapped_extract(**kwargs: Any) -> dict[str, Any]:
        model = kwargs["model"]
        max_length = int(kwargs["max_length"])
        device = str(kwargs["device"])
        batch_size = int(kwargs["batch_size"])
        validate_df1_architecture(model, max_length=max_length)
        vit = model.vit
        block = vit.blocks[TARGET_VISION_BLOCK_ID]
        device_before_rng = _rng_signature(device)
        versions_before = _version_signature(model)
        capture = LocalCounterfactualCapture(
            model.c_layers[1],
            block,
            token_count=EXPECTED_TOKENS,
            embedding_dim=EXPECTED_WIDTH,
        )
        with capture:
            captured = original_extract(**kwargs)
            batch_count = math.ceil(len(kwargs["records"]) / batch_size)
            cf_arrays = capture.finish(expected_batches=batch_count)
        integrity = _state_integrity(device_before_rng, versions_before, model, device)
        alignment = validate_counterfactual_alignment(
            cf_arrays,
            event_pre_cls=captured["pre_cls"],
            event_post_cls=captured["post_cls"],
        )
        report = _counterfactual_report(
            arrays=cf_arrays,
            alignment_checks=alignment,
            capture=capture,
            batch_count=batch_count,
            state_integrity=integrity,
        )
        state["counterfactual_arrays"] = cf_arrays
        state["counterfactual_report"] = report
        return captured

    def wrapped_npz_writer(path: Path, arrays: dict[str, Any]) -> None:
        import numpy as np

        if "counterfactual_arrays" not in state:
            raise LocalCounterfactualError("counterfactual arrays were not captured before NPZ write")
        enriched = dict(arrays)
        enriched.update(state["counterfactual_arrays"])
        metadata = json.loads(str(np.asarray(enriched["metadata_json"]).item()))
        metadata["format"] = NPZ_FORMAT
        metadata["local_counterfactual"] = state["counterfactual_report"]
        enriched["metadata_json"] = np.asarray(
            json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        )
        original_npz_writer(path, enriched)
        state["npz_array_keys"] = sorted(enriched)

    def wrapped_json_writer(path: Path, value: dict[str, Any]) -> None:
        result = dict(value)
        result["format"] = RESULT_FORMAT
        result["local_counterfactual"] = state["counterfactual_report"]
        result["npz"] = dict(result["npz"])
        result["npz"]["local_counterfactual_arrays"] = {
            key: {
                "shape": state["counterfactual_report"]["array_shapes"][key],
                "dtype": "float32",
                "sha256": state["counterfactual_report"]["array_sha256"][key],
            }
            for key in COUNTERFACTUAL_ARRAY_KEYS
        }
        original_json_writer(path, result)

    try:
        base.OUTPUT_RELATIVE_ROOT = OUTPUT_RELATIVE_ROOT
        base.EVENT_CODE_FILES = tuple(
            original_code_files
            + (
                "src/autoresearch_local_counterfactual.py",
                "autoresearch/local_counterfactual_protocol.md",
            )
        )
        base.extract_event_features = wrapped_extract
        base._write_npz_once = wrapped_npz_writer
        base._write_json_once = wrapped_json_writer
        yield
    finally:
        base.extract_event_features = original_extract
        base._write_npz_once = original_npz_writer
        base._write_json_once = original_json_writer
        base.OUTPUT_RELATIVE_ROOT = original_output_root
        base.EVENT_CODE_FILES = original_code_files


def build_parser():
    """Reuse the frozen event driver's explicit-boundary CLI."""
    parser = base.build_parser()
    parser.description = (
        "Capture frozen DF1 event states and compare the actual versus "
        "pre-c_layers[1] input to ViT block 6. The matching checkpoint/cache "
        "control is mandatory."
    )
    for action in parser._actions:
        if action.dest == "output_dir":
            action.help = "only autoresearch/local-counterfactual/df1 is accepted"
    return parser


def run(args: Any) -> Path:
    output_root = base.authorized_output_root(PROJECT_ROOT)
    expected = output_root / OUTPUT_RELATIVE_ROOT / "df1"
    requested = Path(os.path.abspath(args.output_dir))
    if requested != expected:
        raise LocalCounterfactualError(
            f"local counterfactual output is restricted to {expected} (DF1 only)"
        )
    state: dict[str, Any] = {}
    with _patched_base_for_run(state):
        return base.run(args)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result_path = run(args)
    except (base.EvaluatorError, base.EventExtractionError, OSError, RuntimeError, ValueError) as exc:
        print(f"local counterfactual extraction failed: {exc}", file=sys.stderr)
        return 2
    print(result_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

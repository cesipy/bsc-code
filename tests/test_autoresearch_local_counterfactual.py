from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import autoresearch_events as base
import autoresearch_local_counterfactual as local_cf


class AddCrossAttention(torch.nn.Module):
    def __init__(self, amount=3.0):
        super().__init__()
        self.amount = amount

    def forward(self, text, vision):
        return text, vision + self.amount


class InPlaceViTBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))
        self.register_buffer("anchor", torch.tensor(1.0))
        self.forward_calls = 0

    def forward(self, tokens):
        self.forward_calls += 1
        # In-place mutation stresses the distinction between the cloned CF input
        # and the actual model state that proceeds through this block.
        tokens.mul_(self.weight)
        tokens.add_(self.anchor)
        return tokens


class TinyCounterfactualModel(torch.nn.Module):
    def __init__(self, *, tokens=3, width=2):
        super().__init__()
        self.t_biattention_ids = [3, 4, 5]
        self.v_biattention_ids = [5, 6, 7]
        self.config = SimpleNamespace(embedding_dim=width)
        self.c_layers = torch.nn.ModuleList(
            [torch.nn.Identity(), AddCrossAttention(), torch.nn.Identity()]
        )
        self.vit = SimpleNamespace(blocks=torch.nn.ModuleList(
            [torch.nn.Identity() for _ in range(6)] + [InPlaceViTBlock()]
        ))
        self.tokens = tokens

    def run_local_path(self, text, vision):
        text, vision = self.c_layers[1](text, vision)
        vision = self.vit.blocks[6](vision)
        return text, vision


def _hooks_removed(module):
    assert not module._forward_pre_hooks
    assert not module._forward_hooks


def test_capture_runs_same_block_on_pre_ca_tokens_without_changing_actual_output():
    model = TinyCounterfactualModel().eval()
    baseline_model = TinyCounterfactualModel().eval()
    text = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
    vision = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2) + 20
    with torch.inference_mode():
        expected_text, expected_vision = baseline_model.run_local_path(text.clone(), vision.clone())

    param_before = model.vit.blocks[6].weight.detach().clone()
    param_version_before = model.vit.blocks[6].weight._version
    buffer_before = model.vit.blocks[6].anchor.clone()
    buffer_version_before = model.vit.blocks[6].anchor._version
    rng_before = torch.random.get_rng_state().clone()
    capture = local_cf.LocalCounterfactualCapture(
        model.c_layers[1], model.vit.blocks[6], token_count=3, embedding_dim=2
    )
    with capture:
        with torch.inference_mode():
            actual_text, actual_vision = model.run_local_path(text.clone(), vision.clone())
        arrays = capture.finish(expected_batches=1)

    assert torch.equal(actual_text, expected_text)
    assert torch.equal(actual_vision, expected_vision)
    assert torch.equal(torch.from_numpy(arrays["cf_input_cls"]), vision[:, 0])
    assert torch.equal(torch.from_numpy(arrays["actual_input_cls"]), vision[:, 0] + 3)
    assert torch.equal(torch.from_numpy(arrays["actual_output_cls"]), (vision[:, 0] + 3) * 2 + 1)
    assert torch.equal(torch.from_numpy(arrays["cf_output_cls"]), vision[:, 0] * 2 + 1)
    assert model.vit.blocks[6].forward_calls == 2  # one CF pass and one actual pass
    assert torch.equal(model.vit.blocks[6].weight, param_before)
    assert torch.equal(model.vit.blocks[6].anchor, buffer_before)
    assert model.vit.blocks[6].weight._version == param_version_before
    assert model.vit.blocks[6].anchor._version == buffer_version_before
    assert torch.equal(torch.random.get_rng_state(), rng_before)
    assert capture.call_counts == {
        "ca1_pre_hook": 1,
        "ca1_post_hook": 1,
        "block6_actual_pre_hook": 1,
        "block6_actual_post_hook": 1,
        "block6_counterfactual_calls": 1,
    }
    _hooks_removed(model.c_layers[1])
    _hooks_removed(model.vit.blocks[6])


def test_capture_commits_batches_in_order_and_requires_exact_call_counts():
    model = TinyCounterfactualModel().eval()
    capture = local_cf.LocalCounterfactualCapture(
        model.c_layers[1], model.vit.blocks[6], token_count=3, embedding_dim=2
    )
    with capture:
        for offset in (0.0, 10.0):
            text = torch.zeros(1, 3, 2)
            vision = torch.full((1, 3, 2), offset)
            with torch.inference_mode():
                model.run_local_path(text, vision)
        arrays = capture.finish(expected_batches=2)
    assert arrays["cf_input_cls"].shape == (2, 2)
    assert arrays["cf_input_cls"][:, 0].tolist() == [0.0, 10.0]
    assert capture.call_counts["block6_counterfactual_calls"] == 2
    _hooks_removed(model.c_layers[1])
    _hooks_removed(model.vit.blocks[6])

    another = TinyCounterfactualModel().eval()
    incomplete = local_cf.LocalCounterfactualCapture(
        another.c_layers[1], another.vit.blocks[6], token_count=3, embedding_dim=2
    )
    with incomplete:
        with torch.inference_mode():
            with pytest.raises(local_cf.LocalCounterfactualError, match="counts differ"):
                incomplete.finish(expected_batches=1)


def test_capture_rejects_block6_before_ca1_and_removes_hooks_after_failure():
    model = TinyCounterfactualModel().eval()
    capture = local_cf.LocalCounterfactualCapture(
        model.c_layers[1], model.vit.blocks[6], token_count=3, embedding_dim=2
    )
    with capture:
        with pytest.raises(local_cf.LocalCounterfactualError, match="exactly after"):
            model.vit.blocks[6](torch.zeros(1, 3, 2))
    _hooks_removed(model.c_layers[1])
    _hooks_removed(model.vit.blocks[6])

    class FailingBlock(torch.nn.Module):
        def forward(self, tokens):
            raise RuntimeError("intentional counterfactual failure")

    failing = FailingBlock()
    capture = local_cf.LocalCounterfactualCapture(
        model.c_layers[1], failing, token_count=3, embedding_dim=2
    )
    with pytest.raises(RuntimeError, match="intentional"):
        with capture:
            _, vision = model.c_layers[1](torch.zeros(1, 3, 2), torch.zeros(1, 3, 2))
            failing(vision)
    _hooks_removed(model.c_layers[1])
    _hooks_removed(failing)


def test_df1_architecture_validator_rejects_df2_or_changed_input_boundary():
    model = TinyCounterfactualModel(tokens=197, width=768).eval()
    local_cf.validate_df1_architecture(model, max_length=197)
    model.v_biattention_ids = [9, 10, 11]
    with pytest.raises(local_cf.LocalCounterfactualError, match="DF1 vision"):
        local_cf.validate_df1_architecture(model, max_length=197)
    model.v_biattention_ids = [5, 6, 7]
    with pytest.raises(local_cf.LocalCounterfactualError, match="197-token"):
        local_cf.validate_df1_architecture(model, max_length=196)
    model.train()
    with pytest.raises(local_cf.LocalCounterfactualError, match="evaluation mode"):
        local_cf.validate_df1_architecture(model, max_length=197)


def test_alignment_controls_are_exact_and_fail_closed():
    pre = np.zeros((2, 3, 2, 4), dtype=np.float32)
    post = np.zeros_like(pre)
    pre[:, 1, 1, :] = 1
    post[:, 1, 1, :] = 2
    pre[:, 2, 1, :] = 3
    cf_arrays = {
        "cf_input_cls": pre[:, 1, 1].copy(),
        "actual_input_cls": post[:, 1, 1].copy(),
        "actual_output_cls": pre[:, 2, 1].copy(),
        "cf_output_cls": np.full((2, 4), 4, dtype=np.float32),
    }
    checks = local_cf.validate_counterfactual_alignment(
        cf_arrays, event_pre_cls=pre, event_post_cls=post
    )
    assert all(item["matches_event_state_exactly"] for item in checks.values())
    cf_arrays["actual_output_cls"][0, 0] += 1
    with pytest.raises(local_cf.LocalCounterfactualError, match="exactly match"):
        local_cf.validate_counterfactual_alignment(
            cf_arrays, event_pre_cls=pre, event_post_cls=post
        )


def test_runtime_integrity_guard_detects_parameter_or_buffer_mutation():
    class Stateful(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(2))
            self.register_buffer("counter", torch.zeros(1))

    model = Stateful()
    rng_before = local_cf._rng_signature("cpu")
    versions_before = local_cf._version_signature(model)
    assert all(local_cf._state_integrity(rng_before, versions_before, model, "cpu").values())

    rng_before = local_cf._rng_signature("cpu")
    versions_before = local_cf._version_signature(model)
    with torch.no_grad():
        model.counter.add_(1)
    with pytest.raises(local_cf.LocalCounterfactualError, match="state"):
        local_cf._state_integrity(rng_before, versions_before, model, "cpu")


def test_patch_context_restores_every_base_global_after_failure():
    names = (
        "extract_event_features",
        "_write_npz_once",
        "_write_json_once",
        "OUTPUT_RELATIVE_ROOT",
        "EVENT_CODE_FILES",
    )
    before = {name: getattr(base, name) for name in names}
    with pytest.raises(RuntimeError, match="intentional"):
        with local_cf._patched_base_for_run({}):
            assert base.OUTPUT_RELATIVE_ROOT == Path("local-counterfactual")
            assert "src/autoresearch_local_counterfactual.py" in base.EVENT_CODE_FILES
            raise RuntimeError("intentional patch scope failure")
    assert {name: getattr(base, name) for name in names} == before


def test_runner_boundary_is_df1_only_and_parser_reuses_all_explicit_flags():
    df2 = base.authorized_output_root(local_cf.PROJECT_ROOT) / "local-counterfactual" / "df2"
    with pytest.raises(local_cf.LocalCounterfactualError, match="DF1 only"):
        local_cf.run(SimpleNamespace(output_dir=str(df2)))
    parser = local_cf.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
    args = parser.parse_args(
        [
            "--checkpoint", "/checkpoint",
            "--data-root", "/data",
            "--manifest", "/manifest",
            "--output-dir", "/autoresearch/local-counterfactual/df1",
            "--tokenizer-path", "/tokenizer",
            "--reference-result", "/reference.json",
            "--device", "cuda:0",
            "--batch-size", "2",
        ]
    )
    assert args.device == "cuda:0"
    assert args.batch_size == 2

"""CPU contract tests for the opt-in cross-attention SDPA backend."""
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from attention import CrossAttention, CrossAttentionBlock
from config import ViLBERTConfig


DIM = 12
HEADS = 3
PROJECTED_SIZE = 24  # projected width per stream; head dim is 8, legacy dk is 4


def _make_blocks():
    torch.manual_seed(11)
    eager = CrossAttentionBlock(
        dim=DIM,
        heads=HEADS,
        dropout=0.0,
        coattn_hidden_size=PROJECTED_SIZE,
        cross_attention_backend="eager",
    )
    sdpa = CrossAttentionBlock(
        dim=DIM,
        heads=HEADS,
        dropout=0.0,
        coattn_hidden_size=PROJECTED_SIZE,
        cross_attention_backend="sdpa",
    )
    sdpa.load_state_dict(eager.state_dict())
    eager.eval()
    sdpa.eval()
    return eager, sdpa


def _inputs(mask_kind="float"):
    torch.manual_seed(19)
    text = torch.randn(2, 5, DIM)
    vision = torch.randn(2, 7, DIM)

    if mask_kind == "float":
        text_mask = torch.zeros(2, 1, 1, 5)
        text_mask[:, :, :, -2:] = -15.0
        vision_mask = torch.zeros(2, 1, 1, 7)
        vision_mask[:, :, :, -1:] = -9.0
    else:
        # Legacy eager behavior treats bool values as numeric additive biases.
        text_mask = torch.zeros(2, 1, 1, 5, dtype=torch.bool)
        text_mask[:, :, :, -2:] = True
        vision_mask = torch.zeros(2, 1, 1, 7, dtype=torch.bool)
        vision_mask[:, :, :, -1:] = True
    return text, vision, text_mask, vision_mask


@pytest.mark.parametrize("mask_kind", ["float", "legacy_bool_bias"])
def test_sdpa_matches_eager_outputs_and_gradients_with_asymmetric_masks(mask_kind):
    eager, sdpa = _make_blocks()
    text, vision, text_mask, vision_mask = _inputs(mask_kind)
    eager_text = text.clone().requires_grad_()
    eager_vision = vision.clone().requires_grad_()
    sdpa_text = text.clone().requires_grad_()
    sdpa_vision = vision.clone().requires_grad_()

    eager_out = eager(eager_text, eager_vision, text_mask=text_mask, vision_mask=vision_mask)
    sdpa_out = sdpa(sdpa_text, sdpa_vision, text_mask=text_mask, vision_mask=vision_mask)

    # The projected head width (8) differs from the historical attention scale
    # dimension (4), and text/vision have distinct query and key lengths.
    assert eager.cross_attention.dk == sdpa.cross_attention.dk == 4
    assert sdpa.cross_attention.query1.out_features == PROJECTED_SIZE
    assert eager_out[0].shape == sdpa_out[0].shape == (2, 5, DIM)
    assert eager_out[1].shape == sdpa_out[1].shape == (2, 7, DIM)
    for eager_tensor, sdpa_tensor in zip(eager_out, sdpa_out):
        torch.testing.assert_close(sdpa_tensor, eager_tensor, rtol=2e-5, atol=2e-6)

    eager_loss = sum(tensor.square().mean() for tensor in eager_out)
    sdpa_loss = sum(tensor.square().mean() for tensor in sdpa_out)
    eager_loss.backward()
    sdpa_loss.backward()

    torch.testing.assert_close(sdpa_text.grad, eager_text.grad, rtol=3e-5, atol=3e-6)
    torch.testing.assert_close(sdpa_vision.grad, eager_vision.grad, rtol=3e-5, atol=3e-6)
    eager_parameters = dict(eager.named_parameters())
    sdpa_parameters = dict(sdpa.named_parameters())
    assert eager_parameters.keys() == sdpa_parameters.keys()
    for name in eager_parameters:
        eager_grad = eager_parameters[name].grad
        sdpa_grad = sdpa_parameters[name].grad
        # CrossAttention has retained legacy to_out_proj layers that its
        # current forward does not use; preserve that unused-gradient pattern.
        assert (eager_grad is None) == (sdpa_grad is None), name
        if eager_grad is not None:
            torch.testing.assert_close(sdpa_grad, eager_grad, rtol=4e-5, atol=4e-6, msg=name)


@pytest.mark.parametrize(
    ("training", "expected"),
    [(True, [0.23, 0.37]), (False, [0.0, 0.0])],
)
def test_sdpa_uses_attention_dropout_probability_only_during_training(training, expected):
    module = CrossAttention(
        dim=DIM,
        all_head_size=PROJECTED_SIZE,
        heads=HEADS,
        cross_attention_backend="sdpa",
    )
    module.t_attn_dropout.p = 0.23
    module.v_attn_dropout.p = 0.37
    module.train(training)
    text = torch.randn(2, 5, DIM)
    vision = torch.randn(2, 7, DIM)

    with patch("attention.F.scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa_call:
        module(text, vision)

    assert [call.kwargs["dropout_p"] for call in sdpa_call.call_args_list] == expected
    assert [call.kwargs["scale"] for call in sdpa_call.call_args_list] == [4 ** -0.5] * 2


def test_sdpa_respects_frozen_attention_dropout_submodule():
    module = CrossAttention(
        dim=DIM,
        all_head_size=PROJECTED_SIZE,
        heads=HEADS,
        cross_attention_backend="sdpa",
    )
    module.train()
    module.t_attn_dropout.eval()
    module.v_attn_dropout.p = 0.37
    assert module.training
    assert not module.t_attn_dropout.training
    assert module.v_attn_dropout.training

    with patch("attention.F.scaled_dot_product_attention", wraps=F.scaled_dot_product_attention) as sdpa_call:
        module(torch.randn(2, 5, DIM), torch.randn(2, 7, DIM))

    assert [call.kwargs["dropout_p"] for call in sdpa_call.call_args_list] == [0.0, 0.37]


def test_sdpa_casts_additive_mask_to_query_dtype_under_cpu_autocast():
    module = CrossAttentionBlock(
        dim=DIM,
        heads=HEADS,
        dropout=0.0,
        coattn_hidden_size=PROJECTED_SIZE,
        cross_attention_backend="sdpa",
    ).eval()
    text, vision, text_mask, vision_mask = _inputs("float")

    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        text_out, vision_out = module(
            text, vision, text_mask=text_mask, vision_mask=vision_mask
        )

    assert text_out.shape == text.shape
    assert vision_out.shape == vision.shape
    assert torch.isfinite(text_out).all()
    assert torch.isfinite(vision_out).all()


def test_backend_is_opt_in_config_roundtrips_and_does_not_change_state_keys():
    default_config = ViLBERTConfig()
    assert default_config.cross_attention_backend == "eager"
    assert default_config.mlm_selected_tokens_only is False

    sdpa_config = ViLBERTConfig(
        cross_attention_backend="sdpa", mlm_selected_tokens_only=True
    )
    serialized = sdpa_config.to_dict()
    assert serialized["cross_attention_backend"] == "sdpa"
    assert serialized["mlm_selected_tokens_only"] is True
    restored = ViLBERTConfig.from_dict(serialized)
    assert restored.cross_attention_backend == "sdpa"
    assert restored.mlm_selected_tokens_only is True
    assert ViLBERTConfig.from_dict({}).cross_attention_backend == "eager"

    eager, sdpa = _make_blocks()
    assert eager.cross_attention.cross_attention_backend == "eager"
    assert sdpa.cross_attention.cross_attention_backend == "sdpa"
    eager_state = eager.state_dict()
    sdpa_state = sdpa.state_dict()
    assert eager_state.keys() == sdpa_state.keys()
    assert all(eager_state[name].shape == sdpa_state[name].shape for name in eager_state)


def test_invalid_backend_is_rejected_at_config_and_module_boundaries():
    with pytest.raises(ValueError, match="cross_attention_backend"):
        ViLBERTConfig(cross_attention_backend="flash")
    with pytest.raises(ValueError, match="cross_attention_backend"):
        CrossAttentionBlock(
            dim=DIM,
            heads=HEADS,
            coattn_hidden_size=PROJECTED_SIZE,
            cross_attention_backend="flash",
        )

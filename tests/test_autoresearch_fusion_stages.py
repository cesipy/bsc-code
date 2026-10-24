import numpy as np
import pytest
import torch
from torch import nn
from torch.nn import functional as F

import autoresearch_fusion_stages as stages
from autoresearch_events import EventSpec


class ToyOutput(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.projection_text = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.2))
        self.projection_vision = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(0.2))
        self.norm_text = nn.LayerNorm(dim)
        self.norm_vision = nn.LayerNorm(dim)

    def forward(self, text_attn, vision_attn, text_input, vision_input):
        projected_text = self.projection_text(text_attn)
        projected_vision = self.projection_vision(vision_attn)
        text = self.norm_text(text_input + projected_text)
        vision = self.norm_vision(vision_input + projected_vision)
        return text, vision


class ToyCrossAttentionBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.text_from_vision = nn.Linear(dim, dim)
        self.vision_from_text = nn.Linear(dim, dim)
        self.output = ToyOutput(dim)
        self.ff_text = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.ff_vision = nn.Sequential(nn.Linear(dim, dim), nn.GELU())
        self.fail_after_output = False

    def attentions(self, text, vision):
        text_attn = self.text_from_vision(vision.mean(dim=1, keepdim=True)).expand(
            -1, text.shape[1], -1
        )
        vision_attn = self.vision_from_text(text.mean(dim=1, keepdim=True)).expand(
            -1, vision.shape[1], -1
        )
        return text_attn, vision_attn

    def forward(self, text, vision, text_mask=None):
        del text_mask
        text_attn, vision_attn = self.attentions(text, vision)
        text, vision = self.output(text_attn, vision_attn, text, vision)
        if self.fail_after_output:
            raise RuntimeError("intentional toy forward failure")
        return text + self.ff_text(text), vision + self.ff_vision(vision)


class ToyModel(nn.Module):
    def __init__(self, dim=4):
        super().__init__()
        self.word_embeddings = nn.Embedding(32, dim)
        self.image_embeddings = nn.Linear(3, dim)
        self.c_layers = nn.ModuleList([ToyCrossAttentionBlock(dim) for _ in range(3)])
        self.fc = nn.Linear(dim * 2, 1)
        self.forward_calls = 0

    def forward(
        self,
        *,
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values,
        image_attention_mask=None,
    ):
        del text_attention_mask, text_token_type_ids, image_attention_mask
        self.forward_calls += 1
        text = self.word_embeddings(text_input_ids)
        patches = F.adaptive_avg_pool2d(image_pixel_values, (2, 2))
        patches = patches.permute(0, 2, 3, 1).reshape(image_pixel_values.shape[0], 4, 3)
        vision = self.image_embeddings(patches)
        for block in self.c_layers:
            text, vision = block(text, vision)
        return text[:, 0, :], vision[:, 0, :]


def _one_spec():
    return (EventSpec(ordinal_id=0, text_insertion_layer_id=3, vision_insertion_layer_id=5),)


def _assert_close(actual, expected):
    if torch.is_tensor(actual):
        actual = actual.detach().cpu().numpy()
    if torch.is_tensor(expected):
        expected = expected.detach().cpu().numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=1e-6)


def test_capture_records_exact_two_stream_stages_and_update_summaries():
    torch.manual_seed(19)
    block = ToyCrossAttentionBlock(dim=4).eval()
    text = torch.randn(2, 3, 4)
    vision = torch.randn(2, 2, 4)

    text_attn, vision_attn = block.attentions(text, vision)
    text_update = block.output.projection_text(text_attn)
    vision_update = block.output.projection_vision(vision_attn)
    text_pre_ln = text + text_update
    vision_pre_ln = vision + vision_update
    text_post_ln = block.output.norm_text(text_pre_ln)
    vision_post_ln = block.output.norm_vision(vision_pre_ln)
    text_post = text_post_ln + block.ff_text(text_post_ln)
    vision_post = vision_post_ln + block.ff_vision(vision_post_ln)

    capture = stages.CrossAttentionStageCapture([block], _one_spec(), embedding_dim=4)
    with capture:
        capture.begin_batch(2)
        actual_text, actual_vision = block(text, vision)
        captured = capture.finish_batch()

    _assert_close(captured["pre_cls"][:, 0, 0], text[:, 0])
    _assert_close(captured["pre_cls"][:, 0, 1], vision[:, 0])
    _assert_close(captured["projected_update_cls"][:, 0, 0], text_update[:, 0])
    _assert_close(captured["projected_update_cls"][:, 0, 1], vision_update[:, 0])
    _assert_close(captured["pre_ln_cls"][:, 0, 0], text_pre_ln[:, 0])
    _assert_close(captured["pre_ln_cls"][:, 0, 1], vision_pre_ln[:, 0])
    _assert_close(captured["post_ln_cls"][:, 0, 0], text_post_ln[:, 0])
    _assert_close(captured["post_ln_cls"][:, 0, 1], vision_post_ln[:, 0])
    _assert_close(captured["post_cls"][:, 0, 0], text_post[:, 0])
    _assert_close(captured["post_cls"][:, 0, 1], vision_post[:, 0])
    _assert_close(captured["projected_update_full_token_rms"][:, 0, 0], text_update.square().mean((1, 2)).sqrt())
    _assert_close(captured["projected_update_full_token_rms"][:, 0, 1], vision_update.square().mean((1, 2)).sqrt())
    _assert_close(actual_text, text_post)
    _assert_close(actual_vision, vision_post)
    assert capture.pre_call_counts == [1]
    assert capture.post_call_counts == [1]
    assert capture.removed_handle_count == capture.registered_handle_count
    assert not capture.hook_removal_errors


class CountingTokenizer:
    def __init__(self):
        self.calls = 0

    def __call__(self, texts, **_kwargs):
        self.calls += 1
        ids = [[(int(text) + offset) % 32 for offset in range(4)] for text in texts]
        return {
            "input_ids": torch.tensor(ids, dtype=torch.long),
            "attention_mask": torch.ones((len(texts), 4), dtype=torch.long),
            "token_type_ids": torch.zeros((len(texts), 4), dtype=torch.long),
        }


def test_full_extractor_reuses_first_batch_for_no_hook_parity_and_emits_hm_logits():
    torch.manual_seed(91)
    model = ToyModel(dim=4).eval()
    tokenizer = CountingTokenizer()
    records = list(range(stages.MAX_SAMPLES))
    ids = [f"sample-{value:03d}" for value in records]
    image_loads = []

    def load_image(record):
        image_loads.append(record)
        value = float(record % 7) / 7.0
        return torch.full((3, 4, 4), value, dtype=torch.float32)

    result = stages.extract_staged_features(
        model=model,
        tokenizer=tokenizer,
        records=records,
        sample_ids=ids,
        text_loader=lambda row: str(row),
        image_loader=load_image,
        batch_size=2,
        device="cpu",
        max_length=4,
        event_specs=(
            EventSpec(0, 3, 5),
            EventSpec(1, 4, 6),
            EventSpec(2, 5, 7),
        ),
        embedding_dim=4,
        mode="hm-finetuned",
        fusion_method="concat",
    )

    assert tokenizer.calls == stages.MAX_SAMPLES // 2
    assert len(image_loads) == stages.MAX_SAMPLES
    assert model.forward_calls == stages.MAX_SAMPLES // 2 + 1
    assert result["first_batch_parity"]["same_first_batch_reused"] is True
    assert result["first_batch_parity"]["text_final_cls"]["allclose"] is True
    assert result["first_batch_parity"]["vision_final_cls"]["allclose"] is True
    assert result["first_batch_parity"]["hm_logits"]["allclose"] is True
    for name in stages.STAGE_ARRAY_NAMES:
        assert result[name].shape == (500, 3, 2, 4)
        assert np.isfinite(result[name]).all()
    assert result["hm_logits"].shape == (500, 1)
    assert result["removed_hook_count"] == 3 * 8


def test_hook_cleanup_runs_when_block_forward_raises():
    block = ToyCrossAttentionBlock(dim=4).eval()
    block.fail_after_output = True
    capture = stages.CrossAttentionStageCapture([block], _one_spec(), embedding_dim=4)
    with pytest.raises(RuntimeError, match="intentional toy forward failure"):
        with capture:
            capture.begin_batch(1)
            block(torch.randn(1, 2, 4), torch.randn(1, 2, 4))
    assert capture.removed_handle_count == capture.registered_handle_count == 8
    assert not capture.hook_removal_errors
    assert not block._forward_hooks
    assert not block._forward_pre_hooks
    assert not block.output.projection_text._forward_hooks
    assert not block.output.norm_text._forward_hooks
    assert not block.output.norm_text._forward_pre_hooks


def test_capture_rejects_duplicate_and_out_of_order_block_calls():
    blocks = [ToyCrossAttentionBlock(dim=4).eval() for _ in range(2)]
    specs = (EventSpec(0, 3, 5), EventSpec(1, 4, 6))
    capture = stages.CrossAttentionStageCapture(blocks, specs, embedding_dim=4)
    inputs = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))

    with pytest.raises(stages.FusionStageError, match="out-of-order"):
        with capture:
            capture.begin_batch(1)
            blocks[1](*inputs)
    assert capture.removed_handle_count == capture.registered_handle_count == 16

    capture = stages.CrossAttentionStageCapture(blocks[:1], specs[:1], embedding_dim=4)
    with pytest.raises(stages.FusionStageError, match="out-of-order"):
        with capture:
            capture.begin_batch(1)
            blocks[0](*inputs)
            blocks[0](*inputs)
    assert capture.removed_handle_count == capture.registered_handle_count == 8


def test_capture_rejects_missing_event_and_removes_all_hooks():
    blocks = [ToyCrossAttentionBlock(dim=4).eval() for _ in range(2)]
    specs = (EventSpec(0, 3, 5), EventSpec(1, 4, 6))
    capture = stages.CrossAttentionStageCapture(blocks, specs, embedding_dim=4)
    inputs = (torch.randn(1, 2, 4), torch.randn(1, 2, 4))

    with capture:
        capture.begin_batch(1)
        blocks[0](*inputs)
        with pytest.raises(stages.FusionStageError, match="not every c_layers stage"):
            capture.finish_batch()

    assert capture.removed_handle_count == capture.registered_handle_count == 16
    for block in blocks:
        assert not block._forward_hooks
        assert not block._forward_pre_hooks


def test_model_ids_encode_expected_stream_insertion_pairs():
    assert stages.MODEL_INSERTION_IDS == {
        "df1": ((3, 4, 5), (5, 6, 7)),
        "df2": ((3, 4, 5), (9, 10, 11)),
        "df3": ((3, 4, 5), (7, 8, 9)),
        "rev": ((9, 10, 11), (3, 4, 5)),
    }

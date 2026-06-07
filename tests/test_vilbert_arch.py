"""
Architecture-contract tests for ViLBERT.

BertModel.from_pretrained and timm.create_model are both patched to return
real, randomly-initialised models (no internet download required).
This lets us test the structural logic of ViLBERT — cross-attention layer
routing, output shapes, intermediate representations — without pretrained
weights.

Why this matters for refactoring:
  - forward_coattention contains a stateful loop that interleaves unimodal
    transformer layers with CrossAttentionBlocks. Moving or restructuring
    this code can silently change which layers fire and in what order.
  - These tests lock down the observable contracts so such regressions
    surface immediately.
"""
import pytest
import torch
from unittest.mock import patch

import vilbert as vilbert_mod   # import once so patch targets the right module
from config import ViLBERTConfig
from attention import CrossAttentionBlock

ATOL = 2e-4   # tolerance for all golden-value assertions

B          = 2
TEXT_LEN   = 32
EMB_DIM    = 768
NUM_LAYERS = 12   # BERT and ViT both use 12 blocks


# ---------------------------------------------------------------------------
# Fixture: factory that creates a ViLBERT with mocked pretrained weights
# ---------------------------------------------------------------------------

@pytest.fixture
def make_vilbert(random_bert, random_vit):
    """
    Returns a callable: make_vilbert(t_ids, v_ids) → (model, config).

    Patches BertModel.from_pretrained and timm.create_model for the
    duration of ViLBERT.__init__ only.  The returned model holds references
    to random_bert and random_vit as its .bert and .vit attributes.
    """
    def _build(t_ids, v_ids):
        with patch('vilbert.BertModel') as MockBertCls, \
             patch('vilbert.timm') as mock_timm:
            MockBertCls.from_pretrained.return_value = random_bert
            mock_timm.create_model.return_value       = random_vit

            cfg   = ViLBERTConfig(
                text_cross_attention_layers=t_ids,
                vision_cross_attention_layers=v_ids,
            )
            model = vilbert_mod.ViLBERT(cfg)
            model.eval()
        return model, cfg
    return _build


# ---------------------------------------------------------------------------
# Cross-attention layer count
# ---------------------------------------------------------------------------

class TestCLayersCount:
    """len(model.c_layers) must equal len(cross_attention_ids)."""

    def test_single_cross_attention(self, make_vilbert):
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        assert len(model.c_layers) == 1

    def test_six_cross_attentions(self, make_vilbert):
        model, _ = make_vilbert(t_ids=[6, 7, 8, 9, 10, 11], v_ids=[0, 1, 2, 3, 4, 5])
        assert len(model.c_layers) == 6

    def test_late_fusion_single(self, make_vilbert):
        model, _ = make_vilbert(t_ids=[11], v_ids=[11])
        assert len(model.c_layers) == 1

    def test_early_fusion_single(self, make_vilbert):
        model, _ = make_vilbert(t_ids=[0], v_ids=[0])
        assert len(model.c_layers) == 1

    def test_c_layers_are_cross_attention_blocks(self, make_vilbert):
        """Refactoring must not swap CrossAttentionBlock for a different type."""
        model, _ = make_vilbert(t_ids=[6, 8], v_ids=[1, 3])
        for block in model.c_layers:
            assert isinstance(block, CrossAttentionBlock)


# ---------------------------------------------------------------------------
# Attention mask shape
# ---------------------------------------------------------------------------

class TestExtendedAttentionMask:

    def test_mask_broadcast_shape(self, make_vilbert):
        """get_extended_attention_mask must produce (B, 1, 1, T) for broadcasting."""
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        mask     = torch.ones(B, TEXT_LEN)
        extended = model.get_extended_attention_mask(mask)
        assert extended.shape == (B, 1, 1, TEXT_LEN)

    def test_none_mask_returns_none(self, make_vilbert):
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        assert model.get_extended_attention_mask(None) is None


# ---------------------------------------------------------------------------
# Forward pass output shapes
# ---------------------------------------------------------------------------

class TestForwardShapes:

    def test_cls_extraction_shapes(self, make_vilbert, fake_image, fake_text):
        """With extract_cls=True both streams return (B, 768)."""
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        ids, mask, types = fake_text
        with torch.no_grad():
            t_out, v_out = model.forward_coattention(
                text_input_ids=ids,
                text_attention_mask=mask,
                text_token_type_ids=types,
                image_pixel_values=fake_image,
                extract_cls=True,
            )
        assert t_out.shape == (B, EMB_DIM)
        assert v_out.shape == (B, EMB_DIM)

    def test_full_sequence_shapes(self, make_vilbert, fake_image, fake_text):
        """With extract_cls=False streams keep their full sequence lengths."""
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        ids, mask, types = fake_text
        with torch.no_grad():
            t_out, v_out = model.forward_coattention(
                text_input_ids=ids,
                text_attention_mask=mask,
                text_token_type_ids=types,
                image_pixel_values=fake_image,
                extract_cls=False,
            )
        assert t_out.shape == (B, TEXT_LEN, EMB_DIM)
        assert v_out.shape == (B, 197, EMB_DIM)    # 196 patches + CLS

    def test_output_shapes_late_fusion(self, make_vilbert, fake_image, fake_text):
        """Shape contract must hold regardless of where cross-attention fires."""
        model, _ = make_vilbert(t_ids=[11], v_ids=[11])
        ids, mask, types = fake_text
        with torch.no_grad():
            t_out, v_out = model.forward_coattention(
                text_input_ids=ids,
                text_attention_mask=mask,
                text_token_type_ids=types,
                image_pixel_values=fake_image,
                extract_cls=True,
            )
        assert t_out.shape == (B, EMB_DIM)
        assert v_out.shape == (B, EMB_DIM)

    def test_output_shapes_early_fusion(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[0], v_ids=[0])
        ids, mask, types = fake_text
        with torch.no_grad():
            t_out, v_out = model.forward_coattention(
                text_input_ids=ids,
                text_attention_mask=mask,
                text_token_type_ids=types,
                image_pixel_values=fake_image,
                extract_cls=True,
            )
        assert t_out.shape == (B, EMB_DIM)
        assert v_out.shape == (B, EMB_DIM)


# ---------------------------------------------------------------------------
# Intermediate representations
# ---------------------------------------------------------------------------

class TestIntermediateRepresentations:
    """
    save_intermediate_representations=True must return a list of exactly
    NUM_LAYERS entries regardless of fusion config, because both BERT and
    ViT have NUM_LAYERS blocks and every block is visited exactly once.
    """

    def _run(self, model, fake_image, fake_text):
        ids, mask, types = fake_text
        with torch.no_grad():
            t, v, intermediates = model.forward_coattention(
                text_input_ids=ids,
                text_attention_mask=mask,
                text_token_type_ids=types,
                image_pixel_values=fake_image,
                extract_cls=False,
                save_intermediate_representations=True,
            )
        return t, v, intermediates

    def test_length_is_num_layers(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        _, _, reps = self._run(model, fake_image, fake_text)
        assert len(reps) == NUM_LAYERS

    def test_length_late_fusion(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[11], v_ids=[11])
        _, _, reps = self._run(model, fake_image, fake_text)
        assert len(reps) == NUM_LAYERS

    def test_length_early_fusion(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[0], v_ids=[0])
        _, _, reps = self._run(model, fake_image, fake_text)
        assert len(reps) == NUM_LAYERS

    def test_length_six_layers(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[6, 7, 8, 9, 10, 11], v_ids=[0, 1, 2, 3, 4, 5])
        _, _, reps = self._run(model, fake_image, fake_text)
        assert len(reps) == NUM_LAYERS

    def test_required_keys(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        _, _, reps = self._run(model, fake_image, fake_text)
        for entry in reps:
            assert "text_embedding"   in entry
            assert "vision_embedding" in entry
            assert "layer"            in entry
            assert "is_cross_attention" in entry

    def test_embedding_shapes(self, make_vilbert, fake_image, fake_text):
        """Each intermediate must have the same shapes throughout the forward pass."""
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        ids, mask, types = fake_text
        _, _, reps = self._run(model, fake_image, fake_text)
        for entry in reps:
            assert entry["text_embedding"].shape   == (B, TEXT_LEN, EMB_DIM)
            assert entry["vision_embedding"].shape == (B, 197, EMB_DIM)

    def test_returns_three_values(self, make_vilbert, fake_image, fake_text):
        model, _ = make_vilbert(t_ids=[6], v_ids=[0])
        ids, mask, types = fake_text
        with torch.no_grad():
            result = model.forward_coattention(
                text_input_ids=ids,
                text_attention_mask=mask,
                text_token_type_ids=types,
                image_pixel_values=fake_image,
                extract_cls=False,
                save_intermediate_representations=True,
            )
        assert len(result) == 3


# ---------------------------------------------------------------------------
# Fusion configs produce numerically distinct representations
# ---------------------------------------------------------------------------

# ── Golden-value tests ──────────────────────────────────────────────────────
# Model weights: seeded_models fixture (torch.manual_seed(0) for each).
# c_layers weights: torch.manual_seed(42) before ViLBERT().
# Inputs: golden_inputs fixture (torch.manual_seed(5)).
#
# These tests catch regressions that shape tests miss:
#   - wrong layer firing order (early vs late fusion routing)
#   - missing or double-applied cross-attention
#   - wrong CLS extraction position
#   - extended-mask sign error (should ADD mask, not subtract)
# Regenerate with the "Compute golden values" script when architecture
# is intentionally changed.
# ───────────────────────────────────────────────────────────────────────────


@pytest.fixture
def make_seeded_vilbert(seeded_models):
    """Factory using deterministic BERT/ViT weights for golden-value tests."""
    bert, vit = seeded_models

    def _build(t_ids, v_ids, c_seed=42):
        with patch('vilbert.BertModel') as M, patch('vilbert.timm') as T:
            M.from_pretrained.return_value = bert
            T.create_model.return_value    = vit
            torch.manual_seed(c_seed)
            cfg   = ViLBERTConfig(text_cross_attention_layers=t_ids,
                                  vision_cross_attention_layers=v_ids)
            model = vilbert_mod.ViLBERT(cfg)
            model.eval()
        return model
    return _build


class TestExtendedAttentionMaskGolden:
    """Exact values for the extended attention mask computation."""

    def test_all_unmasked_tokens_give_zero(self, make_seeded_vilbert):
        model = make_seeded_vilbert([6], [0])
        mask  = torch.ones(1, 32)
        ext   = model.get_extended_attention_mask(mask)
        # (1 - 1) * -10000 = 0
        assert torch.allclose(ext, torch.zeros_like(ext), atol=1e-6)

    def test_padding_tokens_give_minus_10000(self, make_seeded_vilbert):
        model = make_seeded_vilbert([6], [0])
        # last 8 tokens are padding (mask=0)
        mask       = torch.ones(1, 32)
        mask[0, 24:] = 0.0
        ext = model.get_extended_attention_mask(mask)
        assert torch.allclose(ext[0, 0, 0, :24],  torch.zeros(24),        atol=1e-6)
        assert torch.allclose(ext[0, 0, 0, 24:],  torch.full((8,), -10000.0), atol=1e-1)


class TestViLBERTGoldenValues:
    """
    Numerically exact CLS-token and intermediate-representation values.

    Using seeded BERT/ViT weights + fixed inputs means any change to the
    forward-pass logic (wrong layer order, missing cross-attention, wrong
    CLS index) produces a different number and fails here.
    """

    def test_late_fusion_text_cls(self, make_seeded_vilbert, golden_inputs):
        ids, mask, types, img = golden_inputs
        model = make_seeded_vilbert([11], [11])
        with torch.no_grad():
            t, _ = model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img,
                extract_cls=True,
            )
        expected = torch.tensor([0.19418, 0.81153, 1.55617, 1.67231, -2.93705, 0.38494])
        assert torch.allclose(t[0, :6], expected, atol=ATOL)

    def test_late_fusion_vision_cls(self, make_seeded_vilbert, golden_inputs):
        ids, mask, types, img = golden_inputs
        model = make_seeded_vilbert([11], [11])
        with torch.no_grad():
            _, v = model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img,
                extract_cls=True,
            )
        expected = torch.tensor([0.54828, 0.04422, -0.55454, 1.14164, 1.12521, -0.00411])
        assert torch.allclose(v[0, :6], expected, atol=ATOL)

    def test_early_fusion_text_cls(self, make_seeded_vilbert, golden_inputs):
        ids, mask, types, img = golden_inputs
        model = make_seeded_vilbert([0], [0])
        with torch.no_grad():
            t, _ = model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img,
                extract_cls=True,
            )
        expected = torch.tensor([0.35422, 0.88123, 1.13347, 1.67818, -2.97764, 0.63460])
        assert torch.allclose(t[0, :6], expected, atol=ATOL)

    def test_early_fusion_vision_cls(self, make_seeded_vilbert, golden_inputs):
        ids, mask, types, img = golden_inputs
        model = make_seeded_vilbert([0], [0])
        with torch.no_grad():
            _, v = model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img,
                extract_cls=True,
            )
        expected = torch.tensor([-1.73506, 1.44945, -0.29672, -0.24029, -0.59057, 1.04478])
        assert torch.allclose(v[0, :6], expected, atol=ATOL)

    def test_early_vs_late_text_cls_differs(self, make_seeded_vilbert, golden_inputs):
        """
        Sanity-check: early and late CLS values must differ.
        If the routing loop is broken (e.g. cross-attention always fires last),
        both would return the same number and this fails.
        """
        ids, mask, types, img = golden_inputs
        early_model = make_seeded_vilbert([0],  [0])
        late_model  = make_seeded_vilbert([11], [11])
        with torch.no_grad():
            t_e, _ = early_model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img, extract_cls=True)
            t_l, _ = late_model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img, extract_cls=True)
        assert not torch.allclose(t_e[0, :6], t_l[0, :6], atol=ATOL)

    def test_layer0_intermediate_text(self, make_seeded_vilbert, golden_inputs):
        """Intermediate text embedding at layer index 0 (after first BERT layer)."""
        ids, mask, types, img = golden_inputs
        model = make_seeded_vilbert([11], [11])
        with torch.no_grad():
            _, _, reps = model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img,
                extract_cls=False, save_intermediate_representations=True,
            )
        expected = torch.tensor([0.22181, -0.06676, 0.80527, 1.02941])
        assert torch.allclose(reps[0]["text_embedding"][0, 0, :4], expected, atol=ATOL)

    def test_layer0_intermediate_vision(self, make_seeded_vilbert, golden_inputs):
        """Intermediate vision embedding at layer index 0 (after first ViT block)."""
        ids, mask, types, img = golden_inputs
        model = make_seeded_vilbert([11], [11])
        with torch.no_grad():
            _, _, reps = model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=img,
                extract_cls=False, save_intermediate_representations=True,
            )
        expected = torch.tensor([0.77284, -0.12582, 0.40827, -0.00136])
        assert torch.allclose(reps[0]["vision_embedding"][0, 0, :4], expected, atol=ATOL)


class TestFusionDistinctness:
    """
    Early and late fusion inject cross-attention at different points in the
    forward pass, so their output CLS representations must differ.
    Failing this test means the layer-routing loop was accidentally flattened.
    """

    def test_early_vs_late_text_output_differs(self, make_vilbert, fake_image, fake_text):
        ids, mask, types = fake_text

        early_model, _ = make_vilbert(t_ids=[0],  v_ids=[0])
        late_model,  _ = make_vilbert(t_ids=[11], v_ids=[11])

        with torch.no_grad():
            t_early, _ = early_model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=fake_image,
                extract_cls=True,
            )
            t_late, _ = late_model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=fake_image,
                extract_cls=True,
            )

        assert not torch.allclose(t_early, t_late), (
            "Early and late fusion produced identical text outputs — "
            "the layer-routing loop may have been broken."
        )

    def test_early_vs_late_vision_output_differs(self, make_vilbert, fake_image, fake_text):
        ids, mask, types = fake_text

        early_model, _ = make_vilbert(t_ids=[0],  v_ids=[0])
        late_model,  _ = make_vilbert(t_ids=[11], v_ids=[11])

        with torch.no_grad():
            _, v_early = early_model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=fake_image,
                extract_cls=True,
            )
            _, v_late = late_model.forward_coattention(
                text_input_ids=ids, text_attention_mask=mask,
                text_token_type_ids=types, image_pixel_values=fake_image,
                extract_cls=True,
            )

        assert not torch.allclose(v_early, v_late), (
            "Early and late fusion produced identical vision outputs — "
            "the layer-routing loop may have been broken."
        )

"""
Shape-contract tests for all custom attention blocks in attention.py.

These tests pin the public API of each block so that any refactoring
(renaming, moving, splitting) that silently breaks the interface fails
loudly here rather than at training time.
"""
import pytest
import torch

from attention import (
    Attention_Block,
    CrossAttentionBlock,
    DualAttention_Block,
    FeedForward_Block,
)

B = 2    # batch size
T = 20   # text sequence length
V = 197  # vision patches (196 patches + 1 CLS)
D = 768  # embedding dim (matches production EMBEDDING_DIM)

# Smaller dim used in golden tests so the hardcoded values stay readable
G_D = 32
G_H = 4


class TestAttentionBlock:
    """Self-attention block: (B, N, D) → (B, N, D)."""

    def test_output_shape(self):
        block = Attention_Block(dim=D, heads=8)
        x = torch.randn(B, T, D)
        assert block(x).shape == (B, T, D)

    def test_batch_size_one(self):
        block = Attention_Block(dim=D, heads=8)
        assert block(torch.randn(1, T, D)).shape == (1, T, D)

    def test_dtype_preserved(self):
        block = Attention_Block(dim=D, heads=8)
        x = torch.randn(B, T, D)
        assert block(x).dtype == x.dtype

    @pytest.mark.parametrize("seq_len", [1, 10, 50, 197])
    def test_various_sequence_lengths(self, seq_len):
        block = Attention_Block(dim=D, heads=8)
        assert block(torch.randn(B, seq_len, D)).shape == (B, seq_len, D)


class TestCrossAttentionBlock:
    """
    Cross-attention block: (text, vision) → (text', vision').

    The key contract: each stream's sequence length and embedding dim are
    preserved independently. text and vision do NOT have to be the same length.
    This is the interface that forward_coattention in vilbert.py relies on.
    """

    def test_returns_two_tensors(self):
        block = CrossAttentionBlock(dim=D, heads=8)
        result = block(torch.randn(B, T, D), torch.randn(B, V, D))
        assert isinstance(result, tuple) and len(result) == 2

    def test_output_shapes_asymmetric(self):
        """Text (T tokens) and vision (V patches) have different seq lengths."""
        block = CrossAttentionBlock(dim=D, heads=8)
        text   = torch.randn(B, T, D)
        vision = torch.randn(B, V, D)
        t_out, v_out = block(text, vision)
        assert t_out.shape == (B, T, D)
        assert v_out.shape == (B, V, D)

    def test_output_shapes_equal_len(self):
        block = CrossAttentionBlock(dim=D, heads=8)
        x = torch.randn(B, T, D)
        t_out, v_out = block(x, x)
        assert t_out.shape == x.shape
        assert v_out.shape == x.shape

    def test_batch_size_one(self):
        block = CrossAttentionBlock(dim=D, heads=8)
        t_out, v_out = block(torch.randn(1, T, D), torch.randn(1, V, D))
        assert t_out.shape == (1, T, D)
        assert v_out.shape == (1, V, D)

    def test_with_text_mask_does_not_change_shape(self):
        """Passing an attention mask must not alter output shapes."""
        block = CrossAttentionBlock(dim=D, heads=8)
        text   = torch.randn(B, T, D)
        vision = torch.randn(B, V, D)
        # extended_attention_mask format produced by ViLBERT.get_extended_attention_mask
        mask = torch.zeros(B, 1, 1, T)
        t_out, v_out = block(text, vision, text_mask=mask)
        assert t_out.shape == (B, T, D)
        assert v_out.shape == (B, V, D)

    def test_dtype_preserved(self):
        block = CrossAttentionBlock(dim=D, heads=8)
        text   = torch.randn(B, T, D)
        vision = torch.randn(B, V, D)
        t_out, v_out = block(text, vision)
        assert t_out.dtype == text.dtype
        assert v_out.dtype == vision.dtype


class TestFeedForwardBlock:
    """Point-wise FFN: (B, N, D) → (B, N, D)."""

    def test_output_shape(self):
        block = FeedForward_Block(dim=D, mlp_factor=4)
        x = torch.randn(B, T, D)
        assert block(x).shape == (B, T, D)

    def test_dtype_preserved(self):
        block = FeedForward_Block(dim=D, mlp_factor=4)
        x = torch.randn(B, T, D)
        assert block(x).dtype == x.dtype

    def test_small_mlp_factor(self):
        block = FeedForward_Block(dim=D, mlp_factor=1)
        assert block(torch.randn(B, T, D)).shape == (B, T, D)


class TestDualAttentionBlock:
    """
    Dual self-attention: each stream attends only to itself.
    Shapes must be preserved; streams do not need matching lengths.
    """

    def test_output_shapes(self):
        block = DualAttention_Block(dim=D, heads=8)
        text   = torch.randn(B, T, D)
        vision = torch.randn(B, V, D)
        t_out, v_out = block(text, vision)
        assert t_out.shape == (B, T, D)
        assert v_out.shape == (B, V, D)

    def test_returns_two_tensors(self):
        block = DualAttention_Block(dim=D, heads=8)
        result = block(torch.randn(B, T, D), torch.randn(B, V, D))
        assert isinstance(result, tuple) and len(result) == 2

    def test_dtype_preserved(self):
        block = DualAttention_Block(dim=D, heads=8)
        text   = torch.randn(B, T, D)
        vision = torch.randn(B, V, D)
        t_out, v_out = block(text, vision)
        assert t_out.dtype == text.dtype
        assert v_out.dtype == vision.dtype


# ── Golden-value tests ──────────────────────────────────────────────────────
# Weights: torch.manual_seed(N) immediately before block construction.
# Inputs:  torch.manual_seed(7) immediately before torch.randn.
# All blocks in eval() so dropout is identity.
#
# Purpose: catch numerical regressions that shape tests cannot — wrong
# attention scaling, missing residual, transposed Q/K, wrong mask sign, etc.
# Run the "Compute golden values" script at the top of this section to
# regenerate these constants if the architecture is intentionally changed.
# ───────────────────────────────────────────────────────────────────────────


class TestAttentionBlockGolden:
    """Attention_Block — self-attention on a single stream."""

    def _make(self):
        torch.manual_seed(0)
        block = Attention_Block(dim=G_D, heads=G_H, dropout=0.0)
        block.eval()
        torch.manual_seed(7)
        x = torch.randn(1, 4, G_D)
        return block, x

    def test_token0_values(self):
        block, x = self._make()
        with torch.no_grad():
            out = block(x)
        expected = torch.tensor([-0.30236, 0.18642, -0.13545, -0.12975])
        assert torch.allclose(out[0, 0, :4], expected, atol=1e-4)

    def test_token1_values(self):
        block, x = self._make()
        with torch.no_grad():
            out = block(x)
        expected = torch.tensor([-0.28738, 0.30593, 0.01685, -0.16378])
        assert torch.allclose(out[0, 1, :4], expected, atol=1e-4)

    def test_deterministic_given_seed(self):
        """Two forward passes with the same seed must produce identical output."""
        block1, x1 = self._make()
        block2, x2 = self._make()
        with torch.no_grad():
            assert torch.allclose(block1(x1), block2(x2))


class TestCrossAttentionBlockGolden:
    """CrossAttentionBlock — each stream attends to the other."""

    def _make(self):
        torch.manual_seed(1)
        block = CrossAttentionBlock(dim=G_D, heads=G_H, dropout=0.0)
        block.eval()
        torch.manual_seed(7)
        txt = torch.randn(1, 4, G_D)
        vis = torch.randn(1, 6, G_D)
        return block, txt, vis

    def test_text_output_token0(self):
        block, txt, vis = self._make()
        with torch.no_grad():
            t_out, _ = block(txt, vis)
        expected = torch.tensor([-0.55142, 0.62205, 1.63380, -1.46523])
        assert torch.allclose(t_out[0, 0, :4], expected, atol=1e-4)

    def test_vision_output_token0(self):
        block, txt, vis = self._make()
        with torch.no_grad():
            _, v_out = block(txt, vis)
        expected = torch.tensor([0.59902, 0.39875, 0.37056, -0.69083])
        assert torch.allclose(v_out[0, 0, :4], expected, atol=1e-4)

    def test_zero_text_mask_equals_no_mask(self):
        """
        A zero-valued text_mask should not change output.
        Catches sign errors in mask application:
          correct:   vision_qk += text_mask   (0 leaves scores unchanged)
          wrong:     vision_qk -= text_mask   (same for zeros, but breaks for padding)
        """
        block, txt, vis = self._make()
        zero_mask = torch.zeros(1, 1, 1, 4)
        with torch.no_grad():
            t_no, v_no = block(txt, vis)
            t_m,  v_m  = block(txt, vis, text_mask=zero_mask)
        assert torch.allclose(t_no, t_m,  atol=1e-6)
        assert torch.allclose(v_no, v_m, atol=1e-6)

    def test_partial_padding_mask_changes_vision_output(self):
        """
        A mask that hides ONLY SOME text tokens must change the vision output.
        (Masking ALL tokens is a no-op because softmax is shift-invariant;
        masking a subset genuinely changes the attention distribution.)
        Catches sign errors: if the mask is subtracted instead of added, the
        unmasked tokens get suppressed and the output will differ.
        """
        block, txt, vis = self._make()
        # First 2 tokens valid (0), last 2 tokens padded (-10000)
        partial_mask = torch.zeros(1, 1, 1, 4)
        partial_mask[0, 0, 0, 2:] = -10000.0
        with torch.no_grad():
            _, v_no_mask = block(txt, vis)
            _, v_masked  = block(txt, vis, text_mask=partial_mask)
        assert not torch.allclose(v_no_mask, v_masked, atol=1e-3)


class TestFeedForwardBlockGolden:
    """FeedForward_Block — point-wise FFN with GeLU."""

    def test_token0_values(self):
        torch.manual_seed(2)
        ff = FeedForward_Block(dim=G_D, mlp_factor=2, dropout=0.0)
        ff.eval()
        torch.manual_seed(7)
        x = torch.randn(1, 3, G_D)
        with torch.no_grad():
            out = ff(x)
        expected = torch.tensor([0.08722, 0.30550, 0.01083, 0.25022])
        assert torch.allclose(out[0, 0, :4], expected, atol=1e-4)

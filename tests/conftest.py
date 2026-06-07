import sys
import os
import pytest
import torch

from unittest.mock import patch
from transformers import BertConfig, BertModel
import timm
import vilbert as vilbert_mod
from config import ViLBERTConfig


# Fallback path setup in case pytest.ini pythonpath is not picked up
_src = os.path.join(os.path.dirname(__file__), '..', 'src')
if _src not in sys.path:
    sys.path.insert(0, _src)

BATCH_SIZE   = 2
TEXT_SEQ_LEN = 32   # short; ViLBERT accepts any text length

# Dim / heads used in golden-value tests (small to keep values readable)
GOLDEN_DIM   = 32
GOLDEN_HEADS = 4


@pytest.fixture(scope="session")
def random_bert():
    """Real BertModel with random weights — no download, session-scoped."""
    from transformers import BertConfig, BertModel
    cfg = BertConfig(
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=256,   # tiny FFN to keep memory low
        vocab_size=30522,
    )
    return BertModel(cfg)


@pytest.fixture(scope="session")
def random_vit():
    """Real timm ViT-Base/16 with random weights — no download, session-scoped."""
    import timm
    return timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=0,
        global_pool="",
    )


@pytest.fixture
def fake_image():
    """(B, 3, 224, 224) float32."""
    return torch.randn(BATCH_SIZE, 3, 224, 224)


@pytest.fixture
def fake_text():
    """Returns (input_ids, attention_mask, token_type_ids)."""
    ids   = torch.randint(0, 30522, (BATCH_SIZE, TEXT_SEQ_LEN))
    mask  = torch.ones(BATCH_SIZE, TEXT_SEQ_LEN)
    types = torch.zeros(BATCH_SIZE, TEXT_SEQ_LEN, dtype=torch.long)
    return ids, mask, types


# ---------------------------------------------------------------------------
# Fixtures for golden-value tests
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def seeded_models():
    """
    (bert, vit) with fully deterministic weights for golden-value tests.

    torch.manual_seed is reset before each model so the two seeds are
    independent of each other and of session fixture creation order.
    """
    from transformers import BertConfig, BertModel
    import timm

    torch.manual_seed(0)
    bert_cfg = BertConfig(
        hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
        intermediate_size=256, vocab_size=30522,
    )
    bert = BertModel(bert_cfg)
    bert.eval()

    torch.manual_seed(0)
    vit = timm.create_model(
        "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool=""
    )
    vit.eval()

    return bert, vit


@pytest.fixture
def golden_inputs():
    """Fixed (B=1) inputs for golden-value forward-pass tests."""
    torch.manual_seed(5)
    ids   = torch.randint(0, 30522, (1, TEXT_SEQ_LEN))
    mask  = torch.ones(1, TEXT_SEQ_LEN)
    types = torch.zeros(1, TEXT_SEQ_LEN, dtype=torch.long)
    img   = torch.randn(1, 3, 224, 224)
    return ids, mask, types, img


# ---------------------------------------------------------------------------
# Fresh-model factory for trainer / serialization tests
# ---------------------------------------------------------------------------

@pytest.fixture
def make_fresh_vilbert() -> tuple[vilbert_mod.ViLBERT, "ViLBERTConfig"]:
    """
    Returns make(t_ids, v_ids, **cfg_kwargs) -> (model, config).

    Unlike the session-scoped random_bert/random_vit fixtures, this builds
    FRESH BERT + ViT weights on every call. Trainer/serialization tests run
    backward passes and optimizer steps that mutate weights in place, so they
    must not share the session models that the golden-value tests rely on.
    """

    def _build(t_ids, v_ids, seed=0, **cfg_kwargs):
        torch.manual_seed(seed)
        bert = BertModel(BertConfig(
            hidden_size=768, num_hidden_layers=12, num_attention_heads=12,
            intermediate_size=256, vocab_size=30522,
        ))
        vit = timm.create_model(
            "vit_base_patch16_224", pretrained=False, num_classes=0, global_pool="",
        )
        with patch("vilbert.BertModel") as MockBertCls, \
             patch("vilbert.timm") as mock_timm:
            MockBertCls.from_pretrained.return_value = bert
            mock_timm.create_model.return_value       = vit
            cfg = ViLBERTConfig(
                text_cross_attention_layers=t_ids,
                vision_cross_attention_layers=v_ids,
                **cfg_kwargs,
            )
            model = vilbert_mod.ViLBERT(cfg)
        return model, cfg

    tupl = _build
    return tupl

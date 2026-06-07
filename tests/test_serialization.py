"""
Checkpoint round-trip: save_model -> load_model must preserve both the
weights and the cross-attention placement.

Why this is worth a dedicated fast test: run_finetune asserts that a loaded
checkpoint's cross_attention_layers match the requested config. If
save/load drops or reorders that, the failure currently only shows up
~20 minutes into the pipeline. This catches it in a couple of seconds.

ViLBERT.load_model reconstructs the model via cls(config), which calls
BertModel.from_pretrained + timm.create_model, so those are patched here too
(with throwaway backbones, since load_state_dict overwrites their weights).
"""
from unittest.mock import patch

import torch

import vilbert as vilbert_mod


def _load_patched(path):
    """ViLBERT.load_model with throwaway backbones patched in."""
    from transformers import BertConfig, BertModel
    import timm

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
        return vilbert_mod.ViLBERT.load_model(path, device="cpu")


def test_roundtrip_preserves_weights(make_fresh_vilbert, tmp_path):
    model, _ = make_fresh_vilbert(t_ids=[9, 10, 11], v_ids=[9, 10, 11])
    ckpt = tmp_path / "model.pt"
    model.save_model(str(ckpt))

    loaded = _load_patched(str(ckpt))

    orig = dict(model.state_dict())
    new = dict(loaded.state_dict())
    assert orig.keys() == new.keys()
    for k in orig:
        assert torch.equal(orig[k].cpu(), new[k].cpu()), f"weight mismatch at {k}"


def test_roundtrip_preserves_cross_attention_placement(make_fresh_vilbert, tmp_path):
    t_ids, v_ids = [9, 10, 11], [3, 4, 5]
    model, _ = make_fresh_vilbert(t_ids=t_ids, v_ids=v_ids)
    ckpt = tmp_path / "model.pt"
    model.save_model(str(ckpt))

    loaded = _load_patched(str(ckpt))

    assert loaded.config.text_cross_attention_layers == t_ids
    assert loaded.config.vision_cross_attention_layers == v_ids
    # the injected cross-attention blocks must be rebuilt 1:1
    assert len(loaded.c_layers) == len(model.c_layers)

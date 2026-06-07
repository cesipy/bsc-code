"""
Correctness tests (gaps 1-4) — these verify *what the training code computes*,
not just that it runs. All are deterministic and machine-independent (oracle
comparisons / exact graph facts), so they hold on CPU and GPU alike.

  gap 1  gradient flow      cross-attention blocks AND both backbones get grads
  gap 2  loss math          loss functions match an independent reference;
                            task heads emit the class count the loss expects
  gap 3  evaluate contract  no_grad (weights frozen), correct return arity,
                            model left in eval mode
  gap 4  determinism        same seed + same data -> identical epoch loss
"""
import math

import pytest
import torch
import torch.nn.functional as F

from trainer import HatefulMemesTrainer, MM_IMDB_Trainer, UPMCTrainer
from config import (
    MM_IMDB_NUM_GENRES, UPMC_NUM_CLASSES, POS_COUNT_HM, NEG_COUNT_HM,
)

B = 2
L = 32
T_IDS = V_IDS = [9, 10, 11]


# --------------------------------------------------------------------------
# synthetic inputs
# --------------------------------------------------------------------------

def _direct_text_img(seed=0):
    """Tensors shaped as the model itself expects (no dataset dim-1)."""
    torch.manual_seed(seed)
    text = dict(
        input_ids=torch.randint(0, 30522, (B, L)),
        attention_mask=torch.ones(B, L, dtype=torch.long),
        token_type_ids=torch.zeros(B, L, dtype=torch.long),
    )
    img = torch.randn(B, 3, 224, 224)
    return text, img


def _batch_text_img(seed=0):
    """Tensors shaped as a DataLoader emits them (extra dim-1 the trainer squeezes)."""
    torch.manual_seed(seed)
    text = {
        "input_ids":      torch.randint(0, 30522, (B, 1, L)),
        "attention_mask": torch.ones(B, 1, L, dtype=torch.long),
        "token_type_ids": torch.zeros(B, 1, L, dtype=torch.long),
    }
    img = {"pixel_values": torch.randn(B, 1, 3, 224, 224)}
    return text, img


def hm_batch(labels, seed=0):
    text, img = _batch_text_img(seed)
    return {"label": torch.tensor(labels), "text": text, "img": img}


def imdb_batch(seed=0):
    text, img = _batch_text_img(seed)
    return {"label": torch.randint(0, 2, (B, MM_IMDB_NUM_GENRES)).float(),
            "text": text, "img": img}


def upmc_batch(seed=0):
    text, img = _batch_text_img(seed)
    label = F.one_hot(torch.randint(0, UPMC_NUM_CLASSES, (B,)), UPMC_NUM_CLASSES).float()
    return {"label": label, "text": text, "img": img}


def _forward_cls(model, text, img):
    return model(
        text_input_ids=text["input_ids"],
        text_attention_mask=text["attention_mask"],
        text_token_type_ids=text["token_type_ids"],
        image_pixel_values=img,
        image_attention_mask=None,
    )


# ==========================================================================
# gap 1 — gradient flow
# ==========================================================================

def test_cross_attention_and_backbones_receive_gradients(make_fresh_vilbert):
    """
    A scalar built from the model output must propagate gradients into the
    injected cross-attention blocks and into both the BERT and ViT backbones.
    Guards against a refactor that detaches a stream or drops c_layers from
    the graph (which a finite-loss smoke test would not notice).
    """
    model, _ = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    model.train()
    model.zero_grad()

    text, img = _direct_text_img(seed=0)
    t_emb, v_emb = _forward_cls(model, text, img)
    (t_emb.pow(2).sum() + v_emb.pow(2).sum()).backward()

    # Known dead weight: CrossAttention.to_out_proj1/2 are built in __init__ but
    # commented out of forward (attention.py), so they never receive gradients.
    # The active cross-attention path must train fully; anything *outside* the
    # known-dead set that lacks a gradient is a real regression.
    DEAD = "to_out_proj"
    no_grad = [n for n, p in model.c_layers.named_parameters()
               if p.grad is None or p.grad.abs().sum().item() == 0]
    assert no_grad, "expected the known-dead to_out_proj params to show up"
    assert all(DEAD in n for n in no_grad), (
        f"cross-attention params unexpectedly received no gradient: "
        f"{[n for n in no_grad if DEAD not in n]}"
    )
    # and the active params really do train
    active = [p for n, p in model.c_layers.named_parameters() if DEAD not in n]
    assert all(p.grad is not None and p.grad.abs().sum().item() > 0 for p in active)

    # both backbones must train too
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in model.bert.parameters()), "BERT received no gradient"
    assert any(p.grad is not None and p.grad.abs().sum().item() > 0
               for p in model.vit.parameters()), "ViT received no gradient"


# ==========================================================================
# gap 2 — loss math
# ==========================================================================

def test_hm_loss_uses_class_imbalance_pos_weight(make_fresh_vilbert):
    """Hateful-memes training loss must carry the NEG/POS pos_weight."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    trainer = HatefulMemesTrainer(model=model, config=cfg)

    expected_pw = NEG_COUNT_HM / POS_COUNT_HM
    assert trainer.loss_fn.pos_weight.item() == pytest.approx(expected_pw, rel=1e-6)

    logits = torch.tensor([0.7, -0.4], device=trainer.device)
    labels = torch.tensor([1.0, 0.0], device=trainer.device)
    oracle = F.binary_cross_entropy_with_logits(
        logits, labels,
        pos_weight=torch.tensor(expected_pw, device=trainer.device),
    )
    assert trainer.loss_fn(logits, labels).item() == pytest.approx(oracle.item(), rel=1e-6)


def test_mmimdb_loss_is_plain_multilabel_bce(make_fresh_vilbert):
    """MM-IMDB is multi-label -> unweighted BCEWithLogits over all genres."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    trainer = MM_IMDB_Trainer(model=model, config=cfg)

    logits = torch.randn(B, MM_IMDB_NUM_GENRES, device=trainer.device)
    labels = torch.randint(0, 2, (B, MM_IMDB_NUM_GENRES), device=trainer.device).float()
    oracle = F.binary_cross_entropy_with_logits(logits, labels)
    assert trainer.loss_fn(logits, labels).item() == pytest.approx(oracle.item(), rel=1e-6)


def test_upmc_loss_is_crossentropy_over_class_indices(make_fresh_vilbert):
    """UPMC is single-label 101-class -> CrossEntropy on argmax(one-hot) targets."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    trainer = UPMCTrainer(model=model, config=cfg)

    logits = torch.randn(B, UPMC_NUM_CLASSES, device=trainer.device)
    onehot = F.one_hot(torch.randint(0, UPMC_NUM_CLASSES, (B,)), UPMC_NUM_CLASSES).float()
    targets = torch.argmax(onehot, dim=1).to(trainer.device)
    oracle = F.cross_entropy(logits, targets)
    assert trainer.loss_fn(logits, targets).item() == pytest.approx(oracle.item(), rel=1e-6)


@pytest.mark.parametrize("head_attr, out_dim", [
    ("fc",      1),
    ("fc_imdb", MM_IMDB_NUM_GENRES),
    ("fc_upmc", UPMC_NUM_CLASSES),
])
def test_head_output_dim_matches_task(make_fresh_vilbert, head_attr, out_dim):
    """Each task head must emit exactly the class count its loss expects."""
    model, _ = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    model.eval()
    text, img = _direct_text_img(seed=1)
    with torch.no_grad():
        t_emb, v_emb = _forward_cls(model, text, img)
        fused = torch.cat([t_emb, v_emb], dim=1)   # concat fusion
        out = getattr(model, head_attr)(fused)
    assert out.shape == (B, out_dim)


# ==========================================================================
# gap 3 — evaluate() contract
# ==========================================================================

def _params_snapshot(model):
    return {n: p.detach().clone() for n, p in model.named_parameters()}


def _assert_unchanged(model, snap):
    for n, p in model.named_parameters():
        assert torch.equal(p.detach().cpu(), snap[n].cpu()), f"{n} changed during evaluate"


def test_hm_evaluate_contract(make_fresh_vilbert):
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    trainer = HatefulMemesTrainer(model=model, config=cfg)
    # both label classes present so roc_auc_score is defined
    batches = [hm_batch([0, 1], seed=0), hm_batch([1, 0], seed=1)]

    snap = _params_snapshot(trainer.model)
    out = trainer.evaluate(batches)

    assert len(out) == 3                       # (loss, acc, auc)
    loss, acc, auc = out
    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    assert 0.0 <= auc <= 1.0
    _assert_unchanged(trainer.model, snap)     # no_grad => no weight drift
    assert trainer.model.training is False


@pytest.mark.parametrize("trainer_cls, batch_fn", [
    (MM_IMDB_Trainer, imdb_batch),
    (UPMCTrainer,     upmc_batch),
])
def test_downstream_evaluate_contract(make_fresh_vilbert, trainer_cls, batch_fn):
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS)
    trainer = trainer_cls(model=model, config=cfg)
    batches = [batch_fn(0), batch_fn(1)]

    snap = _params_snapshot(trainer.model)
    out = trainer.evaluate(batches)

    assert len(out) == 2                       # (loss, acc)
    loss, acc = out
    assert math.isfinite(loss)
    assert 0.0 <= acc <= 1.0
    _assert_unchanged(trainer.model, snap)
    assert trainer.model.training is False


# ==========================================================================
# gap 4 — determinism
# ==========================================================================

def test_train_epoch_is_deterministic(make_fresh_vilbert):
    """
    Same weight init + same batches + same RNG seed before the epoch must
    yield the same loss. This is the unit-level guarantee the integration
    test's golden values silently depend on.
    """
    def run():
        model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, seed=0)
        trainer = HatefulMemesTrainer(model=model, config=cfg, gradient_accumulation=1)
        batches = [hm_batch([0, 1], seed=10), hm_batch([1, 0], seed=11)]
        trainer.setup_scheduler(epochs=1, train_dataloader=batches, lr=1e-4)
        torch.manual_seed(1234)   # pin dropout RNG identically for both runs
        return trainer.train_epoch(batches)

    loss_a = run()
    loss_b = run()
    assert loss_a == pytest.approx(loss_b, rel=1e-5, abs=1e-7)

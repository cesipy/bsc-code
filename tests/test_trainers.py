"""
Trainer one-step smoke tests.

Each trainer is driven for one short epoch over a handful of *synthetic*
in-memory batches (no dataset files, no disk). A Python list of batch dicts
stands in for a DataLoader — it supports len() and iteration, which is all
train_epoch needs.

These prove that after a BaseTrainer refactor every trainer still:
  - runs a full forward / backward / optimizer step without crashing,
  - returns a finite scalar loss,
  - actually updates its task head weights.

They use fresh model weights (make_fresh_vilbert) so the in-place updates
never leak into the session-scoped golden-value models.
"""
import math
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from trainer import (
    HatefulMemesTrainer, MM_IMDB_Trainer, UPMCTrainer, PretrainingTrainer,
)
from config import (
    MM_IMDB_NUM_GENRES, UPMC_NUM_CLASSES, WARMUP_ITERATIONS, DECAY_ITERATIONS,
)
import task as tasklib

B = 2
L = 32
T_IDS = V_IDS = [9, 10, 11]


# --------------------------------------------------------------------------
# synthetic batch builders — shapes match what the real datasets emit
# (text/img values carry an extra dim-1 that the trainers squeeze off)
# --------------------------------------------------------------------------

def _text_img(seed=0):
    torch.manual_seed(seed)
    text = {
        "input_ids":      torch.randint(0, 30522, (B, 1, L)),
        "attention_mask": torch.ones(B, 1, L, dtype=torch.long),
        "token_type_ids": torch.zeros(B, 1, L, dtype=torch.long),
    }
    img = {"pixel_values": torch.randn(B, 1, 3, 224, 224)}
    return text, img


def hm_batch(seed=0):
    text, img = _text_img(seed)
    return {"label": torch.randint(0, 2, (B,)), "text": text, "img": img}


def imdb_batch(seed=0):
    text, img = _text_img(seed)
    label = torch.randint(0, 2, (B, MM_IMDB_NUM_GENRES)).float()
    return {"label": label, "text": text, "img": img}


def upmc_batch(seed=0):
    text, img = _text_img(seed)
    label = F.one_hot(torch.randint(0, UPMC_NUM_CLASSES, (B,)), UPMC_NUM_CLASSES).float()
    return {"label": label, "text": text, "img": img}


def ap_batch(seed=0):
    text, img = _text_img(seed)
    return {
        "task": torch.full((B,), tasklib.Task.ALIGNMENT_PREDICTION.value),
        "label": torch.randint(0, 2, (B,)),
        "text": text, "img": img,
    }


def mlm_batch(seed=0):
    text, img = _text_img(seed)
    return {
        "task": torch.full((B,), tasklib.Task.MASKED_LM.value),
        "label": torch.randint(0, 30522, (B, L)),   # per-token targets
        "text": text, "img": img,
    }


def _head_snapshot(module):
    # heads may be nn.Sequential or nn.Linear; grab the first learnable tensor
    return next(module.parameters()).detach().clone()


def _changed(before, after):
    return not torch.equal(before.cpu(), after.cpu())


# --------------------------------------------------------------------------
# downstream trainers — registry: (trainer class, head attr, batch builder)
# the grad-accum + scheduler bookkeeping is currently duplicated across all
# three; these tests pin its behaviour before it is pulled into BaseTrainer.
# --------------------------------------------------------------------------

DOWNSTREAM = [
    (HatefulMemesTrainer, "fc",      hm_batch),
    (MM_IMDB_Trainer,     "fc_imdb", imdb_batch),
    (UPMCTrainer,         "fc_upmc", upmc_batch),
]


# ---- smoke: runs end-to-end, finite loss, head weights move -----------------

@pytest.mark.parametrize("trainer_cls, head_attr, batch_fn", DOWNSTREAM)
def test_downstream_trainer_one_step(make_fresh_vilbert, trainer_cls, head_attr, batch_fn):
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=1e-4)
    trainer = trainer_cls(model=model, config=cfg, gradient_accumulation=1)

    batches = [batch_fn(0), batch_fn(1)]
    trainer.setup_scheduler(epochs=1, train_dataloader=batches, lr=1e-4)

    head = getattr(trainer.model, head_attr)
    before = _head_snapshot(head)

    loss = trainer.train_epoch(batches)

    assert isinstance(loss, float) and math.isfinite(loss)
    after = next(head.parameters())
    assert _changed(before, after), f"{head_attr} weights did not update"


# ---- behavioural guards: deterministic, machine-independent -----------------

@pytest.mark.parametrize("trainer_cls, _head, _batch", DOWNSTREAM)
def test_init_contract(make_fresh_vilbert, trainer_cls, _head, _batch):
    """__init__ wires AdamW over model params at config LR, on the right device."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=3.2e-5)
    trainer = trainer_cls(model=model, config=cfg, gradient_accumulation=1)

    assert isinstance(trainer.optimizer, torch.optim.AdamW)
    assert trainer.optimizer.param_groups[0]["lr"] == pytest.approx(3.2e-5)

    expected_device = "cuda" if torch.cuda.is_available() else "cpu"
    assert trainer.device == expected_device
    assert next(trainer.model.parameters()).device.type == expected_device

    # the optimizer must actually own the model's parameters
    opt_params = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
    model_params = {id(p) for p in trainer.model.parameters()}
    assert opt_params == model_params


@pytest.mark.parametrize("trainer_cls, _head, batch_fn", DOWNSTREAM)
def test_setup_scheduler_computes_steps(make_fresh_vilbert, trainer_cls, _head, batch_fn):
    """total_training_steps = epochs * len(loader) // grad_accum, split warmup/decay."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=1e-4)
    trainer = trainer_cls(model=model, config=cfg, gradient_accumulation=2)

    batches = [batch_fn(i) for i in range(4)]
    trainer.setup_scheduler(epochs=3, train_dataloader=batches, lr=1e-4)

    total = 3 * 4 // 2  # = 6
    assert trainer.scheduler.warmup_iterations == int(WARMUP_ITERATIONS * total)
    assert trainer.scheduler.decay_iterations == int(DECAY_ITERATIONS * total)
    assert trainer.scheduler.learning_rate == pytest.approx(1e-4)


@pytest.mark.parametrize("trainer_cls, _head, batch_fn", DOWNSTREAM)
@pytest.mark.parametrize("grad_accum, n_batches, expected_steps", [
    (1, 4, 4),   # step every batch
    (2, 4, 2),   # step every 2nd
    (4, 4, 1),   # single step at the end
    (3, 4, 2),   # 3rd batch, then trailing flush on the 4th
])
def test_grad_accum_step_cadence(
    make_fresh_vilbert, trainer_cls, _head, batch_fn, grad_accum, n_batches, expected_steps
):
    """optimizer.step fires every grad_accum batches (+ a trailing flush)."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=1e-4)
    trainer = trainer_cls(model=model, config=cfg, gradient_accumulation=grad_accum)

    batches = [batch_fn(i) for i in range(n_batches)]
    trainer.setup_scheduler(epochs=1, train_dataloader=batches, lr=1e-4)

    with patch.object(trainer.optimizer, "step", wraps=trainer.optimizer.step) as spy:
        trainer.train_epoch(batches)

    assert spy.call_count == expected_steps


@pytest.mark.parametrize("trainer_cls, _head, batch_fn", DOWNSTREAM)
def test_scheduler_lr_written_to_optimizer(make_fresh_vilbert, trainer_cls, _head, batch_fn):
    """the LR the scheduler emits must be pushed onto the optimizer param groups."""
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=1e-4)
    trainer = trainer_cls(model=model, config=cfg, gradient_accumulation=1)

    batches = [batch_fn(0), batch_fn(1)]
    trainer.setup_scheduler(epochs=1, train_dataloader=batches, lr=1e-4)

    SENTINEL = 0.0123456
    with patch.object(trainer.scheduler, "get_lr", return_value=SENTINEL):
        trainer.train_epoch(batches)

    for pg in trainer.optimizer.param_groups:
        assert pg["lr"] == pytest.approx(SENTINEL)


# --------------------------------------------------------------------------
# pretraining trainer — exercise the AP and MLM single-batch paths directly
# (the full train_epoch needs three length-matched loaders + the MIM path;
#  the per-task batch methods cover the core forward/backward contract)
# --------------------------------------------------------------------------

def test_pretrain_alignment_prediction_step(make_fresh_vilbert):
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=1e-4)
    trainer = PretrainingTrainer(
        model=model, config=cfg, use_contrastive_loss=False, gradient_accumulation=1,
    )
    before = _head_snapshot(trainer.model.alignment_fc)

    loss = trainer.train_alignment_prediction_batch(ap_batch(0), flag_optimizer=True)

    assert math.isfinite(loss)
    assert _changed(before, next(trainer.model.alignment_fc.parameters()))


def test_pretrain_mlm_step(make_fresh_vilbert):
    model, cfg = make_fresh_vilbert(t_ids=T_IDS, v_ids=V_IDS, learning_rate=1e-4)
    trainer = PretrainingTrainer(
        model=model, config=cfg, use_contrastive_loss=False, gradient_accumulation=1,
    )
    loss = trainer.train_mlm_batch(mlm_batch(0), flag_optimizer=True)

    assert math.isfinite(loss)

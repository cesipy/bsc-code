"""
Fast, pure-logic tests — no model, no GPU, no datasets. Run in milliseconds.

Two contracts that have already bitten us once and are easy to break in a
refactor:

  1. ExperimentConfig -> ViLBERTConfig field mapping (create_config).
     A field silently not being copied (e.g. pretrain_batch_size) only
     surfaced after a 16-minute integration run. This locks every field down.

  2. The LR scheduler's warmup -> cosine-decay -> floor shape.
"""
import math

import pytest

import experiment_tracker
import utils
from config import MIN_LR_FRACTION


class TestCreateConfigMapping:
    """ExperimentTracker.create_config must copy every tunable field across."""

    @pytest.fixture
    def mapped(self):
        exp = experiment_tracker.ExperimentConfig(
            t_biattention_ids=[9, 10, 11],
            v_biattention_ids=[3, 4, 5],
            use_contrastive_loss=True,
            epochs=7,
            batch_size=16,
            pretrain_batch_size=24,
            gradient_accumulation=11,
            learning_rate=2.5e-5,
            seed=4242,
            train_test_ratio=0.7,
            dropout=0.15,
        )
        cfg = experiment_tracker.ExperimentTracker().create_config(exp)
        return exp, cfg

    def test_cross_attention_layers(self, mapped):
        exp, cfg = mapped
        assert cfg.text_cross_attention_layers == exp.t_biattention_ids
        assert cfg.vision_cross_attention_layers == exp.v_biattention_ids

    def test_batch_sizes_decoupled(self, mapped):
        exp, cfg = mapped
        # the bug we hit: pretrain_batch_size must not fall back to a global
        assert cfg.batch_size == exp.batch_size == 16
        assert cfg.pretrain_batch_size == exp.pretrain_batch_size == 24

    def test_scalar_fields(self, mapped):
        exp, cfg = mapped
        assert cfg.epochs == exp.epochs
        assert cfg.gradient_accumulation == exp.gradient_accumulation
        assert cfg.learning_rate == exp.learning_rate
        assert cfg.seed == exp.seed
        assert cfg.train_test_ratio == exp.train_test_ratio
        assert cfg.dropout_prob == exp.dropout
        assert cfg.use_contrastive_loss == exp.use_contrastive_loss


class TestScheduler:
    """utils.Scheduler: warmup ramp -> cosine decay -> min_lr floor."""

    LR = 1e-4
    WARMUP = 10
    DECAY = 100

    @pytest.fixture
    def lrs(self):
        sched = utils.Scheduler(
            warmup_iterations=self.WARMUP,
            decay_iterations=self.DECAY,
            learning_rate=self.LR,
            min_lr_fraction=MIN_LR_FRACTION,
        )
        # collect the schedule over the full run plus a tail past decay
        return [sched.get_lr() for _ in range(self.DECAY + 20)]

    def test_stays_within_bounds(self, lrs):
        # during warmup the LR ramps up from ~0, so it may be below the floor;
        # the only hard bounds are non-negative and never above the base LR
        for lr in lrs:
            assert 0.0 <= lr <= self.LR + 1e-9

    def test_warmup_is_increasing(self, lrs):
        warmup = lrs[: self.WARMUP - 1]
        assert all(b >= a for a, b in zip(warmup, warmup[1:]))

    def test_decay_is_decreasing(self, lrs):
        # between end of warmup and start of floor the cosine decay falls
        decay = lrs[self.WARMUP : self.DECAY]
        assert all(b <= a + 1e-12 for a, b in zip(decay, decay[1:]))

    def test_floor_after_decay(self, lrs):
        min_lr = self.LR * MIN_LR_FRACTION
        for lr in lrs[self.DECAY + 1 :]:
            assert lr == pytest.approx(min_lr)

    def test_peak_near_base_lr(self, lrs):
        # the schedule should reach close to the base LR at the warmup peak
        assert max(lrs) == pytest.approx(self.LR, rel=0.2)

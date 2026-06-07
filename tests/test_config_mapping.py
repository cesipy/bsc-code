"""
Fast, pure-logic tests — no model, no GPU, no datasets. Run in milliseconds.

Two contracts that have already bitten us once and are easy to break in a
refactor:

  1. ViLBERTConfig field defaults and construction.
     A field silently falling back to a wrong global (e.g. pretrain_batch_size
     picking up batch_size) only surfaced after a 16-minute integration run.
     This locks every field down.

  2. The LR scheduler's warmup -> cosine-decay -> floor shape.
"""
import math

import pytest

from config import (
    ViLBERTConfig,
    BATCH_SIZE_PRETRAIN, BATCH_SIZE_DOWNSTREAM,
    GRADIENT_ACCUMULATION, GRADIENT_ACCUMULATION_DOWNSTREAM,
    DROPOUT_PROB, SEED, TRAIN_TEST_RATIO, PRETRAIN_LEARNING_RATE,
    MIN_LR_FRACTION, NUM_WORKERS, PREFETCH, PERSISTENT_WORKERS, PIN_MEMORY,
)
import utils


class TestViLBERTConfigConstruction:
    """ViLBERTConfig must accept all tunable fields and store them correctly."""

    @pytest.fixture
    def cfg(self):
        return ViLBERTConfig(
            text_cross_attention_layers=[9, 10, 11],
            vision_cross_attention_layers=[3, 4, 5],
            use_contrastive_loss=True,
            epochs=7,
            batch_size=16,
            pretrain_batch_size=24,
            gradient_accumulation=11,
            learning_rate=2.5e-5,
            seed=4242,
            train_test_ratio=0.7,
            dropout_prob=0.15,
        )

    def test_cross_attention_layers(self, cfg):
        assert cfg.text_cross_attention_layers == [9, 10, 11]
        assert cfg.vision_cross_attention_layers == [3, 4, 5]

    def test_batch_sizes_decoupled(self, cfg):
        # the bug we hit: pretrain_batch_size must not fall back to batch_size
        assert cfg.batch_size == 16
        assert cfg.pretrain_batch_size == 24

    def test_scalar_fields(self, cfg):
        assert cfg.epochs == 7
        assert cfg.gradient_accumulation == 11
        assert cfg.learning_rate == 2.5e-5
        assert cfg.seed == 4242
        assert cfg.train_test_ratio == 0.7
        assert cfg.dropout_prob == 0.15
        assert cfg.use_contrastive_loss is True

    def test_depth_derived(self, cfg):
        # depth = 12 base + len(text_cross_attention_layers)
        assert cfg.depth == 12 + 3

    def test_defaults(self):
        cfg = ViLBERTConfig(
            text_cross_attention_layers=[0, 1],
            vision_cross_attention_layers=[0, 1],
        )
        assert cfg.pretrain_batch_size == BATCH_SIZE_PRETRAIN
        assert cfg.batch_size == BATCH_SIZE_DOWNSTREAM
        assert cfg.gradient_accumulation == GRADIENT_ACCUMULATION
        assert cfg.dropout_prob == DROPOUT_PROB
        assert cfg.seed == SEED
        assert cfg.train_test_ratio == TRAIN_TEST_RATIO
        assert cfg.num_workers == NUM_WORKERS
        assert cfg.prefetch == PREFETCH
        assert cfg.persistent_workers == PERSISTENT_WORKERS
        assert cfg.pin_memory == PIN_MEMORY

    def test_round_trip(self, cfg):
        d = cfg.to_dict()
        cfg2 = ViLBERTConfig.from_dict(d)
        assert cfg2.text_cross_attention_layers == cfg.text_cross_attention_layers
        assert cfg2.vision_cross_attention_layers == cfg.vision_cross_attention_layers
        assert cfg2.batch_size == cfg.batch_size
        assert cfg2.pretrain_batch_size == cfg.pretrain_batch_size
        assert cfg2.gradient_accumulation == cfg.gradient_accumulation
        assert cfg2.learning_rate == cfg.learning_rate
        assert cfg2.seed == cfg.seed
        assert cfg2.depth == cfg.depth


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

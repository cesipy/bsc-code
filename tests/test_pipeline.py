# includes integration test for the pipeline

import math
import os
import socket

import pytest

import experiment_tracker
from config import (
    OPTIMIZE_CKA, OPTIMIZE_MUTUAL_KNN,
    _GOOD_GPUS, _GPU_PREFIX,
)

SEED = 1567
T_BIATTN_IDS = [9,10,11]
V_BIATTN_IDS = T_BIATTN_IDS
NUM_SAMPLES  = 2000
TASK         = "hateful_memes"

assert OPTIMIZE_MUTUAL_KNN == False, "set OPTIMIZE_MUTUAL_KNN to False for testing"
assert OPTIMIZE_CKA == False, "set OPTIMIZE_CKA to False for testing"
assert int(socket.gethostname().replace(_GPU_PREFIX, "")) in _GOOD_GPUS, "run tests on the 24gb gpus, otherwise the integration tests are not working due to batch sizes"




@pytest.mark.integration
def test_pretrain_integration():
    """
    pretraining["training"][1] = {
        "train_loss_ap":  0.7904494988865086,
        "train_loss_mlm": 9.931544618237586,
        "train_loss_mim": 8.233610275955428,
        "val_loss_ap":    0.7257219024679877,
        "val_acc_ap":     0.505452821242295,
        "val_loss_mlm":   9.361643956466155,
        "val_loss_mim":   7.29743764210831,
    }

    pretraining["alignment"][0] = {
        0:  {"cka": 0.01595042273402214, "mknn": 0.260009765625,  "svcca": 0.2408614569732411, "procrustes": 1.7769853},
        9:  {"cka": 0.5425311923027039,  "mknn": 0.36279296875,   "svcca": 0.4229512303486683, "procrustes": 5.0262456},
        11: {"cka": 0.6358933448791504,  "mknn": 0.38330078125,   "svcca": 0.4797670724229811, "procrustes": 3.5957716},
    }
    """

    t = experiment_tracker.ExperimentTracker()
    pretrain_config = experiment_tracker.ExperimentConfig(
        t_biattention_ids=T_BIATTN_IDS,
        v_biattention_ids=V_BIATTN_IDS,
        use_contrastive_loss=False,
        epochs=1,
        learning_rate=1e-4,
        seed=SEED,
        pretrain_batch_size=24,
        gradient_accumulation=22,
    )

    results_pretrain = t.run_pretrain(
        experiment_config=pretrain_config,
        num_samples=2_000,
        run_alignment_analysis=True,
        alignment_analysis_size=128,
    )


    pretrain = results_pretrain["pretraining"]

    # epoch 1
    epoch_1 = pretrain["training"][1]
    assert epoch_1 == pytest.approx({
        "train_loss_ap":  0.7904494988865086,
        "train_loss_mlm": 9.931544618237586,
        "train_loss_mim": 8.233610275955428,
        "val_loss_ap":    0.7257219024679877,
        "val_acc_ap":     0.505452821242295,
        "val_loss_mlm":   9.361643956466155,
        "val_loss_mim":   7.29743764210831,
    }, rel=1e-3)

    def _check_alignment(layer: dict, cka, mknn, svcca, procrustes):
        assert layer["cka"]        == pytest.approx(cka,        rel=1e-3)
        assert layer["mknn"]       == pytest.approx(mknn,       rel=1e-3)
        assert layer["svcca"]      == pytest.approx(svcca,      rel=1e-3)
        assert layer["procrustes"] == pytest.approx(procrustes, rel=1e-3)

    # ---- alignment metrics: epoch 0 (untrained) ----
    # Model weights are init-seed-dependent, not batch-size-dependent, so exact
    # values are stable across local and remote machines.
    # spot-check layers 0 (unimodal), 9 (first cross-attn), 11 (last)
    a0 = pretrain["alignment"][0]
    _check_alignment(a0[0],  cka=0.01595042273402214, mknn=0.260009765625,  svcca=0.2408614569732411, procrustes=1.7769853)
    _check_alignment(a0[9],  cka=0.5425311923027039,  mknn=0.36279296875,   svcca=0.4229512303486683, procrustes=5.0262456)
    _check_alignment(a0[11], cka=0.6358933448791504,  mknn=0.38330078125,   svcca=0.4797670724229811, procrustes=3.5957716)

    pretrain = results_pretrain["pretraining"]
    assert "training" in pretrain
    a1 = pretrain["alignment"][1]
    assert set(a1.keys()) == set(range(12))
    for layer_idx in (0, 9, 11):
        for metric in ("cka", "mknn", "svcca", "procrustes"):
            assert math.isfinite(float(a1[layer_idx][metric])), f"a1[{layer_idx}][{metric}] not finite"
        assert 0.0 <= a1[layer_idx]["cka"]  <= 1.0
        assert 0.0 <= a1[layer_idx]["mknn"] <= 1.0
        assert a1[layer_idx]["procrustes"]   > 0.0

    cfg = results_pretrain["config"]
    assert cfg["t_biattention_ids"] == T_BIATTN_IDS
    assert cfg["v_biattention_ids"] == V_BIATTN_IDS
    assert cfg["epochs"] == 1
    assert cfg["learning_rate"] == pytest.approx(1e-4)
    assert cfg["seed"] == SEED
    assert cfg["train_test_ratio"] == pytest.approx(0.8)
    assert cfg["use_contrastive_loss"] is False
    assert cfg["dropout"] == pytest.approx(0.08)

    model_path = results_pretrain["model_path"]
    assert isinstance(model_path, str) and model_path.endswith(".pt")
    assert os.path.exists(model_path), f"checkpoint not found: {model_path}"



# @pytest.mark.integration
# not needed, is also integrated in the full pipeline test
# def test_finetune_ingegration():
#     t = experiment_tracker.ExperimentTracker()

#     e_conf = experiment_tracker.ExperimentConfig(
#         t_biattention_ids=T_BIATTN_IDS,
#         v_biattention_ids=V_BIATTN_IDS,
#         epochs=15,
#         learning_rate=3.2e-5,
#         seed=SEED,
#         use_contrastive_loss=False,
#     )

#     restuls_finetune = t.run_finetune(
#         experiment_config=e_conf,
#         pretrained_model_path=pretrained_path, # not sure where to get that from
#         tasks=[TASK]
#     )


@pytest.mark.integration
def test_full_pipeline():
    """
    hateful_memes["training"][1] = {
        "train_loss": 0.9144762974075029,
        "val_loss":   0.6969664286483418,
        "val_acc":    0.4634615384615385,
        "val_auc":    0.5451030101369068,
    }

    hateful_memes["final_test"] = {
        "loss": 0.695854199886322,
        "acc":  0.46266666666666667,
        "auc":  0.5833683101173021,
    }

    hateful_memes["alignment"][0] = {
        0:  {"cka": 0.00501922657713294,  "mknn": 0.041839599609375, "svcca": 0.11175385683774519, "procrustes": 4.4642677},
        9:  {"cka": 0.34816133975982666,  "mknn": 0.1500244140625,   "svcca": 0.3682464814481697,  "procrustes": 17.234404},
        11: {"cka": 0.6311681866645813,   "mknn": 0.341827392578125, "svcca": 0.5717147970916138,  "procrustes": 8.309519},
    }
    """
    # tests pretrain + finetune
    t = experiment_tracker.ExperimentTracker()
    pretrain_config = experiment_tracker.ExperimentConfig(
        t_biattention_ids=T_BIATTN_IDS,
        v_biattention_ids=V_BIATTN_IDS,
        use_contrastive_loss=False,
        epochs=1,
        learning_rate=1e-4,
        seed=SEED,
        pretrain_batch_size=24,
        gradient_accumulation=22,
    )
    finetune_config = experiment_tracker.ExperimentConfig(
        t_biattention_ids=T_BIATTN_IDS,
        v_biattention_ids=V_BIATTN_IDS,
        epochs=1,
        learning_rate=3.2e-5,
        seed=SEED,
        use_contrastive_loss=False,
    )

    results_pretrain = t.run_pretrain(
        experiment_config=pretrain_config,
        num_samples=2_000,
    )

    results_finetune = t.run_finetune(
        experiment_config=finetune_config,
        run_alignment_analysis=True,
        pretrained_model_path=results_pretrain["model_path"],
        tasks=[TASK],
    )

    hm = results_finetune["hateful_memes"]

    epoch_1 = hm["training"][1]
    assert epoch_1 == pytest.approx({
        "train_loss": 0.9144762974075029,
        "val_loss":   0.6969664286483418,
        "val_acc":    0.4634615384615385,
        "val_auc":    0.5451030101369068,
    }, rel=1e-3)

    ft = hm["final_test"]
    assert ft == pytest.approx({
        "loss": 0.695854199886322,
        "acc":  0.46266666666666667,
        "auc":  0.5833683101173021,
    }, rel=1e-3)

    # ---- alignment: spot-check layers 0 (unimodal), 9 (first cross-attn), 11 (last) ----
    def _check_alignment(layer, cka, mknn, svcca, procrustes):
        assert layer["cka"]        == pytest.approx(cka,        rel=1e-3)
        assert layer["mknn"]       == pytest.approx(mknn,       rel=1e-3)
        assert layer["svcca"]      == pytest.approx(svcca,      rel=1e-3)
        assert layer["procrustes"] == pytest.approx(procrustes, rel=1e-3)

    #epoch 0 = alignment on the pretrained model before any finetuning.
    # Weights are fixed (loaded from checkpoint) so values are deterministic.
    a0 = hm["alignment"][0]
    _check_alignment(a0[0],  cka=0.00501922657713294,  mknn=0.041839599609375, svcca=0.11175385683774519, procrustes=4.4642677)
    _check_alignment(a0[9],  cka=0.34816133975982666,  mknn=0.1500244140625,   svcca=0.3682464814481697,  procrustes=17.234404)
    _check_alignment(a0[11], cka=0.6311681866645813,   mknn=0.341827392578125, svcca=0.5717147970916138,  procrustes=8.309519)

    #epoch 1
    a1 = hm["alignment"][1]
    assert set(a1.keys()) == set(range(12))
    for layer_idx in (0, 9, 11):
        for metric in ("cka", "mknn", "svcca", "procrustes"):
            assert math.isfinite(float(a1[layer_idx][metric])), f"a1[{layer_idx}][{metric}] not finite"
        assert 0.0 <= a1[layer_idx]["cka"]  <= 1.0
        assert 0.0 <= a1[layer_idx]["mknn"] <= 1.0
        assert a1[layer_idx]["procrustes"]   > 0.0

    # config and trivial things
    cfg = results_finetune["config"]
    assert cfg["t_biattention_ids"] == T_BIATTN_IDS
    assert cfg["v_biattention_ids"] == V_BIATTN_IDS
    assert cfg["epochs"] == 1
    assert cfg["learning_rate"] == pytest.approx(3.2e-5)
    assert cfg["seed"] == SEED
    assert cfg["use_contrastive_loss"] is False
    ft_path = hm["model_path"]
    assert isinstance(ft_path, str) and ft_path.endswith(".pt")
    assert os.path.exists(ft_path), f"finetuned checkpoint not found: {ft_path}"

import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr


paths = [
    "res/experiments/20251006-205740_experiment_coattn_7-8-9.json"
    # "res/experiments/20251005-195330_experiment_coattn_2-3-4.json",     # finetune hms
    # "res/experiments/20251005-205228_experiment_coattn_4-8-9.json",
    # "res/experiments/20251005-214318_experiment_coattn_0-3-4-5-6-8.json",
    # "res/experiments/20251005-220421_experiment_coattn_1-3-5-6-7.json",
    # "res/experiments/20251005-222404_experiment_coattn_0-1-3-8-11.json",
    # "res/experiments/20251006-125134_experiment_coattn_1-2-4-7-8-10.json",
    # "res/experiments/20251006-132458_experiment_coattn_1-3-5-7-8-9-10.json",
    # below all are imported, Im not sure what task and if this was finetune or not
    # "res/experiments-import/20251003-162327_experiment_coattn_5-6-7.json",
    # "res/experiments-import/20251004-173707_experiment_coattn_5-6-7.json",
    # "res/experiments-import/20251004-191821_experiment_coattn_5-6-7.json",
    # "res/experiments-import/20251004-232753_experiment_coattn_2-3-4.json",
    # "res/experiments-import/20251005-051920_experiment_coattn_2-3-4.json",
    # "res/experiments-import/20251005-063917_experiment_coattn_5-6-7.json",
    # "res/experiments-import/20251005-124607_experiment_coattn_5-6-7.json",
    # "res/experiments-import/20251005-140732_experiment_coattn_9-10-11.json",
]


def was_task_analysed(task, content):
    """check whether task contains alignment data"""
    return content[task]["alignment"]["0"] != {}


def compute_correlations(label, dps, target_key):
    """computes Pearson and Spearman correlation for each metric"""
    target = np.array([dp[target_key] for dp in dps])
    metrics = ["cosine", "cka", "mknn", "rank", "procrustes", "svcca"]

    val_loss = np.array([dp["val_loss"] for dp in dps])
    val_accs = np.array([dp["val_acc"] for dp in dps])
    r, p = pearsonr(val_loss, val_accs)
    print(f"{label} - Pearson between val_loss and val_acc: r={r:+.3f}, p={p:.3f}")

    for metric in metrics:
        metric_vals = np.array([dp[metric] for dp in dps])

        # Pearson
        r, p = pearsonr(metric_vals, target)
        print(f"{label} - Pearson between {metric} and {target_key}: r={r:+.3f}, p={p:.3f}")

    print("")  # spacing

    for metric in metrics:
        metric_vals = np.array([dp[metric] for dp in dps])

        rho, p_s = spearmanr(metric_vals, target)
        print(f"{label} - Spearman between {metric} and {target_key}: r={rho:+.3f}, p={p_s:.3f}")

    print("")


def analyse_all_dps(dps):
    """aggregated correlation across all data points"""
    print("-" * 20 + " all dps together")
    print(f"{len(dps)} datapoints")
    compute_correlations("all tasks", dps, "val_acc")

    val_loss_inv = [{"val_loss": -dp["val_loss"], **dp} for dp in dps]
    compute_correlations("all tasks", val_loss_inv, "val_loss")


def analyse_per_task(dps):
    """analyse correlations per task"""
    print("\n" + "-" * 20 + "separated by task")
    for task in ["hateful_memes", "mm_imdb"]:
        task_dps = [dp for dp in dps if dp["task"] == task]
        if not task_dps:
            continue
        print(f"Task {task}: {len(task_dps)} datapoints")
        compute_correlations(f"Task {task}", task_dps, "val_acc")


def analyse_per_run(dps):
    """analyse correlations per individual run"""
    print("\n" + "-" * 25 + "seperted by run")
    for run_id in sorted(set(dp["id"] for dp in dps)):
        run_dps = [dp for dp in dps if dp["id"] == run_id]
        if not run_dps:
            continue
        task = run_dps[0]["task"]
        print(f"Run {run_id} - Task {task}: {len(run_dps)} datapoints")
        compute_correlations(f"Run {run_id} - Task {task}", run_dps, "val_acc")

def analyse_best_epochs(dps):
    """Find and analyze best epochs per run"""
    best_dps = []
    for run_id in sorted(set(dp['id'] for dp in dps)):
        run_dps = [dp for dp in dps if dp['id'] == run_id]
        best_dp = max(run_dps, key=lambda x: x['val_acc'])
        best_dps.append(best_dp)

    print("Best Epochs Analysis:")
    compute_correlations("Best Epochs", best_dps, "val_acc")

# --- main ---
c = 0
dps = []

for path in paths:
    with open(path, "r") as f:
        content = json.load(f)

    for task in ["hateful_memes", "mm_imdb"]:
        res = content[task]

        if was_task_analysed(task, content):
            # get validation acc and alignment values
            epochs = len(res["alignment"]) -1
            print(f"Task {task} was analysed for {epochs} epochs")

            for epoch in range(1, epochs + 1):
                align = res["alignment"][f"{epoch}"]["11"]

                dp = {
                    "task": task,
                    "epoch": epoch,
                    "val_acc": res["training"][f"{epoch}"]["val_acc"],
                    "val_loss": res["training"][f"{epoch}"]["val_loss"],
                    "cosine": align["cosine"],
                    "cka": align["cka"],
                    "mknn": align["mknn_full_epoch"],
                    "rank": align["rank_full_epoch"],
                    "procrustes": align["procrustes_full_epoch"],
                    "svcca": align["svcca"],
                    "id": c,
                }
                dps.append(dp)

        c += 1





# perform analyses
if dps:
    analyse_all_dps(dps)
    analyse_best_epochs(dps)
    # analyse_per_task(dps)
    # analyse_per_run(dps)

import os
import json
import numpy as np
from scipy.stats import pearsonr, spearmanr


paths = [
    "res/experiments/20251005-195330_experiment_coattn_2-3-4.json",     # finetune hms
    "res/experiments/20251005-205228_experiment_coattn_4-8-9.json",
    # below all are imported, Im not sure what task and if this was finetune or not
    "res/experiments-import/20251003-162327_experiment_coattn_5-6-7.json",
    "res/experiments-import/20251004-173707_experiment_coattn_5-6-7.json",
    "res/experiments-import/20251004-191821_experiment_coattn_5-6-7.json",
    "res/experiments-import/20251004-232753_experiment_coattn_2-3-4.json",
    "res/experiments-import/20251005-051920_experiment_coattn_2-3-4.json",
    "res/experiments-import/20251005-063917_experiment_coattn_5-6-7.json",
    "res/experiments-import/20251005-124607_experiment_coattn_5-6-7.json",
    "res/experiments-import/20251005-140732_experiment_coattn_9-10-11.json",

]

def was_task_analysed(task, content):
    if content[task]["alignment"]["0"] != {}:
        return True
    return False



c = 0
dps = []
for path in paths:
    content = None
    with open(path, "r") as f:
        content = json.load(f)

    for task in ["hateful_memes", "mm_imdb"]:

        res = content[task]


        if was_task_analysed(task, content):
            # get validation acc and alignment values
            epochs = len(res["alignment"]) -1
            print(f"Task {task} was analysed for {epochs} epochs")

            for epoch in range(1, epochs+1):
                cosine = res["alignment"][f"{epoch}"]["11"]["cosine"]
                cka = res["alignment"][f"{epoch}"]["11"]["cka"]
                mknn = res["alignment"][f"{epoch}"]["11"]["mknn_full_epoch"]
                rank = res["alignment"][f"{epoch}"]["11"]["rank_full_epoch"]
                procrustes = res["alignment"][f"{epoch}"]["11"]["procrustes_full_epoch"]

                val_acc = res["training"][f"{epoch}"]["val_acc"]
                val_loss = res["training"][f"{epoch}"]["val_loss"]

                dp = {
                    "task": task,
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "cosine": cosine,
                    "cka": cka,
                    "mknn": mknn,
                    "rank": rank,
                    "procrustes": procrustes,
                    "id": c
                }
                dps.append(dp)

        c += 1

if dps:
    val_accs = np.array([ dp["val_acc"] for dp in dps])
    val_losses_only = np.array([ -dp["val_loss"] for dp in dps])

    print("-" *20 + " all dps together")
    print(f"{val_accs.shape[0]} datapoints")
    for metric in ["cosine", "cka", "mknn", "rank", "procrustes"]:
        metric_only = np.array([ dp[metric] for dp in dps])
        correlation = pearsonr(metric_only, val_accs)
        print(f"Task {task} - Correlation between {metric} and val acc: r={correlation[0]:+.3f}, p={correlation[1]:.3f}")

    print("")
    for metric in ["cosine", "cka", "mknn", "rank", "procrustes"]:
        metric_only = np.array([ dp[metric] for dp in dps])
        correlation_spearman = spearmanr(metric_only, val_accs)
        print(f"Task {task} - Spearman correlation between {metric} and val acc: r={correlation_spearman[0]:+.3f}, p={correlation_spearman[1]:.3f}")
    print("")
    for metric in ["cosine", "cka", "mknn", "rank", "procrustes"]:
        metric_only = np.array([ dp[metric] for dp in dps])
        correlation_loss = pearsonr(metric_only, val_losses_only)
        print(f"Task {task} - Correlation between {metric} and val loss: r={correlation_loss[0]:+.3f}, p={correlation_loss[1]:.3f}")


    print("\n" + "-" *20 + "separated by task")
    for task in ["hateful_memes", "mm_imdb"]:
        task_dps = [dp for dp in dps if dp["task"] == task]
        val_accs = np.array([ dp["val_acc"] for dp in task_dps])
        val_losses_only = np.array([ dp["val_loss"] for dp in task_dps])

        print(f"Task {task}: {val_accs.shape[0]} datapoints")
        for metric in ["cosine", "cka", "mknn", "rank", "procrustes"]:
            metric_only = np.array([ dp[metric] for dp in task_dps])
            correlation = pearsonr(metric_only, val_accs)
            print(f"Task {task} - Correlation between {metric} and val acc: r={correlation[0]:+.3f}, p={correlation[1]:.3f}")
        print("")

    print("\n" + "-"*25 + "seperted by run")
    set_ids = set([dp["id"] for dp in dps])
    for id in set_ids:
        run_dps = [dp for dp in dps if dp["id"] == id]
        val_accs = np.array([ dp["val_acc"] for dp in run_dps])
        val_losses_only = np.array([ dp["val_loss"] for dp in run_dps])
        task = run_dps[0]["task"]

        print(f"Run {id} - Task {task}: {val_accs.shape[0]} datapoints")
        for metric in ["cosine", "cka", "mknn", "rank", "procrustes"]:
            metric_only = np.array([ dp[metric] for dp in run_dps])
            correlation = pearsonr(metric_only, val_accs)
            print(f"Run {id} - Task {task} - Correlation between {metric} and val acc: r={correlation[0]:+.3f}, p={correlation[1]:.3f}")
        print("")
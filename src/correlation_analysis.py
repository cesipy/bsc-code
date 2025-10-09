import os
import json
import warnings
from sklearn.exceptions import ConvergenceWarning

warnings.filterwarnings('ignore', category=ConvergenceWarning)

import numpy as np
from scipy.stats import pearsonr, spearmanr
import experiment_tracker
from vilbert import *
from logger import Logger


import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")

logger = Logger()


def was_task_analysed(task, content):
    """check whether task contains alignment data"""
    return content[task]["alignment"]["0"] != {}

def check_correlation(accs, losses, vals, metric:str,  corr_fn=pearsonr):
    assert len(accs) == len(losses) == len(vals)
    r_a, p_a = corr_fn(accs, vals)
    r_l, p_l = corr_fn(losses, vals)
    info_str1 = f"corr. of {metric:14} with acc : r={r_a:+.3f}, p={p_a:.3f}"
    info_str2 = f"corr. of {metric:14} with loss: r={r_l:+.3f}, p={p_l:.3f}"



def main():

    paths = [
        # on gaming pc
        # "res/checkpoints/20251007-200007_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251007-201826_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-141240_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-144350_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-154319_finetuned_hateful_memes.pt",
        # "res/checkpoints/20251008-160257_finetuned_mm_imdb.pt",
        # "res/checkpoints/20251008-170044_finetuned_hateful_memes.pt"

        # ---------------------------------------------------------
        #uni gpus
        "res/checkpoints/20251006-211344_finetuned_hateful_memes.pt",
        "res/checkpoints/20251006-211344_finetuned_mm_imdb.pt",
        "res/checkpoints/20251006-224233_finetuned_hateful_memes.pt",
        "res/checkpoints/20251006-224233_finetuned_mm_imdb.pt",
        "res/checkpoints/20251007-160301_finetuned_hateful_memes.pt",
        "res/checkpoints/20251007-160855_finetuned_hateful_memes.pt",
        "res/checkpoints/20251007-160855_finetuned_mm_imdb.pt",
        "res/checkpoints/20251007-162505_finetuned_mm_imdb.pt",
        "res/checkpoints/20251007-172924_finetuned_hateful_memes.pt",
        "res/checkpoints/20251007-173620_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-113042_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-113042_finetuned_mm_imdb.pt",
        #not sure about them above, could be bad runs
        "res/checkpoints/20251008-131105_finetuned_mm_imdb.pt",
        "res/checkpoints/20251008-143741_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-143741_finetuned_mm_imdb.pt",
        "res/checkpoints/20251008-161701_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-161701_finetuned_mm_imdb.pt",
        "res/checkpoints/20251008-174253_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-174253_finetuned_mm_imdb.pt",
        "res/checkpoints/20251008-193456_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-193456_finetuned_mm_imdb.pt",
        "res/checkpoints/20251008-211323_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-211323_finetuned_mm_imdb.pt",
        "res/checkpoints/20251008-223850_finetuned_hateful_memes.pt",
        "res/checkpoints/20251008-223850_finetuned_mm_imdb.pt",
        "res/checkpoints/20251009-004449_finetuned_hateful_memes.pt",
        "res/checkpoints/20251009-004449_finetuned_mm_imdb.pt",
        "res/checkpoints/20251009-023655_finetuned_hateful_memes.pt",
        "res/checkpoints/20251009-023655_finetuned_mm_imdb.pt",
        "res/checkpoints/20251009-042241_finetuned_hateful_memes.pt",
        "res/checkpoints/20251009-042241_finetuned_mm_imdb.pt",
        "res/checkpoints/20251009-060800_finetuned_hateful_memes.pt",
        "res/checkpoints/20251009-060800_finetuned_mm_imdb.pt",
        "res/checkpoints/20251009-075441_finetuned_hateful_memes.pt",
        "res/checkpoints/20251009-075441_finetuned_mm_imdb.pt",
    ]
    t = experiment_tracker.ExperimentTracker()
    num_samples = 512
    k=32

    losses = []
    accs   = []
    metrics =   []


    i = 0
    for path in paths:
        model = ViLBERT.load_model(load_path=path, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"model coattentions - t_biattn: {model.config.text_cross_attention_layers}, v_biattn: {model.config.vision_cross_attention_layers}, path: {path}")
        alignment_metrics_h = t.run_alignment_analysis(model=model, num_samples=num_samples, knn_k=k, task="hateful_memes",
            verbose=False, device="cuda" if torch.cuda.is_available() else "cpu",)

        # print(alignment_metrics_h)
        metrics_last_layer_h = alignment_metrics_h[11]      # we only want to have them for the last layer compute correlation between them and acc/loss

        alignment_metrics_m = t.run_alignment_analysis(model=model, num_samples=num_samples, knn_k=k, task="mm_imdb",
            verbose=False, device="cuda" if torch.cuda.is_available() else "cpu",)
        metrics_last_layer_m = alignment_metrics_m[11]      # we only want to have them for the last layer compute correlation between them and acc/loss#


        loss_h, acc_h = t.evaluate(model=model, task="hateful_memes",)
        loss_m, acc_m = t.evaluate(model=model, task="mm_imdb",)
        # loss_h, acc_h = .0,.0
        # loss_m, acc_m = .0,.0

        print(f"model {path}: \n\thateful_memes: val_loss={loss_h:.4f}, val_acc={acc_h:.4f}\n\tmm_imdb      : val_loss={loss_m:.4f}, val_acc={acc_m:.4f}")
        logger.info(f"model {path}: \n\thateful_memes: val_loss={loss_h:.4f}, val_acc={acc_h:.4f}\n\tmm_imdb      : val_loss={loss_m:.4f}, val_acc={acc_m:.4f}")
        losses.extend([-loss_h, -loss_m])
        accs.extend([acc_h, acc_m])

        metrics.append({**metrics_last_layer_h, "task": "hateful_memes", "val_loss": loss_h, "val_acc": acc_h, "id": i})
        metrics.append({**metrics_last_layer_m, "task": "mm_imdb", "val_loss": loss_m, "val_acc": acc_m, "id": i})

    all_metrics = metrics_last_layer_h.keys()
    print(f"all metrics: {all_metrics}")
    accs_array = np.array(accs)
    losses_array = np.array(losses)

    all_metrics = list(metrics[0].keys())
    exclude = ["task", "val_loss", "val_acc", "id"]
    all_metrics = [m for m in all_metrics if m not in exclude]

    print("Pearson Correlations:")
    logger.info("Pearson Correlations:")
    print("-" * 25)
    logger.info("-" * 25)
    for metric in all_metrics:
        metric_array = np.array([m[metric] for m in metrics])
        check_correlation(
            accs=accs_array,
            losses=losses_array,
            metric=metric,
            vals=metric_array,
            corr_fn=pearsonr
        )

    print(f"\n{'='*25}")
    print("Spearman Correlations:")
    print("-" * 25)
    logger.info(f"\n{'='*25}");logger.info("Spearman Correlations:");logger.info("-" * 25)
    for metric in all_metrics:
        metric_array = np.array([m[metric] for m in metrics])
        check_correlation(
            accs=accs_array,
            losses=losses_array,
            metric=metric,
            vals=metric_array,
            corr_fn=spearmanr
        )

    return metrics, accs_array, losses_array








# paths = [
#     "res/experiments/20251006-205740_experiment_coattn_7-8-9.json"
#     # "res/experiments/20251005-195330_experiment_coattn_2-3-4.json",     # finetune hms
#     # "res/experiments/20251005-205228_experiment_coattn_4-8-9.json",
#     # "res/experiments/20251005-214318_experiment_coattn_0-3-4-5-6-8.json",
#     # "res/experiments/20251005-220421_experiment_coattn_1-3-5-6-7.json",
#     # "res/experiments/20251005-222404_experiment_coattn_0-1-3-8-11.json",
#     # "res/experiments/20251006-125134_experiment_coattn_1-2-4-7-8-10.json",
#     # "res/experiments/20251006-132458_experiment_coattn_1-3-5-7-8-9-10.json",
#     # below all are imported, Im not sure what task and if this was finetune or not
#     # "res/experiments-import/20251003-162327_experiment_coattn_5-6-7.json",
#     # "res/experiments-import/20251004-173707_experiment_coattn_5-6-7.json",
#     # "res/experiments-import/20251004-191821_experiment_coattn_5-6-7.json",
#     # "res/experiments-import/20251004-232753_experiment_coattn_2-3-4.json",
#     # "res/experiments-import/20251005-051920_experiment_coattn_2-3-4.json",
#     # "res/experiments-import/20251005-063917_experiment_coattn_5-6-7.json",
#     # "res/experiments-import/20251005-124607_experiment_coattn_5-6-7.json",
#     # "res/experiments-import/20251005-140732_experiment_coattn_9-10-11.json",
# ]


# def compute_correlations(label, dps, target_key):
#     """computes Pearson and Spearman correlation for each metric"""
#     target = np.array([dp[target_key] for dp in dps])
#     metrics = ["cosine", "cka", "mknn", "rank", "procrustes", "svcca"]

#     val_loss = np.array([dp["val_loss"] for dp in dps])
#     val_accs = np.array([dp["val_acc"] for dp in dps])
#     r, p = pearsonr(val_loss, val_accs)
#     print(f"{label} - Pearson between val_loss and val_acc: r={r:+.3f}, p={p:.3f}")

#     for metric in metrics:
#         metric_vals = np.array([dp[metric] for dp in dps])

#         # Pearson
#         r, p = pearsonr(metric_vals, target)
#         print(f"{label} - Pearson between {metric} and {target_key}: r={r:+.3f}, p={p:.3f}")

#     print("")  # spacing

#     for metric in metrics:
#         metric_vals = np.array([dp[metric] for dp in dps])

#         rho, p_s = spearmanr(metric_vals, target)
#         print(f"{label} - Spearman between {metric} and {target_key}: r={rho:+.3f}, p={p_s:.3f}")

#     print("")


# def analyse_all_dps(dps):
#     """aggregated correlation across all data points"""
#     print("-" * 20 + " all dps together")
#     print(f"{len(dps)} datapoints")
#     compute_correlations("all tasks", dps, "val_acc")

#     val_loss_inv = [{"val_loss": -dp["val_loss"], **dp} for dp in dps]
#     compute_correlations("all tasks", val_loss_inv, "val_loss")


# def analyse_per_task(dps):
#     """analyse correlations per task"""
#     print("\n" + "-" * 20 + "separated by task")
#     for task in ["hateful_memes", "mm_imdb"]:
#         task_dps = [dp for dp in dps if dp["task"] == task]
#         if not task_dps:
#             continue
#         print(f"Task {task}: {len(task_dps)} datapoints")
#         compute_correlations(f"Task {task}", task_dps, "val_acc")


# def analyse_per_run(dps):
#     """analyse correlations per individual run"""
#     print("\n" + "-" * 25 + "seperted by run")
#     for run_id in sorted(set(dp["id"] for dp in dps)):
#         run_dps = [dp for dp in dps if dp["id"] == run_id]
#         if not run_dps:
#             continue
#         task = run_dps[0]["task"]
#         print(f"Run {run_id} - Task {task}: {len(run_dps)} datapoints")
#         compute_correlations(f"Run {run_id} - Task {task}", run_dps, "val_acc")

# def analyse_best_epochs(dps):
#     """Find and analyze best epochs per run"""
#     best_dps = []
#     for run_id in sorted(set(dp['id'] for dp in dps)):
#         run_dps = [dp for dp in dps if dp['id'] == run_id]
#         best_dp = max(run_dps, key=lambda x: x['val_acc'])
#         best_dps.append(best_dp)

#     print("Best Epochs Analysis:")
#     compute_correlations("Best Epochs", best_dps, "val_acc")

# # --- main ---
# c = 0
# dps = []

# for path in paths:
#     with open(path, "r") as f:
#         content = json.load(f)

#     for task in ["hateful_memes", "mm_imdb"]:
#         res = content[task]

#         if was_task_analysed(task, content):
#             # get validation acc and alignment values
#             epochs = len(res["alignment"]) -1
#             print(f"Task {task} was analysed for {epochs} epochs")

#             for epoch in range(1, epochs + 1):
#                 align = res["alignment"][f"{epoch}"]["11"]

#                 dp = {
#                     "task": task,
#                     "epoch": epoch,
#                     "val_acc": res["training"][f"{epoch}"]["val_acc"],
#                     "val_loss": res["training"][f"{epoch}"]["val_loss"],
#                     "cosine": align["cosine"],
#                     "cka": align["cka"],
#                     "mknn": align["mknn_full_epoch"],
#                     "rank": align["rank_full_epoch"],
#                     "procrustes": align["procrustes_full_epoch"],
#                     "svcca": align["svcca"],
#                     "id": c,
#                 }
#                 dps.append(dp)

#         c += 1





# # perform analyses
# if dps:
#     analyse_all_dps(dps)
#     analyse_best_epochs(dps)
#     # analyse_per_task(dps)
#     # analyse_per_run(dps)
main()
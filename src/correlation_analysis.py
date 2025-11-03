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
ANALYSIS_SIZE = 1024


def was_task_analysed(task, content):
    """check whether task contains alignment data"""
    return content[task]["alignment"]["0"] != {}

def check_correlation(accs, losses, vals, metric:str,  corr_fn=pearsonr):
    assert len(accs) == len(losses) == len(vals)
    r_a, p_a = corr_fn(accs, vals)
    r_l, p_l = corr_fn(losses, vals)
    info_str1 = f"corr. of {metric:14} with acc : r={r_a:+.3f}, p={p_a:.3f}"
    info_str2 = f"corr. of {metric:14} with loss: r={r_l:+.3f}, p={p_l:.3f}"
    print(info_str1); logger.info(info_str1)
    print(info_str2); logger.info(info_str2)

def analyse_per_task(task:str, paths):
    t = experiment_tracker.ExperimentTracker()
    num_samples = ANALYSIS_SIZE
    k=32

    losses = []
    accs   = []
    losses_val = []
    accs_val = []
    metrics =   []
    metrics_val = []
    metrics_last_layer = None


    for path in paths:
        if task not in path:
            continue
        model = ViLBERT.load_model(load_path=path, device="cuda" if torch.cuda.is_available() else "cpu")
        print(f"model coattentions - t_biattn: {model.config.text_cross_attention_layers}, v_biattn: {model.config.vision_cross_attention_layers}, path: {path}")
        alignment_metrics = t.run_alignment_analysis(
            model=model,
            num_samples=num_samples,
            knn_k=k,
            task=task,
            verbose=False,
            device="cuda" if torch.cuda.is_available() else "cpu",
        )

        # print(alignment_metrics_h)
        metrics_last_layer = alignment_metrics[11]      # we only want to have them for the last layer compute correlation between them and acc/loss
        dict_v= t.evaluate(model=model, task=task,)
        dict_t = t.evaluate(model=model, task=task, dataset="val")
        loss, acc = dict_v["loss"], dict_v["acc"]
        loss_val, acc_val = dict_t["loss"], dict_t["acc"]

        info_str = f"model {path}: \n\t{task}: test loss={loss:.4f}, test acc={acc:.4f}"
        info_str2 = f"model {path}: \n\t{task}:  val loss={loss_val:.4f},  val acc={acc_val:.4f}"
        losses.append(loss); losses_val.append(loss_val)
        accs.append(acc); accs_val.append(acc_val)

        metrics.append({**metrics_last_layer, "task": task, "test_loss": loss, "test_acc": acc, "id": path})
        metrics_val.append({**metrics_last_layer, "task": task, "val_loss": loss_val, "val_acc": acc_val, "id": path})
        print(info_str); logger.info(info_str)
        print(info_str2); logger.info(info_str2)

    if metrics_last_layer is None:
        print(f"No models found for task {task} in the provided paths.")
        return
    all_metrics = metrics_last_layer.keys()
    accs_array = np.array(accs)
    losses_array = np.array(losses)

    all_metrics = list(metrics[0].keys())
    all_metrics_val = list(metrics_val[0].keys())
    exclude_test = ["task", "test_loss", "test_acc", "id"]
    exclude_val = ["task", "val_loss", "val_acc", "id"]

    all_metrics = [m for m in all_metrics if m not in exclude_test]
    all_metrics_val = [m for m in all_metrics_val if m not in exclude_val]

    info_str = f"\n{'='*25}test dataset"
    print(info_str); logger.info(info_str)
    info_str = f"Pearson Correlations for task {task}:"
    print(info_str); logger.info(info_str)
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
    print(f"Spearman Correlations for task {task}:")
    print("-" * 25)
    logger.info(f"\n{'='*25}");logger.info(f"Spearman Correlations for task {task}");logger.info("-" * 25)
    for metric in all_metrics:
        metric_array = np.array([m[metric] for m in metrics])
        check_correlation(
            accs=accs_array,
            losses=losses_array,
            metric=metric,
            vals=metric_array,
            corr_fn=spearmanr
        )

    info_str = f"\n{'='*25}validation dataset"
    print(info_str); logger.info(info_str)
    info_str = f"Pearson Correlations for task {task}:"
    print(info_str); logger.info(info_str)
    print("-" * 25); logger.info("-" * 25)
    accs_array_val = np.array(accs_val)
    losses_array_val = np.array(losses_val)
    for metric in all_metrics_val:
        metric_array = np.array([m[metric] for m in metrics_val])
        check_correlation(
            accs=accs_array_val,
            losses=losses_array_val,
            metric=metric,
            vals=metric_array,
            corr_fn=pearsonr
        )
    print(f"Spearman Correlations for task {task}:");print("-" * 25)
    logger.info(f"\n{'='*25}");logger.info(f"Spearman Correlations for task {task}");logger.info("-" * 25)
    for metric in all_metrics_val:
        metric_array = np.array([m[metric] for m in metrics_val])
        check_correlation(
            accs=accs_array_val,
            losses=losses_array_val,
            metric=metric,
            vals=metric_array,
            corr_fn=spearmanr
        )


def main():

    paths = []
    # below are finetuned on uni gpus
    dir1 = "res/checkpoints/20251010-085859_pretrained_baseline"
    dir2 = "res/checkpoints/20251010-234252_pretrained_early_fusion"
    dir3 = "res/checkpoints/20251011-234349_pretrained_middle_fusion"
    dir4 = "res/checkpoints/20251013-010227_pretrained_late_fusion"
    dir5 = "res/checkpoints/20251014-034432_pretrained_asymmetric_fusion"
    dir6 = "res/checkpoints/20251015-081211_pretrained_optuna1"
    dir7 = "res/checkpoints/20251016-062038_pretrained_optuna2"
    dir8 = "res/checkpoints/20251025-105249_pretrained_bl_full_coattn"
    dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]
    for dir in dirs:
        for i in os.listdir(dir):
            if i.endswith(".pt"):
                paths.append(os.path.join(dir, i))




    analyse_per_task(task="hateful_memes", paths=paths)
    analyse_per_task(task="mm_imdb", paths=paths)
    analyse_per_task(task="upmc_food", paths=paths)





main()
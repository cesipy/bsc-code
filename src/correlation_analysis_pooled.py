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
num_samples = ANALYSIS_SIZE
k=32
t = experiment_tracker.ExperimentTracker()


def was_task_analysed(task, content):
    """check whether task contains alignment data"""
    return content[task]["alignment"]["0"] != {}

def check_correlation(accs,  vals, metric:str,  corr_fn=pearsonr):
    assert len(accs) == len(vals)
    r_a, p_a = corr_fn(accs, vals)
    info_str1 = f"corr. of {metric:14} with perf_value : r={r_a:+.3f}, p={p_a:.3f}"
    print(info_str1); logger.info(info_str1)



def get_metrics_last_layer(task:str, path:str, t:experiment_tracker.ExperimentTracker, lists):
    perf_values, metrics = lists

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
    metrics_last_layer = alignment_metrics[11]      # we only want to have them for the last layer compute correlation between them and perf_value/loss
    # dict_v= t.evaluate(model=model, task=task,)
    # dict_t = t.evaluate(model=model, task=task, dataset="val")


    dl = datasets.get_task_test_dataset(
        task=task,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        seed=1      # should be not important here
    )

    trainer = t.get_task_trainer(
        task=task,
        model=ViLBERT.load_model(load_path=path,
                                 device="cuda" if torch.cuda.is_available() else "cpu",)
    )
    if task == "hateful_memes":
        metric = "auc"
    elif task == "mm_imdb":
        metric = "f1_score_macro"
    elif task == "upmc_food":
        metric = "accuracy"
    else:
        assert False


    metric_res = trainer.get_performance_metric(dataloader=dl, metric=metric)
    info_str = f"model: {path} - metric: {metric}: {metric_res}"
    print(info_str); logger.info(info_str)

    perf_value = metric_res

    perf_values.append(perf_value)
    metrics.append({**metrics_last_layer, "task": task, "test_acc": perf_value, "id": path})
    return metrics, perf_values, metrics_last_layer

def get_metrics_max_layer(task:str, path:str, t:experiment_tracker.ExperimentTracker, lists):
    perf_values, metrics = lists

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

    metrics_max_layer = {}
    metric_keys = alignment_metrics[0].keys()
    print(metric_keys)
    for m in metric_keys:
        all_ms = [alignment_metrics[i][m] for i in range(12)]
        metrics_max_layer[m] = max(all_ms)

    dl = datasets.get_task_test_dataset(
        task=task,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        seed=1      # should be not important here
    )

    trainer = t.get_task_trainer(
        task=task,
        model=ViLBERT.load_model(load_path=path,
                                 device="cuda" if torch.cuda.is_available() else "cpu",)
    )
    if task == "hateful_memes":
        metric = "auc"
    elif task == "mm_imdb":
        metric = "f1_score_macro"
    elif task == "upmc_food":
        metric = "accuracy"
    else:
        assert False

    metric_res = trainer.get_performance_metric(dataloader=dl, metric=metric)
    info_str = f"model: {path} - metric: {metric}: {metric_res}"
    print(info_str); logger.info(info_str)

    perf_value = metric_res

    perf_values.append(perf_value)
    metrics.append({**metrics_max_layer, "task": task, "test_acc": perf_value, "id": path})
    return metrics, perf_values, metrics_max_layer

def analyse_per_task(task:str, paths):

    perf_values   = []
    metrics =   []


    for path in paths:
        if task not in path:
            continue
        lists = [perf_values, metrics]
        # metrics,perf_values, metrics_last_layer = get_metrics_last_layer(task, path, t, lists)
        metrics,perf_values, metrics_last_layer = get_metrics_max_layer(task, path, t, lists)

    if not metrics:  # Add this
        print(f"No models found for task {task} in the provided paths.")
        return

    accs_array = np.array(perf_values)

    all_metrics = metrics_last_layer.keys()
    all_metrics = list(metrics[0].keys())

    exclude_test = ["task", "test_loss", "test_acc", "id"]


    all_metrics = [m for m in all_metrics if m not in exclude_test]


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
            metric=metric,
            vals=metric_array,
            corr_fn=pearsonr
        )
    print(f"Spearman Correlations for task {task}:");print("-" * 25)
    logger.info(f"\n{'='*25}");logger.info(f"Spearman Correlations for task {task}");logger.info("-" * 25)

    for metric in all_metrics:
        metric_array = np.array([m[metric] for m in metrics])
        check_correlation(
            accs=accs_array,
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
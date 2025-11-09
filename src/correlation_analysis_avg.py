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
    print(info_str1); #logger.info(info_str1)

def get_ram_averages(alignment_metrics_list: list[dict]):

    metric_keys = list(alignment_metrics_list[0].keys())
    averaged = {}
    for key in metric_keys:
        values = [m[key] for m in alignment_metrics_list]
        averaged[key] = np.mean(values)

    return averaged



def get_metrics_last_layer(task:str, dir_path:str, t:experiment_tracker.ExperimentTracker, lists):
    alignment_metrics_seed = []
    performance_metrics_seed = []
    for path in os.listdir(dir_path):
        if task not in path:
            continue

        path = os.path.join(dir_path, path)
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

        # perf_value = metric_res
        performance_metrics_seed.append(metric_res)
        alignment_metrics_seed.append(metrics_last_layer)

    performance_value  = np.mean(performance_metrics_seed)
    metrics_last_layer = get_ram_averages(alignment_metrics_seed)


    perf_values.append(performance_value)
    metrics.append({**metrics_last_layer, "task": task, "test_acc": performance_value, "id": dir_path})
    counter = 1

    return metrics, perf_values, metrics_last_layer

def get_metrics_max_layer(task:str, dir_path:str, t:experiment_tracker.ExperimentTracker, lists):
    alignment_metrics_seed = []
    performance_metrics_seed = []
    for path in os.listdir(dir_path):
        if task not in path:
            continue

        path = os.path.join(dir_path, path)
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

        # perf_value = metric_res
        performance_metrics_seed.append(metric_res)
        alignment_metrics_seed.append(metrics_max_layer)

    performance_value = np.mean(performance_metrics_seed)
    metrics_max_layer = get_ram_averages(alignment_metrics_seed)

    perf_values.append(performance_value)
    metrics.append({**metrics_max_layer, "task": task, "test_acc": performance_value, "id": dir_path})
    counter = 1

    return metrics, perf_values, metrics_max_layer



def analyse_per_task(task:str, dirs):

    perf_values   = []
    metrics =   []

    for dir in dirs:
        lists = [perf_values, metrics]
        # metrics,perf_values, metrics_layer = get_metrics_last_layer(task, dir, t, lists)
        metrics, perf_values, metrics_layer = get_metrics_max_layer(task, dir, t, lists)

    if not metrics:
        print(f"No models found for task {task} in the provided paths.")
        return

    accs_array = np.array(perf_values)

    all_metrics = metrics_layer.keys()
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

def get_jsons_for_architecture(path: str):
    paths = []
    for filename in os.listdir(path):
        if filename.endswith(".json"):
            paths.append(os.path.join(path, filename))
    return paths

def correlation_analysis_architecture(json_path):
    """
    calculates correlation between layerwise performance and layerwise representation alignment within one architecture.
    This means e.g. for 'late_fusion' correlation between performance(layer_i) vs alignment(layer_i) for all i in range(12)
    """
    print(json_path)
    with open(json_path, 'r') as f:
        content = json.load(f)

    # print(content[content.keys()[0]].keys())
    performance_values = [ content[key]["metric"] for key in content.keys()]
    cka_values         = [ content[key]["cka"] for key in content.keys()]
    mknn_values        = [ content[key]["mknn"] for key in content.keys()]
    svcca_values       = [ content[key]["svcca"] for key in content.keys()]
    procrustes_values  = [ content[key]["procrustes"] for key in content.keys()]

    for metric in ["cka", "mknn", "svcca", "procrustes"]:
        vals_map = {
            "cka": cka_values,
            "mknn": mknn_values,
            "svcca": svcca_values,
            "procrustes": procrustes_values,
        }
        vals = np.array(vals_map[metric])
        for corr_fn in [pearsonr, spearmanr]:
            check_correlation(
                accs=np.array(performance_values),
                metric=metric,
                vals=vals,
                corr_fn=corr_fn
            )

def correlation_analysis_layerwise_pooled(json_paths: str):
    """
    calculates correlation between layerwise performance and layerwise representation alignment across multiple architectures.
    """
    performance_values = []
    cka_values         = []
    mknn_values        = []
    svcca_values       = []
    procrustes_values  = []

    for path in json_paths:
        with open(path, "r")as f:
            curr_content = json.load(f)

        performance_values.extend([curr_content[key]["metric"] for key in curr_content.keys()])
        cka_values.extend([curr_content[key]["cka"] for key in curr_content.keys()])
        mknn_values.extend([curr_content[key]["mknn"] for key in curr_content.keys()])
        svcca_values.extend([curr_content[key]["svcca"] for key in curr_content.keys()])
        procrustes_values.extend([curr_content[key]["procrustes"] for key in curr_content.keys()])

    for metric in ["cka", "mknn", "svcca", "procrustes"]:
        vals_map = {
            "cka": cka_values,
            "mknn": mknn_values,
            "svcca": svcca_values,
            "procrustes": procrustes_values,
        }
        vals = np.array(vals_map[metric])
        for corr_fn in [pearsonr, spearmanr]:
            check_correlation(
                accs=np.array(performance_values),
                metric=metric,
                vals=vals,
                corr_fn=corr_fn
            )

def get_all_jsons_for_task(dirs: list[str], task:str):

    paths = []

    for dir in dirs:
        for filename in os.listdir(dir):
            if filename.endswith(".json") and task in filename:
                paths.append(os.path.join(dir, filename))
    return paths


def main():
    paths = [
        "plots_probing/20251010-085859_pretrained_baseline",
        "plots_probing/20251013-010227_pretrained_late_fusion",
        "plots_probing/20251016-062038_pretrained_optuna2"
    ]

    # the following is pooled layerwise correlation between performance and alignment across architectures
    tasks = ["hateful_memes", "mm_imdb", "upmc_food"]
    for task in tasks:
        print(f"correlation for {task}")
        correlation_analysis_layerwise_pooled(get_all_jsons_for_task(dirs=paths, task=task))


    # the following are within-architecture correlations. seem to be very good.
    # for path in paths:
    #     print(f"architecture: {path.split('/')[-1]}")
    #     for json_path in get_jsons_for_architecture(path):
    #         correlation_analysis_architecture(json_path)


    # paths = []
    # # below are finetuned on uni gpus
    # dir1 = "res/checkpoints/20251010-085859_pretrained_baseline"
    # dir2 = "res/checkpoints/20251010-234252_pretrained_early_fusion"
    # dir3 = "res/checkpoints/20251011-234349_pretrained_middle_fusion"
    # dir4 = "res/checkpoints/20251013-010227_pretrained_late_fusion"
    # dir5 = "res/checkpoints/20251014-034432_pretrained_asymmetric_fusion"
    # dir6 = "res/checkpoints/20251015-081211_pretrained_optuna1"
    # dir7 = "res/checkpoints/20251016-062038_pretrained_optuna2"
    # dir8 = "res/checkpoints/20251025-105249_pretrained_bl_full_coattn"
    # dirs = [dir1, dir2, dir3, dir4, dir5, dir6, dir7, dir8]




    # analyse_per_task(task="hateful_memes", dirs=dirs)
    # analyse_per_task(task="mm_imdb", dirs=dirs)
    # analyse_per_task(task="upmc_food", dirs=dirs)





main()
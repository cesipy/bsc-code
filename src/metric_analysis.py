import os
import random, time, json
import numpy as np

import experiment_tracker
from config import *
import experiment_tracker_utils as etu
from logger import Logger
import task as tasklib
from vilbert import *
import utils

from scipy.stats import pearsonr, spearmanr


import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")

logger = Logger()

def randomly_generate_config():

    def get_list():
        config = []
        for i in range(12):
            if random.random() < 0.5:
                config.append(i)
        return config
    config_t = get_list()
    config_v = []
    while len(config_v) != len(config_t):
        config_v = get_list()

    return config_t, config_v


def compare_knn_k_values(result, k_values, n_models:int):
    """Compare mknn metric across different K values."""
    print("Spearman correlations between different K values:")
    logger.info("Spearman correlations between different K values:")
    for i, k1 in enumerate(k_values):
        for k2 in k_values[i+1:]:
            data1 = np.array(result[k1]["mknn"])
            data2 = np.array(result[k2]["mknn"])
            r, p = spearmanr(data1, data2)
            result_str = f"  K={k1:3d} vs K={k2:3d}: r={r:.4f}, p={p:.4f}"
            print(result_str)
            logger.info(result_str)

    print(f"\n{'-'*60}")
    print("Within-model K consistency (mean across models):")
    logger.info("\nWithin-model K consistency (mean across models):")

    for i, k1 in enumerate(k_values):
        for k2 in k_values[i+1:]:
            within_corrs = []
            for model_idx in range(n_models):
                start = model_idx * 12
                end = start + 12
                model_data1 = np.array(result[k1]["mknn"][start:end])
                model_data2 = np.array(result[k2]["mknn"][start:end])
                r, _ = spearmanr(model_data1, model_data2)
                within_corrs.append(r)
            mean_r = np.mean(within_corrs)
            std_r = np.std(within_corrs)
            result_str = f"  K={k1:3d} vs K={k2:3d}: mean r={mean_r:.4f} ± {std_r:.4f}"
            print(result_str)
            logger.info(result_str)

def compare_metrics_correlation(result, num_samples, metrics):
    """Compare correlation between different alignment metrics."""
    print(f"\n{'='*50}")
    print(f"Metric correlations at {num_samples} samples")
    print(f"{'='*50}\n")
    logger.info(f"\n{'='*50}")
    logger.info(f"Metric correlations at {num_samples} samples")
    metric_data = {}    # pooled data
    for metric in metrics:
        metric_data[metric] = np.array(result[num_samples][metric])
    print("Spearman correlations between metrics:")
    logger.info("Spearman correlations between metrics:")

    for i, metric1 in enumerate(metrics):
        for metric2 in metrics[i+1:]:  # Only upper triangle
            r, p = spearmanr(metric_data[metric1], metric_data[metric2])
            result_str = f"  {metric1:12} vs {metric2:12}: r={r:.3f}, p={p:.4f}"
            print(result_str)
            logger.info(result_str)

def compare_sample_sizes_(result, size1, size2, metric:str="mknn"):
    info_str= f"{'-'*25}\nanalysis for {metric}"
    print(info_str)
    logger.info(info_str)
    mknns_1 = np.array(result[size1][metric])
    mknns_2 = np.array(result[size2][metric])

    # here is the problem that we violate the independence assumption. in one model, 12 measures are collected.
    r_overall, _ = spearmanr(mknns_1, mknns_2)
    print(f"\nOverall: r={r_overall:.4f}")
    logger.info(f"\nOverall: r={r_overall:.4f}")

    # therefore compare between two models, check if ordering is perserved
    print("\nWithin-model layer ordering:")
    within_corrs = []
    for i in range(8):
        model_vals_1 = mknns_1[i*12:(i+1)*12]
        model_vals_2 = mknns_2[i*12:(i+1)*12]
        r, _ = spearmanr(model_vals_1, model_vals_2)
        within_corrs.append(r)
        print(f"  Model {i}: r={r:.4f}")
        logger.info(f"  Model {i}: r={r:.4f}")

    print(f"\nMean within-model: {np.mean(within_corrs):.4f} ± {np.std(within_corrs):.4f}")
    logger.info(f"\nMean within-model: {np.mean(within_corrs):.4f} ± {np.std(within_corrs):.4f}")


def check_ranking_preservation(results, small_size, large_size, n_models, metric:str):
    n_layers = 12
    ranking_correlations = []
    ranking_ps = []
    for model_idx in range(n_models):
        start_idx = model_idx * n_layers
        end_idx = start_idx + n_layers
        small_metrics = results[small_size][metric][start_idx:end_idx]
        large_metrics = results[large_size][metric][start_idx:end_idx]
        small_ranks = np.argsort(np.argsort(small_metrics))[::-1]  # descending
        large_ranks = np.argsort(np.argsort(large_metrics))[::-1]

        r, p = spearmanr(small_ranks, large_ranks)
        ranking_correlations.append(r)
        ranking_ps.append(p)

    avg_rank_corr = np.mean(ranking_correlations)
    info_str = f"avg rank for {metric:9} ({small_size} vs {large_size}): {avg_rank_corr:.4f} (std:{np.std(ranking_correlations)}), \np={np.mean(ranking_ps):.4f} (std:{np.std(ranking_ps)})"
    print(info_str)
    logger.info(info_str)


def compare_sample_sizes(result, size1, size2):
    mknns_1 = np.array(result[size1]["mknn"])
    ckas_1  = np.array(result[size1]["cka"])
    mknns_2 = np.array(result[size2]["mknn"])
    ckas_2  = np.array(result[size2]["cka"])


    print(f"\nComparing {size1} vs {size2} samples:")
    logger.info(f"\nComparing {size1} vs {size2} samples:")
    print("=" * 40)

    for corr in [pearsonr, spearmanr]:
        r_mknn, p_mknn = corr(mknns_1, mknns_2)
        r_cka, p_cka = corr(ckas_1, ckas_2)
        print(f"{corr.__name__}:")
        print(f"  mknn: r={r_mknn:.4f}, p={p_mknn:.4f}")
        print(f"  cka:  r={r_cka:.4f}, p={p_cka:.4f}")
        print()
        logger.info(f"{corr.__name__}:")
        logger.info(f"  mknn: r={r_mknn:.4f}, p={p_mknn:.4f}")
        logger.info(f"  cka:  r={r_cka:.4f}, p={p_cka:.4f}")
        logger.info("----")

def main():
    t = experiment_tracker.ExperimentTracker()
    # temp_conf = experiment_tracker.ExperimentConfig(
    #     t_biattention_ids=[4,5,6],
    #     v_biattention_ids=[5,6,7],
    #     use_contrastive_loss=False,
    #     epochs=6,
    #     seed=42,
    #     learning_rate=3.5e-5
    # )
    # t.run_finetune(temp_conf, run_visualizations=True, tasks=["hateful_memes"])


    t_biattn_idxs, v_biattn_idxs = randomly_generate_config()

    paths = [
        'res/checkpoints/20251007-122929_finetuned_hateful_memes.pt', 'res/checkpoints/20251007-125349_finetuned_hateful_memes.pt',
        'res/checkpoints/20251007-165717_finetuned_hateful_memes.pt', 'res/checkpoints/20251007-171651_finetuned_mm_imdb.pt',
        'res/checkpoints/20251007-182842_finetuned_hateful_memes.pt', 'res/checkpoints/20251007-184814_finetuned_mm_imdb.pt',
        'res/checkpoints/20251007-200007_finetuned_hateful_memes.pt', 'res/checkpoints/20251007-201826_finetuned_mm_imdb.pt'
    ]
    # seeds = [20, 33]
    # analysis_num_samples = [64, 128, 256, 512, 1024, ]#1700]
    # result = {}
    # for num_samples in analysis_num_samples:

    #     #TODO: still not all
    #     result[num_samples] = {
    #         "mknn":[],
    #         "cka": [],
    #         "cka_rbf": [],
    #         "svcca": [],
    #         "cknna": [],
    #         "cycle_knn": [],
    #         "procrustes": [],
    #         "jaccard": [],
    #         "rsa": [],
    #         "r2": [],
    #     }

    # for path in paths:

    #     model = ViLBERT.load_model(load_path=path)
    #     print(f"model coattentions - t_biattn: {model.config.text_cross_attention_layers}, v_biattn: {model.config.vision_cross_attention_layers}")
    #     for num_samples in analysis_num_samples:
    #         print(f"analysis num samples: {num_samples}")
    #         # problem: gpu is out of memory if num_samples is at 1700!
    #         device = "cpu" if num_samples == 1700 else "cuda"

    #         metrics = t.run_alignment_analysis(
    #             model=model, num_samples=num_samples, task="hateful_memes", device=device)
    #         mknns = [metrics[i]["mknn"] for i in range(12)]
    #         ckas  = [metrics[i]["cka"] for i in range(12)]
    #         cka_rbfs = [metrics[i]["cka_rbf"] for i in range(12)]
    #         svccas = [metrics[i]["svcca"] for i in range(12)]
    #         cknnas = [metrics[i]["cknna"] for i in range(12)]
    #         cycle_knns = [metrics[i]["cycle_knn"] for i in range(12)]
    #         procrustes = [metrics[i]["procrustes"] for i in range(12)]
    #         jaccards = [metrics[i]["jaccard"] for i in range(12)]
    #         rsas = [metrics[i]["rsa"] for i in range(12)]
    #         r2s = [metrics[i]["r2"] for i in range(12)]
    #         result[num_samples]["mknn"].extend(mknns)
    #         result[num_samples]["cka"].extend(ckas)
    #         result[num_samples]["cka_rbf"].extend(cka_rbfs)
    #         result[num_samples]["svcca"].extend(svccas)
    #         result[num_samples]["cknna"].extend(cknnas)
    #         result[num_samples]["cycle_knn"].extend(cycle_knns)
    #         result[num_samples]["procrustes"].extend(procrustes)
    #         result[num_samples]["jaccard"].extend(jaccards)
    #         result[num_samples]["rsa"].extend(rsas)
    #         result[num_samples]["r2"].extend(r2s)
    #     print("-"*25)

    # metrics = ["mknn", "cka", "cka_rbf", "svcca", "cknna", "cycle_knn", "procrustes", "jaccard", "rsa", "r2"]


    # compare_sample_sizes_(result, 64, 256)
    # check_ranking_preservation(result, 64, 1024, n_models=len(paths), metric="cka")
    # check_ranking_preservation(result, 128, 1024, n_models=len(paths), metric="cka")
    # check_ranking_preservation(result, 256, 1024, n_models=len(paths), metric="cka")
    # for metric in metrics:
    #     check_ranking_preservation(result, 512, 1024, n_models=len(paths), metric=metric)

    # for metric in metrics:
    #     utils.visualize_correlation_matrix(
    #         result=result,
    #         metric=metric,
    #         corr_func=pearsonr,
    #         save_path=f"temp/{metric}_pearsonr.png"
    #     )

    #     utils.visualize_correlation_matrix(
    #         result=result,
    #         metric=metric,
    #         corr_func=spearmanr,
    #         save_path=f"temp/{metric}_spearmanr.png"
    #     )
    #     compare_sample_sizes_(result, 512, 1024, metric=metric)


    # print(f"{'-'*25}\nmetric correlation analysis:")
    # logger.info(f"{'-'*25}\nmetric correlation analysis:")

    # compare_metrics_correlation(result, 512, metrics)

    # utils.visualize_metric_correlation_matrix(
    #     result=result,
    #     num_samples=512,
    #     metrics=metrics,
    #     corr_func=spearmanr,
    #     save_path="temp/metric_correlation_spearman.png"
    # )

    # utils.visualize_metric_correlation_matrix(
    #     result=result,
    #     num_samples=512,
    #     metrics=metrics,
    #     corr_func=pearsonr,
    #     save_path="temp/metric_correlation_pearson.png"
    # )

    # ---- knn_k correlation test
    info_str = f"\n\n{'-'*25}\n now running k importantce on mknn!"
    print(info_str); logger.info(info_str)
    result = {}  # reset for knn_k correlation test
    knn_ks = [5, 10, 16, 32, 64, 128, 256]      # 512 is here to big, as i have only that many samples

    for k in knn_ks:
        result[k] = {
            "mknn":[],
            "running_time": []
        }
    for path in paths:
        model = ViLBERT.load_model(load_path=path)
        print(f"model coattentions - t_biattn: {model.config.text_cross_attention_layers}, v_biattn: {model.config.vision_cross_attention_layers}, path: {path}")
        for k in knn_ks:
            t_s = time.time()
            info_str = f"knn_k correlation test with k={k}"
            print(info_str); logger.info(info_str)

            metrics = t.run_alignment_analysis(
                model=model, num_samples=512, task="hateful_memes", device="cuda", knn_k=k, verbose=False)
            # only need to analyse mknn
            result[k]["mknn"].extend([metrics[i]["mknn"] for i in range(12)])

            t_e = time.time()

            result[k]["running_time"].append(t_e-t_s)
        print("-"*25 + "\n\n")

    for k in knn_ks:
        times = result[k]["running_time"]
        info_str = f"knn_k={k}: avg running time: {np.mean(times):.2f}s ± {np.std(times):.2f}s over {len(times)} models"
        print(info_str); logger.info(info_str)


    compare_knn_k_values(result, knn_ks, n_models=len(paths))
    utils.visualize_k_correlation_matrix(
        result=result,
        k_values=knn_ks,
        corr_func=spearmanr,
        save_path="temp/knn_k_correlation_spearman.png"
    )
    utils.visualize_k_correlation_matrix(
        result=result,
        k_values=knn_ks,
        corr_func=pearsonr,
        save_path="temp/knn_k_correlation_pearson.png"
    )



if __name__ == "__main__":
    main()
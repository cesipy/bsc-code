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

os.environ["TOKENIZERS_PARALLELISM"] = "false"

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


def compare_sample_sizes(result, size1, size2):
    """Compare alignment metrics between two different sample sizes."""
    mknns_1 = np.array(result[size1]["mknn"])
    ckas_1  = np.array(result[size1]["cka"])
    mknns_2 = np.array(result[size2]["mknn"])
    ckas_2  = np.array(result[size2]["cka"])
    print(mknns_1)
    print(mknns_2)

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
    # t_biattn_idxs, v_biattn_idxs = randomly_generate_config()

    paths = ['res/checkpoints/20251007-122929_finetuned_hateful_memes.pt', 'res/checkpoints/20251007-125349_finetuned_hateful_memes.pt', ]
    seeds = [20, 33]
    analysis_num_samples = [64, 128, 256, 512, 1024, ]#1700]
    result = {}
    for num_samples in analysis_num_samples:

        #TODO: still not all
        result[num_samples] = {
            "mknn":[],
            "cka": [],
            "cka_rbf": [],
            "svcca": [],
            "cknna": [],
            "cycle_knn": [],
            "procrustes": [],
            "jaccard": [],
            "rsa": [],
            "r2": [],
        }

    for path in paths:

        model = ViLBERT.load_model(load_path=path)
        print(f"model coattentions - t_biattn: {model.config.text_cross_attention_layers}, v_biattn: {model.config.vision_cross_attention_layers}")
        for num_samples in analysis_num_samples:
            print(f"analysis num samples: {num_samples}")
            # problem: gpu is out of memory if num_samples is at 1700!
            device = "cpu" if num_samples == 1700 else "cuda"

            metrics = t.analyse_alignment(model=model, num_samples=num_samples, task="hateful_memes", device=device)
            mknns = [metrics[i]["mknn"] for i in range(12)]
            ckas  = [metrics[i]["cka"] for i in range(12)]
            cka_rbfs = [metrics[i]["cka_rbf"] for i in range(12)]
            svccas = [metrics[i]["svcca"] for i in range(12)]
            cknnas = [metrics[i]["cknna"] for i in range(12)]
            cycle_knns = [metrics[i]["cycle_knn"] for i in range(12)]
            procrustes = [metrics[i]["procrustes"] for i in range(12)]
            jaccards = [metrics[i]["jaccard"] for i in range(12)]
            rsas = [metrics[i]["rsa"] for i in range(12)]
            r2s = [metrics[i]["r2"] for i in range(12)]
            # print(f"temp mknns: {mknns}")
            # print(metrics)
            # with open("temp.json", "w")as f:
            #     def convert_to_native(obj):
            #         if isinstance(obj, dict):
            #             return {k: convert_to_native(v) for k, v in obj.items()}
            #         elif isinstance(obj, list):
            #             return [convert_to_native(item) for item in obj]
            #         elif isinstance(obj, (np.integer, np.floating)):
            #             return obj.item()
            #         elif isinstance(obj, np.ndarray):
            #             return obj.tolist()
            #         return obj

            #     json.dump(convert_to_native(metrics), f, indent=4)
            # exit(0)
            result[num_samples]["mknn"].extend(mknns)
            result[num_samples]["cka"].extend(ckas)
            result[num_samples]["cka_rbf"].extend(cka_rbfs)
            result[num_samples]["svcca"].extend(svccas)
            result[num_samples]["cknna"].extend(cknnas)
            result[num_samples]["cycle_knn"].extend(cycle_knns)
            result[num_samples]["procrustes"].extend(procrustes)
            result[num_samples]["jaccard"].extend(jaccards)
            result[num_samples]["rsa"].extend(rsas)
            result[num_samples]["r2"].extend(r2s)
        print("-"*25)



    # compare_sample_sizes(result, 64, 256)
    # compare_sample_sizes(result, 64, 1024)
    # compare_sample_sizes(result, 128, 1024)

    for i in range(len(analysis_num_samples)):
        for j in range(i+1, len(analysis_num_samples)):
            size1 = analysis_num_samples[i]
            size2 = analysis_num_samples[j]
            compare_sample_sizes(result, size1, size2)
    for metric in ["mknn", "cka", "cka_rbf", "svcca", "cknna", "cycle_knn", "procrustes", "jaccard", "rsa", "r2"]:

        utils.visualize_correlation_matrix(
            result=result,
            metric=metric,
            corr_func=pearsonr,
            save_path=f"temp/{metric}_pearsonr.png"
        )

        utils.visualize_correlation_matrix(
            result=result,
            metric=metric,
            corr_func=spearmanr,
            save_path=f"temp/{metric}_spearmanr.png"
        )

    #TODO: matrix for correlation











    # ------------------------------------------------------------
    # paths = []

    # for i in range(3):
    #     t_biattn, v_biattn = randomly_generate_config()
    #     infostr = f"Run {i+1}/3 with t_biattn={t_biattn}, v_biattn={v_biattn}"
    #     logger.info(infostr)
    #     print(infostr)
    #     test_config = experiment_tracker.ExperimentConfig(
    #         t_biattention_ids=t_biattn,
    #         v_biattention_ids=v_biattn,
    #         use_contrastive_loss=False,
    #         epochs=5,
    #         seed=15,
    #     )


    #     res = t.run_finetune(experiment_config=test_config, tasks=["hateful_memes"], run_alignment_analysis=False)
    #     paths.append(res["hateful_memes"]["model_path"])

    # print(paths)




if __name__ == "__main__":
    main()
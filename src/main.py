
import os, torch
import json

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



def main():
    t = experiment_tracker.ExperimentTracker()



     configs = [
        {
            "name": "early_fusion",
            "t_biattention_ids": [2, 3, 4],
            "v_biattention_ids": [2, 3, 4],
        },
        {
            "name": "middle_fusion",
            "t_biattention_ids": [5, 6, 7],
            "v_biattention_ids": [5, 6, 7],
        },
        {
            "name": "late_fusion",
            "t_biattention_ids": [9, 10, 11],
            "v_biattention_ids": [9, 10, 11],
        },
        {
            "name": "asymmetric_fusion",
            "t_biattention_ids": [6, 7, 8, 9],
            "v_biattention_ids": [3, 5, 7, 9],
        },
    ]

    seeds = [42, 123, 456]
    analysis_num_samples = [100, 400, 1000]
    paths = []

    total_r = len(seeds) * len(configs)
    c = 0
    trained_models = {}
    for config in configs:
        for seed in seeds:
            key = (config['name'], seed)
            c+=1

            logger.info(f"{'-'*25}\nTraining {config['name']}, seed={seed}, run:{c}/{total_r}")

            finetune_config_hm = experiment_tracker.ExperimentConfig(
                t_biattention_ids=config["t_biattention_ids"],
                v_biattention_ids=config["v_biattention_ids"],
                use_contrastive_loss=False,
                epochs=5,
                learning_rate=3.2e-5,
                seed=seed,
            )

            finetune_config_mm_imdb = experiment_tracker.ExperimentConfig(
                t_biattention_ids=config["t_biattention_ids"],
                v_biattention_ids=config["v_biattention_ids"],
                use_contrastive_loss=False,
                epochs=6,
                learning_rate=3.8e-5,
                seed=seed,
            )

            results_hm = t.run_finetune(
                experiment_config=finetune_config_hm,
                run_visualizations=False,
                run_alignment_analysis=False,  # Skip for now
                tasks=["hateful_memes"],
            )
            results_mm_imdb = t.run_finetune(
                experiment_config=finetune_config_mm_imdb,
                run_visualizations=False,
                run_alignment_analysis=False,  # Skip for now
                tasks=["mm_imdb"],
            )

            trained_models[key] = {
                "hateful_memes": results_hm["hateful_memes"]["model_path"],
                "mm_imdb": results_mm_imdb["mm_imdb"]["model_path"],
            }
            paths.append(results_hm["hateful_memes"]["model_path"])
            paths.append(results_mm_imdb["mm_imdb"]["model_path"])

            # print("completed training")
            # path = "res/checkpoints/20251007-113613_finetuned_hateful_memes.pt"
            # # path = results_hm["hateful_memes"]["model_path"]
            # model = ViLBERT.load_model(path,
            #     "cuda" if torch.cuda.is_available() else "cpu")
            # t.analyse_alignment(model, num_samples=1000, task="hateful_memes")
    print(paths)

    with open("trained_models.json", "w") as f:
        json.dump(trained_models, f, indent=4)
    #------------------------------------------------------------
    paths = []
    seed = 123

    for i in range(3):
        t_biattn, v_biattn = randomly_generate_config()
        infostr = f"Run {i+1}/3 with t_biattn={t_biattn}, v_biattn={v_biattn}"
        logger.info(infostr)
        print(infostr)
        test_config = experiment_tracker.ExperimentConfig(
            t_biattention_ids=t_biattn,
            v_biattention_ids=v_biattn,
            use_contrastive_loss=False,
            epochs=4,
            seed=seed,
            learning_rate=3.55e-5
        )
        seed += 1

        test_config_mm = experiment_tracker.ExperimentConfig(
            t_biattention_ids=t_biattn,
            v_biattention_ids=v_biattn,
            use_contrastive_loss=False,
            epochs=6,
            seed=seed,
            learning_rate=4e-5
        )

        seed+=1


        res = t.run_finetune(experiment_config=test_config, tasks=["hateful_memes"], run_alignment_analysis=False)
        paths.append(res["hateful_memes"]["model_path"])
        res2 = t.run_finetune(experiment_config=test_config_mm, tasks=["mm_imdb"], run_alignment_analysis=False)
        paths.append(res2["mm_imdb"]["model_path"])

    print(paths)




>>>>>>> 3ca026b13de2b7296b0ab8f803c18c2c5304f838
if __name__ == "__main__":
    main()
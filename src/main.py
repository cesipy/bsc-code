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

import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")


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
    print("good hm")
    t,v = t.get_biattentions(num_coattn_layers= 2,
        t_center= 4.204866112455652,
        t_spread= 3.149664541279545,
        v_center= 7.075933466334327,
        v_spread= 1.5021909737136068)
    print(t,v)

    print("good both")
    t,v= t.get_biattentions(num_coattn_layers =4,
        t_center= 9.41235168443222,
        t_spread =2.928327618277501,
        v_center =7.845874844937789,
        v_spread =2.6834436760417772)

    configs = [
        {
            "name": "early_fusion",
            "t_biattention_ids": [3, 4, 6],
            "v_biattention_ids": [3, 4, 6],
        },
        # {
        #     "name": "middle_fusion",
        #     "t_biattention_ids": [5, 6, 7],
        #     "v_biattention_ids": [5, 6, 7],
        # },
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

    test_conf = configs[2] # late
    conf = experiment_tracker.ExperimentConfig(
        t_biattention_ids=[5,6,9],
        v_biattention_ids=[5,6,9],
        use_contrastive_loss=False,
        epochs=4,
        seed=42,
        learning_rate=4e-5
    )


    # t.run_finetune(experiment_config=conf, tasks=["upmc_food"], run_alignment_analysis=True, run_visualizations=True)
    # t.run_finetune(experiment_config=conf_contrastive, tasks=["hateful_memes"], run_alignment_analysis=True, run_visualizations=True)
    # print("completed all finetuning")

    seeds = [4261, 1213, ]
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
                epochs=6,
                learning_rate=4e-5,
                seed=seed,
            )
            finetune_config_upmc = experiment_tracker.ExperimentConfig(
                t_biattention_ids=config["t_biattention_ids"],
                v_biattention_ids=config["v_biattention_ids"],
                use_contrastive_loss=False,
                epochs=6,
                learning_rate=4e-5,
                seed=seed,
            )

            results_hm = t.run_finetune(
                experiment_config=finetune_config_hm,
                run_visualizations=False,
                run_alignment_analysis=False,
                tasks=["mm_imdb"],
            )
            results_upmc = t.run_finetune(
                experiment_config=finetune_config_upmc,
                run_visualizations=False,
                run_alignment_analysis=False,
                tasks=["upmc_food"],
            )

            paths.append(results_hm["mm_imdb"]["model_path"])
            paths.append(results_upmc["upmc_food"]["model_path"])

            # print("completed training")

    print(paths)

    with open("trained_models.json", "w") as f:
        json.dump(trained_models, f, indent=4)
    # #------------------------------------------------------------
    # paths = []
    # seed = 123

    # for i in range(3):
    #     t_biattn, v_biattn = randomly_generate_config()
    #     infostr = f"Run {i+1}/3 with t_biattn={t_biattn}, v_biattn={v_biattn}"
    #     logger.info(infostr)
    #     print(infostr)
    #     test_config = experiment_tracker.ExperimentConfig(
    #         t_biattention_ids=t_biattn,
    #         v_biattention_ids=v_biattn,
    #         use_contrastive_loss=False,
    #         epochs=4,
    #         seed=seed,
    #         learning_rate=3.55e-5
    #     )
    #     seed += 1

    #     test_config_mm = experiment_tracker.ExperimentConfig(
    #         t_biattention_ids=t_biattn,
    #         v_biattention_ids=v_biattn,
    #         use_contrastive_loss=False,
    #         epochs=6,
    #         seed=seed,
    #         learning_rate=4e-5
    #     )

    #     seed+=1


    #     res = t.run_finetune(experiment_config=test_config, tasks=["hateful_memes"], run_alignment_analysis=False)
    #     paths.append(res["hateful_memes"]["model_path"])
    #     res2 = t.run_finetune(experiment_config=test_config_mm, tasks=["mm_imdb"], run_alignment_analysis=False)
    #     paths.append(res2["mm_imdb"]["model_path"])

    # print(paths)


if __name__ == "__main__":
    main()
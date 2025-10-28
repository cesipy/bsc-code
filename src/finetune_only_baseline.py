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



logger = Logger()



def main():
    t = experiment_tracker.ExperimentTracker()

    configs = [
        {
            "name": "baseline",
            "t_biattention_ids": [],
            "v_biattention_ids": [],
        },
        {
            "name": "early_fusion",
            "t_biattention_ids": [3, 4, 5],
            "v_biattention_ids": [3, 4, 5],
        },
        {
            "name": "middle_fusion",
            "t_biattention_ids": [6, 7, 8],
            "v_biattention_ids": [6, 7, 8],
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
        {
            "name": "optuna1",
            "t_biattention_ids": [3,6],
            "v_biattention_ids": [6,8],
        },
        {
            "name": "optuna2",
            "t_biattention_ids": [7, 9, 10, 11],
            "v_biattention_ids": [6, 7, 9, 10],
        },
        {
            "name": "baseline_full",
            "t_biattention_ids": [0,1,2,3,4,5,6,7,8,9,10,11],
            "v_biattention_ids": [0,1,2,3,4,5,6,7,8,9,10,11],
        }
    ]

    results = {}

    # tasks = ["mm_imdb", "hateful_memes", "upmc_food"]
    tasks = ["hateful_memes"]       # only train on hm first
    seeds = [1568, 1569, 1570]

    total_runs = len(configs) * len(seeds)* len(tasks)
    c = 1
    for config in configs:
        for task in tasks:
            aucs = []
            accs = []

            for seed in seeds:
                info_str = f"{c:2}/{total_runs}: finetuning on {task} with seed {seed} and coattn placements of {config['t_biattention_ids']} (text) and {config['v_biattention_ids']} (vision)"
                print(info_str); logger.info(info_str)

                e_conf = experiment_tracker.ExperimentConfig(
                    t_biattention_ids=config["t_biattention_ids"],
                    v_biattention_ids=config["v_biattention_ids"],
                    epochs=15,
                    learning_rate=3.5e-5 if task == "hateful_memes" else 4.2e-5,
                    seed=seed,
                    use_contrastive_loss=False
                )

                res = t.run_finetune(experiment_config=e_conf,
                    # run_alignment_analysis=True,
                    # run_visualizations=True,
                    pretrained_model_path=None,
                    tasks=[task]
                )
                c+=1

                final_test_dict = res[task]["final_test"]
                acc = final_test_dict["acc"]
                accs.append(acc)
                # should work for hm!
                if final_test_dict.get("auc") is not None:
                    auc = final_test_dict["auc"]
                    aucs.append(auc)


            if config["name"] not in results:
                results[config["name"]] = {}
            if task not in results[config["name"]]:
                results[config["name"]][task] = {}

            acc_mean, acc_std = np.mean(accs), np.std(accs)
            results[config["name"]][task]["acc"] = {
                "mean": acc_mean,
                "std": acc_std
            }
            if aucs:
                auc_mean, auc_std = np.mean(aucs), np.std(aucs)
                results[config["name"]][task]["auc"] = {
                    "mean": auc_mean,
                    "std": auc_std
                }
                info_str = f"acc_mean: {acc_mean:.4f} (std: {acc_std:.4f}), auc_mean: {auc_mean:.4f} (std: {auc_std:.4f})"
            else:
                info_str = f"acc_mean: {acc_mean:.4f} (std: {acc_std:.4f})"

            print(info_str); logger.info(info_str)
















if __name__ == "__main__":
    main()
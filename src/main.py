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

from analyses import metric_evolution, dim_red
import performance_metric_collection

import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")



logger = Logger()

ft_epochs = 4
pt_epochs = 7


def main():

    path_prefix = "single_plots/dim_red"
    dim_red.main()
    # results = metric_evolution.load_results("metric_evolution/pretrained_early_fusion/intra_modal_results_cka.json")
    # metric_evolution.plot_intra_modal_metrics_single(results,
    #     t_biattn_ids=[3, 4, 5], v_biattn_ids=[3, 4, 5],
    #     save_path=os.path.join(path_prefix, "early_fusion.png")
    # )

    # results = metric_evolution.load_results("metric_evolution/pretrained_middle_fusion/intra_modal_results_cka.json")
    # metric_evolution.plot_intra_modal_metrics_single(results,
    #     t_biattn_ids=[6,7,8], v_biattn_ids=[6,7,8],
    #     save_path=os.path.join(path_prefix, "middle_fusion.png")
    # )

    # results = metric_evolution.load_results("metric_evolution/pretrained_late_fusion/intra_modal_results_cka.json")
    # metric_evolution.plot_intra_modal_metrics_single(results,
    #     t_biattn_ids=[9,10,11], v_biattn_ids=[9,10,11],
    #     save_path=os.path.join(path_prefix, "late_fusion.png")
    # )
    exit()


    t = experiment_tracker.ExperimentTracker()

    configs = [

        # {
        #     "name": "hybrid-1",
        #     "t_biattention_ids": [3, 4, 10],
        #     "v_biattention_ids": [3, 4, 10],
        # },
        # {
        #     "name":  "hybrid-2",
        #     "t_biattention_ids": [4, 10, 11],
        #     "v_biattention_ids": [4, 10, 11],
        # }
        # {
        #     "name": "baseline",
        #     "t_biattention_ids": [],
        #     "v_biattention_ids": [],
        # },
        # {
        #     "name": "early_fusion",
        #     "t_biattention_ids": [3, 4, 5],
        #     "v_biattention_ids": [3, 4, 5],
        # },
        # {
        #     "name": "middle_fusion",
        #     "t_biattention_ids": [6, 7, 8],
        #     "v_biattention_ids": [6, 7, 8],
        # },
        # {
        #     "name": "late_fusion",
        #     "t_biattention_ids": [9, 10, 11],
        #     "v_biattention_ids": [9, 10, 11],
        # },
        # {
        #     "name": "asymmetric_fusion",
        #     "t_biattention_ids": [6, 7, 8, 9],
        #     "v_biattention_ids": [3, 5, 7, 9],
        # },
        # {
        #     "name": "optuna1",
        #     "t_biattention_ids": [3,6],
        #     "v_biattention_ids": [6,8],
        # },
        # {
        #     "name": "optuna2",
        #     "t_biattention_ids": [7, 9, 10, 11],
        #     "v_biattention_ids": [6, 7, 9, 10],
        # },
        # {
        #     "name": "baseline_full",
        #     "t_biattention_ids": [0,1,2,3,4,5,6,7,8,9,10,11],
        #     "v_biattention_ids": [0,1,2,3,4,5,6,7,8,9,10,11],
        # }
        # {
        #     "name": "hybrid_six",
        #     "t_biattention_ids": [3, 4,5, 9, 10,11],
        #     "v_biattention_ids": [3, 4, 5, 9, 10,11],
        # },
        {
           "name": "early_fusion_full",
           "t_biattention_ids": [4,5,6,7,8,9,10,11],
              "v_biattention_ids": [4,5,6,7,8,9,10,11],
        }

    ]


    paths = []
    seed=1567
    for config in configs:
        info_str = f"{'-'*25}\ntraining with coattn placements of {config['t_biattention_ids']} (text) and {config['v_biattention_ids']} (vision)"
        print(info_str); logger.info(info_str)
        pretrain_config = experiment_tracker.ExperimentConfig(
            t_biattention_ids=config["t_biattention_ids"],
            v_biattention_ids=config["v_biattention_ids"],
            use_contrastive_loss=False,
            epochs=pt_epochs,
            learning_rate=1e-4,
            seed=seed,
        )

        results_pretrain = t.run_pretrain(
            experiment_config=pretrain_config,
            # run_alignment_analysis=True,
            # run_visualizations=True,
            num_samples=500_000,
        )
        paths = paths + [results_pretrain["model_path"]]



    print(paths)
    for pretrained_path in paths:
        modl = ViLBERT.load_model(pretrained_path)
        t_biattns = modl.config.text_cross_attention_layers
        v_biattns = modl.config.vision_cross_attention_layers

        tasks = ["hateful_memes", "upmc_food", "mm_imdb"]

        seeds = [1568, 1569, 1570]

        c = 1
        logger.info("test")

        for task in tasks:
            for seed in seeds:
                info_str=f"{c:2}/{len(tasks)*len(seeds)}: finetuning on {task} with seed {seed}"
                print(info_str); logger.info(info_str)
                e_conf = experiment_tracker.ExperimentConfig(
                    t_biattention_ids=t_biattns,
                    v_biattention_ids=v_biattns,
                    epochs=15,
                    learning_rate=3.2e-5 if task == "hateful_memes" else 4e-5,
                    seed=seed,
                    use_contrastive_loss=False
                )
                res = t.run_finetune(experiment_config=e_conf,
                    # run_alignment_analysis=True,
                    # run_visualizations=True,
                    pretrained_model_path=pretrained_path,
                    tasks=[task]
                    )
                path = res[task]["model_path"]
                paths.append(path)
                c+=1


        info_str =f"finished with paths: {paths}"

if __name__ == "__main__":
    main()
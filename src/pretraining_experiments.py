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
seed=1567
ft_epochs = 4
pt_epochs = 6


def main():
    t = experiment_tracker.ExperimentTracker()
    info_str = f"running baseline experiments with no co-attention layers"
    print(info_str); logger.info(info_str)
    baseline = experiment_tracker.ExperimentConfig(
        t_biattention_ids=[],
        v_biattention_ids=[],
        use_contrastive_loss=False,
        epochs=pt_epochs,
        learning_rate=1e-4,
        seed=seed,
    )
    baseline_ft = experiment_tracker.ExperimentConfig(
        t_biattention_ids=[],
        v_biattention_ids=[],
        use_contrastive_loss=False,
        epochs=ft_epochs,
        learning_rate=3e-5,
        seed=seed,
    )

    baseline_results = t.run_pretrain(experiment_config=baseline,
        tiny_fraction=True,
        # num_samples=500_000,
        # run_alignment_analysis=True,
        # run_visualizations=True,
        )

    t.run_finetune(
        experiment_config=baseline_ft,
        # run_visualizations=True,
        # run_alignment_analysis=True,
        tasks=["hateful_memes"],
        pretrained_model_path=baseline_results["model_path"]
    )

    configs = [
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
    ]

    paths = []

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


        # for health checking, proper analysis afterwards
        finetune_config_hm = experiment_tracker.ExperimentConfig(
            t_biattention_ids=config["t_biattention_ids"],
            v_biattention_ids=config["v_biattention_ids"],
            use_contrastive_loss=False,
            epochs=ft_epochs,
            learning_rate=3e-5,
            seed=seed,
        )


        results_pretrain = t.run_pretrain(
            experiment_config=pretrain_config,
            run_alignment_analysis=True,
            run_visualizations=True,
            num_samples=500_000,
        )

        results_hm = t.run_finetune(
            experiment_config=finetune_config_hm,
            run_visualizations=False,
            run_alignment_analysis=False,
            tasks=["hateful_memes"],
            pretrained_model_path=results_pretrain["model_path"]
        )
        # results_mm_imdb = t.run_finetune(
        #     experiment_config=finetune_config_mm_imdb,
        #     run_visualizations=False,
        #     run_alignment_analysis=False,
        #     tasks=["mm_imdb"],
        # )

        paths = paths + [results_pretrain["model_path"]]



    print(paths)

if __name__ == "__main__":
    main()
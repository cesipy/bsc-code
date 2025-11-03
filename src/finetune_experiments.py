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
import performance_metric_collection

import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")



logger = Logger()
seeds = [1568, 1569, 1570]
ft_epochs = 4
pt_epochs = 7


def main():
    t = experiment_tracker.ExperimentTracker()
    # pretrained_path = "res/checkpoints/pretrains/20251010-234252_pretrained_early_fusion.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251011-234349_pretrained_middle_fusion.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251013-010227_pretrained_late_fusion.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251014-034432_pretrained_asymmetric_fusion.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251015-081211_pretrained_optuna1.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251016-062038_pretrained_optuna2.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251010-085859_pretrained_baseline.pt"
    # pretrained_path = "res/checkpoints/pretrains/20251025-105249_pretrained_bl_full_coattn.pt"
    pretrained_path = "res/checkpoints/pretrains/20251030-192145_pretrained_latefusion_cka.pt"

    modl = ViLBERT.load_model(pretrained_path)
    t_biattns = modl.config.text_cross_attention_layers
    v_biattns = modl.config.vision_cross_attention_layers


    tasks = ["mm_imdb", "hateful_memes", "upmc_food"]
    paths = []
    c = 1

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
                run_alignment_analysis=True,
                run_visualizations=True,
                pretrained_model_path=pretrained_path,
                tasks=[task]
                )
            path = res[task]["model_path"]
            paths.append(path)
            c+=1


    info_str =f"finished with paths: {paths}"

    performance_metric_collection.get_performance_metrics(dirs=["res/checkpoints/20251030-192145_pretrained_latefusion_cka"])




if __name__ == "__main__":
    main()
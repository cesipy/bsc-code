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



seed=1567


def main():
    t = experiment_tracker.ExperimentTracker()

    pt_config_late_fusion = experiment_tracker.ExperimentConfig(
        t_biattention_ids=[9,10,11],
        v_biattention_ids=[9,10,11],
        use_contrastive_loss=False,
        seed=seed,
        learning_rate=3.2e-5,
        epochs=8,
    )

    t.run_finetune(
        experiment_config=pt_config_late_fusion,
        # run_alignment_analysis=True,
        # run_visualizations=True,
        tasks=["hateful_memes"],
        pretrained_model_path="res/checkpoints/pretrains/20251030-192145_pretrained_latefusion_cka.pt"
    )












if __name__ == "__main__":
    main()
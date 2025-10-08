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







def main():
    t = experiment_tracker.ExperimentTracker()
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




if __name__ == "__main__":
    main()
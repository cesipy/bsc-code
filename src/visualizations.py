import os

import numpy as np
import torch

from experiment_tracker import ExperimentTracker, ExperimentConfig

import analysis
import datasets
from config import *
import task as tasklib
import vilbert

def get_paths_per_task(dir:str, task:str):
    assert task in tasklib.all_task_list

    return [
       os.path.join(dir, filename) for filename in os.listdir(dir) if task in filename and filename.endswith(".pt")
    ]

def get_task(path) -> str:
    for task in tasklib.all_task_list:
        if task in path:
            return task
    assert False





def main():
    t = ExperimentTracker()
    dirs = [
        "res/checkpoints/20251010-085859_pretrained_baseline",
        "res/checkpoints/20251010-234252_pretrained_early_fusion",
        "res/checkpoints/20251011-234349_pretrained_middle_fusion",
        "res/checkpoints/20251013-010227_pretrained_late_fusion",
        "res/checkpoints/20251014-034432_pretrained_asymmetric_fusion",
        "res/checkpoints/20251015-081211_pretrained_optuna1",
        "res/checkpoints/20251016-062038_pretrained_optuna2",
        "res/checkpoints/20251025-105249_pretrained_bl_full_coattn",
    ]

    task = "hateful_memes"
    for dir in dirs:
        for task in ["hateful_memes", "mm_imdb", "upmc_food"]:
            paths = get_paths_per_task(dir, task)

            for i, path in enumerate(paths):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = vilbert.ViLBERT.load_model(path, device=device)
                config = model.config
                
                dl = t.get_task_alignment_dataloader(
                    task=task,
                    config=config)

                dirname = os.path.join("visualizations_test", f"{'_'.join(dir.split('/')[-1].split('_')[1:])}_{task}/{i}")
                os.makedirs(dirname, exist_ok=True)
                analysis.run_alignment_visualization(dl, model,dir_name=dirname, filename_extension="latefusion_hm", k=10)







if __name__ == "__main__":
    main()
import os

import numpy as np
import torch

from  experiment_tracker import ExperimentTracker, ExperimentConfig

import analysis
import datasets
from config import *
import task as tasklib
import vilbert
import metrics


import matplotlib.pyplot as plt

def plot_intra_modal_metrics(results, save_path=None):
    """Plot text-text and vision-vision CKA metrics for all tasks side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tasks = ["hateful_memes", "mm_imdb", "upmc_food"]
    task_titles = ["Hateful Memes", "MM-IMDB", "UPMC Food"]

    for ax, task, title in zip(axes, tasks, task_titles):
        tt_means, vv_means, tt_stds, vv_stds = results[task]

        x = range(len(tt_means))
        tt_means = np.array(tt_means)
        vv_means = np.array(vv_means)
        tt_stds = np.array(tt_stds)
        vv_stds = np.array(vv_stds)

        ax.plot(x, tt_means, marker='o', label='Text-Text CKA', linewidth=2, color='C0')
        ax.fill_between(x, tt_means - tt_stds, tt_means + tt_stds, alpha=0.3, color='C0')

        ax.plot(x, vv_means, marker='s', label='Vision-Vision CKA', linewidth=2, color='C1')
        ax.fill_between(x, vv_means - vv_stds, vv_means + vv_stds, alpha=0.3, color='C1')

        ax.set_xlabel('Layer Transition', fontsize=12)
        ax.set_ylabel('CKA Similarity', fontsize=12)
        ax.set_title(title, fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")

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

def compute_similiarity(embeds1, embeds2, metric="cka"):
    res = None
    if metric == "cka":
        res = metrics.AlignmentMetrics.cka(embeds1, embeds2)

    return  res


def intra_modal_analysis(data: tuple[dict]):

    tt_means = []
    tt_stds  = []
    vv_means = []
    vv_stds  = []
    for i in range(0, len(data[0].keys())-1):
        buffer_tt = []
        buffer_vv = []
        for j in range(len(data)):

            text_embeds   = data[j][i]["text_embeddings"]
            vision_embeds = data[j][i]["vision_embeddings"]

            text_embeds_comp = data[j][i+1]["text_embeddings"]
            vision_embeds_comp = data[j][i+1]["vision_embeddings"]

            buffer_tt.append(compute_similiarity(text_embeds, text_embeds_comp, metric="cka"))
            buffer_vv.append(compute_similiarity(vision_embeds, vision_embeds_comp, metric="cka"))
        tt_means.append(np.mean(buffer_tt))
        tt_stds.append(np.std(buffer_tt))
        vv_means.append(np.mean(buffer_vv))
        vv_stds.append(np.std(buffer_vv))

    return tt_means, vv_means, tt_stds, vv_stds






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
        print("_".join((dir.split("/")[-1].split("_")[2:])))
        results = {}
        for task in ["hateful_memes", "mm_imdb", "upmc_food"]:
            paths = get_paths_per_task(dir, task)
            current_data = []
            for i, path in enumerate(paths):
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = vilbert.ViLBERT.load_model(path, device=device)
                config = model.config

                dl = t.get_task_alignment_dataloader(
                    task=task,
                    config=config)

                dirname = os.path.join("metric_evolution", f"{'_'.join(dir.split('/')[-1].split('_')[1:])}")
                os.makedirs(dirname, exist_ok=True)
                data = analysis.get_alignment_data(dl, model, device)
                current_data.append(data)


            tt_means, vv_means, tt_stds, vv_stds = intra_modal_analysis(current_data)
            results[task] = (tt_means, vv_means, tt_stds, vv_stds)



        plot_intra_modal_metrics(
            results,
            save_path=os.path.join(dirname, "intra_modal_cka.png")
        )








if __name__ == "__main__":
    main()
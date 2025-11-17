import os

import numpy as np
import torch
import matplotlib.pyplot as plt; import json

from  experiment_tracker import ExperimentTracker, ExperimentConfig

import analysis
import datasets
from config import *
import task as tasklib
import vilbert
import metrics


def plot_intra_modal_metrics(results, t_biattn_ids, v_biattn_ids, save_path=None):
    """Plot text-text and vision-vision CKA metrics for all tasks side by side"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    tasks = ["hateful_memes", "mm_imdb", "upmc_food"]
    task_titles = ["Hateful Memes", "MM-IMDB", "UPMC Food"]
    # magma colors, for spicy effect
    colors = {"text": "#FCCE25", "vision": "#A52C60"}

    for ax, task, title in zip(axes, tasks, task_titles):
        tt_means, vv_means, tt_stds, vv_stds = results[task]

        x = np.arange(len(tt_means)) + 1  # shift x axix by one
        tt_means = np.array(tt_means)
        vv_means = np.array(vv_means)
        tt_stds = np.array(tt_stds)
        vv_stds = np.array(vv_stds)

        # coattns - assuming symmetric
        for layer_id in t_biattn_ids:
            ax.axvspan(layer_id +  - 0.3, layer_id +  + 0.3, color='black', alpha=0.15)

        ax.plot(x, tt_means, marker='o', label='Text-Text CKA',
                linewidth=2.5, color=colors["text"], markersize=6)
        ax.fill_between(x, tt_means - tt_stds, tt_means + tt_stds,
                        alpha=0.3, color=colors["text"])

        ax.plot(x, vv_means, marker='o', label='Vision-Vision CKA',
                linewidth=2.5, color=colors["vision"], markersize=6)
        ax.fill_between(x, vv_means - vv_stds, vv_means + vv_stds,
                        alpha=0.3, color=colors["vision"])

        ax.set_xlabel('Layer', fontsize=16, fontweight='bold')
        ax.set_ylabel('CKA Sim.', fontsize=16, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=13, loc='best')
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=13)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    plt.close()


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



def save_results(results: dict, path):
    serializable_results = {}
    for task, (tt_means, vv_means, tt_stds, vv_stds) in results.items():
        serializable_results[task] = {
            "tt_means": [float(x) for x in tt_means],
            "vv_means": [float(x) for x in vv_means],
            "tt_stds": [float(x) for x in tt_stds],
            "vv_stds": [float(x) for x in vv_stds]
        }

    with open(path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {path}")


def load_results( path) -> dict:
    with open(path, "r") as f:
        content = json.load(f)

    results = {
    }
    for t, k in content.items():
        results[t] = (
            content[t]["tt_means"],
            content[t]["vv_means"],
            content[t]["tt_stds"],
            content[t]["vv_stds"],
        )
    return results

def main(dirs):
    t = ExperimentTracker()


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
                t_biattn_ids = model.t_biattention_ids
                v_biattn_ids = model.v_biattention_ids

                dl = t.get_task_alignment_dataloader(
                    task=task,
                    config=config)

                dirname = os.path.join("metric_evolution", f"{'_'.join(dir.split('/')[-1].split('_')[1:])}")
                os.makedirs(dirname, exist_ok=True)
                data = analysis.get_alignment_data(dl, model, device)
                current_data.append(data)


            tt_means, vv_means, tt_stds, vv_stds = intra_modal_analysis(current_data)
            results[task] = (tt_means, vv_means, tt_stds, vv_stds)

        save_results(results, os.path.join(dirname, "intra_modal_results.json"))
        plot_intra_modal_metrics(
            results,
            t_biattn_ids,
            v_biattn_ids,
            save_path=os.path.join(dirname, "intra_modal_cka.png")
        )





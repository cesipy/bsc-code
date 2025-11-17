import os, json

import numpy as np
import torch

from  experiment_tracker import ExperimentTracker, ExperimentConfig

import analysis
import datasets
from config import *
import task as tasklib
import vilbert
import metrics

from sklearn.decomposition import PCA
from skdim.id import TwoNN
import matplotlib.pyplot as plt

def plot_dim_reduction(results_dict: dict, model_name, t_biattn_ids, v_biattn_ids, save_path):
    """
    results_dict: {task: {modality: {layer: mean}, modality_stds: {layer: std}, ...}}
    Creates one figure with subplots for each task, both modalities together
    Includes shaded std regions around lines
    """
    tasks = sorted(results_dict.keys())
    fig, axes = plt.subplots(1, len(tasks), figsize=(6*len(tasks), 5))

    if len(tasks) == 1:
        axes = [axes]  # ensure iterable


    # magma colors
    colors = {"text": "#FCCE25", "vision": "#A52C60"}

    for task_idx, task in enumerate(tasks):
        ax = axes[task_idx]
        task_data = results_dict[task]

        # coattentn- assumes symmetric placement
        for layer_id in t_biattn_ids:
            ax.axvspan(layer_id + 1 - 0.3, layer_id + 1 + 0.3, color="black", alpha=0.15)

        for modality_name in ["text", "vision"]:
            modality_means = task_data[modality_name]
            modality_stds = task_data[f"{modality_name}_stds"]

            layers = np.array(sorted(modality_means.keys())) + 1  # Shift by +1
            dims = np.array([modality_means[l - 1] for l in layers])  # Access original keys
            stds = np.array([modality_stds[l - 1] for l in layers])

            ax.plot(layers, dims, marker='o', label=modality_name.upper(),
                   linewidth=2.5, color=colors[modality_name], markersize=6)
            ax.fill_between(layers, dims - stds, dims + stds,
                           alpha=0.3, color=colors[modality_name])

        ax.set_xlabel("Layer", fontsize=16, fontweight='bold')
        ax.set_ylabel("Eff. Dim.", fontsize=16, fontweight='bold')
        ax.set_title(task.replace("_", " ").title(), fontsize=18, fontweight='bold')
        ax.legend(fontsize=13, loc='best')
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=13)

    # fig.suptitle(f"Model: {model_name}", fontsize=20, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
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


def dim_red_data(data: list):
    """
    data: list of data dicts from multiple runs
    Returns: dict with means and stds per layer
    """
    results = {"text": {}, "vision": {},
               "text_stds": {}, "vision_stds": {}}
    n_layers = len(data[0].keys())

    for i in range(n_layers):
        buffer_text = []
        buffer_vision = []

        for data_point in data:
            text_embeds = data_point[i]["text_embeddings"]
            vision_embeds = data_point[i]["vision_embeddings"]

            pca_text = PCA(n_components=0.95)
            pca_vision = PCA(n_components=0.95)
            pca_text.fit(text_embeds)
            pca_vision.fit(vision_embeds)
            values_t = pca_text.n_components_
            values_v = pca_vision.n_components_

            # unfortunately it did not work here, some numerical instabilities
            # twonn_t = TwoNN()
            # twonn_v = TwoNN()
            # values_t = twonn_t.fit_transform(text_embeds)
            # values_v = twonn_v.fit_transform(vision_embeds)

            buffer_text.append(values_t)
            buffer_vision.append(values_v)

        results["text"][i] = np.mean(buffer_text)
        results["vision"][i] = np.mean(buffer_vision)
        results["text_stds"][i] = np.std(buffer_text)
        results["vision_stds"][i] = np.std(buffer_vision)

        print(f"    layer{i}: text={results['text'][i]:.1f}±{results['text_stds'][i]:.2f} | vision={results['vision'][i]:.1f}±{results['vision_stds'][i]:.2f}")

    return results


def save_results(results: dict, path):
    ser_res = {}
    for task, data in results.items():
        ser_res[task] = {}
        for key, val in data.items():
            ser_res[task][key] = {str(k): float(v) for k, v in val.items()}

    with open(path,"w")as f:
        f.write( json.dumps(ser_res, indent=2) )

def main(dirs):
    t = ExperimentTracker()


    for dir in dirs:
        model_name = "_".join(dir.split("/")[-1].split("_")[2:])
        print(f"\n{model_name}")
        results = {}

        for task in ["hateful_memes", "mm_imdb", "upmc_food"]:
            paths = get_paths_per_task(dir, task)
            current_data = []

            for path in paths:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                model = vilbert.ViLBERT.load_model(path, device=device)
                config = model.config

                dl = t.get_task_alignment_dataloader(task=task, config=config)
                data = analysis.get_alignment_data(dl, model, device)
                current_data.append(data)

            print(f"  {task}:")
            results[task] = dim_red_data(current_data)


        dirname = os.path.join("dim_red", model_name)
        os.makedirs(dirname, exist_ok=True)
        save_path = os.path.join(dirname, "dimension_reduction.png")
        save_results(results, os.path.join(dirname, "dim_red_results.json"))
        plot_dim_reduction(results, model_name,
            t_biattn_ids=model.t_biattention_ids,
            v_biattn_ids=model.v_biattention_ids,
            save_path=save_path)



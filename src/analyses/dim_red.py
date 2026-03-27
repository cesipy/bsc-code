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


def plot_dim_reduction_single(results_dict: dict, model_name, t_biattn_ids, v_biattn_ids, save_path):
    """
    results_dict: {task: {modality: {layer: mean}, modality_stds: {layer: std}, ...}}
    Creates one plot for hateful_memes with both modalities
    Includes shaded std regions around lines
    """
    task = "hateful_memes"  # Changed from list
    title = "Hateful Memes"  # Changed from list
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    colors = {"text": "#FCCE25", "vision": "#A52C60"}
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
    ax.set_title(title, fontsize=18, fontweight='bold')  # Changed from task_titles[task_idx]
    ax.legend(fontsize=13, loc='best')
    ax.grid(alpha=0.3)
    ax.tick_params(labelsize=13)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved: {save_path}")
    plt.close()

def plot_dim_reduction(results_dict: dict, model_name, t_biattn_ids, v_biattn_ids, save_path):
    """
    results_dict: {task: {modality: {layer: mean}, modality_stds: {layer: std}, ...}}
    Creates one figure with subplots for each task, both modalities together
    Includes shaded std regions around lines
    """
    tasks = ["hateful_memes", "mm_imdb", "upmc_food"]
    task_titles = ["Hateful Memes", "MM-IMDB", "UPMC Food"]
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
        ax.set_title(task_titles[task_idx], fontsize=18, fontweight='bold')
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


# this function plots the singular value spectrum to analyze the tail behavior...
def plot_spectral_decay(text_embeds, vision_embeds, layer_idx, task_name, save_path=None):
    plt.figure(figsize=(10, 6))

    # helper to process embeddings...
    def get_spectrum(embeds):
        # center the embeddings...
        embeds = embeds - embeds.mean(axis=0)
        # compute singular values using torch for speed...
        if isinstance(embeds, np.ndarray):
            embeds = torch.from_numpy(embeds)

        _, S, _ = torch.linalg.svd(embeds, full_matrices=False)
        return S.cpu().numpy()

    s_text = get_spectrum(text_embeds)
    s_vision = get_spectrum(vision_embeds)

    # plot log singular values...
    plt.plot(np.log(s_text), label='TEXT Spectrum', color="#FCCE25", linewidth=2)
    plt.plot(np.log(s_vision), label='VISION Spectrum', color="#A52C60", linewidth=2)

    # mark the 95% threshold to show where your previous metric cut off...
    def get_cutoff_index(S, threshold=0.95):
        eigenvalues = S ** 2
        explained = eigenvalues / eigenvalues.sum()
        cumulative = np.cumsum(explained)
        return np.argmax(cumulative > threshold)

    cut_t = get_cutoff_index(s_text)
    cut_v = get_cutoff_index(s_vision)

    plt.axvline(cut_t, color="#FCCE25", linestyle='--', alpha=0.5, label=f'Text 95% Cutoff ({cut_t})')
    plt.axvline(cut_v, color="#A52C60", linestyle='--', alpha=0.5, label=f'Vision 95% Cutoff ({cut_v})')

    plt.xlabel('Singular Value Index')
    plt.ylabel('Log Magnitude (Singular Values)')
    plt.title(f'Spectral Decay: {task_name} - Layer {layer_idx}')
    plt.legend()
    plt.grid(alpha=0.3)

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved spectral plot: {save_path}")
    plt.close()

def save_results(results: dict, path):
    ser_res = {}
    for task, data in results.items():
        ser_res[task] = {}
        for key, val in data.items():
            ser_res[task][key] = {str(k): float(v) for k, v in val.items()}

    with open(path,"w")as f:
        f.write( json.dumps(ser_res, indent=2) )

def load_results(path):
    with open(path,"r")as f:
        ser_res = json.loads(f.read())

    results = {}
    for task, data in ser_res.items():
        results[task] = {}
        for key, val in data.items():
            results[task][key] = {int(k): float(v) for k, v in val.items()}

    return results

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

            if len(current_data) > 0:
                layer_to_inspect = 6

                # aggregate data from all runs for this layer...
                text_all = np.concatenate([run[layer_to_inspect]["text_embeddings"] for run in current_data], axis=0)
                vision_all = np.concatenate([run[layer_to_inspect]["vision_embeddings"] for run in current_data], axis=0)

                spec_save_path = os.path.join("dim_red", model_name, f"spectral_decay_{task}_L{layer_to_inspect}.png")
                plot_spectral_decay(text_all, vision_all, layer_to_inspect, task, spec_save_path)


        dirname = os.path.join("dim_red", model_name)
        os.makedirs(dirname, exist_ok=True)
        save_path = os.path.join(dirname, "dimension_reduction.png")
        save_results(results, os.path.join(dirname, "dim_red_results.json"))
        plot_dim_reduction(results, model_name,
            t_biattn_ids=model.t_biattention_ids,
            v_biattn_ids=model.v_biattention_ids,
            save_path=save_path)



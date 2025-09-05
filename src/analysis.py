import time

import torch; from torch import nn; from torch.utils.data import DataLoader
from ckatorch.core import cka_batch, cka_base
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tqdm import tqdm


import cca_core
from vilbert import ViLBERT
from config import *
import utils
from logger import Logger

logger = Logger()


def jaccard_similarity(
    X: torch.Tensor,        # [n, d]
    Y: torch.Tensor,
    k: int = 10,
):
    n = X.shape[0]
    assert Y.shape[0] == n

    knns1 = pairwise_knn(X, k)   # [n, k]
    knns2 = pairwise_knn(Y, k)   # [n, k]

    sims = []

    for i in range(n):
        nearest_neighbors1 = set(knns1[i].cpu().numpy())
        nearest_neighbors2 = set(knns2[i].cpu().numpy())

        denom = len(nearest_neighbors1.intersection(nearest_neighbors2))
        nom = len(nearest_neighbors1.union(nearest_neighbors2))

        sim = 0.0
        if nom > 0:
            sim = denom / nom

        sims.append(sim)

    avg = sum(sims) / len(sims)
    return avg



def _visualize_jaccard(measure_per_layer: dict, num_layers: int, k: int = 10):

    jaccard_cross_modal = np.zeros((num_layers, num_layers))
    jaccard_text_text = np.zeros((num_layers, num_layers))
    jaccard_vision_vision = np.zeros((num_layers, num_layers))



    for i in tqdm(range(num_layers), leave=False, desc="computing neighborhood measures"):
        for j in range(num_layers):
            # Cross-modal (text layer i CLS vs vision layer j CLS)
            text_cls = measure_per_layer[i]["text_embeddings"][:, 0, :]
            vision_cls = measure_per_layer[j]["vision_embeddings"][:, 0, :]

            jaccard_cross_modal[i,j] = jaccard_similarity(
                X=text_cls, Y=vision_cls, k=k
            )
            # text2text
            text_cls_i = measure_per_layer[i]["text_embeddings"][:, 0, :]
            text_cls_j = measure_per_layer[j]["text_embeddings"][:, 0, :]

            jaccard_text_text[i,j] = jaccard_similarity(
                X=text_cls_i, Y=text_cls_j, k=k
            )

            # vis2vis
            vision_cls_i = measure_per_layer[i]["vision_embeddings"][:, 0, :]
            vision_cls_j = measure_per_layer[j]["vision_embeddings"][:, 0, :]

            jaccard_vision_vision[i,j] = jaccard_similarity(
                X=vision_cls_i, Y=vision_cls_j, k=k
            )

    # all the plotting from genai
    fig1, axes1 = plt.subplots(1, 3, figsize=(18, 5))


    # k-NN Jaccard Plot
    im1 = axes1[0].imshow(jaccard_cross_modal, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes1[0].set_title(f'Cross-Modal k-NN Jaccard (k={k})', fontsize=14, pad=20)
    axes1[0].set_xlabel('Vision Layer'); axes1[0].set_ylabel('Text Layer')

    im2 = axes1[1].imshow(jaccard_text_text, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes1[1].set_title(f'Text k-NN Jaccard (k={k})', fontsize=14, pad=20)
    axes1[1].set_xlabel('Text Layer'); axes1[1].set_ylabel('Text Layer')

    im3 = axes1[2].imshow(jaccard_vision_vision, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes1[2].set_title(f'Vision k-NN Jaccard (k={k})', fontsize=14, pad=20)
    axes1[2].set_xlabel('Vision Layer'); axes1[2].set_ylabel('Vision Layer')



    # Add ticks and colorbars
    for ax in axes1:
        ax.set_xticks(range(num_layers)); ax.set_yticks(range(num_layers))
        ax.set_xticklabels(range(num_layers)); ax.set_yticklabels(range(num_layers))


    plt.colorbar(im1, ax=axes1[0], shrink=0.8, label='Jaccard Similarity')
    plt.colorbar(im2, ax=axes1[1], shrink=0.8)
    plt.colorbar(im3, ax=axes1[2], shrink=0.8)



    plt.tight_layout()

    # Save plots
    timestamp = int(time.time())
    fig1.savefig(f"res/plots/jaccard_matrices_{timestamp}.png", dpi=300, bbox_inches='tight', facecolor='white')


    plt.show()

    return (jaccard_cross_modal, jaccard_text_text, jaccard_vision_vision,)


def _visualize_cka(measure_per_layer: dict, num_layers: int):

    cross_modal_matrix = np.zeros((num_layers, num_layers))
    text_text_matrix = np.zeros((num_layers, num_layers))
    vision_vision_matrix = np.zeros((num_layers, num_layers))



    for i in tqdm(range(num_layers), leave=False, desc="computing cka matrix"):
        for j in range(num_layers):
            current_text = measure_per_layer[i]["text_embeddings"]
            current_vision = measure_per_layer[j]["vision_embeddings"]
            cross_modal_matrix[i,j] = cka(
                text_embedding=current_text,
                vision_embedding=current_vision
            )
            print(f"layer {i}-{j}, cka: {cross_modal_matrix[i,j]:.4f}")
            # weird results, not sure if this impl is correct at all...
            # cross_modal_matrix[i,j] = cka_base(
            #     x=current_text.reshape(current_text.shape[0], -1),        #[batch_size, num_tokens * embedding_dim]
            #     y=current_vision.reshape(current_vision.shape[0], -1),    #[batch_size, num_patches * embedding_dim]
            #     # x=current_text.reshape(-1, current_text.shape[-1]),    # [batch*tokens, 768]
            #     # y=current_vision.reshape(-1, current_vision.shape[-1]), # [batch*patches, 768]
            #     kernel="rbf",
            #     method="hsic"
            # ).item()

            text_i = measure_per_layer[i]["text_embeddings"]
            text_j = measure_per_layer[j]["text_embeddings"]
            text_text_matrix[i,j] = cka(
                text_embedding=text_i,
                vision_embedding=text_j
            )
            # text_text_matrix[i,j] = cka_base(
            #     x=text_i.reshape(text_i.shape[0], -1),
            #     y=text_j.reshape(text_j.shape[0], -1),
            #     kernel="rbf",
            #     method="hsic")

            vision_i = measure_per_layer[i]["vision_embeddings"]
            vision_j = measure_per_layer[j]["vision_embeddings"]
            vision_vision_matrix[i,j] = cka(
                text_embedding=vision_i,
                vision_embedding=vision_j
            )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(cross_modal_matrix, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[0].set_xticks(range(num_layers))
    axes[0].set_yticks(range(num_layers))
    axes[0].set_xticklabels(range(num_layers))
    axes[0].set_yticklabels(range(num_layers))
    axes[0].set_xlabel('Vision Layer', fontsize=12)
    axes[0].set_ylabel('Text Layer', fontsize=12)
    axes[0].set_title('Cross-Modal CKA', fontsize=14, pad=20)
    axes[0].tick_params(labelsize=10)

    im2 = axes[1].imshow(text_text_matrix, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[1].set_xticks(range(num_layers))
    axes[1].set_yticks(range(num_layers))
    axes[1].set_xticklabels(range(num_layers))
    axes[1].set_yticklabels(range(num_layers))
    axes[1].set_xlabel('Text Layer', fontsize=12)
    axes[1].set_ylabel('Text Layer', fontsize=12)
    axes[1].set_title('Text Layer-to-Layer CKA', fontsize=14, pad=20)
    axes[1].tick_params(labelsize=10)

    im3 = axes[2].imshow(vision_vision_matrix, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[2].set_xticks(range(num_layers))
    axes[2].set_yticks(range(num_layers))
    axes[2].set_xticklabels(range(num_layers))
    axes[2].set_yticklabels(range(num_layers))
    axes[2].set_xlabel('Vision Layer', fontsize=12)
    axes[2].set_ylabel('Vision Layer', fontsize=12)
    axes[2].set_title('Vision Layer-to-Layer CKA', fontsize=14, pad=20)
    axes[2].tick_params(labelsize=10)

    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('CKA (Linear)', rotation=270, labelpad=15)

    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('CKA (Linear)', rotation=270, labelpad=15)

    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('CKA (Linear)', rotation=270, labelpad=15)

    plt.tight_layout()

    timestamp = int(time.time())
    filename = f"res/plots/all_cka_matrices_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')


    plt.show()

    return cross_modal_matrix, text_text_matrix, vision_vision_matrix

def _visualize_mutual_knn(measure_per_layer: dict, num_layers: int, k: int = 10):

    cross_modal_matrix = np.zeros((num_layers, num_layers))
    text_text_matrix = np.zeros((num_layers, num_layers))
    vision_vision_matrix = np.zeros((num_layers, num_layers))

    for i in tqdm(range(num_layers), leave=False, desc="computing mknn matrix"):
        for j in range(num_layers):
            # Cross-modal (text layer i CLS vs vision layer j CLS)
            text_cls = measure_per_layer[i]["text_embeddings"][:, 0, :]
            vision_cls = measure_per_layer[j]["vision_embeddings"][:, 0, :]
            cross_modal_matrix[i,j] = mutual_knn_alignment_gpu_advanced(
                Z1=text_cls, Z2=vision_cls, k=k
            )

            # Text-to-text
            text_cls_i = measure_per_layer[i]["text_embeddings"][:, 0, :]
            text_cls_j = measure_per_layer[j]["text_embeddings"][:, 0, :]
            text_text_matrix[i,j] = mutual_knn_alignment_gpu_advanced(
                Z1=text_cls_i, Z2=text_cls_j, k=k
            )

            # Vision-to-vision
            vision_cls_i = measure_per_layer[i]["vision_embeddings"][:, 0, :]
            vision_cls_j = measure_per_layer[j]["vision_embeddings"][:, 0, :]
            vision_vision_matrix[i,j] = mutual_knn_alignment_gpu_advanced(
                Z1=vision_cls_i, Z2=vision_cls_j, k=k
            )

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    im1 = axes[0].imshow(cross_modal_matrix, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[0].set_xticks(range(num_layers))
    axes[0].set_yticks(range(num_layers))
    axes[0].set_xticklabels(range(num_layers))
    axes[0].set_yticklabels(range(num_layers))
    axes[0].set_xlabel('Vision Layer', fontsize=12)
    axes[0].set_ylabel('Text Layer', fontsize=12)
    axes[0].set_title(f'Cross-Modal Mutual k-NN (k={k})', fontsize=14, pad=20)
    axes[0].tick_params(labelsize=10)

    im2 = axes[1].imshow(text_text_matrix, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[1].set_xticks(range(num_layers))
    axes[1].set_yticks(range(num_layers))
    axes[1].set_xticklabels(range(num_layers))
    axes[1].set_yticklabels(range(num_layers))
    axes[1].set_xlabel('Text Layer', fontsize=12)
    axes[1].set_ylabel('Text Layer', fontsize=12)
    axes[1].set_title(f'Text Mutual k-NN (k={k})', fontsize=14, pad=20)
    axes[1].tick_params(labelsize=10)

    im3 = axes[2].imshow(vision_vision_matrix, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[2].set_xticks(range(num_layers))
    axes[2].set_yticks(range(num_layers))
    axes[2].set_xticklabels(range(num_layers))
    axes[2].set_yticklabels(range(num_layers))
    axes[2].set_xlabel('Vision Layer', fontsize=12)
    axes[2].set_ylabel('Vision Layer', fontsize=12)
    axes[2].set_title(f'Vision Mutual k-NN (k={k})', fontsize=14, pad=20)
    axes[2].tick_params(labelsize=10)

    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
    cbar1.set_label('Mutual k-NN', rotation=270, labelpad=15)

    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
    cbar2.set_label('Mutual k-NN', rotation=270, labelpad=15)

    cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
    cbar3.set_label('Mutual k-NN', rotation=270, labelpad=15)

    plt.tight_layout()

    timestamp = int(time.time())
    filename = f"res/plots/mutual_knn_matrices_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

    plt.show()

    return cross_modal_matrix, text_text_matrix, vision_vision_matrix


def get_visualisation_data(dataloader: DataLoader, model: ViLBERT):

    model.eval()
    with torch.no_grad():
        torch.cuda.empty_cache()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    measure_per_layer = {}
    for i in range(model.depth):
        measure_per_layer[i] = {
            "text_embeddings": [],
            "vision_embeddings": [],
            "is_cross_attention": None,
            "layer": i
        }

    with torch.no_grad():
        for batch in dataloader:

            text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
            label = batch["label"].to(device)

            # print(f"img shape: {image['pixel_values'].shape}, ")

            preds, intermediate_representations = model(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                save_intermediate_representations=True
            )

            #generate dummy reprs
            # intermediate_representations = [
            #     {
            #         "text_embedding": torch.randn(16, 197, 768),
            #         "vision_embedding": torch.randn(16, 197, 768),
            #         "is_cross_attention": i in [0, 2],
            #         "layer": i
            #     } for i in range(4)
            # ]

            # shape: [bs, num_tokens, dim]

            for repr_dict in intermediate_representations:
                repr_dict["text_embedding"] = repr_dict["text_embedding"].detach().cpu()
                repr_dict["vision_embedding"] = repr_dict["vision_embedding"].detach().cpu()


            for repr_dict in intermediate_representations:
                layer = repr_dict["layer"]
                measure_per_layer[layer]["text_embeddings"].append(repr_dict["text_embedding"])
                measure_per_layer[layer]["vision_embeddings"].append(repr_dict["vision_embedding"])
                measure_per_layer[layer]["is_cross_attention"] = repr_dict["is_cross_attention"]

            del intermediate_representations, text, image, preds

            if len(measure_per_layer[0]["text_embeddings"]) > 20:
                torch.cuda.empty_cache()
                break

    for layer in measure_per_layer.keys():
        measure_per_layer[layer]["text_embeddings"] = torch.cat(measure_per_layer[layer]["text_embeddings"], dim=0)
        measure_per_layer[layer]["vision_embeddings"] = torch.cat(measure_per_layer[layer]["vision_embeddings"], dim=0)
        # print(f"layer {layer}, text shape: {measure_per_layer[layer]['text_embeddings'].shape}, vision shape: {measure_per_layer[layer]['vision_embeddings'].shape}")

    print(f"shape of collected visualization data: {measure_per_layer[0]['text_embeddings'].shape}, {measure_per_layer[0]['vision_embeddings'].shape}")
    model.train()
    return measure_per_layer        # dict

def visualize_cka(
    dataloader: DataLoader,
    model: ViLBERT
    ):

    measures_per_layer: dict = get_visualisation_data(dataloader, model)
    _visualize_jaccard(measures_per_layer, model.depth, k=10)
    _visualize_cka(measures_per_layer, model.depth)
    _visualize_mutual_knn(measures_per_layer, model.depth, k=10)

def analyse_alignment(dataloader: DataLoader, model: ViLBERT):
    model.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = {}
    for i in range(model.depth):
        layers[i] = {
            "text_embeddings": [],
            "vision_embeddings": [],
            "is_cross_attention": i in model.cross_attention_layers,
            "layer": i
        }

    for i, batch in enumerate(dataloader):
        text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
        image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
        label = batch["label"].to(device)




        with torch.no_grad():
            preds, intermediate_representations =model.forward(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                save_intermediate_representations=True
            )

            for repr_dict in intermediate_representations:
                layer = repr_dict["layer"]
                cls_text = repr_dict["text_embedding"][:,0,:].detach().cpu()   # [bs, dim]
                cls_vision = repr_dict["vision_embedding"][:,0,:].detach().cpu() # [bs, dim]
                layers[layer]["text_embeddings"].append(cls_text)
                layers[layer]["vision_embeddings"].append(cls_vision)

            del intermediate_representations
            del text
            del image
            del preds



    mknn_values = {}

    for i in range(model.depth):
        layers[i]["text_embeddings"] = torch.cat(layers[i]["text_embeddings"], dim=0)
        layers[i]["vision_embeddings"] = torch.cat(layers[i]["vision_embeddings"], dim=0)
        current_text = layers[i]["text_embeddings"]
        current_vision = layers[i]["vision_embeddings"]

        # cka_cls = cka_base(
        #     x=text_cls,
        #     y=vision_cls,
        #     kernel="linear",
        #     method="hsic"
        # )
        # device = "cuda" if torch.cuda.is_available() else "cpu"

        # cka_seq = cka_base(
        #     x=text_seq.to(device),
        #     y=vision_seq.to(device),
        #     kernel="linear",
        #     method="hsic"
        # ).cpu().item()
        # print(f"Layer {i}: CKA (CLS) = {cka_cls:.4f}, CKA (Seq) = {cka_seq:.4f}")


    #     # print(f"shape of cls: text {text_cls.shape}, vision {vision_cls.shape}")
    #
    #     # print(f"shape of full-seq: text {text_seq.shape}, vision {vision_seq.shape}")
    #     mknn_seq = mutual_knn_alignment_gpu_advanced(
    #         Z1=text_seq,
    #         Z2=vision_seq,
    #         k=10
    #     )

    #     print(f"Layer {i}: mKNN (CLS) = {mknn_cls:.4f}, mKNN (Seq) = {mknn_seq:.4f}")

        mknn_cls = mutual_knn_alignment_gpu_advanced(
            Z1=current_text,
            Z2=current_vision,
            k=10
        )

        mknn_values[i] = mknn_cls



    with torch.no_grad():
        torch.cuda.empty_cache()


    layer_sims = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
            label = batch["label"].to(device)

            # print(f"img shape: {image['pixel_values'].shape}, ")

            preds, intermediate_representations = model(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                save_intermediate_representations=True
            )

            #generate dummy reprs
            # intermediate_representations = [
            #     {
            #         "text_embedding": torch.randn(16, 197, 768),
            #         "vision_embedding": torch.randn(16, 197, 768),
            #         "is_cross_attention": i in [0, 2],
            #         "layer": i
            #     } for i in range(4)
            # ]

            for repr_dict in intermediate_representations:
                repr_dict["text_embedding"] = repr_dict["text_embedding"].detach().cpu()
                repr_dict["vision_embedding"] = repr_dict["vision_embedding"].detach().cpu()



            current_layer_sims: list[dict] = process_intermediate_repr(
                intermediate_reprs=intermediate_representations,
                pooling_method="cls",
            )

            del intermediate_representations
            del text
            del image

            layer_sims.extend(current_layer_sims)

            if i % 10 == 0:
                torch.cuda.empty_cache()

    analyse(
        layer_similarities=layer_sims,
        num_layers=model.depth,
        mknn_values=mknn_values)
    model.train()

    with torch.no_grad():
        torch.cuda.empty_cache()

def knn(row: int, Z, k):
    # print(f"shape: {Z[row].unspeeze(0).shape}")

    # get knns for row in Z
    distances = torch.cdist(Z[row].unsqueeze(0), Z)
    # print(f"distances shape: {distances.shape}")
    knns_vals, knns_inds = torch.topk(distances, k=k+1, largest=False)  # include self
    # print(f"knn indices shape: {knns_inds.shape}")
    return knns_inds[0, 1:]

def pairwise_knn(
    X: torch.tensor,
    k:int
):
    distances = torch.cdist(X, X)

    # self is included
    knns_vals, knns_inds = torch.topk(distances, k=k+1, largest=False, dim=1)
    return knns_inds[:, 1:]  # remove self

def mutual_knn_alignment(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):
    # Z is matrix, each row is one sample

    def align(Z1, Z2, k):
        # a(i,j) = 1[j in knn(i, Z1) and i in knn(j, Z2); and i != j]
        counter = 0
        for i in range(Z1.shape[0]):
            knns_i_z1 = set(knn(i, Z1, k).cpu().numpy())  # explicit .cpu()
            for j in range(Z2.shape[0]):
                if i != j:
                    knns_j_z2 = set(knn(j, Z2, k).cpu().numpy())  # explicit .cpu()
                    if j in knns_i_z1 and i in knns_j_z2:
                        counter += 1
        return counter

    mknn = align(Z1, Z2, k) / ((align(Z1, Z1, k) * align(Z2, Z2, k)) ** 0.5)
    return mknn

# genai came up with the gpu optimized version of it
def mutual_knn_alignment_gpu(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):

    n = Z1.shape[0]

    def align_gpu(Z1, Z2, k):
        # Precompute all k-NN indices at once
        dists1 = torch.cdist(Z1, Z1)  # [n, n]
        dists2 = torch.cdist(Z2, Z2)  # [n, n]

        # Get k+1 nearest (including self), then remove self
        _, knn1 = torch.topk(dists1, k + 1, largest=False, dim=1)  # [n, k+1]
        _, knn2 = torch.topk(dists2, k + 1, largest=False, dim=1)  # [n, k+1]

        knn1 = knn1[:, 1:]  # [n, k] - remove self
        knn2 = knn2[:, 1:]  # [n, k] - remove self

        # Create masks for the conditions: j in knn(i, Z1) and i in knn(j, Z2)
        total = 0
        for i in range(n):
            for j in range(n):
                if i != j:
                    # Check if j in knn(i, Z1)
                    j_in_knn_i = (knn1[i] == j).any()
                    # Check if i in knn(j, Z2)
                    i_in_knn_j = (knn2[j] == i).any()

                    if j_in_knn_i and i_in_knn_j:
                        total += 1

        return total

    mknn = align_gpu(Z1, Z2, k) / ((align_gpu(Z1, Z1, k) * align_gpu(Z2, Z2, k)) ** 0.5)
    return mknn

# even morew sophisticated version works! from genai
def mutual_knn_alignment_gpu_advanced(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):
    def align_gpu_vectorized(Z1, Z2, k):
        n = Z1.shape[0]

        # Precompute all k-NN indices
        dists1 = torch.cdist(Z1, Z1)
        dists2 = torch.cdist(Z2, Z2)

        _, knn1 = torch.topk(dists1, k + 1, largest=False, dim=1)  # [n, k+1]
        _, knn2 = torch.topk(dists2, k + 1, largest=False, dim=1)  # [n, k+1]

        knn1 = knn1[:, 1:]  # [n, k] - remove self
        knn2 = knn2[:, 1:]  # [n, k] - remove self

        # Create boolean masks entirely on GPU
        # For each i,j: is j in knn(i, Z1)?
        mask1 = (knn1.unsqueeze(2) == torch.arange(n, device=Z1.device).unsqueeze(0).unsqueeze(0)).any(dim=1)  # [n, n]

        # For each i,j: is i in knn(j, Z2)?
        mask2 = (knn2.unsqueeze(2) == torch.arange(n, device=Z1.device).unsqueeze(0).unsqueeze(0)).any(dim=1)  # [n, n]

        # Exclude diagonal (i != j)
        eye_mask = ~torch.eye(n, dtype=torch.bool, device=Z1.device)

        # Count mutual k-NN: j in knn(i,Z1) AND i in knn(j,Z2) AND i!=j
        mutual_mask = mask1 & mask2.T & eye_mask

        return mutual_mask.sum().item()

    mknn = align_gpu_vectorized(Z1, Z2, k) / ((align_gpu_vectorized(Z1, Z1, k) * align_gpu_vectorized(Z2, Z2, k)) ** 0.5)
    return mknn


# simpler version
def mutual_knn_alignment_simple(
    Z1: torch.Tensor,
    Z2: torch.Tensor,
    k: int = 10
):
    """
    computes the mutual k-NN alignment metric as described in
    "Understanding the Emergence of Multimodal Representation Alignment".

    Args:
        Z1, Z2: [n_samples, embedding_dim] tensors
        k: number of nearest neighbors

    """
    assert Z1.shape[0] == Z2.shape[0]

    n_samples = Z1.shape[0]

    # Compute pairwise distances
    dists1 = torch.cdist(Z1, Z1)  # [n, n]
    dists2 = torch.cdist(Z2, Z2)  # [n, n]

    # Find k nearest neighbors (excluding self), genai gave me this function
    _, neighbors1 = torch.topk(dists1, k + 1, largest=False, dim=1)
    _, neighbors2 = torch.topk(dists2, k + 1, largest=False, dim=1)

    #rm self
    neighbors1 = neighbors1[:, 1:]  # [n, k]
    neighbors2 = neighbors2[:, 1:]  # [n, k]

    total_mutual = 0
    for i in range(n_samples):
        nn1_i = set(neighbors1[i].cpu().numpy())
        nn2_i = set(neighbors2[i].cpu().numpy())

        # nn1_i \cap nn2_i
        mutual_count = len(nn1_i.intersection(nn2_i))
        total_mutual += mutual_count


    align_mknn = total_mutual / (n_samples * k)
    return align_mknn


def cosine_similarity_indv(
    text_embedding:torch.Tensor,
    vision_embedding: torch.Tensor
) -> float:

    norm_text_embedding = nn.functional.normalize(text_embedding, dim=-1)
    norm_vision_embedding = nn.functional.normalize(vision_embedding, dim=-1)

    # print(f"norm Text embedding shape: {norm_text_embedding.shape},Vision embedding shape: {norm_vision_embedding.shape}")

    sim = torch.dot(norm_text_embedding, norm_vision_embedding)
    return sim.item()

def cosine_similarity_batch(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
) -> torch.Tensor:


    # sim = torch.nn.functional.cosine_similarity(text_embedding, vision_embedding, dim=1)
    # return sim

    norm_text = nn.functional.normalize(text_embedding, dim=-1)
    norm_vision = nn.functional.normalize(vision_embedding, dim=-1)


    sim = torch.sum(norm_text * norm_vision, dim=-1)

    return sim  # [bs] tensor of similarities

def svcca_similarity(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
):

    text_embedding_numpy = text_embedding.detach().cpu().numpy()
    vision_embedding_numpy = vision_embedding.detach().cpu().numpy()

    # print(f"text embedding shape: {text_embedding_numpy.shape}, vision embedding shape: {vision_embedding_numpy.shape}")


    #cca processing needs numpy input, numpy is different to tensors
    # needs shape [dim, tokens * bs]
    # each row: one feature dimension; dim[i]
    # each column: one datapoint/token/patch

    text_input = text_embedding_numpy.reshape(-1, text_embedding_numpy.shape[-1])
    text_input = text_input.transpose()     # [dim, tokens*bs]
    vision_input = vision_embedding_numpy.reshape(-1, vision_embedding_numpy.shape[-1])
    vision_input = vision_input.transpose()  # [dim, patches*bs]

    try:
        # print(f"reshaped text input shape: {text_input.shape}")
        result = cca_core.get_cca_similarity(
            text_input,
            vision_input,
            verbose=False
        )

        # single value result
        # from https://github.com/google/svcca/blob/master/tutorials/001_Introduction.ipynb
        result = np.mean(result["cca_coef1"])
        return result
    except np.linalg.LinAlgError as e:
        print(f"LinAlgError during SVCCA computation: {e}")
        return 0.0

def mutual_nearest_neighbor_alignment(text_embeds, vision_embeds, k=5):
    # this measurement is from the paper
    # The Platonic Representation Hypothesis
    # where they got values of about 0.16. theoretically the range of this measurement is 0 to 1.
    # but they also noted, that their alignment score was increasing, but not even near the top value of 1.
    batch_size = text_embeds.shape[0]

    text_dists = torch.cdist(text_embeds, text_embeds)
    vision_dists = torch.cdist(vision_embeds, vision_embeds)

    # largest=False => exclude self
    _, text_neighbors = text_dists.topk(k+1, largest=False)
    _, vision_neighbors = vision_dists.topk(k+1, largest=False)



    #remove self (index 0) from neighbors
    text_neighbors = text_neighbors[:, 1:]  # [batch_size, k]
    vision_neighbors = vision_neighbors[:, 1:]  # [batch_size, k]

    # was manually counting, genai gave me this more efficient method
    total_overlap = 0
    for i in range(batch_size):
        # Use torch operations to count intersections
        text_nn = text_neighbors[i]  # [k]
        vision_nn = vision_neighbors[i]  # [k]

        # Count how many elements in text_nn are also in vision_nn
        overlap = torch.isin(text_nn, vision_nn).sum().item()
        total_overlap += overlap

    return total_overlap / (batch_size * k)

def cka(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
) -> float:
    """
    Compute Centered Kernel Alignment (CKA) between two representations.

    Args:
        text_embedding:  shape [batch_size, embedding_dim]
        vision_embedding: shape [batch_size, embedding_dim]

    """
    cka_score = cka_batch(text_embedding, vision_embedding)   # needs shape [bs, tokens, dim], but i have different tokens
    # => i could pad tokenizer to 197?
    # cka_score = cka_base(
    #     x=text_embedding,
    #     y=vision_embedding,
    #     kernel="linear",         # Use linear kernel (standard for CKA)
    #     unbiased=False,          # Use biased version (standard)
    #     method="fro_norm"        # Use Frobenius norm method
    # )
    return cka_score.item()

def process_intermediate_repr(
    intermediate_reprs: list[dict],
    pooling_method:str ="cls"
    ):
    """
    processes intermediate representations and
    computes quantitative measurements.

    input: intermediate_reprs: list of dicts
    dict= {
        "layer": int,
        "text_embedding": torch.Tensor,     # shape [bs, num_tokens, dim]
        "vision_embedding": torch.Tensor,   # shape [bs, num_patches+1, dim]
        "is_cross_attention": bool
    }
    """
    assert pooling_method in ["cls", "mean"], "invalid pooling method specified!!"

    layers_sims = []

    for i, representation in enumerate(intermediate_reprs):
        # print(f"shape text: {representation['text_embedding'].shape}, shape image: {representation['vision_embedding'].shape}")


        cka_sim = cka(
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )

        max_similarity_tp = max_similarity_token_patch(
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )
        max_similarity_pt = max_similarity_patch_token(
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )
        max_similarity_tp = max_similarity_tp.mean().item()
        max_similarity_pt = max_similarity_pt.mean().item()
        # print(f"temp value: {max_simil_avg}")

        # currently not working?
        # #TODO: FIX
        svcca_sim = 0.0
        # svcca_sim = svcca_similarity(
        #     text_embedding=representation["text_embedding"],
        #     vision_embedding=representation["vision_embedding"]
        # )

        # cka_sim = 0.0
        # max_similarity_tp = 0.0
        # max_similarity_pt = 0.0
        # svcca_sim = 0.0



        if pooling_method == "cls":
            text_embedding = representation["text_embedding"][:, 0, :]
            vision_embedding = representation["vision_embedding"][:, 0, :]

        elif pooling_method == "mean":
            text_embedding = torch.mean(representation["text_embedding"], dim=1)
            vision_embedding = torch.mean(representation["vision_embedding"], dim=1)

        #shape vision: [bs, 768]
        #shape text: [bs, 768]

        # TODO: currently only processing the first example in batch.
        # maybe something more sophisticated needed?

        text_embedding_sample = text_embedding
        vision_embedding_sample = vision_embedding
        is_corss_attention = representation["is_cross_attention"]
        layer = representation["layer"]


        cos_sim = cosine_similarity_batch(
            text_embedding=text_embedding_sample,
            vision_embedding=vision_embedding_sample
        )




        layers_sims.append(
            {
                "layer": layer,
                "is_cross_attention": is_corss_attention,
                "cosine_similarity": torch.mean(cos_sim).item(),
                "cka_similarity": cka_sim,
                "max_similarity_tp": max_similarity_tp,
                "max_similarity_pt": max_similarity_pt,
                "svcca_similarity": svcca_sim,
            }
        )


        # print(f"Layer: {layer}, Cross Attention: {is_corss_attention}, "
            #   f"Cosine Similarity: {cos_sim}")
    return layers_sims

def max_similarity_token_patch(
    text_embedding, # [bs, num_tokens, dim]
    vision_embedding # [bs, num_patches+1, dim]
    ):

    text_embedding = text_embedding[:, 1:, :]  # [bs, num_tokens-1, dim]
    vision_embedding = vision_embedding[:, 1:, :]  # [bs, num_patches, dim]

    text_norm = nn.functional.normalize(text_embedding, dim=-1)
    vision_norm = nn.functional.normalize(vision_embedding, dim=-1)

    # pairwise similarities => matrix of similarities. here take the maximum for each token
    # basically doing:
    # sim_matrix[b, i, j] = text_norm[b, i, :] · vision_norm[b, j, :]
    # = cosine_similarity(text_token_i, image_patch_j)
    sim_matrix = torch.bmm(text_norm, vision_norm.transpose(1, 2))  # [bs, num_tokens-1, num_patches]

    max_sims, _ = sim_matrix.max(dim=2)
    return max_sims.mean(dim=1)

def max_similarity_patch_token(
    text_embedding,
    vision_embedding,
):
    max_sims = max_similarity_token_patch(
        text_embedding=vision_embedding,
        vision_embedding=text_embedding
    )

    return max_sims



def analyse(layer_similarities: list[dict], num_layers: int, mknn_values:dict):
    """input format:
        {
            "layer": layer,
            "is_cross_attention": is_cross_attention,
            "cosine_similarity": float,
            "cka_similarity": float,
            "max_similarity": float
        }

        mknn_values:
        {
            "i": mknn-value,
        }

    """

    layers = {}
    for i in range(num_layers):
        layers[f"layer{i}"] = {
            "is_cross_attention": False,
            "similarity_measures": [],
            "full_epoch_measures": mknn_values[i]
        }

    for similarity_measure in layer_similarities:
        layer = similarity_measure["layer"]
        is_cross_attention = similarity_measure["is_cross_attention"]


        layers[f"layer{layer}"]["similarity_measures"].append(similarity_measure)
        layers[f"layer{layer}"]["is_cross_attention"] = is_cross_attention

    for layer_name in layers:
        is_cross_attention = layers[layer_name]["is_cross_attention"]
        measures = layers[layer_name]["similarity_measures"]
        full_epoch_measure = layers[layer_name]["full_epoch_measures"]

        if measures:

            cos_values = [m["cosine_similarity"] for m in measures]
            cka_values = [m["cka_similarity"] for m in measures]
            max_similarity_values_tp = [m["max_similarity_tp"] for m in measures]
            max_similarity_values_pt = [m["max_similarity_pt"] for m in measures]
            svcca_values = [m["svcca_similarity"] for m in measures]



            avg_cosine = sum(cos_values) / len(cos_values)
            avg_cka = sum(cka_values) / len(cka_values)
            avg_max_similarity_tp = sum(max_similarity_values_tp) / len(max_similarity_values_tp)
            avg_max_similarity_pt = sum(max_similarity_values_pt) / len(max_similarity_values_pt)
            avg_svcca = sum(svcca_values) / len(svcca_values)


            info_str = f"layer {layer_name} (co-attn-{is_cross_attention}): cosine={avg_cosine:.4f}, CKA={avg_cka:.4f}, max_sim_tp={avg_max_similarity_tp:.4f}, max_sim_pt={avg_max_similarity_pt:.4f}, SVCCA={avg_svcca:.4f}, mknn_full_epoch={full_epoch_measure:.4f}"
            logger.info(info_str)
            print(info_str)

def cka_custom(X, Y):
    """
    Compute Centered Kernel Alignment (CKA) between two matrices.

    Args:
        X: torch.Tensor of shape [n_samples, features_x]
        Y: torch.Tensor of shape [n_samples, features_y]

    Returns:
        CKA score (float between 0 and 1)
    """
    n = X.shape[0]

    # Compute linear Gram matrices
    K_X = torch.mm(X, X.T)  # [n, n]
    K_Y = torch.mm(Y, Y.T)  # [n, n]

    # Center the Gram matrices
    # H = I - (1/n) * ones_matrix
    ones = torch.ones(n, n, device=X.device)
    H = torch.eye(n, device=X.device) - (1/n) * ones

    K_X_centered = torch.mm(torch.mm(H, K_X), H)
    K_Y_centered = torch.mm(torch.mm(H, K_Y), H)

    # Compute CKA using Frobenius norm formula
    numerator = torch.trace(torch.mm(K_X_centered, K_Y_centered))

    norm_X = torch.norm(K_X_centered, p='fro')
    norm_Y = torch.norm(K_Y_centered, p='fro')
    denominator = norm_X * norm_Y

    if denominator == 0:
        return torch.tensor(0.0)

    cka_score = numerator / denominator
    return cka_score.item()



if __name__ == "__main__":
    # data1 = torch.rand( 250, 768)
    # data2 = torch.rand( 250,768)
    # start = time.time()
    # mknn = mutual_knn_alignment(Z1=data1, Z2=data2, k=5)
    # end = time.time()
    # diff_cpu = end - start

    # start = time.time()
    # mknn_gpu = mutual_knn_alignment_gpu(Z1=data1, Z2=data2, k=5)
    # end = time.time()
    # diff_gpu = end - start

    # print(f"mknn: {mknn}, mknn_gpu: {mknn_gpu}, diff = {mknn-mknn_gpu}")
    # print(f"cpu time: {diff_cpu}, gpu time: {diff_gpu}")

    # mknn_sim = mutual_knn_alignment(Z1=data1, Z2=data1, k=5)
    # mknn_sim_gpu = mutual_knn_alignment_gpu(Z1=data1, Z2=data1, k=5)
    # print(f"mknn identical: {mknn_sim}, mknn_gpu identical: {mknn_sim_gpu}, diff = {mknn_sim-mknn_sim_gpu}")

    data1 = torch.rand( 10000, 768)
    data2 = torch.rand( 10000,768)

    cka_identical = cka_base(
        x=data1,
        y=data1,
        kernel="linear",         # Use linear kernel (standard for CKA)
        unbiased=False,          # Use biased version (standard)
        method="fro_norm"        # Use Frobenius norm method
    )
    cka_different = cka_base(
        x=data1,
        y=data2,
        kernel="linear",         # Use linear kernel (standard for CKA)
        unbiased=False,          # Use biased version (standard)
        method="fro_norm"        # Use Frobenius norm method
    )
    print("shape:", data1.shape, data2.shape)
    print(f"cka identical: {cka_identical}, cka different: {cka_different}")
    # start = time.time()
    # mknn_gpu = mutual_knn_alignment_gpu_advanced(Z1=data1, Z2=data2, k=5)
    # end = time.time()
    # diff_gpu = end - start

    # print(f"mknn_gpu_advanced: {mknn_gpu}, time: {diff_gpu}")
    # mknn_simple = mutual_knn_alignment_simple(Z1=data1, Z2=data2, k=5)
    # print(f"mknn_simple: {mknn_simple}, diff = {mknn_gpu - mknn_simple}")

    # mknn_ident = mutual_knn_alignment_gpu_advanced(Z1=data1, Z2=data1, k=5)
    # print(f"mknn_gpu_advanced identical: {mknn_ident}")
    # mknn_simple_ident = mutual_knn_alignment_simple(Z1=data1, Z2=data1, k=5)
    # print(f"mknn_simple identical: {mknn_simple_ident}, diff = {mknn_ident - mknn_simple_ident}")

    # sim_identical =cka(data1, data1)
    # sim_different = cka(data1, data2)


    # print(f"sim indentical: {sim_identical}, sim different: {sim_different}")

    # svcca_sim_identical = svcca_similarity(
    #     text_embedding=data1,
    #     vision_embedding=data1
    # )
    # svcca_sim_different = svcca_similarity(
    #     text_embedding=data1,
    #     vision_embedding=data2
    # )
    # print(f"svcca sim identical: {svcca_sim_identical}, svcca sim different: {svcca_sim_different}")

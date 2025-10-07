import time

import torch; from torch import nn; from torch.utils.data import DataLoader
import torch.nn.functional as F
from ckatorch.core import cka_batch, cka_base
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#the platonic hypothesis metrics
import metrics; import metrics_llmrepsim

from tqdm import tqdm


import cca_core
from vilbert import ViLBERT
from config import *
import utils
from logger import Logger
import measures

logger = Logger()




def _visualize_new_measures(
    measure_per_layer: dict,
    num_layers: int,
    k: int = 10,
    dir_name: str = None,
    filename_extension: Optional[str]=None
):
    """
    Creates and saves heatmaps for Rank Similarity and Orthogonal Procrustes.
    """

    rank_sim_cross_modal = np.zeros((num_layers, num_layers))
    procrustes_dist_cross_modal = np.zeros((num_layers, num_layers))

    for i in tqdm(range(num_layers), leave=False, desc="Computing Rank Sim & Procrustes"):
        for j in range(num_layers):
            text_cls = measure_per_layer[i]["text_embeddings"]
            vision_cls = measure_per_layer[j]["vision_embeddings"]

            rank_sim_cross_modal[i, j] = measures.rank_similarity(
                X=text_cls, Y=vision_cls, k=k
            )
            procrustes_dist_cross_modal[i, j] = measures.orthogonal_procrustes_distance(
                X=text_cls, Y=vision_cls
            )


    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Rank Similarity Plot
    im1 = axes[0].imshow(rank_sim_cross_modal, cmap='magma', vmin=0, vmax=1, aspect='equal')
    axes[0].set_title(f'Cross-Modal Rank Similarity (k={k})')
    axes[0].set_xlabel('Vision Layer')
    axes[0].set_ylabel('Text Layer')
    plt.colorbar(im1, ax=axes[0], shrink=0.8, label='Rank Similarity')

    # Orthogonal Procrustes Plot
    im2 = axes[1].imshow(procrustes_dist_cross_modal, cmap='magma_r', aspect='equal') # _r reverses colormap
    axes[1].set_title('Cross-Modal Procrustes Distance')
    axes[1].set_xlabel('Vision Layer')
    axes[1].set_ylabel('Text Layer')
    plt.colorbar(im2, ax=axes[1], shrink=0.8, label='Distance (Lower is Better)')

    for ax in axes:
        ax.set_xticks(range(num_layers))
        ax.set_yticks(range(num_layers))

    plt.tight_layout()
    timestamp = int(time.time())
    if dir_name is None:
        fig.savefig(f"res/plots/new_measures_matrices_{timestamp}.png", dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved new measures heatmaps to res/plots/new_measures_matrices_{timestamp}.png")
    else:
        filename = f"{dir_name}/{filename_extension}_new_measures_matrices.png"
        fig.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved new measures heatmaps to {filename}")

    plt.close(fig)

def _visualize_jaccard(
    measure_per_layer: dict,
    num_layers: int,
    k: int = 10,
    dir_name: str = None,
    filename_extension: Optional[str]=None
    ):

    jaccard_cross_modal = np.zeros((num_layers, num_layers))
    jaccard_text_text = np.zeros((num_layers, num_layers))
    jaccard_vision_vision = np.zeros((num_layers, num_layers))



    for i in tqdm(range(num_layers), leave=False, desc="computing neighborhood measures"):
        for j in range(num_layers):
            # Cross-modal (text layer i CLS vs vision layer j CLS)
            text_cls = measure_per_layer[i]["text_embeddings"]
            vision_cls = measure_per_layer[j]["vision_embeddings"]

            jaccard_cross_modal[i,j] = measures.jaccard_similarity(
                X=text_cls, Y=vision_cls, k=k
            )
            # text2text
            text_cls_i = measure_per_layer[i]["text_embeddings"]
            text_cls_j = measure_per_layer[j]["text_embeddings"]

            jaccard_text_text[i,j] = measures.jaccard_similarity(
                X=text_cls_i, Y=text_cls_j, k=k
            )

            # vis2vis
            vision_cls_i = measure_per_layer[i]["vision_embeddings"]
            vision_cls_j = measure_per_layer[j]["vision_embeddings"]

            jaccard_vision_vision[i,j] = measures.jaccard_similarity(
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
    if dir_name is  None:
        timestamp = int(time.time())
        fig1.savefig(f"res/plots/jaccard_matrices_{timestamp}.png", dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved jaccard heatmaps to res/plots/jaccard_matrices_{timestamp}.png")
    else:
        timestamp = int(time.time())
        filename = f"{dir_name}/{filename_extension}_jaccard_matrices_.png"
        fig1.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
        logger.info(f"Saved jaccard heatmaps to {filename}")

    plt.close(fig1)

    return (jaccard_cross_modal, jaccard_text_text, jaccard_vision_vision,)

def _visualize_cka(
    measure_per_layer: dict,
    num_layers: int,
    dir_name: str = None,
    filename_extension: Optional[str]=None
    ):

    cross_modal_matrix = np.zeros((num_layers, num_layers))
    text_text_matrix = np.zeros((num_layers, num_layers))
    vision_vision_matrix = np.zeros((num_layers, num_layers))



    for i in tqdm(range(num_layers), leave=False, desc="computing cka matrix"):
        for j in range(num_layers):
            current_text = measure_per_layer[i]["text_embeddings"]
            current_vision = measure_per_layer[j]["vision_embeddings"]
            cross_modal_matrix[i,j] = measures.cka(
                text_embedding=current_text,
                vision_embedding=current_vision
            )
            # print(f"layer {i}-{j}, cka: {cross_modal_matrix[i,j]:.4f}")
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
            text_text_matrix[i,j] = measures.cka(
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
            vision_vision_matrix[i,j] = measures.cka(
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
    if dir_name is None:
        filename = f"res/plots/all_cka_matrices_{timestamp}.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')
    else:
        filename = f"{dir_name}/{filename_extension}_all_cka_matrices.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

    logger.info(f"Saved CKA heatmaps to {filename}")

    plt.close(fig)
    return cross_modal_matrix, text_text_matrix, vision_vision_matrix

def _visualize_mutual_knn(
    measure_per_layer: dict,
    num_layers: int,
    k: int = 10,
    dir_name: str = None,
    filename_extension: Optional[str]=None
    ):

    cross_modal_matrix = np.zeros((num_layers, num_layers))
    text_text_matrix = np.zeros((num_layers, num_layers))
    vision_vision_matrix = np.zeros((num_layers, num_layers))

    for i in tqdm(range(num_layers), leave=False, desc="computing mknn matrix"):
        for j in range(num_layers):
            # Cross-modal (text layer i CLS vs vision layer j CLS)
            text_cls = measure_per_layer[i]["text_embeddings"]
            vision_cls = measure_per_layer[j]["vision_embeddings"]
            cross_modal_matrix[i,j] = measures.mutual_knn_alignment_gpu_advanced(
                Z1=text_cls, Z2=vision_cls, k=k
            )

            # Text-to-text
            text_cls_i = measure_per_layer[i]["text_embeddings"]
            text_cls_j = measure_per_layer[j]["text_embeddings"]
            text_text_matrix[i,j] = measures.mutual_knn_alignment_gpu_advanced(
                Z1=text_cls_i, Z2=text_cls_j, k=k
            )

            # Vision-to-vision
            vision_cls_i = measure_per_layer[i]["vision_embeddings"]
            vision_cls_j = measure_per_layer[j]["vision_embeddings"]
            vision_vision_matrix[i,j] = measures.mutual_knn_alignment_gpu_advanced(
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
    if dir_name is  None:
        filename = f"res/plots/mutual_knn_matrices_{timestamp}.png"
    else:
        filename = f"{dir_name}/{filename_extension}_mutual_knn_matrices.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight', facecolor='white')

    logger.info(f"Saved mutual k-NN heatmaps to {filename}")

    return cross_modal_matrix, text_text_matrix, vision_vision_matrix


def get_visualisation_data(
    dataloader: DataLoader,
    model: ViLBERT,
    num_samples:int,
    cls_only:bool=False
    #TODO: how many samples to collec
    ):

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

    batch_size = dataloader.batch_size
    sample_counter  = 0

    with torch.no_grad():
        for batch in dataloader:

            text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
            label = batch["label"].to(device)

            # print(f"img shape: {image['pixel_values'].shape}, ")

            text_embedding, image_embedding, intermediate_representations = model(
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
            if cls_only:
                for repr_dict in intermediate_representations:
                    repr_dict["text_embedding"]= repr_dict["text_embedding"][:,0,:].detach().cpu()   # [bs,  dim]
                    repr_dict["vision_embedding"] = repr_dict["vision_embedding"][:,0,:].detach().cpu()
            else:
                for repr_dict in intermediate_representations:
                    repr_dict["text_embedding"] = repr_dict["text_embedding"].detach().cpu()
                    repr_dict["vision_embedding"] = repr_dict["vision_embedding"].detach().cpu()


            for repr_dict in intermediate_representations:
                layer = repr_dict["layer"]
                measure_per_layer[layer]["text_embeddings"].append(repr_dict["text_embedding"])
                measure_per_layer[layer]["vision_embeddings"].append(repr_dict["vision_embedding"])
                measure_per_layer[layer]["is_cross_attention"] = repr_dict["is_cross_attention"]

            del intermediate_representations, text, image, text_embedding, image_embedding

            sample_counter += batch_size
            if sample_counter >= num_samples:
                break

    for layer in measure_per_layer.keys():
        measure_per_layer[layer]["text_embeddings"] = torch.cat(measure_per_layer[layer]["text_embeddings"], dim=0)
        measure_per_layer[layer]["vision_embeddings"] = torch.cat(measure_per_layer[layer]["vision_embeddings"], dim=0)
        # print(f"layer {layer}, text shape: {measure_per_layer[layer]['text_embeddings'].shape}, vision shape: {measure_per_layer[layer]['vision_embeddings'].shape}")

    print(f"shape of collected visualization data: {measure_per_layer[0]['text_embeddings'].shape}, {measure_per_layer[0]['vision_embeddings'].shape}")
    model.train()
    return measure_per_layer        # dict

def run_alignment_visualization(
    dataloader: DataLoader,
    model: ViLBERT,
    dir_name=None,
    filename_extension:Optional[str]=None
    ):
    if filename_extension:
        assert dir_name is not None

    measures_per_layer_full_seq: dict = get_visualisation_data(
        dataloader=dataloader,
        model=model,
        num_samples=NUM_SAMPLES_FULL_SEQ
    )

    measures_per_layer_cls: dict = get_visualisation_data(
        dataloader=dataloader,
        model=model,
        num_samples=NUM_SAMPLES_CLS,
        cls_only=True
    )
    if dir_name is None:
        _visualize_new_measures(measures_per_layer_cls, model.depth, k=KNN_K,)
        _visualize_jaccard(measures_per_layer_cls, model.depth, k=KNN_K)
        _visualize_cka(measures_per_layer_full_seq, model.depth)
        _visualize_mutual_knn(measures_per_layer_cls, model.depth, k=KNN_K)
    else:
        _visualize_new_measures(measures_per_layer_cls, model.depth, k=KNN_K,
            dir_name=dir_name, filename_extension=filename_extension)
        _visualize_jaccard(measures_per_layer_cls, model.depth, k=KNN_K,
            dir_name=dir_name, filename_extension=filename_extension)
        _visualize_cka(measures_per_layer_full_seq, model.depth,
            dir_name=dir_name, filename_extension=filename_extension)
        _visualize_mutual_knn(measures_per_layer_cls, model.depth, k=KNN_K,
            dir_name=dir_name, filename_extension=filename_extension)









def get_alignment_data(dataloader: DataLoader, model:ViLBERT):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    with torch.no_grad():
        torch.cuda.empty_cache()

        layers = {}



    for i in range(model.depth):
        layers[i] = {
            "text_embeddings": [],
            "vision_embeddings": [],
            "is_cross_attention": True,#i in model.cross_attention_layers,
            "layer": i
        }




    for i, batch in enumerate(dataloader):
        text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
        image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
        label = batch["label"].to(device)

        with torch.no_grad():
            text_embedding, image_embedding, intermediate_representations =model.forward(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                save_intermediate_representations=True
            )

            tmp_txt = intermediate_representations[-1]["text_embedding"][:, 0, :]
            # print(tmp_txt.shape)
            # print(f"len intermed: {len(intermediate_representations)}")
            # print(f"shape of intermediates: {intermediate_representations[0]}")
            # print(f"{intermediate_representations[-1]['text_embedding'].shape}")
            # print(tmp_txt.shape, text_embedding.shape)

            cmp_shape_t = text_embedding.shape
            cmp_shape_v = image_embedding.shape

            assert torch.equal(text_embedding , intermediate_representations[-1]["text_embedding"][:, 0, :])
            assert torch.equal(image_embedding , intermediate_representations[-1]["vision_embedding"][:, 0, :])
            del text, image, text_embedding, image_embedding
            for entry in intermediate_representations:
                text_embedding = entry["text_embedding"].detach().cpu()   # [bs, dim]
                vision_embedding = entry["vision_embedding"].detach().cpu() # [bs, dim]
                layer = entry["layer"]

                text_cls =  text_embedding[:,0,:]   # [bs, dim]
                vision_cls = vision_embedding[:,0,:] # [bs, dim]
                # print(f"vision_cls shape: {vision_cls.shape}, text_cls shape: {text_cls.shape}")
                assert text_cls.shape == cmp_shape_t
                assert vision_cls.shape == cmp_shape_v

                layers[layer]["text_embeddings"].append(text_cls)
                layers[layer]["vision_embeddings"].append(vision_cls)

    for layer in layers.keys():
        layers[layer]["text_embeddings"] = torch.cat(layers[layer]["text_embeddings"], dim=0)
        layers[layer]["vision_embeddings"] = torch.cat(layers[layer]["vision_embeddings"], dim=0)

    return layers

def calculate_metrics_old(text_embeddings, vision_embeddings):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text_embeddings = text_embeddings.to(device)
    vision_embeddings = vision_embeddings.to(device)

    norm_t = F.normalize(text_embeddings, dim=-1)
    norm_v = F.normalize(vision_embeddings, dim=-1)

    mknn = measures.mutual_knn_alignment_gpu_advanced(
        Z1=norm_t,
        Z2=norm_v,
        k=KNN_K
    )

    rank_sim = measures.rank_similarity(
        X=norm_t,
        Y=norm_v,
        k=KNN_K
    )

    procrustes_dist = measures.orthogonal_procrustes_distance(
        X=text_embeddings,
        Y=vision_embeddings
    )

    cka_linear = measures.cka(
        text_embedding=text_embeddings.unsqueeze(0),
        vision_embedding=vision_embeddings.unsqueeze(0)
    )

    # svcca = measures.svcca_similarity(text_embedding=text_embeddings, vision_embedding=vision_embeddings)
    svcca = 0.0

    linear_r2 = measures.linear_r2_alignment(
        X=text_embeddings,
        Y=vision_embeddings
    )

    rsa = measures.rsa_similarity(
        X=text_embeddings,
        Y=vision_embeddings
    )

    return {
        "mknn": mknn,
        "rank_sim": rank_sim,
        "procrustes_dist": procrustes_dist,
        "cka_linear": cka_linear,
        "svcca": svcca,
        "linear_r2": linear_r2,
        "rsa": rsa
    }

def calculate_metrics(text_embeddings, vision_embeddings):
    norm_t = F.normalize(text_embeddings, dim=-1)
    norm_v = F.normalize(vision_embeddings, dim=-1)

    cycle_knn = metrics.AlignmentMetrics.cycle_knn(feats_A=norm_t, feats_B=norm_v, topk=KNN_K)
    mknn = metrics.AlignmentMetrics.measure("mutual_knn", text_embeddings, vision_embeddings, topk=KNN_K)
    cka = metrics.AlignmentMetrics.cka(text_embeddings, vision_embeddings)
    cka_rbf = metrics.AlignmentMetrics.cka(text_embeddings, vision_embeddings, kernel_metric="rbf")
    unbiased_cka = metrics.AlignmentMetrics.unbiased_cka(text_embeddings, vision_embeddings)
    svcca = metrics.AlignmentMetrics.svcca(text_embeddings, vision_embeddings, cca_dim=10)
    cknna = metrics.AlignmentMetrics.cknna(norm_t, norm_v, topk=KNN_K)

    return {
        "cycle_knn": cycle_knn,
        "mknn": mknn,
        "cka": cka,
        "cka_rbf": cka_rbf,
        "unbiased_cka": unbiased_cka,
        "svcca": svcca,
        "cknna": cknna
    }

def get_additional_metrics(text_embeddings, vision_embeddings):
    # really big differences
    procrustes_ = measures.orthogonal_procrustes_distance(X=text_embeddings, Y=vision_embeddings)
    procrustes = metrics_llmrepsim.orthogonal_procrustes(R=text_embeddings, Rp=vision_embeddings)

    # they are pretty similar,
    jaccard_ = measures.jaccard_similarity(X=text_embeddings, Y=vision_embeddings)
    jaccard = metrics_llmrepsim.jaccard_similarity(R=text_embeddings, Rp=vision_embeddings)

    # moderate diffs
    rsa_ = measures.rsa_similarity(X=text_embeddings, Y=vision_embeddings)
    rsa= metrics_llmrepsim.representational_similarity_analysis(R=text_embeddings, Rp=vision_embeddings)
    # print(f"old procrustes: {procrustes_} vs. new: {procrustes}; diff: {procrustes - procrustes_}")
    # print(f"old jaccard: {jaccard_} vs. new: {jaccard}; diff: {jaccard - jaccard_}")
    # print(f"old rsa: {rsa_} vs. new: {rsa}; diff: {rsa - rsa_}")


    return {
        "procrustes": procrustes,
        "jaccard": jaccard,
        "rsa": rsa
    }


def analyse_alignment(dataloader: DataLoader, model: ViLBERT):
    layers_data = get_alignment_data(dataloader=dataloader, model=model)

    num_layers = len(layers_data.keys())
    for i in range(num_layers):
        data = layers_data[i]
        text_embeddings = data["text_embeddings"]
        vision_embeddings = data["vision_embeddings"]

        metrics_old = calculate_metrics_old(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings)

        metrics_new = calculate_metrics(
            text_embeddings=text_embeddings,
            vision_embeddings=vision_embeddings)

        metrics_add = get_additional_metrics(text_embeddings=text_embeddings, vision_embeddings=vision_embeddings)


        info_str = (f"layer {i:2}: mknn = {metrics_old['mknn']:4.2f}, "
               f"cka(lin) = {metrics_old['cka_linear']:5.2f}, "
               f"svcca = {metrics_old['svcca']:5.2f}, "
               f"rank-sim = {metrics_old['rank_sim']:5.2f}, "
               f"procrust = {metrics_old['procrustes_dist']:7.2f}, "
        )
        info_str2 = (f"layer {i:2}: mknn = {metrics_new['mknn']:4.2f}, "
                f"cka(lin)= {metrics_new['cka']:5.2f}, "
                f"svcca= {metrics_new['svcca']:4.2f}, "
                f"cka(rbf)= {metrics_new['cka_rbf']:5.2f}, "
                f"cka(unb)= {metrics_new['unbiased_cka']:5.2f}, "
                f"cknn-a= {metrics_new['cknna']:4.2f}, "
                f"cycleknn= {metrics_new['cycle_knn']:4.2f}")

        # print(info_str)
        print(info_str2)
        logger.info(info_str)
        logger.info(info_str2)

    return_dict = {}
    for i in range(num_layers):
        return_dict[i] = {
            "mknn": metrics_new["mknn"],
            "cka": metrics_new["cka"],
            "cka_rbf": metrics_new["cka_rbf"],
            "unbiased_cka": metrics_new["unbiased_cka"],
            "svcca": metrics_new["svcca"],
            "cknna": metrics_new["cknna"],
            "cycle_knn": metrics_new["cycle_knn"],
            "procrustes": metrics_add["procrustes"],
            "jaccard": metrics_add["jaccard"],
            "rsa": metrics_add["rsa"],
            "r2": metrics_old["linear_r2"]
        }
    return return_dict



def analyse_alignment_(dataloader: DataLoader, model: ViLBERT):
    get_alignment_data(dataloader=dataloader, model=model)
    model.eval()

    with torch.no_grad():
        torch.cuda.empty_cache()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    layers = {}
    for i in range(model.depth):
        layers[i] = {
            "text_embeddings": [],
            "vision_embeddings": [],
            "is_cross_attention": True,#i in model.cross_attention_layers,
            "layer": i
        }

    for i, batch in enumerate(dataloader):
        text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
        image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
        label = batch["label"].to(device)




        with torch.no_grad():
            text_embedding, image_embedding, intermediate_representations =model.forward(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                save_intermediate_representations=True
            )
            # print(f"intermed_reps shape: {intermediate_representations[0]}")
            print(f"intermed len: {len(intermediate_representations)}")
            for repr_dict in intermediate_representations:
                layer = repr_dict["layer"]
                cls_text = repr_dict["text_embedding"][:,0,:].detach().cpu()   # [bs, dim]
                cls_vision = repr_dict["vision_embedding"][:,0,:].detach().cpu() # [bs, dim]
                layers[layer]["text_embeddings"].append(cls_text)
                layers[layer]["vision_embeddings"].append(cls_vision)


                norm_t = F.normalize(cls_text, dim=-1)
                norm_v = F.normalize(cls_vision, dim=-1)

                mknn_own = measures.mutual_knn_alignment_gpu_advanced(
                    Z1=norm_t,
                    Z2=norm_v,
                    k=KNN_K
                )
                mknn_n = metrics.AlignmentMetrics.measure(
                    "mutual_knn",
                    norm_t,
                    norm_v,
                    topk=KNN_K)

                print(f"Diff: {mknn_own - mknn_n}; Layer {layer}, mKNN (old impl) = {mknn_own:.4f}, mKNN (new impl) = {mknn_n:.4f}")

            del intermediate_representations
            del text
            del image
            del text_embedding, image_embedding



    mknn_values = {}
    rank_values = {}
    procrustes_values = {}



    for i in range(model.depth):
        # print(f"curr i: {i}\n")
        # print(f"shape of collected embeddings for layer {i}: text {layers[i]['text_embeddings'][0].shape}, vision {layers[i]['vision_embeddings'][0].shape}")
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

        mknn_cls = measures.mutual_knn_alignment_gpu_advanced(
            Z1=torch.nn.functional.normalize(current_text, dim=-1),
            Z2=torch.nn.functional.normalize(current_vision, dim=-1),
            # Z1=current_text,
            # Z2=current_vision,
            k=KNN_K
        )
        mknn_n = metrics.AlignmentMetrics.measure(
            "mutual_knn",
            F.normalize(current_text, dim=-1),
            F.normalize(current_vision, dim=-1),
            topk=KNN_K
        )
        ranked = measures.rank_similarity(
            # X=current_text,
            # Y=current_vision,
            X=torch.nn.functional.normalize(current_text, dim=-1),
            Y=torch.nn.functional.normalize(current_vision, dim=-1),
            k=KNN_K)
        procrustes = measures.orthogonal_procrustes_distance(
            X=current_text,
            Y=current_vision,
        )

        print(f"diff of mknns: {mknn_n - mknn_cls}")

        mknn_values[i] = mknn_n # mknn_cls
        rank_values[i] = ranked
        procrustes_values[i] = procrustes




    with torch.no_grad():
        torch.cuda.empty_cache()


    layer_sims = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):

            text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
            label = batch["label"].to(device)

            # print(f"img shape: {image['pixel_values'].shape}, ")

            text_embedding, image_embedding, intermediate_representations = model(
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

    measures_per_layer = analyse(
        layer_similarities=layer_sims,
        num_layers=model.depth,
        mknn_values=mknn_values,
        rank_values=rank_values,
        procrustes_values=procrustes_values,
    )
    model.train()

    with torch.no_grad():
        torch.cuda.empty_cache()

    return measures_per_layer








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

        # logger.info(f"dim before calculating cka: {representation['text_embedding'].shape}, {representation['vision_embedding'].shape}")
        cka_sim = measures.cka(
            # text_embedding=torch.nn.functional.normalize(representation["text_embedding"], dim=-1),
            # vision_embedding=torch.nn.functional.normalize(representation["vision_embedding"], dim=-1)
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )

        max_similarity_tp = measures.max_similarity_token_patch(
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )
        max_similarity_pt = measures.max_similarity_patch_token(
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )
        max_similarity_tp = max_similarity_tp.mean().item()
        max_similarity_pt = max_similarity_pt.mean().item()
        # print(f"temp value: {max_simil_avg}")

        # currently not working?
        # #TODO: FIX

        svcca_sim = 0.0
        svcca_sim = measures.svcca_similarity(
            # torch.nn.functional.normalize(representation["text_embedding"], dim=-1),
            # torch.nn.functional.normalize(representation["vision_embedding"], dim=-1)
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )

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


        cos_sim = measures.cosine_similarity_batch(
            text_embedding=text_embedding_sample,
            vision_embedding=vision_embedding_sample
        )

        rsa_sim = measures.rsa_similarity(
            X=text_embedding_sample,
            Y=vision_embedding_sample
        )

        linear_r2 = measures.linear_r2_alignment(
            X=text_embedding_sample,
            Y=vision_embedding_sample
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
                "rsa_similarity": rsa_sim,
                "linear_r2": linear_r2
            }
        )


        # print(f"Layer: {layer}, Cross Attention: {is_corss_attention}, "
            #   f"Cosine Similarity: {cos_sim}")
    return layers_sims




def analyse(
    layer_similarities: list[dict],
    num_layers: int,
    mknn_values:dict,
    rank_values:dict,
    procrustes_values:dict,
):
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

        same for the others

    """
    temp_dict = {}
    for i in range(num_layers):
        temp_dict[i] = {}

    layers = {}
    for i in range(num_layers):
        layers[f"layer{i}"] = {
            "is_cross_attention": False,
            "similarity_measures": [],
            "full_epoch_measures": mknn_values[i],
            "rank_measures": rank_values[i],
            "procrustes_measures": procrustes_values[i],
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
        rank_measure = layers[layer_name]["rank_measures"]
        procrustes_measure = layers[layer_name]["procrustes_measures"]

        if measures:

            cos_values = [m["cosine_similarity"] for m in measures]
            cka_values = [m["cka_similarity"] for m in measures]
            max_similarity_values_tp = [m["max_similarity_tp"] for m in measures]
            max_similarity_values_pt = [m["max_similarity_pt"] for m in measures]
            svcca_values = [m["svcca_similarity"] for m in measures]
            linear_r2_values = [m["linear_r2"] for m in measures]
            rsa_values = [m["rsa_similarity"] for m in measures]



            avg_cosine = sum(cos_values) / len(cos_values)
            avg_cka = sum(cka_values) / len(cka_values)
            avg_max_similarity_tp = sum(max_similarity_values_tp) / len(max_similarity_values_tp)
            avg_max_similarity_pt = sum(max_similarity_values_pt) / len(max_similarity_values_pt)
            avg_svcca = sum(svcca_values) / len(svcca_values)
            avg_linear_r2 = sum(linear_r2_values) / len(linear_r2_values)
            avg_rsa = sum(rsa_values) / len(rsa_values)


            metrics = {
                "cosine": avg_cosine,
                "CKA": avg_cka,
                "max_sim_tp": avg_max_similarity_tp,
                "max_sim_pt": avg_max_similarity_pt,
                "SVCCA": avg_svcca,
                "mknn_full_epoch": full_epoch_measure,
                "rank_full_epoch": rank_measure,
                "procrustes_full_epoch": procrustes_measure,
                "linear_r2": avg_linear_r2,
                "rsa": avg_rsa
            }

            # info_str = f"layer {layer_name} (co-attn-{is_cross_attention}): " + \
            #         ", ".join([f"{k}={v:.4f}" for k, v in metrics.items()])
            info_str = (
                f"layer {layer_name:7} (co-attn-{is_cross_attention:2}): "
                f"cosine={avg_cosine:7.3f}, "
                f"CKA={avg_cka:.3f}, "
                # f"max_sim_tp={avg_max_similarity_tp:.4f}, "
                # f"max_sim_pt={avg_max_similarity_pt:.4f}, "
                f"SVCCA={avg_svcca:.3f}, "
                f"mknn={full_epoch_measure:.3f}, "
                f"rank={rank_measure:.3f}, "
                f"procrustes={procrustes_measure:7.2f}",
                f"linear_r2={avg_linear_r2:5.2f}, "
                f"rsa={avg_rsa:5.2f}"

            )
            print(info_str)
            logger.info(info_str)

            layer_num = int(layer_name.replace("layer", ""))
            temp_dict[layer_num] = metrics

    return_dict = {}
    for layer in range(num_layers):
        return_dict[layer] = {
            "is_cross_attention": layers[f"layer{layer}"]["is_cross_attention"],
            "cosine": temp_dict[layer]["cosine"],
            "cka": temp_dict[layer]["CKA"],
            "max_sim_tp": temp_dict[layer]["max_sim_tp"],
            "max_sim_pt": temp_dict[layer]["max_sim_pt"],
            "svcca": temp_dict[layer]["SVCCA"],
            "mknn_full_epoch": temp_dict[layer]["mknn_full_epoch"],
            "rank_full_epoch": temp_dict[layer]["rank_full_epoch"],
            "procrustes_full_epoch": temp_dict[layer]["procrustes_full_epoch"]
        }
    return return_dict

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

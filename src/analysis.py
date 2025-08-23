import torch; from torch import nn
from ckatorch.core import cka_batch, cka_base
import numpy as np
import cca_core

from config import *
import utils
from logger import Logger

logger = Logger()


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

        svcca_sim = svcca_similarity(
            text_embedding=representation["text_embedding"],
            vision_embedding=representation["vision_embedding"]
        )



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

        mutual_nearest_neighbor_alignment_score = mutual_nearest_neighbor_alignment(
            text_embeds=text_embedding,
            vision_embeds=vision_embedding,
            k=5
        )
        # print(f"mutual nearest neighbor alignment score: {mutual_nearest_neighbor_alignment_score}")


        layers_sims.append(
            {
                "layer": layer,
                "is_cross_attention": is_corss_attention,
                "cosine_similarity": torch.mean(cos_sim).item(),
                "cka_similarity": cka_sim,
                "max_similarity_tp": max_similarity_tp,
                "max_similarity_pt": max_similarity_pt,
                "svcca_similarity": svcca_sim,
                "mutual_nearest_neighbor_alignment": mutual_nearest_neighbor_alignment_score
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



def analyse(layer_similarities: list[dict], num_layers: int):
    """input format:
        {
            "layer": layer,
            "is_cross_attention": is_cross_attention,
            "cosine_similarity": float,
            "cka_similarity": float,
            "max_similarity": float
        }
    """

    layers = {}
    for i in range(num_layers):
        layers[f"layer{i}"] = {
            "is_cross_attention": False,
            "similarity_measures": []
        }

    for similarity_measure in layer_similarities:
        layer = similarity_measure["layer"]
        is_cross_attention = similarity_measure["is_cross_attention"]


        layers[f"layer{layer}"]["similarity_measures"].append(similarity_measure)
        layers[f"layer{layer}"]["is_cross_attention"] = is_cross_attention

    for layer_name in layers:
        is_cross_attention = layers[layer_name]["is_cross_attention"]
        measures = layers[layer_name]["similarity_measures"]

        if measures:

            cos_values = [m["cosine_similarity"] for m in measures]
            cka_values = [m["cka_similarity"] for m in measures]
            max_similarity_values_tp = [m["max_similarity_tp"] for m in measures]
            max_similarity_values_pt = [m["max_similarity_pt"] for m in measures]
            svcca_values = [m["svcca_similarity"] for m in measures]
            mutual_nn_values = [m["mutual_nearest_neighbor_alignment"] for m in measures]


            avg_cosine = sum(cos_values) / len(cos_values)
            avg_cka = sum(cka_values) / len(cka_values)
            avg_max_similarity_tp = sum(max_similarity_values_tp) / len(max_similarity_values_tp)
            avg_max_similarity_pt = sum(max_similarity_values_pt) / len(max_similarity_values_pt)
            avg_svcca = sum(svcca_values) / len(svcca_values)
            avg_mutual_nn = sum(mutual_nn_values) / len(mutual_nn_values)

            info_str = f"layer {layer_name} (co-attn-{is_cross_attention}): cosine={avg_cosine:.4f}, CKA={avg_cka:.4f}, max_sim_tp={avg_max_similarity_tp:.4f}, max_sim_pt={avg_max_similarity_pt:.4f}, SVCCA={avg_svcca:.4f}, mutual_nn={avg_mutual_nn:.4f}"
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
    data1 = torch.rand(10, 192, 768)
    data2 = torch.rand(10, 192,768)

    sim_identical =cka(data1, data1)
    sim_different = cka(data1, data2)


    print(f"sim indentical: {sim_identical}, sim different: {sim_different}")

    svcca_sim_identical = svcca_similarity(
        text_embedding=data1,
        vision_embedding=data1
    )
    svcca_sim_different = svcca_similarity(
        text_embedding=data1,
        vision_embedding=data2
    )
    print(f"svcca sim identical: {svcca_sim_identical}, svcca sim different: {svcca_sim_different}")

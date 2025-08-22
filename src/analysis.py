import torch; from torch import nn
from ckatorch.core import cka_batch, cka_base

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
            }
        )


        # print(f"Layer: {layer}, Cross Attention: {is_corss_attention}, "
            #   f"Cosine Similarity: {cos_sim}")
    return layers_sims



def analyse(layer_similarities: list[dict], num_layers: int):
    """input format:
        {
            "layer": layer,
            "is_cross_attention": is_cross_attention,
            "cosine_similarity": float,
            "cka_similarity": float,
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

            avg_cosine = sum(cos_values) / len(cos_values)
            avg_cka = sum(cka_values) / len(cka_values)

            info_str = f"layer {layer_name} (co-attn-{is_cross_attention}): cosine={avg_cosine:.4f}, CKA={avg_cka:.4f}"
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

    data1 = torch.rand(100, 1, 768)
    data2 = torch.rand(100, 1, 768)

    sim_identical = cka_batch(data1, data1)
    sim_different = cka_batch(data1, data2)

    print(f"sim indentical: {sim_identical}, sim different: {sim_different}")

    data_m1 = torch.rand(100,768)
    data_m2 = torch.rand(100,768)

    sim_identical = cka_base(x=data_m1, y= data_m1, kernel="linear", unbiased=False, method="fro_norm")
    sim_different = cka_base(x=data_m1, y=data_m2, kernel="linear", unbiased=False, method="fro_norm")

    print(f"sim indentical: {sim_identical}, sim different: {sim_different}")

    sim_identical = cka_custom(data_m1, data_m1)
    sim_different = cka_custom(data_m1, data_m2)

    print(f"sim indentical: {sim_identical}, sim different: {sim_different}")

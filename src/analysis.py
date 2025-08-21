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

        # Store the entire similarity_measure dict instead of just cosine
        layers[f"layer{layer}"]["similarity_measures"].append(similarity_measure)
        layers[f"layer{layer}"]["is_cross_attention"] = is_cross_attention

    for layer_name in layers:
        is_cross_attention = layers[layer_name]["is_cross_attention"]
        measures = layers[layer_name]["similarity_measures"]

        if measures:  # Only process if we have measurements
            # Extract cosine and CKA values
            cos_values = [m["cosine_similarity"] for m in measures]
            cka_values = [m["cka_similarity"] for m in measures]

            avg_cosine = sum(cos_values) / len(cos_values)
            avg_cka = sum(cka_values) / len(cka_values)

            info_str = f"layer {layer_name} (co-attn-{is_cross_attention}): cosine={avg_cosine:.4f}, CKA={avg_cka:.4f}"
            logger.info(info_str)
            print(info_str)


if __name__ == "__main__":
    data1 = torch.rand(10240,  768)  # Example shape
    data2 = torch.rand(10240,  768)  # Example shape

    sim_identical =cka(data1, data1)
    sim_different = cka(data1, data2)

    print(f"sim indentical: {sim_identical}, sim different: {sim_different}")

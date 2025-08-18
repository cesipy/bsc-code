import torch; from torch import nn

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

    norm_text = nn.functional.normalize(text_embedding, dim=-1)
    norm_vision = nn.functional.normalize(vision_embedding, dim=-1)


    sim = torch.sum(norm_text * norm_vision, dim=-1)

    return sim  # [bs] tensor of similarities



def cka(
    text_embedding: torch.Tensor,
    vision_embedding: torch.Tensor
    ) -> float:

    ...

def process_intermediate_repr(intermediate_reprs: list[dict]):
    """
    processes intermediate representations and
    computes quantitative measurements.

    input: intermediate_reprs: list of dicts
    dict= {
        "layer": int,
        "text_embedding": torch.Tensor,
        "vision_embedding": torch.Tensor,
        "is_cross_attention": bool
    }
    """
    layers_sims = []


    for i, representation in enumerate(intermediate_reprs):
        #shape vision: [bs, 768]
        #shape text: [bs, 768]

        # TODO: currently only processing the first example in batch.
        # maybe something more sophisticated needed?

        text_embedding_sample = representation["text_embedding"]
        vision_embedding_sample = representation["vision_embedding"]
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
            }
        )


        # print(f"Layer: {layer}, Cross Attention: {is_corss_attention}, "
            #   f"Cosine Similarity: {cos_sim}")
    return layers_sims



def analyse(layer_similarities: list[dict], num_layers: int):
    """input format:
        {
            "layer": layer,
            "is_cross_attention": is_corss_attention,
            "cosine_similarity": torch.mean(cos_sim).item(),
        }
    """

    layers = {}
    for i in range(num_layers):
        layers[f"layer{i}"] = {
            "is_cross_attention": True,
            "similarity_measures": []
        }

    for similarity_measure in layer_similarities:
        layer = similarity_measure["layer"]
        is_cross_attention = similarity_measure["is_cross_attention"]
        cos_sim = similarity_measure["cosine_similarity"]

        layers[f"layer{layer}"]["similarity_measures"].append(cos_sim)
        layers[f"layer{layer}"]["is_cross_attention"] = is_cross_attention

    for layer in layers:
        is_cross_attention = layers[layer]["is_cross_attention"]
        measures = layers[layer]["similarity_measures"]

        # if is_cross_attention:
        avg_cosine = sum(measures) / len(measures)
        info_str = f"layer {layer} (cross attention): avg cosine sim.: {avg_cosine:.4f}"
        logger.info(info_str)
        print(info_str)



import os, torch
import json

import os
import random, time, json
import numpy as np


import experiment_tracker
from config import *
import experiment_tracker_utils as etu
from logger import Logger
import task as tasklib
from vilbert import *
import utils

from analyses import metric_evolution

import metrics

import warnings     # should ignore all warnings,
warnings.filterwarnings("ignore")


os.environ["TOKENIZERS_PARALLELISM"] = "false"

logger = Logger()



def main():
    model = ViLBERT.load_model(load_path="res/checkpoints/20251112-102745_pretrained_early_early_fusion/20251113-145942_finetuned_upmc_food.pt")
    model.train()
    device = "cpu"
    dl = datasets.get_task_test_dataset("upmc_food", batch_size=4, num_workers=NUM_WORKERS, seed=1)

    for data_dict in dl:
        label = data_dict["label"].to(device)
        text = {k: v.squeeze(1).to(device) for k, v in data_dict["text"].items()}
        image = {k: v.squeeze(1).to(device) for k, v in data_dict["img"].items()}

        label = torch.argmax(label, dim=1)  # crossentropy wants class indices, not one-hot

        text_embedding, image_embedding = model(
            text_input_ids = text["input_ids"],
            text_attention_mask = text["attention_mask"],
            text_token_type_ids= text.get("token_type_ids", None),
            image_pixel_values= image["pixel_values"],
            image_attention_mask= image.get("attention_mask", None),
        )


        print(f"shape text_embeddings: {text_embedding.shape}")
        cka = metrics.AlignmentMetrics.cka_tensor(text_embedding, image_embedding)
        print(f"cka: {cka}")
        cka_loss = -cka
        cka_loss.backward()

        weights =  model.bert.encoder.layer[4].attention.self.query.weight
        print(weights.grad)
        # weights = weights[0, :]
        print(weights)
        print(f"weight shape: {weights.shape}")
        print(f"grad: {weights.grad}")



        break










if __name__ == "__main__":
    metric_evolution.main()
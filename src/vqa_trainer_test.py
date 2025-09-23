from transformers import (
     # ViT stuff
    BaseImageProcessor,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
    BertTokenizerFast
)
from config import *

import utils
import datasets; from datasets import UPMC_Dataset
from torch.utils.data import DataLoader
import augments_transforms
from vilbert import ViLBERT, ViLBERTConfig
import torch
from torch import nn
from trainer import UPMCTrainer

def main():
    utils.set_seeds(SEED)

    train_dataloader, val_dataloader = datasets.get_upmc_datasets(
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        train_test_ratio=TRAIN_TEST_RATIO,
        max_samples=None,
    )
    # for batch in train_dataloader:
    #     print("batch")
    #     print(batch)
    #     break
    config = ViLBERTConfig()
    model = ViLBERT(config=config)

    trainer = UPMCTrainer(
        model=model,
        config=config,
        gradient_accumulation=GRADIENT_ACCUMULATION_DOWNSTREAM,
        use_contrastive_loss=True,
    )

    # hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
    #     batch_size=BATCH_SIZE_ANALYSIS,
    #     num_workers=4,
    #     pin_memory=False,
    #     prefetch_factor=4,
    #     num_samples=1000
    # )


    # print("Starting VQA training...")
    trainer.train(
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        epochs=15,

    )

    # print("VQA training completed!")

if __name__ == "__main__":
    main()
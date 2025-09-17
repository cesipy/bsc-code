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
from datasets import VqaDataset; import datasets
from torch.utils.data import DataLoader
from trainer.vqa_trainer import VQATrainer
from vilbert import ViLBERT, ViLBERTConfig
import torch
from torch import nn

def main():
    utils.set_seeds(SEED)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


    # todo get method for it
    train_dataset = VqaDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=True
    )
    val_dataset = VqaDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=False
    )

    print(f"num of trainsamples: {len(train_dataset)}")
    print(f"num of valsamples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        shuffle=True,
        prefetch_factor=PREFETCH,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(SEED),
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        shuffle=False,
        prefetch_factor=PREFETCH,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(SEED),
    )
    config = ViLBERTConfig()
    model = ViLBERT(config=config)

    trainer = VQATrainer(
        model=model,
        config=config,
        gradient_accumulation=GRADIENT_ACCUMULATION_DOWNSTREAM
    )

    hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
        batch_size=BATCH_SIZE_ANALYSIS,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4,
        num_samples=1000
    )


    print("Starting VQA training...")
    trainer.train(
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        epochs=8,
        hm_dataloader=hm_dataloader,
        cc_dataloader=cc_dataloader,
    )

    print("VQA training completed!")

if __name__ == "__main__":
    main()
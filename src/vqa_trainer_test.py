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

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    transform = augments_transforms.get_hateful_memes_train_augmentation_albumation(get_advanced=False)

    # TODO: get method for it
    train_dataset = UPMC_Dataset(
        csv_path="res/data/UPMC_Food-101/upmcfood_trainval.csv",
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=True,
        img_path="res/data/UPMC_Food-101/images",
        train_test_ratio=TRAIN_TEST_RATIO,
        transform=transform,
        max_samples=6000,       # TODO
    )
    val_dataset = UPMC_Dataset(
        csv_path="res/data/UPMC_Food-101/upmcfood_trainval.csv",
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=False,
        img_path="res/data/UPMC_Food-101/images",
        train_test_ratio=TRAIN_TEST_RATIO,
    )

    print(f"num of trainsamples: {len(train_dataset)}")
    print(f"num of valsamples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        # batch_size=BATCH_SIZE_DOWNSTREAM,
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
        epochs=8,

    )

    # print("VQA training completed!")

if __name__ == "__main__":
    main()
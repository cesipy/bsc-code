import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, ViTImageProcessor

import utils
from config import *
from vilbert import ViLBERT
from datasets import HM_Dataset, MM_IMDB_Dataset; import datasets
from trainer import HatefulMemesTrainer, MM_IMDB_Trainer, VQATrainer
from logger import Logger
import augments_transforms

logger = Logger()


def finetune_down_stream_task(task_name, use_contrastive_loss):
    assert task_name in ["hateful_memes", "mm_imdb", "easy_vqa"]
    utils.set_seeds(SEED)

    config = ViLBERTConfig()
    model = ViLBERT(config=config)
    # model = Baseline(config=config)

    if task_name == "hateful_memes":
        train_loader, val_loader = datasets.get_hateful_memes_datasets(
            train_test_ratio=TRAIN_TEST_RATIO,
            batch_size=BATCH_SIZE_DOWNSTREAM,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            use_train_augmentation=True,
        )

        trainer = HatefulMemesTrainer(
            model=model,
            config=config,
            use_contrastive_loss=use_contrastive_loss,
            use_cosine_loss=False,
            gradient_accumulation=GRADIENT_ACCUMULATION,
        )


    elif task_name == "mm_imdb":

        train_loader, val_loader = datasets.get_mmimdb_datasets(
            train_test_ratio=TRAIN_TEST_RATIO,
            batch_size=BATCH_SIZE_DOWNSTREAM,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            use_train_augmentation=True,
        )
        trainer = MM_IMDB_Trainer(
            model=model,
            config=config,
            gradient_accumulation=GRADIENT_ACCUMULATION,
            use_contrastive_loss=use_contrastive_loss,
        )



    else:
        train_loader, val_loader = datasets.get_easyvqa_datasets(
            batch_size=BATCH_SIZE_DOWNSTREAM,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            use_train_augmentation=False,
        )

        trainer = VQATrainer(
            model=model,
            config=config,
            gradient_accumulation=GRADIENT_ACCUMULATION,
            use_contrastive_loss=use_contrastive_loss,
        )

    trainer.train(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=DOWNSTREAM_EPOCHS,
    )

    # TODO: fix alignment dataset, add easy vqa


if __name__ == "__main__":
    task_name = "mm_imdb"#"hateful_memes"
    use_contrastive = True
    finetune_down_stream_task(task_name=task_name, use_contrastive_loss=use_contrastive)

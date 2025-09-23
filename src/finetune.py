import os
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast, ViTImageProcessor
import argparse
import argcomplete

import utils
from config import *
from vilbert import ViLBERT
from datasets import HM_Dataset, MM_IMDB_Dataset; import datasets
from trainer import HatefulMemesTrainer, MM_IMDB_Trainer, VQATrainer
from logger import Logger
import augments_transforms

logger = Logger()


def finetune_down_stream_task(
    task_name:str,
    use_contrastive_loss:bool,
    analyze_alignment:bool,
    ):
    assert task_name in ["hateful_memes", "mm_imdb", "easy_vqa"]
    utils.set_seeds(SEED)

    config = ViLBERTConfig()
    model = ViLBERT(config=config)
    # model = Baseline(config=config)

    dataloader_hm, dataloader_cc, dataloader_imdb = None, None, None

    if analyze_alignment:
        dataloader_hm, dataloader_cc, dataloader_imdb = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            num_samples=2000
        )

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
        analysis_data_loader = dataloader_hm


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
        analysis_data_loader = dataloader_imdb



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

    # TOD: implement alignment analysis on easyvqa
    trainer.train(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=DOWNSTREAM_EPOCHS,
        analyze_alignment=analyze_alignment,
        dataloader=analysis_data_loader,
        cc_dataloader=dataloader_cc,
    )

    checkpoints_path = "res/chechpoints"

    trainer.model.save_model(save_path=checkpoints_path + f"/{task_name}_finetuned.pt")

    # TODO: fix alignment dataset, add easy vqa


if __name__ == "__main__":
    task_name = "mm_imdb"#"hateful_memes"
    use_contrastive = True

    p = argparse.ArgumentParser("train on different downstream tasks")
    p.add_argument("--use_contrastive", action="store_true",)
    p.add_argument("--task", type=str, default="mm_imdb",
                   choices=["hateful_memes", "mm_imdb", "easy_vqa"],  
                   help="which downstream task to finetune on; hateful_memes, mm_imdb, easy_vqa")
    p.add_argument("--analyze", action="store_true",
                   help="run alignment analysis")

    argcomplete.autocomplete(p)
    args = p.parse_args()
    assert args.task in ["hateful_memes", "mm_imdb", "easy_vqa"]

    finetune_down_stream_task(
        task_name=args.task,
        use_contrastive_loss=args.use_contrastive,
        analyze_alignment=args.analyze
        )


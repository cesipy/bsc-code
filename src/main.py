import sys, os; sys.path.append('src')
import time
import os
# problem when running training on loaded models after pretraining.
# occurs because of parallelism in data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import gc
import argparse
from typing import Optional

import torch; from torch.utils.data import DataLoader, Dataset

from transformers import (
    BertTokenizerFast, PreTrainedTokenizerFast, ViTImageProcessor
)

import utils
from task import Task
import datasets; from datasets import HM_Dataset, PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
from config import *
from vilbert import ViLBERT
from trainer import HatefulMemesTrainer, PretrainingTrainer
from logger import Logger

import warnings
# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

logger = Logger()


machine = os.getenv("MACHINE_TYPE", default="home")     # remote or home


@utils.memory_cleanup
def pretrain(tasks:Optional[Task]=[Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM]):

    utils.set_seeds(SEED)
    path = "res/data/conceptual-captions/train.csv"
    val_path = "res/data/conceptual-captions/validation.csv"
    data_list = datasets.generate_data_list_pretrain(path=path, max_number=100_000)
    validation_list = datasets.generate_data_list_pretrain(path=val_path)
    data_list = data_list[:80_000]
    # validation_list = validation_list[:1000]

    # train_idx = int(len(data_list) * TRAIN_TEST_RATIO)
    # train_data = data_list[:train_idx]
    # val_data   = data_list[train_idx:]
    train_data = data_list
    val_data   = validation_list[:]

    print(len(train_data), len(val_data))

    train_loader_ap, val_loader_ap, \
    train_loader_mlm, val_loader_mlm, \
    train_loader_mim, val_loader_mim \
        =  datasets.get_dataloaders_pretrain(
        train_data=train_data,
        val_data=val_data,
        num_workers=NUM_WORKERS,
        prefetch=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        pin_memory=PIN_MEMORY,
        use_contrastive_ap=USE_CONTRASTIVE_LOSS,
        batch_size=BATCH_SIZE_PRETRAIN,
    )

    print(f"Dataset len: \n\t train: {len(train_loader_ap.dataset)}\n\t val: {len(val_loader_ap.dataset)}")

    print(f"batchsize: {BATCH_SIZE_PRETRAIN}, bs-analysis: {BATCH_SIZE_ANALYSIS}")

    config = ViLBERTConfig(
        pretraining_tasks=tasks[:]
    )

    model = ViLBERT(config=config)
    if FREEZE_UNIMODAL_ENCODERS:
        utils.freeze_all_layers(model.bert)
        utils.freeze_all_layers(model.vit)
    utils.params_summary(model=model)
    trainer = PretrainingTrainer(
        model=model,
        config=config,
        tasks=tasks,
        use_contrastive_ap=USE_CONTRASTIVE_LOSS,
        gradient_accumulation=GRADIENT_ACCUMULATION

    )


    hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
        batch_size=BATCH_SIZE_ANALYSIS,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4,
        num_samples=1500
    )

    trainer.train(
        train_dataloaderAP=train_loader_ap,
        test_dataloaderAP=val_loader_ap,
        train_dataloaderMLM=train_loader_mlm,
        test_dataloaderMLM=val_loader_mlm,
        train_dataloaderMIM=train_loader_mim,
        test_dataloaderMIM=val_loader_mim,
        epochs=PRETRAIN_EPOCHS,
        hm_dataloader=hm_dataloader,
        cc_dataloader=cc_dataloader
    )

    task_string = ""
    tasks_vals = [task.value for task in tasks]
    tasks_vals.sort()
    for val in tasks_vals:
        task_string += f"{val}"
    tmsp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"res/checkpoints/pretrained_{task_string}_{tmsp}.pt"
    trainer.model.save_model(save_path=filename)

    del model

    model = ViLBERT.load_model("test.pt")
    logger.info("finished training. \n\n " + 20*"-")

def parse_args():
    p = argparse.ArgumentParser(description="pretrain")
    # p.add_argument("--tasks", type="str", default=None)
    p.add_argument("--no-mim", action='store_true', help="disable masked image modeling")
    p.add_argument("--no-mlm", action='store_true', help="disable masked language modeling")
    p.add_argument("--no-ap", action='store_true', help="disable alignment prediction")


    active_tasks_tmp = []
    res = p.parse_args()

    if not res.no_mim:
        active_tasks_tmp.append(Task.MASKED_IM)
    if not res.no_mlm:
        active_tasks_tmp.append(Task.MASKED_LM)
    if not res.no_ap:
        active_tasks_tmp.append(Task.ALIGNMENT_PREDICTION)


    if len(active_tasks_tmp) == 0:
        raise ValueError("At least one task must be active for pretraining")

    return active_tasks_tmp





if __name__ == "__main__":
    try:
        # pretain()

        tasks = parse_args()
        pretrain(tasks=tasks)

        # train_and_eval_on_downstream_task(pretrained_model_path=None)
        # train_and_eval_on_downstream_task(pretrained_model_path="res/checkpoints/pretrained_4.pt")
    except Exception as e :
        logger.error(f"Error during pretraining or training on downstream task: {e}")
        raise e

import os
# problem when running training on loaded models after pretraining.
# occurs because of parallelism in data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # You already have this


import gc
import torch; from torch.utils.data import DataLoader, Dataset

from transformers import (
    BertTokenizerFast, PreTrainedTokenizerFast, ViTImageProcessor
)

import utils
from task import Task
import datasets; from datasets import CustomDataset, PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
from config import *
from vilbert import ViLBERT
from trainer import Trainer, PretrainingTrainer
from logger import Logger

import argparse

import analysis

import warnings
# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

machine = os.getenv("MACHINE_TYPE", default="home")     # remote or home
logger = Logger()

@utils.memory_cleanup
def train_and_eval_on_downstream_task(pretrained_model_path:str):
    if pretrained_model_path==None or not os.path.exists(pretrained_model_path) :
        # use fresh vilbert
        info_str = f"Pretrained model path {pretrained_model_path} does not exist, using fresh model."
        print(info_str)
        logger.info(info_str)

        config = ViLBERTConfig()
        model = ViLBERT(config=config)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cp = ViLBERT.from_pretrained_checkpoint(checkpoint_path=pretrained_model_path, device=device)
        info_str = f"Loaded model from {pretrained_model_path} with config: {cp['config']}"
        print(info_str)
        logger.info(info_str)

    utils.freeze_all_layers(model.vit)
    utils.freeze_all_layers(model.bert)

    path = "res/data/hateful_memes_data/train.jsonl"
    val_path = "res/data/hateful_memes_data/test.jsonl"

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    config = ViLBERTConfig()

    #TODO: also freeze co-attention layers here
    utils.params_summary(model=model)
    train_data_list = datasets.generate_data_list(path)
    # val_data_list = datasets.generate_data_list(val_path)

    # train_data_list = train_data_list[:1000]

    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:]
    # train_data = train_data_list
    # val_data = val_data_list


    if machine == "remote":
        bs = 64    # obout 23.3gb vrman
        config.learning_rate = 6e-5#5e-6     # TODO: make this cleaner
    else:
        bs = 64
        config.learning_rate = 3e-5

    print(bs)

    num_workers = 4
    pin_memory= True
    prefetch_factor = 3

    train_data = train_data
    train_dataset = CustomDataset(train_data, tokenizer=tokenizer, image_processor=image_processor)
    val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        )

    trainer = Trainer(model, config)
    trainer.train(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=4
    )

    del model, trainer, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Training and evaluation on downstream task finished, cleaning up memory\n\n"+ 25*"-")



def analyse_on_cc(pretrained_model_path: str):
    path = "res/data/conceptual-captions/validation.csv"
    data_list = datasets.generate_data_list_pretrain(path=path,)

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    dataset = datasets.PretrainDatasetAP(data=data_list, tokenizer=tokenizer,
                                         image_processor=image_processor,
                                         preprocessing_prediction_alignment=False)

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        prefetch_factor=4,
        num_workers=4
    )

    if pretrained_model_path==None or not os.path.exists(pretrained_model_path) :
        # use fresh vilbert
        info_str = f"Pretrained model path {pretrained_model_path} does not exist, using fresh model."
        print(info_str)
        logger.info(info_str)

        config = ViLBERTConfig()
        model = ViLBERT(config=config)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cp = ViLBERT.from_pretrained_checkpoint(checkpoint_path=pretrained_model_path, device=device)
        info_str = f"Loaded model from {pretrained_model_path} with config: {cp['config']}"
        print(info_str)
        logger.info(info_str)

        layer_sims = []

    for batch in dataloader:
        current_task = batch["task"][0].item()  # Get task type
        text = {k: v.squeeze(1).to(device) for k, v in batch["text"].items()}
        image = {k: v.squeeze(1).to(device) for k, v in batch["img"].items()}
        label = batch["label"].to(device)

        preds, intermediate_representations = model(
            text_input_ids=text["input_ids"],
            text_attention_mask=text["attention_mask"],
            text_token_type_ids=text.get("token_type_ids", None),
            image_pixel_values=image["pixel_values"],
            image_attention_mask=image.get("attention_mask", None),
            save_intermediate_representations=True
        )

        current_layer_sims: list[dict] = analysis.process_intermediate_repr(intermediate_representations)

        layer_sims.extend(current_layer_sims)

    analysis.analyse(layer_similarities=layer_sims, num_layers=model.depth)


if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="train on hateful memes")
        p.add_argument("--path", type=str, default=None,
                    help="Path to pretrained model checkpoint (optional)")
        # train_and_eval_on_downstream_task(pretrained_model_path=p.parse_args().path)
        analyse_on_cc(pretrained_model_path=p.parse_args().path)
    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        raise e

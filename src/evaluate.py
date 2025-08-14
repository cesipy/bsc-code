import os
# problem when running training on loaded models after pretraining. 
# occurs because of parallelism in data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"


import gc
import torch; from torch.utils.data import DataLoader, Dataset

from transformers import (
    BertTokenizerFast, PreTrainedTokenizerFast, ViTImageProcessor
)

import utils; from utils import Task
import datasets; from datasets import CustomDataset, PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
from config import *
from vilbert import ViLBERT
from trainer import Trainer, PretrainingTrainer
from logger import Logger

import argparse

import warnings
# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

logger = Logger()

@utils.memory_cleanup
def train_and_eval_on_downstream_task(pretrained_model_path:str):
    if pretrained_model_path==None or not os.path.exists(pretrained_model_path) : 
        # use fresh vilbert 
        info_str = f"Pretrained model path {pretrained_model_path} does not exist, using fresh model."
        print(info_str)
        logger.info(info_str)
        
        config = Config()
        model = ViLBERT()
    
    else:    
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cp = ViLBERT.from_pretrained_checkpoint(checkpoint_path=pretrained_model_path, device=device)
        info_str = f"Loaded model from {pretrained_model_path} with config: {cp['config']}"
        print(info_str)
        logger.info(info_str)
        
    # utils.freeze_all_layers(model.vit)
    # utils.freeze_all_layers(model.bert)
        
    path = "res/data/hateful_memes_data/train.jsonl"
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    config = Config()
    config.learning_rate = 2e-5#5e-6     # TODO: make this cleaner
    
    #TODO: also freeze co-attention layers here
    utils.params_summary(model=model)
    train_data_list = datasets.generate_data_list(path)
    
    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:] 
    
    train_data = train_data 
    train_dataset = CustomDataset(train_data, tokenizer=tokenizer, image_processor=image_processor)
    val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    trainer = Trainer(model, config)
    trainer.train(
        train_dataloader=train_loader, 
        test_dataloader=val_loader, 
        epochs=4
    )
    
    del model, trainer, train_dataset, val_dataset, train_loader, val_loader
    torch.cuda.empty_cache()
    gc.collect()
    logger.info("Training and evaluation on downstream task finished, cleaning up memory")
    
    
if __name__ == "__main__": 
    
    p = argparse.ArgumentParser(description="train on hateful memes")
    p.add_argument("--path", type=str, default=None,
                   help="Path to pretrained model checkpoint (optional)")
    train_and_eval_on_downstream_task(pretrained_model_path=p.parse_args().path)
    
    
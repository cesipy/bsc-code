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

import warnings
# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

logger = Logger()

@utils.memory_cleanup
def pretain(): 
    logger.info("starting pretraining")
    epochs = 4
    num_workers = 10
    prefetch= 4
    path = "res/data/conceptual-captions/train.csv"
    val_path = "res/data/conceptual-captions/validation.csv"
    data_list = datasets.generate_data_list_pretrain(path=path)
    validation_list = datasets.generate_data_list_pretrain(path=val_path)
    # data_list = data_list[:10_000]
    # validation_list = validation_list[:1_000]
    
    # train_idx = int(len(data_list) * TRAIN_TEST_RATIO)
    # train_data = data_list[:train_idx]
    # val_data   = data_list[train_idx:]
    train_data = data_list
    val_data   = validation_list
    
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    preprocessing_prediction_alignment = False
    train_dataset_ap = PretrainDatasetAP(
        train_data, 
        tokenizer=tokenizer, 
        image_processor=image_processor, 
        preprocessing_prediction_alignment=preprocessing_prediction_alignment
    )
    val_dataset_ap   = PretrainDatasetAP(
        val_data, 
        tokenizer=tokenizer, 
        image_processor=image_processor,
        preprocessing_prediction_alignment=preprocessing_prediction_alignment
    )
    
    train_dataset_mlm = PretrainDatasetMLM(
        train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    
    val_dataset_mlm   = PretrainDatasetMLM(
        val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )
    
    
    train_loader_ap = DataLoader(
        dataset=train_dataset_ap, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=prefetch
    )
    val_loader_ap = DataLoader(
        dataset=val_dataset_ap, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=prefetch
    )
    
    train_loader_mlm = DataLoader(
        dataset=train_dataset_mlm,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch
    )
    val_loader_mlm = DataLoader(
        dataset=val_dataset_mlm,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch
    )
    
    model = ViLBERT()
    utils.params_summary(model=model)
    
    trainer = PretrainingTrainer(
        model=model, 
        config=Config(), 
    )
    
    trainer.train(
        train_dataloaderAP=train_loader_ap,
        test_dataloaderAP=val_loader_ap,
        train_dataloaderMLM=train_loader_mlm,
        test_dataloaderMLM=val_loader_mlm,
        epochs=epochs, 
        train_only_ap=False
    )
    
    del model, trainer, train_dataset_ap, val_dataset_ap, train_dataset_mlm, val_dataset_mlm
    del train_loader_ap, val_loader_ap, train_loader_mlm, val_loader_mlm
    torch.cuda.empty_cache()
    gc_collected = gc.collect()
    print(f"gc collected items: {gc_collected}")
    logger.info("pretraining finished, cleaning up memory")
    
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

    path = "res/data/hateful_memes_data/train.jsonl"
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    config = Config()
    config.learning_rate = 1e-5     # TODO: make this cleaner
    
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
    
@utils.memory_cleanup
def pretrain_(): 
    epochs = 4
    num_workers = 10
    prefetch= 4
    path = "res/data/conceptual-captions/train.csv"
    val_path = "res/data/conceptual-captions/validation.csv"
    data_list = datasets.generate_data_list_pretrain(path=path)
    validation_list = datasets.generate_data_list_pretrain(path=val_path)
    # data_list = data_list[:200]
    # validation_list = validation_list[:1000]
    
    # train_idx = int(len(data_list) * TRAIN_TEST_RATIO)
    # train_data = data_list[:train_idx]
    # val_data   = data_list[train_idx:]
    train_data = data_list
    val_data   = validation_list
    
    train_loader_ap, val_loader_ap, \
    train_loader_mlm, val_loader_mlm, \
    train_loader_mim, val_loader_mim \
        =  datasets.get_dataloaders(
        train_data=train_data, 
        val_data=val_data, 
        num_workers=num_workers, 
        prefetch=prefetch,
        persistent_workers=False, 
        pin_memory=False
    )
        
    print(f"Dataset len: \n\t train: {len(train_loader_ap.dataset)}\n\t val: {len(val_loader_ap.dataset)}")
    
    
    model = ViLBERT()
    utils.params_summary(model=model)
    trainer = PretrainingTrainer(
        model=model, 
        config=Config(), 
    )
    
    trainer.train(
        train_dataloaderAP=train_loader_ap,
        test_dataloaderAP=val_loader_ap,
        train_dataloaderMLM=train_loader_mlm,
        test_dataloaderMLM=val_loader_mlm,
        train_dataloaderMIM=train_loader_mim,
        test_dataloaderMIM=val_loader_mim,
        epochs=epochs, 
        tasks=[Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM],
    )
    
    
    

if __name__ == "__main__":
    # pretain()
    pretrain_()
    # train_and_eval_on_downstream_task(pretrained_model_path=None)
    # train_and_eval_on_downstream_task(pretrained_model_path="res/checkpoints/pretrained_4.pt")

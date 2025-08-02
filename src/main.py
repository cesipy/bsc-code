from transformers import (
    BertTokenizerFast, PreTrainedTokenizerFast, ViTImageProcessor
)

import utils
import datasets; from datasets import CustomDataset, PretrainDataset
from config import *
from vilbert import ViLBERT
from trainer import Trainer, PretrainingTrainer
from torch.utils.data import DataLoader, Dataset

def pretain(): 
    # path = "res/data/conceptual-captions/validation.csv"
    path = "res/data/conceptual-captions/train.csv"
    data_list = datasets.generate_data_list_pretrain(path=path)
    # data_list = data_list[:1000]
    train_idx = int(len(data_list) * TRAIN_TEST_RATIO)
    train_data = data_list[:train_idx]
    val_data   = data_list[train_idx:]
    
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    preprocessing_prediction_alignment = False
    train_dataset = PretrainDataset(
        train_data, 
        tokenizer=tokenizer, 
        image_processor=image_processor, 
        preprocessing_prediction_alignment=preprocessing_prediction_alignment
    )
    val_dataset   = PretrainDataset(
        val_data, 
        tokenizer=tokenizer, 
        image_processor=image_processor,
        preprocessing_prediction_alignment=preprocessing_prediction_alignment
    )
    
    train_loader = DataLoader(
        dataset=train_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=10, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4
    )
    val_loader = DataLoader(
        dataset=val_dataset, 
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=10, 
        pin_memory=True, 
        persistent_workers=True,
        prefetch_factor=4
    )
    
    model = ViLBERT()
    utils.params_summary(model=model)
    
    trainer = PretrainingTrainer(
        model=model, 
        config=Config()
    )
    
    trainer.train(
        train_dataloader=train_loader, 
        test_dataloader=val_loader, 
        epochs=10
    )
    
    

def main(): 
    path = "res/data/hateful_memes_data/train.jsonl"
    config = Config()
    model = ViLBERT()
    
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    utils.params_summary(model=model)
    train_data_list = datasets.generate_data_list(path)
    
    #TODO: remove this, only temp
    # train_data_list = train_data_list[:100]  
    
    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:] 
    
    train_data = train_data[:100] 
    train_dataset = CustomDataset(train_data, tokenizer=tokenizer, image_processor=image_processor)
    val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)
    
    # print(f"train dataset- length: {len(train_dataset)}, head: {train_dataset.data[:5]}")
    # print(f"val dataset- length: {len(val_dataset)}, head: {val_dataset.data[:5]}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    trainer = Trainer(model, config)
    trainer.train(
        train_dataloader=train_loader, 
        test_dataloader=val_loader, 
        epochs=10
    )


if __name__ == "__main__":
    # main()
    pretain()
from transformers import (
    BertTokenizerFast, PreTrainedTokenizerFast, ViTImageProcessor
)

import utils
import datasets; from datasets import CustomDataset
from config import *
from vilbert import ViLBERT
from trainer import Trainer
from torch.utils.data import DataLoader, Dataset


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
    main()
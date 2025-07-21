import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils
import datasets
from vilbert import ViLBERT
from config import * 

from transformers import (
    #BERT stuff
    BertModel, 
    BertConfig, 
    BertTokenizer, 
    BertTokenizerFast, 
    
    # ViT stuff
    ViTConfig, 
    ViTModel,
    ViTImageProcessor,
    
    # type hinting stuff
    PreTrainedTokenizerFast,
)

from datasets import CustomDataset
import datasets



class Trainer(): 
    def __init__(self, model: ViLBERT, config: Config): 
        self.lr = config.learning_rate
        self.model = model
        self.config = config
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=0.01           #TODO: make weight decay configurable in config
        )
        
        self.loss_fn = nn.BCEWithLogitsLoss()  
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
    
    def train_epoch(self, data_loader: DataLoader): 
        
        total_loss = 0 
        
        num_batches = 0
        for batch in data_loader: 
            num_batches += 1
            self.optimizer.zero_grad()
            
    
            data_dict = batch
            
            label = data_dict["label"].to(self.device)
            # itsnot possible to send dicts to device, so do it for every value in dict. 
            text = {k: v.squeeze(1).to(self.device) for k, v in data_dict["text"].items()}
            image = {k: v.squeeze(1).to(self.device) for k, v in data_dict["img"].items()}
        
            
            preds = self.model(
                text_input_ids= text["input_ids"],
                text_attention_mask= text["attention_mask"],
                text_token_type_ids= text.get("token_type_ids", None),
                image_pixel_values= image["pixel_values"],
                image_attention_mask= image.get("attention_mask", None),
            )
            preds = preds.squeeze()
            label = label.float()
            
            # print(f"shape preds: {preds.shape}, shape label: {label.shape}")
            
            
            loss = self.loss_fn(preds, label)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        
            
        return total_loss / num_batches
            
                
            
                    
    def train(self, train_dataloader: DataLoader, epochs: int): 
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}")
        

def main(): 
    path = "res/data/hateful_memes_data/train.jsonl"
    config = Config()
    model = ViLBERT()
    
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    utils.params_summary()
    train_data_list = datasets.generate_data_list(path)
    
    #TODO: remove this, only temp
    # train_data_list = train_data_list[:100]  
    
    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:]  
     
    
    train_dataset = CustomDataset(train_data, tokenizer=tokenizer, image_processor=image_processor)
    val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)
    
    # print(f"train dataset- length: {len(train_dataset)}, head: {train_dataset.data[:5]}")
    # print(f"val dataset- length: {len(val_dataset)}, head: {val_dataset.data[:5]}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    trainer = Trainer(model, config)
    trainer.train(train_loader, 10)


if __name__ == "__main__":
    main()
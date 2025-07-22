import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from vilbert import ViLBERT
from config import * 


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
        self.model.train()
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
            
                   
    def train(
        self, 
        train_dataloader: DataLoader, 
        test_dataloader:  DataLoader,
        epochs: int
    ): 
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            test_loss  = self.evaluate(test_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")
        
    def evaluate(self, dataloader: DataLoader): 
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        with torch.no_grad(): 
            for batch in dataloader: 
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
                
                loss = self.loss_fn(preds, label)
                total_loss += loss.item()
                num_batches += 1
                
        return total_loss / num_batches
                
                
                    
                
                
        
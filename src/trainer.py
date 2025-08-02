import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

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
                
                
                    
class PretrainingTrainer:
    def __init__(
        self, 
        model: ViLBERT, 
        config: Config,
    ): 
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=0.01
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        
        self.model = torch.compile(self.model)  
        
        self.loss_fn_alignment = nn.BCEWithLogitsLoss()
        self.loss_fn_mlm = nn.CrossEntropyLoss()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler = torch.amp.grad_scaler.GradScaler(device=self.device)
        
    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in dataloader:
                num_batches += 1
                
                current_task = batch["task"][0].item()  # Get task type
                text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
                image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
                label = batch["label"].to(self.device)

                # Handle different task types
                if current_task == 2:  # MLM task (Task.MASKED_LM.value = 2)
                    mlm_logits = self.model.forward_pretrain(
                        text_input_ids=text["input_ids"],
                        text_attention_mask=text["attention_mask"],
                        text_token_type_ids=text.get("token_type_ids", None),
                        image_pixel_values=image["pixel_values"],
                        image_attention_mask=image.get("attention_mask", None),
                        tasks=["mlm"]
                    )
                    preds = mlm_logits.view(-1, mlm_logits.size(-1))
                    label_flat = label.view(-1)
                    loss = self.loss_fn_mlm(preds, label_flat)
                    
                else:  # Alignment prediction task
                    prediction_logits = self.model.forward_pretrain(
                        text_input_ids=text["input_ids"],
                        text_attention_mask=text["attention_mask"],
                        text_token_type_ids=text.get("token_type_ids", None),
                        image_pixel_values=image["pixel_values"],
                        image_attention_mask=image.get("attention_mask", None),
                        tasks=["alignment_prediction"]
                    )
                    label = label.float().unsqueeze(1)
                    loss = self.loss_fn_alignment(prediction_logits, label)

                total_loss += loss.item()
                    
        return total_loss / num_batches
    
    
    # TODO: handle other pretraining tasks
    def train_epoch_prediction(self, data_loader: DataLoader): 
        self.model.train()
        
        total_loss  = 0
        num_batches = 0
        tasks = ["alignment_prediction"]
        
        for batch in tqdm(data_loader):
            
            self.optimizer.zero_grad()
            num_batches += 1
            
            # handle data here
            task = batch["task"]
            text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
            label = batch["label"].to(self.device).float()
            label = label.unsqueeze(1)

            with torch.amp.autocast(device_type=self.device):
                #TODO fix tasks management. now is hardcoded
                prediciton_logits = self.model.forward_pretrain(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),
                    tasks=tasks
                )

                # print(prediciton_logits.shape, label.shape)
                # both have shape [batch_size, 1]

                loss = self.loss_fn_alignment(prediciton_logits, label)
                
            # loss.backward()
            # self.optimizer.step()
            total_loss += loss.item()
            
            # from karpathy video, 
            # https://www.youtube.com/watch?v=l8pRSuU81PU
            scaled_loss = self.scaler.scale(loss)
            scaled_loss.backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            

        return total_loss / num_batches
    
    def train_epoch_mlm(self, data_loader: DataLoader): 
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        task = ["mlm"]
        
        for batch in tqdm(data_loader): 
            self.optimizer.zero_grad()
            
            current_task = batch["task"]            # [bs]
            text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()} # text['input_ids'] shape: torch.Size([128, 192]
            image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
            label = batch["label"].to(self.device) 
            label = label      # [bs,  TOKENIZER_MAX_LEN]
            

            mlm_logits = self.model.forward_pretrain(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                tasks=task
            )
                        
            preds = mlm_logits                      #[bs, seq_len, vocab_size]
            preds = preds.view(-1, preds.size(-1))  # [bs*seq_len, vocab_size]
            label = label.view(-1)                  # [bs*seq_len]
            loss = self.loss_fn_mlm(preds, label)
            
            total_loss += loss.item()
            num_batches += 1
            loss.backward()
            self.optimizer.step()
            
        return total_loss / num_batches
                
    def train(
        self, 
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int
    ): 
        for epoch in range(epochs):
            # train_loss = self.train_epoch_prediction(train_dataloader)
            train_loss = self.train_epoch_mlm(data_loader=train_dataloader)
            test_loss = self.evaluate(test_dataloader)
            print(f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")
            import math
            approx_test_acc = math.exp(-test_loss)
            approx_train_acc = math.exp(-train_loss)
            print(f"Approx train acc: {approx_train_acc:.4f}, approx test acc: {approx_test_acc:.4f}")
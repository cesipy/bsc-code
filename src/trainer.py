import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vilbert import ViLBERT
from config import * 
import utils
from task import Task
from logger import Logger



from info_nce import InfoNCE, info_nce


class Trainer(): 
    def __init__(self, model: ViLBERT, config: ViLBERTConfig): 
        self.lr = config.learning_rate
        self.model = model
        self.config = config
        self.logger = Logger()
        
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
            test_loss, acc  = self.evaluate(test_dataloader)
            info_str = f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f},  accuracy: {acc:.4f}"
            print(info_str)
            self.logger.info(info_str)
        
    def evaluate(self, dataloader: DataLoader): 
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        
        total_preds = 0
        correct_preds = 0
        
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
                
                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()  # Convert to binary
                correct_preds += (preds == label).sum().item()
                total_preds   += label.size(0)
        if total_preds == 0:
            acc = 0
        else:
            acc = correct_preds / total_preds
                          
        return total_loss / num_batches, acc
                
                
                    
class PretrainingTrainer:
    def __init__(
        self, 
        model: ViLBERT, 
        config: ViLBERTConfig,
    ): 
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=config.learning_rate, 
            weight_decay=0.01
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        
        # self.model = torch.compile(self.model)  
        
        self.loss_fn_alignment = nn.BCEWithLogitsLoss()
        self.loss_fn_mlm = nn.CrossEntropyLoss()
        # self.loss_fn_mim = utils.InfoNCE(temperature=0.07)
        self.loss_fn_mim = InfoNCE()
        
        self.scaler = torch.amp.grad_scaler.GradScaler(device=self.device)
        self.config = config
        self.logger = Logger()
        
        
    def evaluate_mlm(self, dataloader: DataLoader):
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
                


                total_loss += loss.item()
                    
        return total_loss / num_batches
    
    
    def evaluate_ap(self, dataloader: DataLoader):
        self.model.eval()
        
        total_loss = 0
        num_batches = 0
        total_preds = 0
        correct_preds = 0
        
        with torch.no_grad():
            for batch in dataloader:
                num_batches += 1
                
                current_task = batch["task"][0].item()  # Get task type
                text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
                image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
                label = batch["label"].to(self.device)

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
                
                preds = torch.sigmoid(prediction_logits)
                preds = (preds > 0.5).float()  # Convert to binary
                correct_preds += (preds == label).sum().item()
                total_preds   += label.size(0)
                total_loss += loss.item()
                
        if total_preds == 0: 
            acc = 0
        else: 
            acc = correct_preds / total_preds
                    
        return total_loss / num_batches, acc
    

    def evaluate_mim(self, dataloader: DataLoader): 
        self.model.eval()
        
        total_loss  = 0
        num_batches = 0
        
        with torch.no_grad(): 
            for batch in dataloader: 
                
                tasks = ["mim"]

                text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
        
                original_image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
                masked_image = {k: v.squeeze(1).to(self.device) for k, v in batch["masked_img"].items()}
                
                # Mask indices - [bs, 196] where 1 = masked, 0 = unmasked
                masked_patches_idxs = batch["masked_patches_idxs"].to(self.device)  # [bs, 196]
                
                
                text_seqs_masked, img_seqs_masked = self.model.forward_pretrain(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=masked_image["pixel_values"],
                    image_attention_mask=masked_image.get("attention_mask", None),
                    tasks=tasks
                )
                
                text_seqs_unmasked, img_seqs_unmasked = self.model.forward_pretrain(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=original_image["pixel_values"],
                    image_attention_mask=original_image.get("attention_mask", None),
                    tasks=tasks
                )
                
                loss = self.compute_mim_loss(
                    img_seqs_unmasked=img_seqs_unmasked,
                    img_seqs_masked=img_seqs_masked,
                    masked_patches_idxs=masked_patches_idxs
                )
                
                total_loss += loss.item()
                num_batches += 1
                
            return total_loss / num_batches
                
            
    

    def train_epoch_prediction(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            num_batches += 1
            
            loss = self.train_epoch_prediction_batch(batch)
            total_loss += loss
            
        return total_loss / num_batches
    
    def train_epoch_mlm(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in dataloader:
            num_batches += 1
            
            loss = self.train_epoch_mlm_batch(batch)
            total_loss += loss
            
        return total_loss / num_batches
    
    
    def train_epoch_prediction_batch(self, batch): 
        """trains only one batch"""
        
        tasks = ["alignment_prediction"]

        self.optimizer.zero_grad()
        
        # handle data here
        current_task = batch["task"]
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
    
        
        # from karpathy video, 
        # https://www.youtube.com/watch?v=l8pRSuU81PU
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch_mlm_batch(self, batch): 
        """trains only one batch"""

        task = ["mlm"]
        
        self.optimizer.zero_grad()
        
        current_task = batch["task"]            # [bs]
        text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()} # text['input_ids'] shape: torch.Size([128, 192]
        image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
        label = batch["label"].to(self.device) 
        label = label      # [bs,  TOKENIZER_MAX_LEN]
        
        with torch.amp.autocast(device_type=self.device):
            mlm_logits = self.model.forward_pretrain(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                tasks=task
            )
                    
            preds = mlm_logits                      # [bs, seq_len, vocab_size]
            preds = preds.view(-1, preds.size(-1))  # [bs*seq_len, vocab_size]
            label = label.view(-1)                  # [bs*seq_len]
            loss = self.loss_fn_mlm(preds, label)
        

        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    def train_epoch_mim_batch(self, batch): 
        
        self.optimizer.zero_grad()
        
        task = ["mim"]
        
        current_task = batch["task"]            # [bs]
        text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
        
        original_image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
        masked_image = {k: v.squeeze(1).to(self.device) for k, v in batch["masked_img"].items()}
        
        # Mask indices - [bs, 196] where 1 = masked, 0 = unmasked
        masked_patches_idxs = batch["masked_patches_idxs"].to(self.device)  # [bs, 196]
        
        with torch.amp.autocast(device_type=self.device):
            text_seqs_masked, img_seqs_masked = self.model.forward_pretrain(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=masked_image["pixel_values"],
                image_attention_mask=masked_image.get("attention_mask", None),
                tasks=task
            )
            
            text_seqs_unmasked, img_seqs_unmasked = self.model.forward_pretrain(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=original_image["pixel_values"],
                image_attention_mask=original_image.get("attention_mask", None),
                tasks=task
            )
            
            loss = self.compute_mim_loss(
                img_seqs_unmasked=img_seqs_unmasked, 
                img_seqs_masked=img_seqs_masked, 
                masked_patches_idxs=masked_patches_idxs
            )
            
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return loss.item()
    
    # TODO: clean this up
    def train_epoch_mim(
        self, 
        dataloader_mim: DataLoader,
    ): 
        total_loss = 0
        num_batches = 0
        for batch in dataloader_mim: 
            loss = self.train_epoch_mim_batch(batch)
            total_loss += loss
            num_batches += 1
            
        return total_loss / num_batches
        

    def train_epoch(
        self, 
        dataloader_ap: DataLoader,
        dataloader_mlm: DataLoader,
        dataloader_mim: DataLoader,
        tasks: list[Task],
    ):
        assert len(dataloader_ap) == len(dataloader_mlm), "something is wrong, are the same dataset!"
        assert len(dataloader_ap) == len(dataloader_mim), "something is wrong, are the same dataset!"
        self.model.train()
        total_loss_ap  = 0
        total_loss_mlm = 0
        total_loss_mim = 0
        num_batches = 0
        
        total_batches = len(dataloader_ap)
        
        for batch_ap, batch_mlm, batch_mim in tqdm(
            zip(dataloader_ap, dataloader_mlm, dataloader_mim), 
            total=total_batches):
            
            loss_ap = 0
            loss_mlm = 0
            loss_mim = 0
            if Task.ALIGNMENT_PREDICTION in tasks:
                # alignment prediction
                loss_ap = self.train_epoch_prediction_batch(batch_ap)
                
            if Task.MASKED_LM in tasks:
                # masked language modeling
                loss_mlm = self.train_epoch_mlm_batch(batch_mlm)
                
            if Task.MASKED_IM in tasks:
                loss_mim = self.train_epoch_mim_batch(batch_mim)

            total_loss_ap += loss_ap
            total_loss_mlm += loss_mlm
            total_loss_mim += loss_mim
            num_batches += 1
            
        avg_loss_ap = total_loss_ap / num_batches
        avg_loss_mlm = total_loss_mlm / num_batches
        avg_loss_mim = total_loss_mim / num_batches
        return avg_loss_ap, avg_loss_mlm, avg_loss_mim
    
    
    def train_mim(
        self, 
        train_dataloader: DataLoader,
        test_datalaoder: DataLoader,
        epochs: int
    ): 
        for i in range(epochs): 
            train_loss = self.train_epoch_mim(train_dataloader)
            val_loss   = self.evaluate_mim(test_datalaoder)
            info_str = f"Epoch {i+1}/{epochs}, train loss MIM: {train_loss:.4f}, validation loss MIM: {val_loss:.4f}"
            print(info_str)
                
    def train(
        self, 
        train_dataloaderAP: DataLoader,     #alignment prediction
        test_dataloaderAP: DataLoader,      #alignment prediction
        train_dataloaderMLM: DataLoader,    #masked language modeling
        test_dataloaderMLM: DataLoader,     #masked language modeling
        train_dataloaderMIM: DataLoader,    #masked image modeling
        test_dataloaderMIM:  DataLoader,        #masked image modeling
        epochs: int, 
    ): 
        info_str = f"training with tasks: {self.config.pretraining_tasks}"
        self.logger.info(info_str)
        print(info_str)
        
        train_losses_ap= []
        validation_losses_ap = []
        train_losses_mlm = []
        validation_losses_mlm = []
        train_losses_mim = []
        validation_losses_mim = []
        
        for epoch in range(epochs):            
            t_loss_ap, t_loss_mlm, t_loss_mim = self.train_epoch(
                dataloader_ap=train_dataloaderAP,
                dataloader_mlm=train_dataloaderMLM, 
                dataloader_mim=train_dataloaderMIM,
                tasks=self.config.pretraining_tasks,
            )
            
            v_loss_ap, acc = self.evaluate_ap(test_dataloaderAP)
            v_loss_mlm = self.evaluate_mlm(test_dataloaderMLM)
            v_loss_mim = self.evaluate_mim(test_dataloaderMIM)
            
            # if Task.ALIGNMENT_PREDICTION in tasks:
            #     v_loss_ap, acc = self.evaluate_ap(test_dataloaderAP)
            # else:
            #     v_loss_ap, acc = 0, 0
                
            # if Task.MASKED_LM in tasks:
            #     v_loss_mlm = self.evaluate_mlm(test_dataloaderMLM)
            # else:
            #     v_loss_mlm = 0
                
            # if Task.MASKED_IM in tasks:
            #     v_loss_mim = self.evaluate_mim(test_dataloaderMIM)
            # else:
            #     v_loss_mim = 0
            # print(f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}")
            info_str = (
                f"Epoch {epoch+1}/{epochs}, "
                f"\n\ttrain loss MLM: {t_loss_mlm:.4f}, "
                f"\n\ttest loss MLM: {v_loss_mlm:.4f}, "
                f"\n\ttrain loss AP: {t_loss_ap:.4f}, "
                f"\n\ttest loss AP: {v_loss_ap:.4f}, "
                f"\n\taccuracy AP: {acc:.4f}"
                f"\n\ttrain loss MIM: {t_loss_mim:.4f}, "
                f"\n\ttest loss MIM: {v_loss_mim:.4f}"
            )
            print(info_str)
            self.logger.info(info_str)
            
            self.__save_checkpoint(
                filepath=f"res/checkpoints/pretrained_{epoch+1}.pt", 
                epoch=epoch, 
                train_loss_ap=t_loss_ap,            # do i really need those?
                train_loss_mlm=t_loss_mlm,
            )
            
            train_losses_ap.append(t_loss_ap)
            validation_losses_ap.append(v_loss_ap)
            train_losses_mlm.append(t_loss_mlm)
            validation_losses_mlm.append(v_loss_mlm)
            train_losses_mim.append(t_loss_mim)
            validation_losses_mim.append(v_loss_mim)
        
        utils.plot_losses(
            train_losses_ap=train_losses_ap,
            validation_losses_ap=validation_losses_ap,
            train_losses_mlm=train_losses_mlm,
            validation_losses_mlm=validation_losses_mlm,
            train_losses_mim=train_losses_mim,
            validation_losses_mim=validation_losses_mim,
        )
                
    def __save_checkpoint(self, filepath, epoch, train_loss_ap, train_loss_mlm):
        """Save model checkpoint with training state
        generated by genAI"""
        
        if hasattr(self.model, "_orig_mod"):
            # even if model is compiled, the orig_mod is also updated when training
            # compiled model cannot be loaded.
            model_state_dict = self.model._orig_mod.state_dict()
        else: 
            model_state_dict = self.model.state_dict()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scaler_state_dict': self.scaler.state_dict(),
            'train_loss_ap': train_loss_ap,
            'train_loss_mlm': train_loss_mlm,
            'config': self.config.__dict__ if hasattr(self, 'config') else None
        }
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved to {filepath}")
        self.logger.info(f"Checkpoint saved to {filepath}")
        
        
    
    
        

    def compute_mim_loss(
        self, 
        img_seqs_masked, 
        img_seqs_unmasked, 
        masked_patches_idxs,
    ): 
        
        # print(f"shape img_seqs_masked: {img_seqs_masked.shape}, shape img_seqs_unmasked: {img_seqs_unmasked.shape}, shape masked_patches_idxs: {masked_patches_idxs.shape}")
        
        # img_seqs_masked: [bs, num_patches+1, 768]
        # img_seqs_unmasked: [bs, num_patches+1, 768]
        # masked_patches_idxs: [bs, num_patches]
        # the other have the same dimensions.
        
        batch_size = img_seqs_masked.shape[0]
        
        # i implemented a sequential code, that iterated over all the batches
        # and then computed the loss. 
        # genAI came up with the more efficient method for gpus
        
        # only compute loss for the unmasked patches, so masked_patches_idxs = 0
        unmasked_patches: torch.Tensor[bool] = (masked_patches_idxs == 0)  # [bs, 196] -> if unmasked = 0
        num_unmasked = unmasked_patches.sum(dim=1, keepdim=True)  # [bs, 1]
        # print(f"num_unmasked: {num_unmasked}")
        # print(f"shape unmasked_patches: {unmasked_patches.shape}")
        # print(f"unmasked_patches: {unmasked_patches}")
        
        # add CLS token (always unmasked) to the beginning
        cls_mask = torch.ones(batch_size, 1, device=self.device, dtype=torch.bool)  # [bs, 1]
        full_unmasked_mask: torch.Tensor[bool] = torch.cat([cls_mask, unmasked_patches], dim=1)  # [bs, 197]
        
        # apply mask to select only unmasked positions for all samples
        masked_feats = img_seqs_masked[full_unmasked_mask]      # [total_unmasked, 768]
        unmasked_feats = img_seqs_unmasked[full_unmasked_mask]  # [total_unmasked, 768]
        
        # print(f"shape: masked_feats: {masked_feats.shape}, unmasked_feats: {unmasked_feats.shape}")
        assert masked_feats.shape == unmasked_feats.shape 
        # has shape between bs * ~177: [bs*~177, 768]
        # loss = torch.nn.functional.mse_loss(masked_feats, unmasked_feats)
        loss = self.loss_fn_mim(masked_feats, unmasked_feats)
        
        return loss
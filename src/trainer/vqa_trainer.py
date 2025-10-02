import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformers import (
     # ViT stuff
    BaseImageProcessor,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
    BertTokenizerFast
)

from vilbert import ViLBERT
from config import *
import utils
from logger import Logger
from .base_trainer import BaseTrainer

from datasets import VqaDataset

class VQATrainer(BaseTrainer):
    def __init__(
        self,
        model: ViLBERT,
        config: ViLBERTConfig,
        gradient_accumulation: int = 1,
    ):
        self.lr = config.learning_rate
        self.model = model
        self.config = config
        self.logger = Logger()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.scheduler = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01           #TODO: make configurable
        )

        self.model = self.model.to(self.device)
        self.loss_fn = nn.CrossEntropyLoss()
        self.gradient_accumulation = gradient_accumulation

    def setup_scheduler(self, epochs: int, train_dataloader: DataLoader, lr=None):
        if lr is None:
            lr = self.lr

        total_training_steps = epochs * len(train_dataloader) // self.gradient_accumulation
        self.scheduler = utils.Scheduler(
            warmup_iterations=int(WARMUP_ITERATIONS * float(total_training_steps)),
            decay_iterations=int(DECAY_ITERATIONS * float(total_training_steps)),
            learning_rate=lr,
            min_lr_fraction=MIN_LR_FRACTION,
        )

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int,
        hm_dataloader=None,
        cc_dataloader=None,
    ):
        self.setup_scheduler(epochs=epochs, train_dataloader=train_dataloader)

        for epoch in range(epochs):
            train_loss = self.train_epoch(dataloader=train_dataloader)
            test_loss, acc = self.evaluate(test_dataloader)

            info_str = f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f}, accuracy: {acc:.4f}"
            print(info_str)
            self.logger.info(info_str)

            import analysis
            info_str = "alignment analysis on test ds"
            analysis.analyse_alignment(model=self.model, dataloader=test_dataloader)

    def train_epoch(self, dataloader: DataLoader):

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Training VQA")):
            num_batches += 1


            label = batch["label"].to(self.device)  # Shape: [batch_size]
            text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
            image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}

            text_embedding, image_embedding = self.model(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
            )

            # same approach as in vilbert paper. instead of concat, do hadamard
            fused_representation = self.get_final_representation(text_embedding, image_embedding)
            pred = self.model.fc_vqa(fused_representation)  # [bs, 13]

            loss = self.loss_fn(pred, label)  # pred: logits, label-class indices

            loss /= self.gradient_accumulation
            loss.backward()

            if (batch_idx + 1) % self.gradient_accumulation == 0 or (batch_idx + 1) == len(dataloader):
                lr = self.scheduler.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.step()
                self.optimizer.zero_grad()

                #TODO: remove
                # if (batch_idx + 1) % 10 == 0:
                #     break

            total_loss += loss.item() * self.gradient_accumulation
        return total_loss / num_batches

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating VQA", leave=False):
                label = batch["label"].to(self.device)
                text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
                image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}

                text_embedding, image_embedding = self.model(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),

                )

                fused_representation = self.get_final_representation(text_embedding, image_embedding)
                pred = self.model.fc_vqa(fused_representation)

                loss = self.loss_fn(pred, label)
                total_loss += loss.item()
                num_batches += 1


                pred_classes = torch.argmax(pred, dim=1)
                all_preds.append(pred_classes.cpu())
                all_labels.append(label.cpu())

        avg_loss = total_loss / num_batches

        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)
        acc = (all_preds == all_labels).float().mean().item()

        return avg_loss, acc


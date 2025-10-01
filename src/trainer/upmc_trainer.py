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

from datasets import UPMC_Dataset, PretrainDatasetAP
import analysis
from datasets import HM_Dataset, PretrainDatasetAP, MM_IMDB_Dataset; import datasets

import augments_transforms

from info_nce import InfoNCE, info_nce



class UPMCTrainer(BaseTrainer):
    def __init__(
        self,
        model: ViLBERT,
        config: ViLBERTConfig,
        gradient_accumulation: int = 1,
        use_contrastive_loss:bool=False,
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
        self.info_nce = InfoNCE()
        self.gradient_accumulation = gradient_accumulation
        self.use_contrastive_loss = use_contrastive_loss

        self.total_losses = []
        self.info_losses = []
        self.normal_losses = []

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int,
        analyze_alignment: bool = False,
        dataloader: UPMC_Dataset=None,
        cc_dataloader: PretrainDatasetAP=None,
    ):

        self.setup_scheduler(epochs=epochs, train_dataloader=train_dataloader)
        total_training_steps = epochs * len(train_dataloader) // self.gradient_accumulation
        self.scheduler = utils.Scheduler(
            warmup_iterations=int(WARMUP_ITERATIONS * float(total_training_steps)),
            decay_iterations=int(DECAY_ITERATIONS * float(total_training_steps)),
            learning_rate=self.lr,
            min_lr_fraction=MIN_LR_FRACTION,
        )

        if analyze_alignment and (dataloader is not None or cc_dataloader is not None):
            info_str = "alignment for mm_imdb:"
            print(info_str)
            self.logger.info(info_str)
            analysis.analyse_alignment(dataloader, self.model)
            # analysis.visualize_cka(dataloader=hm_dataloader, model=self.model)

            info_str = "\n----------\nalignment for conceptual captions:"
            print(info_str)
            self.logger.info(info_str)
            analysis.analyse_alignment(cc_dataloader, self.model)
            # analysis.visualize_cka(dataloader=cc_dataloader, model=self.model)




        for epoch in range(epochs):
            train_loss = self.train_epoch(dataloader=train_dataloader)
            test_loss, acc  = self.evaluate(test_dataloader)
            info_str = f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f},  accuracy: {acc:.4f}"
            print(info_str)
            self.logger.info(info_str)

            if analyze_alignment and (dataloader is not None and cc_dataloader is not None):
                info_str = "alignment for mm_imdb:"
                print(info_str)
                self.logger.info(info_str)
                analysis.analyse_alignment(dataloader, self.model)
                # analysis.visualize_cka(dataloader=hm_dataloader, model=self.model)

                info_str = "\n----------\nalignment for conceptual captions:"
                print(info_str)
                self.logger.info(info_str)
                analysis.analyse_alignment(cc_dataloader, self.model)
                # analysis.visualize_cka(dataloader=cc_dataloader, model=self.model)


    def setup_scheduler(self, epochs:int, train_dataloader: DataLoader, lr=None):
        if lr is None:
            lr = self.lr

        total_training_steps = epochs * len(train_dataloader) // self.gradient_accumulation
        self.scheduler = utils.Scheduler(
            warmup_iterations=int(WARMUP_ITERATIONS * float(total_training_steps)),
            decay_iterations=int(DECAY_ITERATIONS * float(total_training_steps)),
            learning_rate=lr,
            min_lr_fraction=MIN_LR_FRACTION,
        )

    def train_epoch(self, dataloader: DataLoader):
        info_str = f"simulated batchsize: {dataloader.batch_size * self.gradient_accumulation}, actual batchsize: {dataloader.batch_size}"
        print(info_str)
        self.logger.info(info_str)

        self.model.train()

        total_loss = 0
        num_batches = 0



        buffer_info_loss = []
        buffer_normal_loss = []
        buffer_total_loss = []

        for batch_indx, batch in enumerate(tqdm(dataloader, desc="Training")):
            num_batches += 1

            data_dict = batch

            label = data_dict["label"].to(self.device)
            text = {k: v.squeeze(1).to(self.device) for k, v in data_dict["text"].items()}
            image = {k: v.squeeze(1).to(self.device) for k, v in data_dict["img"].items()}

            label = torch.argmax(label, dim=1)  # crossentropy wants class indices, not one-hot

            if self.use_contrastive_loss:

                text_embedding, image_embedding = self.model(
                    text_input_ids = text["input_ids"],
                    text_attention_mask = text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                )
                fused_representation = self.get_final_representation(text_embedding, image_embedding)
                # assert combined.shape == (dataloader.batch_size, self.config.embedding_dim)
                preds = self.model.fc_upmc(fused_representation)

                # print(f"preds: {preds.shape}, label: {label.shape}")
                loss_normal = self.loss_fn(preds, label)

                loss_info = self.info_nce(text_embedding, image_embedding)

                loss = utils.get_weighted_loss(info_nce_loss=loss_info, normal_loss=loss_normal, naive_weighting=True)

                buffer_info_loss.append(loss_info.item())
                buffer_normal_loss.append(loss_normal.item())
                buffer_total_loss.append(loss.item())


                if (batch_indx +1) % 10 == 0:
                    avg_info_loss = sum(buffer_info_loss) / len(buffer_info_loss)
                    avg_normal_loss = sum(buffer_normal_loss) / len(buffer_normal_loss)
                    avg_total_loss = sum(buffer_total_loss) / len(buffer_total_loss)

                    self.info_losses.append(avg_info_loss)
                    self.normal_losses.append(avg_normal_loss)
                    self.total_losses.append(avg_total_loss)

                    buffer_info_loss = []
                    buffer_normal_loss = []
                    buffer_total_loss = []
            else:
                text_embedding, image_embedding = self.model(
                    text_input_ids = text["input_ids"],
                    text_attention_mask = text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                )

                fused_representation = self.get_final_representation(text_embedding, image_embedding)
                # assert combined.shape == (dataloader.batch_size, self.config.embedding_dim)
                preds = self.model.fc_upmc(fused_representation)
                # print(f"preds: {preds.shape}, label: {label.shape}")


                loss = self.loss_fn(preds, label)

            loss /= self.gradient_accumulation

            loss.backward()

            if (batch_indx +1) % self.gradient_accumulation == 0 or (batch_indx+1) == len(dataloader):
                lr = self.scheduler.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation

        if self.use_contrastive_loss:
            utils.visualize_loss(
                info_losses=self.info_losses,
                normal_losses=self.normal_losses,
                total_losses=self.total_losses,
            )
        return total_loss / num_batches

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for i,batch in enumerate(tqdm(dataloader, desc="Evaluating UPMC", leave=False)):
                label = batch["label"].to(self.device)
                text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
                image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
                label = torch.argmax(label, dim=1)
                text_embedding, image_embedding = self.model(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),
                )

                fused_representation = self.get_final_representation(text_embedding, image_embedding)

                pred = self.model.fc_upmc(fused_representation)

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
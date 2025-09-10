import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from vilbert import ViLBERT
from config import *
import utils
from task import Task
from logger import Logger

from transformers import (
     # ViT stuff
    BaseImageProcessor,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
    BertTokenizerFast
)


import analysis
from datasets import CustomDataset, PretrainDatasetAP, MM_IMDB_Dataset





class MM_IMDB_Trainer():
    def __init__(
        self,
        model: ViLBERT,
        config: ViLBERTConfig,
        gradient_accumulation:int=1,        # how many batches to accumulate!
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
            weight_decay=0.01           #TODO: make weight decay configurable in config
        )

        self.model = self.model.to(self.device)

        self.loss_fn = nn.BCEWithLogitsLoss()

        self.gradient_accumulation = gradient_accumulation

    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        epochs: int,
        #for alingment testing
        hm_dataloader: CustomDataset=None,
        cc_dataloader: PretrainDatasetAP=None,
    ):
        total_training_steps = epochs * len(train_dataloader) // self.gradient_accumulation
        self.scheduler = utils.Scheduler(
            warmup_iterations=int(WARMUP_ITERATIONS * float(total_training_steps)),
            decay_iterations=int(DECAY_ITERATIONS * float(total_training_steps)),
            learning_rate=self.lr,
            min_lr_fraction=MIN_LR_FRACTION,
        )

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            test_loss, acc  = self.evaluate(test_dataloader)
            info_str = f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f},  accuracy: {acc:.4f}"
            print(info_str)
            self.logger.info(info_str)

    def train_epoch(self, data_loader: DataLoader):

        info_str = f"simulated batchsize: {data_loader.batch_size * self.gradient_accumulation}, actual batchsize: {data_loader.batch_size}"
        print(info_str)
        self.logger.info(info_str)

        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch_indx, batch in enumerate(tqdm(data_loader, desc="Training")):
            num_batches += 1

            data_dict = batch

            label = data_dict["label"].to(self.device)
            # its not possible to send dicts to device, so do it for every value in dict.
            text = {k: v.squeeze(1).to(self.device) for k, v in data_dict["text"].items()}
            image = {k: v.squeeze(1).to(self.device) for k, v in data_dict["img"].items()}

            preds, text_embedding, image_embedding = self.model(
                text_input_ids= text["input_ids"],
                text_attention_mask= text["attention_mask"],
                text_token_type_ids= text.get("token_type_ids", None),
                image_pixel_values= image["pixel_values"],
                image_attention_mask= image.get("attention_mask", None),
                output_invididual_embeddings=True
            )

            combined = text_embedding * image_embedding

            pred = self.model.fc_imdb(combined)
            # print(f"pred shape: {pred.shape}")

            # print(pred)

            loss = self.loss_fn(pred, label.float())

            loss /= self.gradient_accumulation
            loss.backward()



            if (batch_indx + 1) % self.gradient_accumulation == 0 or (batch_indx + 1) == len(data_loader):
                lr = self.scheduler.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation  # to account for division above

        return total_loss / num_batches

    def evaluate(self, data_loader: DataLoader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        all_preds = []
        all_labels = []

        loss_fn = nn.BCEWithLogitsLoss()

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                data_dict = batch

                label = data_dict["label"].to(self.device)
                text = {k: v.squeeze(1).to(self.device) for k, v in data_dict["text"].items()}
                image = {k: v.squeeze(1).to(self.device) for k, v in data_dict["img"].items()}

                preds, text_embedding, image_embedding = self.model(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),
                    output_invididual_embeddings=True
                )

                combined = text_embedding * image_embedding
                pred = self.model.fc_imdb(combined)

                loss = loss_fn(pred, label)
                total_loss += loss.item()
                num_batches += 1

                # For multi-label classification (convert logits to binary predictions)
                preds = (torch.sigmoid(pred) > 0.5).float()
                all_preds.append(preds.cpu())
                all_labels.append(label.cpu())

        avg_loss = total_loss / num_batches

        # Calculate metrics
        all_preds = torch.cat(all_preds, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # For multi-label classification, we can calculate:
        # 1. Exact match accuracy (all genres must match)
        exact_match = (all_preds == all_labels).all(dim=1).float().mean().item()

        # 2. Hamming accuracy (percentage of correct genre predictions)
        hamming_acc = (all_preds == all_labels).float().mean().item()

        # Choose which metric to return
        acc = hamming_acc  # or exact_match depending on your preference

        return avg_loss, acc



def main():
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    path = "res/data/mm-imdb/images.h5"
    csv_path = "res/data/mm-imdb/mmimdb_trainval.csv"

    train_dataset = MM_IMDB_Dataset(
        csv_path=csv_path,
        img_path=path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        train_test_ratio=0.2,
        is_train=True
    )

    val_dataset = MM_IMDB_Dataset(
        csv_path=csv_path,
        img_path=path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,
        prefetch_factor=4,
        num_workers=4,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=16,
        shuffle=False,
        prefetch_factor=4,
        num_workers=4,
    )

    config = ViLBERTConfig()
    model = ViLBERT(config=config)

    trainer = MM_IMDB_Trainer(model=model, config=config, gradient_accumulation=32)
    trainer.train(
        train_dataloader=train_dataloader,
        test_dataloader=val_dataloader,
        epochs=3
    )



if __name__ == "__main__":
    main()
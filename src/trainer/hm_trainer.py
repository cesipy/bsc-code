# from .base_trainer import (
#     torch, DataLoader,
#     BaseTrainer, Logger, utils,config,
#     ViLBERTConfig, ViLBERT,
#     info_nce, InfoNCE,
# )
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
from .base_trainer import BaseTrainer


import analysis
from datasets import HM_Dataset, PretrainDatasetAP, MM_IMDB_Dataset; import datasets

import augments_transforms

from info_nce import InfoNCE, info_nce



def alignment_loss_cosine(text_emb, vision_emb):
    cosine_sim = torch.nn.functional.cosine_similarity(
        text_emb,
        vision_emb, dim=1
        )

    return -cosine_sim.mean()


class HatefulMemesTrainer(BaseTrainer):

    def __init__(
        self,
        model: ViLBERT,
        config: ViLBERTConfig,
        use_contrastive_loss:bool=False,
        use_cosine_loss:bool=False,
        gradient_accumulation:int=1,        # how many batches to accumulate!
        ):
        self.lr = config.learning_rate
        self.model = model
        self.config = config
        self.logger = Logger()

        self.scheduler = None

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=0.01           #TODO: make weight decay configurable in config
        )
        assert (use_contrastive_loss and use_cosine_loss) == False, "can only use one of the two losses at once"

        self.loss_fn = torch.nn.BCEWithLogitsLoss()
        self.info_nce = InfoNCE()
        self.cosine_loss = alignment_loss_cosine
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

        self.use_contrastive_loss = use_contrastive_loss
        self.use_cosine_loss = use_cosine_loss

        if self.use_contrastive_loss or self.use_cosine_loss:
            info_str = f"using contrastive loss: {self.use_contrastive_loss}, using cosine loss: {self.use_cosine_loss}"
            print(info_str)
            self.logger.info(info_str)

        self.gradient_accumulation = gradient_accumulation

        self.total_losses = []
        self.info_losses = []
        self.normal_losses = []




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
        for batch_indx,batch in enumerate(tqdm(dataloader,total=len(dataloader), desc="training")):
            num_batches += 1

            data_dict = batch

            label = data_dict["label"].to(self.device)
            # its not possible to send dicts to device, so do it for every value in dict.
            text = {k: v.squeeze(1).to(self.device) for k, v in data_dict["text"].items()}
            image = {k: v.squeeze(1).to(self.device) for k, v in data_dict["img"].items()}

            # #-------
            # #TODO: temp debugging

            # pxl_vals = image["pixel_values"]
            # print(f"Min: {pxl_vals.min().item()}, Max: {pxl_vals.max().item()}, Mean: {pxl_vals.mean().item()}, Std: {pxl_vals.std().item()}")
            # #------

            if self.use_contrastive_loss:

                text_embedding, vision_embedding = self.model(
                    text_input_ids= text["input_ids"],
                    text_attention_mask= text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                )


                fused_representation = self.get_final_representation(text_embedding, vision_embedding)
                preds = self.model.fc(fused_representation)
                preds = preds.squeeze()     #[bs]
                label = label.float()       #[bs]


                loss_info = self.info_nce(text_embedding, vision_embedding)
                loss_normal = self.loss_fn(preds, label)
                # print(f"loss info: {loss_info}, loss normal: {loss_normal}")

                # loss = loss_normal + 0.3 * loss_info   # with info nce
                # #loss info: 1.9934707880020142, loss normal: 0.6653898358345032

                loss = utils.get_weighted_loss(
                    info_nce_loss=loss_info,
                    normal_loss=loss_normal,
                    naive_weighting=True,
                )
                # print(f"weighted loss: {loss}", end="\n\n------\n")

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





            elif self.use_cosine_loss:

                text_embedding, image_embedding = self.model(
                    text_input_ids= text["input_ids"],
                    text_attention_mask= text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                )
                fused_representation = self.get_final_representation(text_embedding, image_embedding)
                preds = self.model.fc(fused_representation)

                preds = preds.squeeze()
                label = label.float()


                loss_cosine = self.cosine_loss(text_embedding, image_embedding)
                loss_normal = self.loss_fn(preds, label)


                loss = loss_normal + 0.3 * loss_cosine  #cosine loss




            else:       # normal loss, no contrastive, no infonce loss term

                text_embedding, image_embedding = self.model(
                    text_input_ids= text["input_ids"],
                    text_attention_mask= text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                )
                fused_representation = self.get_final_representation(text_embedding, image_embedding)
                preds = self.model.fc(fused_representation)


                preds = preds.squeeze()
                label = label.float()
                loss_normal = self.loss_fn(preds, label)

                loss = loss_normal

            loss /= self.gradient_accumulation
            loss.backward()

            if (batch_indx + 1) % self.gradient_accumulation == 0 or (batch_indx + 1) == len(dataloader):
                lr = self.scheduler.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                self.optimizer.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation  # to account for division above



        # temp:
        if self.use_contrastive_loss:
            utils.visualize_loss(info_losses=self.info_losses, normal_losses=self.normal_losses, total_losses=self.total_losses)


        return total_loss / num_batches

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


    def train(
        self,
        train_dataloader: DataLoader,
        test_dataloader:  DataLoader,
        epochs: int,
        analyze_alignment: bool=False,
        dataloader: HM_Dataset=None,
        cc_dataloader: PretrainDatasetAP=None,
    ):
        self.setup_scheduler(epochs=epochs, train_dataloader=train_dataloader)
        # analysis.analyse_alignment(dataloader=hm_dataloader, model=self.model)
        # analysis.visualize_cka(dataloader=hm_dataloader, model=self.model)
        # do one check with the alignment dataloaders before starting training
        if analyze_alignment and (dataloader is not None and cc_dataloader is not None):
            info_str = "\n\nbefore training, evaluating on uninitialized model"
            print(info_str)
            self.logger.info(info_str)
            info_str = "alignment for hateful memes:"
            print(info_str)
            self.logger.info(info_str)
            analysis.analyse_alignment(dataloader, self.model)

            info_str = "\n----------\nalignment for conceptual captions:"
            print(info_str)
            self.logger.info(info_str)
            analysis.analyse_alignment(cc_dataloader, self.model)

                # info_str = "finished!" + "\n" + 20*"-"
                # print(info_str)
                # self.logger.info(info_str)

                # analysis.visualize_cka(dataloader=hm_dataloader, model=self.model)

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_dataloader)
            test_loss, acc  = self.evaluate(test_dataloader)
            info_str = f"Epoch {epoch+1}/{epochs}, train loss: {train_loss:.4f}, test loss: {test_loss:.4f},  accuracy: {acc:.4f}"
            print(info_str)
            self.logger.info(info_str)

            #TODO: only temp!
            checkpoints_path = "res/checkpoints"
            self.model.save_model(save_path=checkpoints_path + f"/hm_finetuned_e{epoch}.pt")


            if analyze_alignment and (dataloader is not None and cc_dataloader is not None):
                info_str = "alignment for hateful memes:"
                print(info_str)
                self.logger.info(info_str)
                analysis.analyse_alignment(dataloader, self.model)
                # analysis.visualize_cka(dataloader=hm_dataloader, model=self.model)

                info_str = "\n----------\nalignment for conceptual captions:"
                print(info_str)
                self.logger.info(info_str)
                analysis.analyse_alignment(cc_dataloader, self.model)
                # analysis.visualize_cka(dataloader=cc_dataloader, model=self.model)



                # info_str = "CKA alignment analysis on CC dataset:"
                # print(info_str)
                # self.logger.info(info_str)
                # # analyze_all_layers_alignment(
                # #     model=self.model,
                # #     dataloader=cc_dataloader,
                # # )
                # analyze_all_layers_alignment(
                #     model=self.model,
                #     dataloader=hm_dataloader
                # )

    def evaluate(self, dataloader: DataLoader):
        self.model.eval()

        total_loss = 0
        num_batches = 0

        total_preds = 0
        correct_preds = 0

        # layer_sims = []

        with torch.no_grad():
            for batch in dataloader:
                data_dict = batch

                label = data_dict["label"].to(self.device)
                # its not possible to send dicts to device, so do it for every value in dict.
                text = {k: v.squeeze(1).to(self.device) for k, v in data_dict["text"].items()}
                image = {k: v.squeeze(1).to(self.device) for k, v in data_dict["img"].items()}

                # preds, intermediate_representations = self.model(
                #     text_input_ids= text["input_ids"],
                #     text_attention_mask= text["attention_mask"],
                #     text_token_type_ids= text.get("token_type_ids", None),
                #     image_pixel_values= image["pixel_values"],
                #     image_attention_mask= image.get("attention_mask", None),
                #     save_intermediate_representations=True
                # )
                text_embedding, image_embedding = self.model(
                    text_input_ids= text["input_ids"],
                    text_attention_mask= text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                    save_intermediate_representations=False
                )
                fused_representation = self.get_final_representation(text_embedding, image_embedding)
                preds = self.model.fc(fused_representation)

                preds = preds.squeeze()
                label = label.float()

                loss = self.loss_fn(preds, label)
                total_loss += loss.item()
                num_batches += 1

                preds = torch.sigmoid(preds)
                preds = (preds > 0.5).float()  # convert to binary
                correct_preds += (preds == label).sum().item()
                total_preds   += label.size(0)

        # analysis.analyse(layer_similarities=layer_sims, num_layers=self.model.depth)

        if total_preds == 0:
            acc = 0
        else:
            acc = correct_preds / total_preds

        return total_loss / num_batches, acc
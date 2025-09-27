import torch
from torch import nn
from tqdm import tqdm

from vilbert import ViLBERT
from config import *
import utils
from task import Task
from logger import Logger


import analysis
from datasets import HM_Dataset, PretrainDatasetAP


from .base_trainer import (
    DataLoader, Dataset, BaseTrainer
)
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

        self.loss_fn = nn.BCEWithLogitsLoss()
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
                preds = self.model.fc(text_embedding * vision_embedding)
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

                text_embedding, vision_embedding = self.model(
                    text_input_ids= text["input_ids"],
                    text_attention_mask= text["attention_mask"],
                    text_token_type_ids= text.get("token_type_ids", None),
                    image_pixel_values= image["pixel_values"],
                    image_attention_mask= image.get("attention_mask", None),
                )
                preds = self.model.fc(text_embedding * vision_embedding)

                preds = preds.squeeze()
                label = label.float()


                loss_cosine = self.cosine_loss(text_embedding, vision_embedding)
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
                preds = self.model.fc(text_embedding * image_embedding)


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
                preds = self.model.fc(text_embedding * image_embedding)

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

class PretrainingTrainer:
    def __init__(
        self,
        model: ViLBERT,
        config: ViLBERTConfig,
        use_contrastive_ap:bool,
        tasks: list[Task] = [
            Task.ALIGNMENT_PREDICTION,
            Task.MASKED_LM,
            Task.MASKED_IM
        ],
        gradient_accumulation:int=1,
    ):
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=0.01,
        )
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = model.to(self.device)
        self.tasks = tasks

        # self.model = torch.compile(self.model)
        self.use_contrastive_ap = use_contrastive_ap

        self.loss_fn_alignment = nn.BCEWithLogitsLoss()
        self.loss_fn_mlm = nn.CrossEntropyLoss()
        # self.loss_fn_mim = utils.InfoNCE(temperature=0.07)
        self.loss_fn_mim = InfoNCE()
        self.loss_info_nce = InfoNCE()

        self.scaler = torch.amp.grad_scaler.GradScaler(device=self.device)
        self.config = config
        self.logger = Logger()
        self.scheduler = None

        self.gradient_accumulation = gradient_accumulation


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

                text_seqs, img_seqs = self.model.forward_pretrain(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),
                    tasks=["mlm"]
                )
                mlm_logits = self.model.mlm(text_seqs)
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

                text_embedding, image_embedding = self.model.forward_pretrain(
                    text_input_ids=text["input_ids"],
                    text_attention_mask=text["attention_mask"],
                    text_token_type_ids=text.get("token_type_ids", None),
                    image_pixel_values=image["pixel_values"],
                    image_attention_mask=image.get("attention_mask", None),
                    tasks=["alignment_prediction"]
                )
                label = label.float().unsqueeze(1)
                if self.use_contrastive_ap:

                    loss = self.loss_info_nce(text_embedding, image_embedding)

                    similarities = torch.mm(
                        torch.nn.functional.normalize(text_embedding, p=2, dim=1),
                        torch.nn.functional.normalize(image_embedding, p=2, dim=1).T
                    )

                    # Correct predictions are on the diagonal
                    predicted = similarities.argmax(dim=1)
                    target = torch.arange(len(similarities), device=self.device)

                    correct_preds += (predicted == target).sum().item()
                    total_preds += len(similarities)
                    total_loss += loss.item()
                else:
                    # shared_embedding = torch.cat([text_embedding, image_embedding], dim=1)
                    prediction_logits = self.model.alignment_fc(text_embedding)
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

            loss = self.train_alignment_prediction_batch(batch)
            total_loss += loss

        return total_loss / num_batches

    def train_epoch_mlm(self, dataloader: DataLoader):
        self.model.train()
        total_loss = 0
        num_batches = 0

        for batch in dataloader:
            num_batches += 1

            loss = self.train_mlm_batch(batch)
            total_loss += loss

        return total_loss / num_batches




    def train_alignment_prediction_batch(self, batch, flag_optimizer:bool=True):
        """trains only one batch"""

        tasks = ["alignment_prediction"]

        # handle data here
        current_task = batch["task"]
        text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()}
        image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
        label = batch["label"].to(self.device).float()
        label = label.unsqueeze(1)

        with torch.amp.autocast(device_type=self.device):
            #TODO fix tasks management. now is hardcoded
            text_embedding, image_embedding = self.model.forward_pretrain(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                tasks=tasks
            )

            # print(prediciton_logits.shape, label.shape)
            # both have shape [batch_size, 1]
            if self.use_contrastive_ap:
                loss = self.loss_info_nce(text_embedding, image_embedding)

            else:
                # both embeddings are only cls, so shape is [bs, dim]
                # shared_embedding = torch.cat([text_embedding, image_embedding], dim=1)
                # only on text representation
                prediction_logits = self.model.alignment_fc(text_embedding)
                loss = self.loss_fn_alignment(prediction_logits, label)

            loss /= self.gradient_accumulation

        # from karpathy video,
        # https://www.youtube.com/watch?v=l8pRSuU81PU
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()

        if flag_optimizer:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item() * self.gradient_accumulation

    def train_mlm_batch(self, batch, flag_optimizer:bool=True):
        """trains only one batch"""

        task = ["mlm"]


        current_task = batch["task"]            # [bs]
        text = {k: v.squeeze(1).to(self.device) for k, v in batch["text"].items()} # text['input_ids'] shape: torch.Size([128, 192]
        image = {k: v.squeeze(1).to(self.device) for k, v in batch["img"].items()}
        label = batch["label"].to(self.device)
        label = label      # [bs,  TOKENIZER_MAX_LEN]

        with torch.amp.autocast(device_type=self.device):
            text_seqs, vision_seqs = self.model.forward_pretrain(
                text_input_ids=text["input_ids"],
                text_attention_mask=text["attention_mask"],
                text_token_type_ids=text.get("token_type_ids", None),
                image_pixel_values=image["pixel_values"],
                image_attention_mask=image.get("attention_mask", None),
                tasks=task
            )

            # in vilbert they only use text_seqs alone
            mlm_logits = self.model.mlm(text_seqs)

            preds = mlm_logits                      # [bs, seq_len, vocab_size]
            preds = preds.view(-1, preds.size(-1))  # [bs*seq_len, vocab_size]
            label = label.view(-1)                  # [bs*seq_len]
            loss = self.loss_fn_mlm(preds, label)

        loss /= self.gradient_accumulation
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()

        if flag_optimizer:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item() * self.gradient_accumulation

    def train_mim_batch(self, batch, flag_optimizer:bool=True):

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

        loss /= self.gradient_accumulation
        scaled_loss = self.scaler.scale(loss)
        scaled_loss.backward()

        if flag_optimizer:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return loss.item() * self.gradient_accumulation

    # TODO: clean this up
    def train_epoch_mim(
        self,
        dataloader_mim: DataLoader,
    ):
        total_loss = 0
        num_batches = 0
        for batch in dataloader_mim:
            loss = self.train_mim_batch(batch)
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

        info_str = f"simulated batchsize: {dataloader_ap.batch_size * self.gradient_accumulation}, actual batchsize: {dataloader_ap.batch_size}"
        print(info_str)
        self.logger.info(info_str)


        for batch_indx, (batch_ap, batch_mlm, batch_mim) in enumerate(
            tqdm(
                zip(dataloader_ap, dataloader_mlm, dataloader_mim),
                total=total_batches
                )
            ):

            # currently set to false, as no step should happen in batch
            # too many steps and zeroing out!
            flag_optimizer=False


            loss_ap = 0
            loss_mlm = 0
            loss_mim = 0
            if Task.ALIGNMENT_PREDICTION in tasks:
                # alignment prediction
                loss_ap = self.train_alignment_prediction_batch(batch_ap, flag_optimizer=flag_optimizer)

            if Task.MASKED_LM in tasks:
                # masked language modeling
                loss_mlm = self.train_mlm_batch(batch_mlm, flag_optimizer=flag_optimizer)

            if Task.MASKED_IM in tasks:
                loss_mim = self.train_mim_batch(batch_mim, flag_optimizer=flag_optimizer)


            if (batch_indx+1) % self.gradient_accumulation == 0 or (batch_indx + 1) == total_batches:
                lr = self.scheduler.get_lr()
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr


                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

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
        hm_dataloader: HM_Dataset=None,
        cc_dataloader: PretrainDatasetAP=None,
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

        total_training_steps = epochs * len(train_dataloaderAP) // self.gradient_accumulation

        self.scheduler = utils.Scheduler(
            warmup_iterations=int(WARMUP_ITERATIONS * float(total_training_steps)),
            decay_iterations=int(DECAY_ITERATIONS * float(total_training_steps)),
            learning_rate=self.config.learning_rate,
            min_lr_fraction=MIN_LR_FRACTION,
        )

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
            task_string = ""
            tasks_vals = [task.value for task in self.tasks]
            tasks_vals.sort()
            for val in tasks_vals:
                task_string += f"{val}"
            filepath = f"res/checkpoints/pretrained_epoch{epoch+1}_task{task_string}.pt"
            self.__save_checkpoint(
                filepath=filepath,
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


            if hm_dataloader is not None and cc_dataloader is not None:
                info_str = "alignment for hateful memes:"
                print(info_str)
                self.logger.info(info_str)
                analysis.analyse_alignment(hm_dataloader, self.model)
                # TODO: rename to proper name
                analysis.visualize_cka(dataloader=hm_dataloader, model=self.model)

                info_str = "alignment for conceptual captions:"
                print(info_str)
                self.logger.info(info_str)
                analysis.analyse_alignment(cc_dataloader, self.model)


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

        self.model.save_model(save_path=filepath)






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
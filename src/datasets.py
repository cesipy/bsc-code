import os
import json
import typing;
import csv
import random



from transformers import (
     # ViT stuff
    BaseImageProcessor,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
    BertTokenizerFast
)
import torchvision; from torchvision import transforms

import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from PIL import Image, UnidentifiedImageError

import utils
from task import Task
from config import *

import warnings

from logger import Logger

logger = Logger()

# disable PIL's decompression bomb warning, bc i get the following:
# DecompressionBombWarning: Image size (93950400 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
warnings.filterwarnings("ignore", ".*DecompressionBombWarning.*", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", ".*Palette images with Transparency.*", category=UserWarning)


Image.MAX_IMAGE_PIXELS = None


def process_single_image(path:str) -> torch.Tensor:
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0  # norm to [0, 1]

    img = np.transpose(img, (2, 0, 1))  # change to CxHxW format (the opencv format)
    img_tensor = torch.from_numpy(img.astype(np.float32))

    return img_tensor

def get_alignment_dataloaders(
    batch_size,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    num_samples:int = 1000
    )-> typing.Tuple[DataLoader, DataLoader]:
    """
    returns tuple of dataloader in the following order:
    dataloader-hateful-memes, dataloader-conceputal-captions
    """

    path_cc = "res/data/conceptual-captions/validation.csv"
    path_hm = "res/data/hateful_memes_data/train.jsonl"

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    data_list_hm = generate_data_list(path_hm)
    random.shuffle(data_list_hm)
    # TODO: not needed for every sample!

    assert num_samples <= len(data_list_hm)

    #TODO: it now uses the complete dataset, not the train-test split here for alignment analysis
    data_list_cc = generate_data_list_pretrain(path=path_cc, max_number=None)
    random.shuffle(data_list_cc)
    data_list_hm = data_list_hm[:num_samples]
    data_list_cc = data_list_cc[:num_samples]

    dataset_hm = CustomDataset(
        data=data_list_hm,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    dataset_cc = PretrainDatasetAP(
        data=data_list_cc,
        tokenizer=tokenizer,
        image_processor=image_processor,
        preprocessing_prediction_alignment=False
    )

    dataloader_hm = DataLoader(
        dataset=dataset_hm,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    dataloader_cc = DataLoader(
        dataset=dataset_cc,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )

    return dataloader_hm, dataloader_cc





def get_dataloaders(
    train_data,
    val_data,
    batch_size: int,
    num_workers,
    prefetch,
    persistent_workers,
    pin_memory=True,
    use_contrastive_ap: bool=False
    ):
    """
    Returns: (
        train_loader_ap,
        val_loader_ap,
        train_loader_mlm,
        val_loader_mlm,
        train_loader_mim,
        val_loader_mim
    )
    """
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    print(f"dataset lengths: train: {len(train_data)}; val: {len(val_data)}")

    train_dataset_mim = PretrainDatasetMIM(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms_weak=utils.transforms_unmasked,
        transforms_strong=utils.transforms_masked
    )
    val_dataset_mim   = PretrainDatasetMIM(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms_weak=utils.transforms_unmasked,
        transforms_strong=utils.transforms_masked,
    )

    train_dataset_ap = PretrainDatasetAP(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        preprocessing_prediction_alignment=False,
        use_contrastive_ap_loss=use_contrastive_ap
    )


    val_dataset_ap   = PretrainDatasetAP(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        preprocessing_prediction_alignment=False,
    )

    train_dataset_mlm = PretrainDatasetMLM(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    val_dataset_mlm   = PretrainDatasetMLM(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    train_loader_mim = DataLoader(
        dataset=train_dataset_mim,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch
    )

    val_loader_mim = DataLoader(
        dataset=val_dataset_mim,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch
    )

    train_loader_ap = DataLoader(
        dataset=train_dataset_ap,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch
    )

    val_loader_ap = DataLoader(
        dataset=val_dataset_ap,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch
    )

    train_loader_mlm = DataLoader(
        dataset=train_dataset_mlm,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch
    )

    val_loader_mlm = DataLoader(
        dataset=val_dataset_mlm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch
    )

    return (
        train_loader_ap,
        val_loader_ap,
        train_loader_mlm,
        val_loader_mlm,
        train_loader_mim,
        val_loader_mim,
    )

def get_image_embedding(path: str, image_processor: BaseImageProcessor, transform=None):


    try:
        with Image.open(path) as image:
            # Resize if too large, some images in cc are too large
            # i need to resize them to mitigate warnings
            # vit sizes them down anyways
            width, height = image.size
            max_size = 4096  # Max dimension
            if width > max_size or height > max_size:
                if width > height:
                    new_width = max_size
                    new_height = int(height * max_size / width)
                else:
                    new_height = max_size
                    new_width = int(width * max_size / height)
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            image = image.convert("RGB")

            # TODO: fix this
            if not transform:
                # performs the transformation (torchvision transform) for
                # the vit model
                # currently only normalization, as dataset is already resized
                image = utils.vit_transform(image).unsqueeze(0)

            else:
                image = transform(image)
            # logger.info(f"processed image shape: {image.shape}")
            # to keep the format of transformers-package, which i was using before
            return {"pixel_values": image}
    except Exception as e:
        print(f"Error processing image {path}. Skipping.")
        print(f"error: {e}")
        return None


def get_text_embedding(text:str, tokenizer: PreTrainedTokenizerFast):
    # it is necessary to pad and trucate all to 128
    # https://huggingface.co/docs/transformers/en/pad_truncation
    # TODO: parameters configurable, no magic numbers directly in code
    return tokenizer(
        text,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=TOKENIZER_MAX_LEN
    )

class CustomDataset(Dataset):
    def __init__(
        self,
        data: typing.List[typing.Tuple[str, int, str]],
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
        transforms=None

    ):
        self.logger = Logger()
        self.transform = None
        self.tokenizer = tokenizer
        self.image_processor = image_processor

        os.remove(PREPROCESSED_PATH) if os.path.exists(PREPROCESSED_PATH) else None
        # self.data = self.__preprocess_data(data)
        self.data = data

        # TODO: caching preprocessed, or even do memory pinning -
        # with open(PREPROCESSED_PATH, "wb") as f:
        #     pickle.dump(self.data, f)

        self.transforms = transforms



    def __preprocess_data(self, data:typing.List[typing.Tuple[str, int, str]]):
        data_tensor = []
        data_dicts = []
        for i, dp in enumerate(data):
            if i % 500 == 0:
                info_str = f"Processing {i}/{len(data)} images"
                print(info_str)
                self.logger.info(info_str)
            img_path, label, text = dp

            img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
            text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

            if img_embeddings is None or text_embeddings is None:
                continue

            # img_tensor = process_single_image(img_path)
            label_tensor = torch.tensor(label, dtype=torch.long)
            # text_tensor = torch.tensor(text, dtype=torch.float32) # does not make any sense
            dict_entry = {
                "img": img_embeddings,
                "label": label_tensor,
                "text": text_embeddings,

            }
            data_dicts.append(dict_entry)
            # data_tensor.append((img_tensor, label_tensor, text))

        return data_dicts


    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """returns dictionary of form:
        {
            "img": img_tensor,
            "label": label_tensor,
            "text": text_tensor
        }
        """

        data = self.data[index]
        img_path, label, text = data

        img_embeddings = get_image_embedding(
            img_path,
            image_processor=self.image_processor,
            # transform=utils.vit_transform_full
            transform=utils.vit_transform        # not vit transform full anymore - no now i resized the images in the dataset
        )
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)


        label_tensor = torch.tensor(label, dtype=torch.long)


        if self.transforms:
            img_tensor = img_embeddings["pixel_values"].squeeze(0)
            img_tensor = self.transforms(img_tensor)
            img_embeddings["pixel_values"] = img_tensor.unsqueeze(0)

        return {
            "img": img_embeddings,
            "label": label_tensor,
            "text": text_embeddings,

        }



class PretrainDatasetAP(Dataset):
    def __init__(
        self,
        data: typing.List[typing.Tuple[str, str]], # path, text/caption
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
        preprocessing_prediction_alignment: bool,    # whether to generate the dataset at first, or at runtime
        use_contrastive_ap_loss: bool=False
        ):
        self.transform = None
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.invalid_image_counter = 0

        self.preprocessing_prediction_alignment = preprocessing_prediction_alignment
        self.use_contrastive_ap_loss = use_contrastive_ap_loss
        self.data = self.__generate_pretrain_dataset(data)
        #TODO
        # self.data = self.exclude_invald_photos(self.data)

        # memory overload, load per element
        # self.data = self.__preprocess_data(self.data)

    def __preprocess_data(self, data:typing.List[typing.Tuple[Task, str, str, int]]):
        # TODO: only temp!
        data = data[:500]
        data_tensor = []
        for i, dp in enumerate(data):
            if i % 500 == 0:
                info_str = f"Processing {i}/{len(data)} images"
                print(info_str)
                self.logger.info(info_str)
            task, text, img_path, label = dp

            img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
            text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

            if img_embeddings is None or text_embeddings is None:
                continue

            label_tensor = torch.tensor(label, dtype=torch.long)

            dict_entry = {
                "task": task,
                "img": img_embeddings,
                "label": label_tensor,
                "text": text_embeddings,
            }
            data_tensor.append(dict_entry)

        return data_tensor

    def __generate_pretrain_dataset(
        self,
        data: typing.List[typing.Tuple[str, str]],
    ):

        data_alignment_prediction = self.__generate_pretrain_dataset_alignment_prediction(data)
        # data_mlm = self.__generate_pretrain_dataset_mlm(data)

        # data_list = data_alignment_prediction + data_mlm
        data_list = data_alignment_prediction

        random.shuffle(data_list)  # shuffle the dataset to mix tasks
        return data_list

    def __generate_pretrain_dataset_alignment_prediction(
        self,
        data: typing.List[typing.Tuple[str, str]],  # path, text/caption
    ):
        data_list = data
        true_alignment = [(Task.ALIGNMENT_PREDICTION, dp[0], dp[1], 1)
                          for dp in data_list]

        if self.preprocessing_prediction_alignment:

            false_alignment = []
            dataset_length = len(true_alignment)
            for i in range(dataset_length):
                random_idx = i
                while random_idx == i:
                    random_idx = random.randint(0, dataset_length - 1)

                dp_path = data_list[i][0]
                text = data_list[random_idx][1]
                # shape: task, path, text, label
                false_alignment.append((Task.ALIGNMENT_PREDICTION, dp_path, text, 0 ))

            data_list = true_alignment + false_alignment
            random.shuffle(data_list)

            return data_list
        else:
            return true_alignment


    def mask_tokens(self, token_ids, tokenizer: PreTrainedTokenizerFast, mask_prob=0.15):
        masked_tokens = token_ids.clone()
        #-100 for loss function to ignore it.
        output_labels = torch.full_like(token_ids, -100, dtype=torch.long)

        for i, token in enumerate(token_ids[0]):
            if token == tokenizer.pad_token_id:
                continue

            prob1 = random.random()

            if prob1 < mask_prob:
                prob2 = random.random()

                # 80 % are replaced with [MASK]
                mask_token_id = tokenizer.mask_token_id
                if prob2 < 0.8:
                    masked_tokens[0,i] = mask_token_id

                # 10 % are replaced with random token
                elif prob2 < 0.9 and prob2 >= 0.8:
                    random_token_id = random.randint(1, tokenizer.vocab_size - 1)
                    masked_tokens[0, i] = random_token_id
                else:
                    # 10 % are left unchanged
                    masked_tokens[0, i] = token_ids[0, i]

                output_labels[0, i] = token.item()     # these will be predicted
            else:
                # will be ignored by loss funciton
                # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html - is default ignore_index
                pass

        return masked_tokens, output_labels



    def handle_fallback(self, ):
        self.invalid_image_counter += 1
        if self.invalid_image_counter > 10:
            raise ValueError("Too many invalid images encountered. Stopping.")
        idx = random.randint(0, len(self.data) - 1)

        return self.__getitem__(idx)

    def __getitem__(self, index):
        # it is not possible to preprocess the data because the memory cannot hold all images simultanously.
        # with the num_workers flag true and other optimizations it should work with loading it
        # on demand.

        dp = self.data[index]
        task, img_path, text, label = dp

        assert task == Task.ALIGNMENT_PREDICTION, "something is completely wrong in the data processing step"

        if not self.preprocessing_prediction_alignment and not self.use_contrastive_ap_loss:
            if random.random() < 0.5:

                # swap out text with some other caption
                random_idx = index
                while random_idx == index:
                    random_idx = random.randint(0, len(self.data) - 1)

                text = self.data[random_idx][2]  # get text from random index
                label = 0
            else:
                label = 1

        img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

        if random.random() <0.35:
            transform = torchvision.transforms.Compose(
                [
                # transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([
                    transforms.GaussianBlur(3, sigma=(0.1, 2.0))
                    ], p=0.3),
                ]
            )

            img_tensor = img_embeddings["pixel_values"].squeeze(0)
            img_tensor = transform(img_tensor)
            img_embeddings["pixel_values"] = img_tensor.unsqueeze(0)

        if img_embeddings is None or text_embeddings is None:
            # this should not happen anymore, as downloading conc.capt. checks for faulty imgs
            return self.handle_fallback()

        self.invalid_image_counter = 0
        label_tensor = torch.tensor(label, dtype=torch.long)

        return {
            "task": task.value,     # torch cannot handle custom classes
            "img": img_embeddings,
            "label": label_tensor,
            "text": text_embeddings,
        }

    def __len__(self,):
        return len(self.data)


class PretrainDatasetMLM(Dataset):
    def __init__(
        self,
        data: typing.List[typing.Tuple[str, str]], # path, text/caption
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,

        ):
        self.logger = Logger()
        self.transform = None
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.invalid_image_counter = 0


        self.data = self.__generate_pretrain_dataset(data)
        #TODO
        # self.data = self.exclude_invald_photos(self.data)

        # memory overload, load per element
        # self.data = self.__preprocess_data(self.data)

    def __preprocess_data(self, data:typing.List[typing.Tuple[Task, str, str, int]]):
        # TODO: only temp!
        data = data[:500]
        data_tensor = []
        for i, dp in enumerate(data):
            if i % 500 == 0:
                print(f"Processing {i}/{len(data)} images")
            task, text, img_path, label = dp

            img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
            text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

            if img_embeddings is None or text_embeddings is None:
                continue

            label_tensor = torch.tensor(label, dtype=torch.long)

            dict_entry = {
                "task": task,
                "img": img_embeddings,
                "label": label_tensor,
                "text": text_embeddings,
            }
            data_tensor.append(dict_entry)

        return data_tensor

    def __generate_pretrain_dataset(
        self,
        data: typing.List[typing.Tuple[str, str]],
    ):
        # data_alignment_prediction = self.__generate_pretrain_dataset_alignment_prediction(data)
        data_mlm = self.__generate_pretrain_dataset_mlm(data)

        data_list = data_mlm

        random.shuffle(data_list)  # shuffle the dataset to mix tasks
        return data_list

    def __generate_pretrain_dataset_mlm(
        self,
        data: typing.List[typing.Tuple[str, str]],      # path, text
    ):
        # only simply generation, real masking is happening in __getitem__
        # task, path, text, placeholder label
        return [ (Task.MASKED_LM, dp[0], dp[1], 3) for dp in data ]



    def mask_tokens(self, token_ids, tokenizer: PreTrainedTokenizerFast, mask_prob=0.15):
        masked_tokens = token_ids.clone()
        #-100 for loss function to ignore it.
        output_labels = torch.full_like(token_ids, -100, dtype=torch.long)

        for i, token in enumerate(token_ids[0]):
            if token == tokenizer.pad_token_id:
                continue

            prob1 = random.random()

            if prob1 < mask_prob:
                prob2 = random.random()

                # 80 % are replaced with [MASK]
                mask_token_id = tokenizer.mask_token_id
                if prob2 < 0.8:
                    masked_tokens[0,i] = mask_token_id

                # 10 % are replaced with random token
                elif prob2 < 0.9 and prob2 >= 0.8:
                    random_token_id = random.randint(1, tokenizer.vocab_size - 1)
                    masked_tokens[0, i] = random_token_id
                else:
                    # 10 % are left unchanged
                    masked_tokens[0, i] = token_ids[0, i]

                output_labels[0, i] = token.item()     # these will be predicted
            else:
                # will be ignored by loss funciton
                # https://docs.pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html - is default ignore_index
                pass

        return masked_tokens, output_labels



    def handle_fallback(self, ):
        self.invalid_image_counter += 1
        if self.invalid_image_counter > 10:
            raise ValueError("Too many invalid images encountered. Stopping.")
        idx = random.randint(0, len(self.data) - 1)

        return self.__getitem__(idx)

    def __getitem__(self, index):
        # it is not possible to preprocess the data because the memory cannot hold all images simultanously.
        # with the num_workers flag true and other optimizations it should work with loading it
        # on demand.

        dp = self.data[index]
        task, img_path, text, label = dp

        img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

        if img_embeddings is None or text_embeddings is None:
            # this should not happen anymore, as downloading conc.capt. checks for faulty imgs
            return self.handle_fallback()

        self.invalid_image_counter = 0

        assert task == Task.MASKED_LM, "something is completely wrong in the data processing step"

        input_ids = text_embeddings["input_ids"].clone()
        masked_input_ids, mlm_label = self.mask_tokens(input_ids, tokenizer=self.tokenizer)
        assert masked_input_ids.shape == mlm_label.shape

        text_embeddings["input_ids"] = masked_input_ids

        return {
            "task": task.value,     # torch cannot handle custom classes
            "img": img_embeddings,
            "label": mlm_label,     # label is the masked tokens
            "text": text_embeddings,
        }


    def __len__(self,):
        return len(self.data)


class PretrainDatasetMIM(Dataset):
    def __init__(
        self,
        data: typing.List[typing.Tuple[str, str]], # path, text/caption
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
        transforms_weak=None,
        transforms_strong=None,

    ):
        self.transform = True if (transforms_weak or transforms_strong) else False
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.invalid_image_counter = 0

        self.transforms_weak = transforms_weak
        self.transforms_strong = transforms_strong

        self.data = self.__generate_pretrain_dataset(data)

    def __generate_pretrain_dataset(
        self,
        data: typing.List[typing.Tuple[str, str]],
    ):
        data_mim = self.__generate_pretrain_dataset_mim(data)

        random.shuffle(data_mim)
        return data_mim

    def __generate_pretrain_dataset_mim(self, data):
        # returns task, path, text/caption, placeholder label
        return [
            (Task.MASKED_IM, dp[0], dp[1], 3) for dp in data
        ]

    def handle_fallback(self, ):
        self.invalid_image_counter += 1
        if self.invalid_image_counter > 10:
            raise ValueError("Too many invalid images encountered. Stopping.")
        idx = random.randint(0, len(self.data) - 1)

        return self.__getitem__(idx)



    def __mask_image(self, img: torch.Tensor):
        img_numpy = utils.img_tensor_to_numpy(image_tensor=img)

        # TODO: add masking_prob
        masked_img, masked_patches_idxs = utils.mask_image(img=img_numpy)
        return masked_img, masked_patches_idxs

    def __len__(self):
        return len(self.data)


    def __getitem__(self, idx):
        dp = self.data[idx]
        task, img_path, text, label = dp

        img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

        if img_embeddings is None or text_embeddings is None:
            # this should not happen anymore, as downloading conc.capt. checks for faulty imgs
            return self.handle_fallback()

        original_img = img_embeddings["pixel_values"].squeeze(0)  # Remove batch dim for transforms

        if self.transforms_weak and self.transforms_strong:
            weak_img = self.transforms_weak(original_img)
            img_embeddings["pixel_values"] = weak_img.unsqueeze(0)  # Add batch dim back


            strong_img = self.transforms_strong(original_img)
            masked_img, masked_patches_idxs = self.__mask_image(strong_img)
        else:

            masked_img, masked_patches_idxs = self.__mask_image(original_img)


        assert task == Task.MASKED_IM, "something is completely wrong in the data processing step"

        return {
            "task": task.value,     # torch cannot handle custom classes
            "img" : img_embeddings,
             "masked_img": {"pixel_values": masked_img.unsqueeze(0)},
            "masked_patches_idxs": masked_patches_idxs,
            "text": text_embeddings,
        }


def generate_data_list(path: str):
    dir_name = os.path.dirname(path)
    print("dirname: ", dir_name)

    def read_jsonl(file_path):
        data = []
        with open(file_path, "r") as file:
            for line in file:
                data.append(json.loads(line.strip()))
        return data

    records = read_jsonl(path)

    data_list = []

    for i in records:
        image_path = i["img"]
        label      = i["label"]
        text       = i["text"]

        # if not exists
        if not os.path.exists(os.path.join(dir_name, image_path)):
            print(f"Image {image_path} does not exist in {dir_name}. Skipping.")
            continue

        dp = (os.path.join(dir_name, image_path), label, text)
        data_list.append(dp)

    # sort by filename
    data_list.sort(
        key = lambda triple: triple[0].split("/")[-1].split(".")[0]
    )

    return data_list


def generate_data_list_pretrain(path: str, max_number=None):
    data_list = []
    counter = 0
    with open(path) as fd:
        rd = csv.reader(fd, quotechar='"')
        next(rd)

        for row in rd:
            text = row[0]
            path = row[1]

            if not os.path.exists(path):
                print(f"Image {path} does not exist. Skipping.")
                continue

            counter +=1

            data_list.append((path, text))
            if max_number and counter >= max_number:
                break

    return data_list



# def main():
#     dir_name = "res/data/hateful_memes_data/img"

#     img_paths = []

#     for file_name in os.listdir(dir_name):
#         if file_name.endswith(".png"):
#             img_paths.append(os.path.join(dir_name, file_name))

#     # print(img_paths)
#     # srot them
#     img_paths.sort(key=lambda path: int(
#                                     path.split("/")[-1]        # extracts only filename in the dir. so fom "res/data/img/001.png" => "001.png"
#                                         .split(".")[0]))
#     # print(img_paths)

#     train_data_list = generate_data_list("res/data/hateful_memes_data/train.jsonl")
#     # not working as there are no labels?
#     # test_data_list = generate_data_list("res/data/hateful_memes_data/test.jsonl")

#     train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
#     train_data = train_data_list[:train_idx]
#     val_data   = train_data_list[train_idx:]

#     tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


#     train_dataset = CustomDataset(train_data, tokenizer=tokenizer, image_processor=image_processor)
#     val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)


#     print(f"train dataset- length: {len(train_dataset)}, head: {train_dataset.data[:5]}")
#     print(f"val dataset- length: {len(val_dataset)}, head: {val_dataset.data[:5]}")
#     # print(f"train data - length: {len(train_data)}, head: {train_data[:5]}")
#     # print(f"val data - length: {len(val_data)}, head: {val_data[:5]}")

#     train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
#     val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)


def main():
    path = "res/data/conceptual-captions/validation.csv"

    dl = generate_data_list_pretrain(path)

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    dataset = PretrainDatasetMIM(
        data = dl,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    print(f"Dataset length: {len(dataset)}")

    '''
    Dataset length: 478
    Task.ALIGNMEN_PREDICTION {'input_ids': tensor([[  101, 10658,  2160,  2013,  1037, 16641,   102,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
                0,     0,     0,     0,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
            0, 0, 0, 0, 0, 0, 0, 0]])} torch.Size([1, 3, 224, 224]) tensor(1)
    '''
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=10,
        pin_memory=True,
        persistent_workers=True
    )

    for batch in dataloader:
        ...

        # break  # only one batch for testing



if __name__ == "__main__":
    main()
# import sys, os; sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import os
import json
import typing;
import csv
import random
import io


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
import pandas as pd
import h5py

from PIL import Image, UnidentifiedImageError

import utils
from task import Task
from config import *
import augments_transforms
import warnings

from logger import Logger

logger = Logger()

# disable PIL's decompression bomb warning, bc i get the following:
# DecompressionBombWarning: Image size (93950400 pixels) exceeds limit of 89478485 pixels, could be decompression bomb DOS attack.
warnings.filterwarnings("ignore", ".*DecompressionBombWarning.*", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", ".*Palette images with Transparency.*", category=UserWarning)


Image.MAX_IMAGE_PIXELS = None

def _process_image(img: Image.Image, transform=None):

    assert isinstance(img, Image.Image)
    basic_vit_transform = augments_transforms.get_minimal_vit_transform()

    if transform:
        img = transform(img)

    # performs only normlalization

    img_t : torch.tensor = basic_vit_transform(img).unsqueeze(0)


    return { "pixel_values": img_t }



def process_image(
    img: typing.Union[str, Image.Image],
    transform=None
    ) -> typing.Optional[typing.Dict[str, torch.Tensor]]:
    """
    processes a single image, performs necessary processing for ViT.

    parameters:
        img: either a path to the image, or a PIL image
        transform: optional torchvision transform to apply after the basic vit transform

    returns:
        dictionary of form {"pixel_values": img_tensor} where img_tensor is of shape (1, 3, 224, 224)

    """

    if isinstance(img, str):
        try:
            with Image.open(img) as image:
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
                img = image
        except Exception as e:
            print(f"Error processing image {img}. Skipping.")
            print(f"error: {e}")
            return None

    # print(type(img))
    assert isinstance(img, Image.Image)
    return _process_image(img, transform=transform)





def process_single_image(path:str) -> torch.Tensor:
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0  # norm to [0, 1]

    img = np.transpose(img, (2, 0, 1))  # change to CxHxW format (the opencv format)
    img_tensor = torch.from_numpy(img.astype(np.float32))

    return img_tensor


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

            # TODO: fix this, this is a bit clunky and unintuitive
            if not transform:
                # performs the transformation (torchvision transform) for
                # the vit model
                # currently only normalization, as dataset is already resized
                minimal_vit_transform = augments_transforms.get_minimal_vit_transform()
                image = minimal_vit_transform(image).unsqueeze(0)

            else:
                image = transform(image)
            # logger.info(f"processed image shape: {image.shape}")
            # to keep the format of transformers-package, which i was using before
            return {"pixel_values": image}
    except Exception as e:
        print(f"Error processing image {path}. Skipping.")
        print(f"error: {e}")
        return None


def get_text_embedding(text:str, tokenizer:PreTrainedTokenizerFast):
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


# def main():
#     path = "res/data/conceptual-captions/validation.csv"

#     dl = generate_data_list_pretrain(path)

#     tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

#     dataset = PretrainDatasetMIM(
#         data = dl,
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#     )

#     print(f"Dataset length: {len(dataset)}")

#     '''
#     Dataset length: 478
#     Task.ALIGNMEN_PREDICTION {'input_ids': tensor([[  101, 10658,  2160,  2013,  1037, 16641,   102,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0,     0,     0,
#                 0,     0,     0,     0,     0,     0,     0,     0]]), 'token_type_ids': tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0]]), 'attention_mask': tensor([[1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
#             0, 0, 0, 0, 0, 0, 0, 0]])} torch.Size([1, 3, 224, 224]) tensor(1)
#     '''
#     dataloader = DataLoader(
#         dataset,
#         batch_size=1,
#         shuffle=True,
#         num_workers=10,
#         pin_memory=True,
#         persistent_workers=True
#     )

#     for batch in dataloader:
#         ...

#         # break  # only one batch for testing



# if __name__ == "__main__":
#     tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")


#     path = "res/data/mm-imdb/images.h5"
#     csv_path = "res/data/mm-imdb/mmimdb_trainval.csv"
#     # get_mm_imdb_data(path=path)

#     dataset = MM_IMDB_Dataset(
#         csv_path=csv_path,
#         img_path=path,
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#     )

#     dataloader = torch.utils.data.DataLoader(dataset=dataset, )

#     for i, batch in enumerate(dataloader):
#         print(batch.keys())
#         shp = batch["label"].shape[1]
#         assert shp == 23

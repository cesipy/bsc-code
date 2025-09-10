import torch; from torch.utils.data import Dataset, DataLoader
import torchvision; from torchvision import transforms
from PIL import Image

from torchvision.transforms import InterpolationMode
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


import typing
import pandas as pd
import numpy as np

import h5py

from transformers import (
     # ViT stuff
    BaseImageProcessor,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
    BertTokenizerFast
)

import utils

from logger import Logger
from config import *
from .dataset_utils import get_image_embedding, get_text_embedding
import augments_transforms



class MM_IMDB_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        img_path: str,
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
        is_train:bool=True,
        train_test_ratio:float = TRAIN_TEST_RATIO,
    ):

        assert os.path.exists(img_path)
        self.img_data = h5py.File(img_path, "r")


        csv_data = pd.read_csv(csv_path)
        csv_data = csv_data.sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle the data

        train_test_split_idx = int(train_test_ratio * len(csv_data))
        print(len(csv_data))

        print(csv_data.columns)
        print(f"train-test split idx: {train_test_split_idx}")

        self.is_train = is_train
        if is_train:
            self.csv_data = csv_data[:train_test_split_idx]
        else:
            self.csv_data = csv_data[train_test_split_idx:]

        self.tokenizer = tokenizer
        self.image_processor = image_processor


    def __len__(self,):
        return len(self.csv_data)



    def __getitem__(self, idx):
        tup = self.csv_data.iloc[idx]       #['img_index', 'genre', 'caption']
        img_idx = tup["img_index"]

        img = self.img_data["images"][img_idx]  # 3, 256, 160 / c, h, w
        img = np.transpose(img, (1,2,0))            # h, w, c

        genre = tup["genre"]
        parsed_genre = utils.genre_parsing(genre)   # multi-hot vector
        genre = torch.tensor(parsed_genre, dtype=torch.float32)
        caption = tup["caption"]

        caption_embeddings = get_text_embedding(caption, tokenizer=self.tokenizer)
        # print(f"img shape: {img.shape}")
        #debugging

        # img = torch.tensor(img, dtype=torch.float32)

        img_pre: Image = Image.fromarray(img.astype(np.uint8))


        # print(f"type image: {type(img)}")
        #custom image embedding handling here
        #TODO: clean up
        transform_mm_imdb = transforms. Compose(
            [
                transforms.Resize(224),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
            ]
        )

        transform_augmentation = transforms.Compose([

            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),

            transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.95, 1.05),
                shear=2
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.2),

            transforms.RandomGrayscale(p=0.1),
            # transforms.RandomHorizontalFlip(p=0.2),
            transforms.RandomErasing(),

        ])


        img_processed = transform_mm_imdb(img_pre)
        if self.is_train:
            img_processed = transform_augmentation(img_processed)


        img_processed = { "pixel_values": img_processed.unsqueeze(0) }

        dict = {
            "img": img_processed,
            "label": genre,
            "text": caption_embeddings,
        }


        return dict


    def __del__(self):
        if self.img_data != None:
            self.img_data.close()
            self.img_data = None


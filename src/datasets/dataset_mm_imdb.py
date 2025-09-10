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
from .dataset_utils import get_image_embedding, get_text_embedding; from . import dataset_utils
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
        transform: typing.Optional[torchvision.transforms.Compose] = None
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
        self.transform = transform


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

        if self.is_train:

            img_processed = dataset_utils.process_image(img_pre, transform=self.transform)
        else:
            transform_resize_only = transforms.Compose(
                [
                    transforms.Resize(224),
                    transforms.CenterCrop(224)
                ]
            )
            img_processed = dataset_utils.process_image(img=img_pre, transform=transform_resize_only)


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


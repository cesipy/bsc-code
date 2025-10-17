import torch; from torch.utils.data import Dataset, DataLoader
import torchvision; from torchvision import transforms
from PIL import Image

from torchvision.transforms import InterpolationMode
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

import albumentations as A

import typing
import pandas as pd
import numpy as np

import h5py
from .dataset_utils import BaseImageProcessor, PreTrainedTokenizerFast

import utils

from logger import Logger
from config import *
from .dataset_utils import get_image_embedding, get_text_embedding; from . import dataset_utils
import augments_transforms

class UPMC_Dataset(torch.utils.data.Dataset):
    def __init__(
        self,
        csv_path: str,
        img_path: str,
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
        is_train:bool=True,
        is_test:bool=False,
        train_test_ratio:float = TRAIN_TEST_RATIO,
        transform: typing.Optional[torchvision.transforms.Compose] = None,
        max_samples: typing.Optional[int] = None,

    ):
        csv_data = pd.read_csv(csv_path)
        train_test_split_idx = int(train_test_ratio * len(csv_data))
        assert os.path.exists(img_path)
        assert not (is_train and is_test)

        self.data = csv_data.sample(frac=1, random_state=46).reset_index(drop=True)  # shuffle the data
        self.imgs_path = img_path

        # print(self.data.columns)
        col_class = self.data.loc[:, "class"].unique()  # np.array[str]
        # print(f"classes {len(col_class)}")  # 101, same as the one_hot

        if is_test==True:
            if max_samples is not None:
                assert max_samples < len(self.data)
                self.data = self.data[:max_samples]
            else:
                self.data = csv_data
        else:
            if max_samples != None:
                assert max_samples < len(csv_data)
                self.data = self.data[:max_samples]
            elif max_samples == None and is_train:
                self.data = csv_data[:train_test_split_idx]
            elif max_samples == None and not is_train:
                self.data = csv_data[train_test_split_idx:]
            else:
                print("smth is completely wrong! panicking!!!")
                exit(0)

        self.is_train = is_train
        self.transform = transform
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __len__(self,):
        return len(self.data)

    def __getitem__(self, idx):
        tup = self.data.iloc[idx]
        img_path, label, class_, text = tup

        img_path = os.path.join(self.imgs_path, img_path)

        transform_resize_only = A.Compose([
            A.Resize(224, 224),
            A.CenterCrop(224, 224),
        ])


        if self.is_train:
            base_transforms = list(transform_resize_only.transforms)
            add_transforms = list(self.transform.transforms)

            transform = A.Compose(base_transforms + add_transforms)
            img_processed = dataset_utils.process_image(img=img_path, transform=transform)
        else:
            # transform_resize_only = transforms.Compose(
            #     [
            #         transforms.Resize(224),
            #         transforms.CenterCrop(224)
            #     ]
            # )

            img_processed = dataset_utils.process_image(img=img_path, transform=transform_resize_only)

        text_processed = dataset_utils.get_text_embedding(text, tokenizer=self.tokenizer)
        label_one_hot = utils.genre_parsing(label)

        class_processed = dataset_utils.get_text_embedding(class_, tokenizer=self.tokenizer)
        assert sum(label_one_hot) == 1

        label_one_hot_t = torch.tensor(label_one_hot, dtype=torch.long)
        dict = {
            "img": img_processed,
            "label": label_one_hot_t,
            "text": text_processed,
        }

        return dict


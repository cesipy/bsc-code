
# import sys, os; sys.path.append('src')
import torch; from torch.utils.data import Dataset, DataLoader
import typing

from .dataset_utils import BaseImageProcessor, PreTrainedTokenizerFast

from logger import Logger
import augments_transforms
from config import *

from .dataset_utils import get_image_embedding, get_text_embedding
from . import dataset_utils


class HM_Dataset(Dataset):
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

        # img_embeddings = get_image_embedding(
        #     img_path,
        #     image_processor=self.image_processor,
        #     transform=augments_transforms.get_minimal_vit_transform()      # not vit transform full anymore - no now i resized the images in the dataset
        # )
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

        if self.transforms:
           img_embeddings = dataset_utils.process_image(
               img=img_path,
               transform=self.transforms,

           )
        else:
              img_embeddings = dataset_utils.process_image(
                img=img_path,
                transform=None
              )

        label_tensor = torch.tensor(label, dtype=torch.long)


        return {
            "img": img_embeddings,
            "label": label_tensor,
            "text": text_embeddings,

        }

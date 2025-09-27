import torch; from torch.utils.data import Dataset, DataLoader
import typing

import torchvision; from torchvision import transforms
import random

from . import dataset_utils

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
from .dataset_utils import get_image_embedding, get_text_embedding, process_image;  from . import dataset_utils
import augments_transforms
from task import Task

class ConceptualCaptionsDataset(Dataset):
    """ dataset for alignment analysis and other downstream operstions. """

    def __init__(
        self,
        data: typing.List[typing.Tuple[str, str]], # path, text/caption
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor,
    ):
        self.data =self.__generate_data(data)
        self.tokenizer = tokenizer
        self.image_processor = image_processor

    def __generate_data(self, data):
        # placeholder task, path, text, placeholder label
        data_ = [ ( Task.PLACEHOLDER, dp[0], dp[1], 3)  for dp in data ]
        p = data_[0]
        print( f" dp0:{p[1]}, dp1:{p[2]}" )

        return data_

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        tup = self.data[idx]

        img_embedding = process_image(img=tup[1], transform=None)
        text_embedding = get_text_embedding(text=tup[2], tokenizer=self.tokenizer)

        label = torch.tensor(tup[3], dtype=torch.long)


        dict_entry = {
            "img": img_embedding,
            "text": text_embedding,
            "label": label,
        }
        return dict_entry




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
            if random.random() < 0.5:       # TODO: create cc evaluation dataset

                # swap out text with some other caption
                random_idx = index
                while random_idx == index:
                    random_idx = random.randint(0, len(self.data) - 1)

                text = self.data[random_idx][2]  # get text from random index
                label = 0
            else:
                label = 1

        img_embeddings = dataset_utils.process_image(img=img_path, transform=None)
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)

        if random.random() <0.5:
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
            "img": img_embeddings,
            "text": text_embeddings,
            "task": task.value,     # torch cannot handle custom classes
            "label": label_tensor,
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

        img_embeddings = dataset_utils.process_image(img=img_path, transform=None)
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

        # img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
        img_embeddings = dataset_utils.process_image(img=img_path, transform=None)
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





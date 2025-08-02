import os
import json
import typing
import pickle
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

import cv2 
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

from PIL import Image, UnidentifiedImageError

from utils import Task
from config import * 

import warnings

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

def get_image_embedding(path: str, image_processor: BaseImageProcessor):
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
            image = image_processor(images=image, return_tensors="pt")
            return image
    except Exception:
        # print(f"Error processing image {path}. Skipping.")
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
        
    ):
        self.transform = None
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        
        os.remove(PREPROCESSED_PATH) if os.path.exists(PREPROCESSED_PATH) else None
        self.data = self.__preprocess_data(data)
        
        # TODO: caching preprocessed, or even do memory pinning -
        # with open(PREPROCESSED_PATH, "wb") as f:
        #     pickle.dump(self.data, f)
        
        
    def __preprocess_data(self, data:typing.List[typing.Tuple[str, int, str]]): 
        data_tensor = []
        data_dicts = []
        for i, dp in enumerate(data):
            if i % 500 == 0: 
                print(f"Processing {i}/{len(data)} images")
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
       
        # TODO: transformation 
        if self.transform:
            img_tensor = self.transform(img_tensor)
        
        return data
    

class PretrainDatasetAP(Dataset): 
    def __init__(
        self,
        data: typing.List[typing.Tuple[str, str]], # path, text/caption
        tokenizer: PreTrainedTokenizerFast,
        image_processor: BaseImageProcessor, 
        preprocessing_prediction_alignment: bool,    # whether to generate the dataset at first, or at runtime
        ): 
        self.transform = None
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.invalid_image_counter = 0
        
        self.preprocessing_prediction_alignment = preprocessing_prediction_alignment
        
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
        true_alignment = [(Task.ALIGNMEN_PREDICTION, dp[0], dp[1], 1)
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
                false_alignment.append((Task.ALIGNMEN_PREDICTION, dp_path, text, 0 ))
                
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
        
        assert task == Task.ALIGNMEN_PREDICTION, "something is completely wrong in the data processing step"
    
        if not self.preprocessing_prediction_alignment: 
            if random.random() < 0.5: 

                # swap out text with some other caption
                random_idx = index
                while random_idx == index:
                    random_idx = random.randint(0, len(self.data) - 1)
                
                text = self.data[random_idx][2]  # get text from random index
                label = 0    
            
        img_embeddings = get_image_embedding(img_path, image_processor=self.image_processor)
        text_embeddings = get_text_embedding(text, tokenizer=self.tokenizer)
        
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
            # print(f"Image {image_path} does not exist in {dir_name}. Skipping.")
            continue
        
        dp = (os.path.join(dir_name, image_path), label, text)
        data_list.append(dp)

    # sort by filename
    data_list.sort(
        key = lambda triple: triple[0].split("/")[-1].split(".")[0] 
    )
    
    return data_list
    
    
def generate_data_list_pretrain(path: str): 
    data_list = []
    with open(path) as fd: 
        rd = csv.reader(fd, quotechar='"')
        next(rd)
        
        for row in rd:
            text = row[0]
            path = row[1]
            
            if not os.path.exists(path):
                # print(f"Image {path} does not exist. Skipping.")
                continue
            
            data_list.append((path, text))
            
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
    
    dataset = PretrainDataset(
        data=dl, 
        tokenizer=tokenizer,
        image_processor=image_processor, 
        preprocessing_prediction_alignment=True
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
    
    for i in dataloader: 
        text_decoded = tokenizer.decode(i["text"]["input_ids"][0][0], skip_special_tokens=True)
        
        print(f"Task: {i['task']}")
        print(f"Text: '{text_decoded}'")
        print(f"Image shape: {i['img']['pixel_values'].shape}")
        print(f"Label: {i['label'].item()}")
        print("-" * 30)
    
    
if __name__ == "__main__":
    main()
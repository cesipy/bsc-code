import os
import json
import typing
import pickle

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

from PIL import Image


from config import * 

def process_single_image(path:str) -> torch.Tensor: 
    img = cv2.imread(path)
    img = cv2.resize(img, IMG_SIZE)
    img = img.astype(np.float32) / 255.0  # norm to [0, 1]

    img = np.transpose(img, (2, 0, 1))  # change to CxHxW format (the opencv format)
    img_tensor = torch.from_numpy(img.astype(np.float32))
    
    return img_tensor

def get_image_embedding(path:str, image_processor: BaseImageProcessor): 
    image = Image.open(path).convert("RGB")
    image = image_processor(images=image, return_tensors="pt")
    
    return image

def get_text_embedding(text:str, tokenizer: PreTrainedTokenizerFast): 
    return tokenizer(text, return_tensors="pt")
    


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
        
        # if self.transform:
        #     img_tensor = self.transform(img_tensor)
        
        return data
        

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
    

        
        
def main(): 
    dir_name = "res/data/hateful_memes_data/img"
    
    img_paths = []
    
    for file_name in os.listdir(dir_name):
        if file_name.endswith(".png"):
            img_paths.append(os.path.join(dir_name, file_name))
            
    # print(img_paths)
    # srot them 
    img_paths.sort(key=lambda path: int(
                                    path.split("/")[-1]        # extracts only filename in the dir. so fom "res/data/img/001.png" => "001.png"
                                        .split(".")[0]))
    # print(img_paths)
    
    train_data_list = generate_data_list("res/data/hateful_memes_data/train.jsonl")
    # not working as there are no labels?
    # test_data_list = generate_data_list("res/data/hateful_memes_data/test.jsonl")
    
    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:]  
     
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    
    train_dataset = CustomDataset(train_data, tokenizer=tokenizer, image_processor=image_processor)
    val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)
    
    
    print(f"train dataset- length: {len(train_dataset)}, head: {train_dataset.data[:5]}")
    print(f"val dataset- length: {len(val_dataset)}, head: {val_dataset.data[:5]}") 
    # print(f"train data - length: {len(train_data)}, head: {train_data[:5]}")
    # print(f"val data - length: {len(val_data)}, head: {val_data[:5]}")
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    

    
    
if __name__ == "__main__":
    main()
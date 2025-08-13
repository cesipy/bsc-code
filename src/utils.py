import math
import random
import typing

import numpy as np
import torch
from torch import nn
import gc; import time
from functools import wraps
import matplotlib.pyplot as plt

from PIL import Image

from enum import Enum
from config import *

from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform


# just for testing purposes here
# to avoid too many unused objects
config = resolve_data_config({}, model=VIT_MODEL_NAME)
vit_transform = create_transform(**config)

def get_image_embedding(path: str, image_processor=None):
    
    
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
            # image = image_processor(images=image, return_tensors="pt")
            # return image
            image = vit_transform(image).unsqueeze(0) 
            # logger.info(f"processed image shape: {image.shape}")
            # to keep the format of transformers-package, which i was using before
            return {"pixel_values": image}      
    except Exception as e:
        print(f"Error processing image {path}. Skipping.")
        print(f"error: {e}")
        return None


class Task(Enum): 
    ALIGNMEN_PREDICTION = 1
    MASKED_LM = 2
    MASKED_IM = 3       # masked image modelling
    

def memory_cleanup(func): 
    @wraps(func)
    def wrapper_func(*args, **kwargs):
        try: 
            result = func(*args, **kwargs)
        
            # repeatedly collect garbabe
            for i in range(3): 
                gc.collect()
                
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                
            gc.collect()
            
            return result
        except Exception as e: 
            print(f"Error in {func.__name__}: {e}")
            for _ in range(5):
                gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            
            raise e
    return wrapper_func

def img_to_tensor(image: Image) -> torch.Tensor:
    """Convert a PIL Image to a PyTorch tensor."""
    if not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image.")
    
    # Convert image to RGB if not already in that mode
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert the image to a tensor
    img_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()  # Change to CxHxW format
    return img_tensor

def img_tensor_to_numpy(image_tensor: torch.Tensor) -> np.ndarray:
    """Convert a PyTorch tensor to a NumPy array."""
    if not isinstance(image_tensor, torch.Tensor):
        raise TypeError("Input must be a PyTorch tensor.")
    
    # Ensure the tensor is in CxHxW format
    if image_tensor.dim() != 3 or image_tensor.size(0) not in [1, 3]:
        raise ValueError("Tensor must be in CxHxW format.")
    
    img_numpy = image_tensor.permute(1, 2, 0).numpy()
    
    # Convert to uint8 if needed (for PIL Image compatibility)
    if img_numpy.dtype != np.uint8:
        img_numpy = (img_numpy * 255).clip(0, 255).astype(np.uint8)
    
    return img_numpy

def img_numpy_to_tensor(image_numpy: np.ndarray) -> torch.Tensor:
    """Convert a NumPy array to a PyTorch tensor."""
    if not isinstance(image_numpy, np.ndarray):
        raise TypeError("Input must be a NumPy array.")
    
    # Ensure the array is in HxWxC format
    if image_numpy.ndim != 3 or image_numpy.shape[2] not in [1, 3]:
        raise ValueError("Array must be in HxWxC format.")
    
    # Convert the NumPy array to a tensor and change to CxHxW format
    img_tensor = torch.tensor(image_numpy).permute(2, 0, 1).float()
    return img_tensor


def mask_image(
    img: np.typing.NDArray, 
    masking_prob=0.1
) -> typing.Tuple[torch.Tensor, torch.Tensor]:
    # img is [h, w, channels]
    img_numpy = np.array(img) 
    patch_size = 16 
    
    counter = 0
    
    assert 196 == img_numpy.shape[0] // patch_size * img_numpy.shape[1] // patch_size
    masked_patches_idxs = torch.full((14*14,), 0)
    # print(f"masked_patches_idxs: {masked_patches_idxs.shape}")
    
    for i in range(0, img_numpy.shape[0], patch_size):
        for j in range(0, img_numpy.shape[1], patch_size):
            rand = random.random()
            if rand < masking_prob: 
                # mask with black
                img_numpy[i:i+patch_size, j:j+patch_size, :] = 0
                masked_patches_idxs[counter] = 1
            
            counter += 1
        

    masked_img = img_numpy_to_tensor(img_numpy)
    
    return masked_img, masked_patches_idxs
    


def mask_image_test(image_path: str): 
    img = Image.open(image_path).convert("RGB")
    
    # no need to resize, as it is already resized in the preprocessing step
    # using the transformations
    img = img.resize(IMG_SIZE)
    img_numpy = np.array(img)   # h, w, c
    
    masked_img, masked_patches_idxs = mask_image(
        img_numpy,)
    
    masked_img = Image.fromarray(masked_img)
    masked_img.save("masked_image.jpg", format="JPEG")
    print(f"masked idxs: {masked_patches_idxs}")
    
    
def save_image(img: torch.tensor, path="saved_image.jpg"): 
    img_numpy = img_tensor_to_numpy(img)
    img_pil = Image.fromarray(img_numpy)
    img_pil.save(path, format="JPEG")
    
    


class GeLU(nn.Module):
    def __init__(self): 
        super(GeLU, self).__init__()
        
    def forward(self, x): 
        return self.__custom_gelu(x)
    
    def __custom_gelu(self, x): 
        """Implementation of the gelu activation function. I found this in the vilbert code: (line 111) in https://github.com/facebookresearch/vilbert-multi-task/blob/main/vilbert/vilbert.py
            For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
            0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
            Also see https://arxiv.org/abs/1606.08415
        """
        return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))
    

    
    
class InfoNCE(nn.Module): 
    # inspiration from: https://github.com/arashkhoeini/infonce/blob/main/infonce/infonce.py
    # temperature setting like in clip paper
    def __init__(self, temperature:float = 0.07): 
        super(InfoNCE, self).__init__()
        self.temperature = temperature
        
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        
        
    # TODO: not quite sure if this implementation is correct
    def forward(self, x, target_labels): 
        
        x = torch.nn.functional.normalize(x, dim=-1)
        target_labels = torch.nn.functional.normalize(target_labels, dim=-1)
        
        n = target_labels.size(0)
        logits = torch.matmul(x, target_labels.T) / self.temperature
        
        labels = torch.arange(n, device=x.device)
        
        # loss2 with logits.T. same as axis=1 in clip paper
        # loss2 = self.ce(logits, labels, axis=1)
        loss1 = self.cross_entropy_loss(logits, labels)
        loss2 = self.cross_entropy_loss(logits.T, labels)
        
        loss = (loss1 + loss2) / 2
        
        return loss
    
    
        

    
    
    
def freeze_all_layers(model: nn.Module): 
    for param in model.parameters(): 
        param.requires_grad = False
        
def params_summary(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters(): 
        total_params += p.numel()
    
        if p.requires_grad: 
            trainable_params += p.numel()
            
    print(f"trainable params: {trainable_params}/{total_params}")
        
        
def force_memory_cleanup():    
    # Force garbage collection multiple times
    for _ in range(3):
        gc.collect()
    
    # Clear all GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
    time.sleep(1)
        
    # Force another garbage collection
    gc.collect()
    
    
    print(f"GPU memory after cleanup: {torch.cuda.memory_allocated()/1024**3:.2f}GB / {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")



def plot_losses(
    train_losses_ap, 
    validation_losses_ap, 
    train_losses_mlm, 
    validation_losses_mlm
):
    epochs = range(1, len(train_losses_ap) + 1)
    
    assert len(train_losses_ap) == len(validation_losses_ap) == len(train_losses_mlm) == len(validation_losses_mlm)
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses_ap, 'b-', label='Train AP')
    plt.plot(epochs, validation_losses_ap, 'r-', label='Val AP')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Alignment Prediction Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_losses_mlm, 'g-', label='Train MLM')
    plt.plot(epochs, validation_losses_mlm, 'orange', label='Val MLM')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Masked Language Modeling Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('res/training_losses.png', dpi=150, bbox_inches='tight')
    # plt.show()
    
    
if __name__ == "__main__": 
    img_data: torch.Tensor =  get_image_embedding("res/test_img.jpg")
    img_tensor = img_data["pixel_values"].squeeze(0)  # [3,224,224]
    img_numpy = img_tensor_to_numpy(img_tensor)
    
    masked_image, masked_patches_idxs = mask_image(img_numpy)
    
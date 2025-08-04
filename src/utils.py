import math

import torch
from torch import nn
import gc; import time
from functools import wraps
import matplotlib.pyplot as plt

from enum import Enum


class Task(Enum): 
    ALIGNMEN_PREDICTION = 1
    MASKED_LM = 2
    MASKED_IM = 3
    

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
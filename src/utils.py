import math

import torch
from torch import nn

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


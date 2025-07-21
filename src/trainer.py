import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

import utils
import datasets
from vilbert import ViLBERT
from config import * 


class Trainer(): 
    def __init__(self, model: ViLBERT, config: Config): 
        pass
    
    def train_epoch(self, ): 
        ... 
        
    def train(self, epochs: int): 
        ...
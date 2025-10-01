from abc import ABC, abstractmethod
import typing
import torch
from logger import Logger
from vilbert import ViLBERT, ViLBERTConfig
from info_nce import InfoNCE, info_nce
import utils
from config import *
import analysis
from datasets import *
import tqdm


from torch.utils.data import DataLoader, Dataset

class BaseTrainer(ABC):


    @abstractmethod
    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        epochs: int,
        hm_dataloader: Dataset=None,
        cc_dataloader: Dataset=None,
    ):
        pass

    @abstractmethod
    def train_epoch(self, dataloader: DataLoader):
        pass


    @abstractmethod
    def setup_scheduler(
        self,
        epochs: int,
        train_dataloader: DataLoader,
        lr: typing.Optional[float]= None
    ):
        pass



    @abstractmethod
    def evaluate(self, dataloader: DataLoader):
        pass

    def get_final_representation(self, text_embedding: torch.tensor, image_embedding: torch.tensor,
        fusion_method=CLS_FUSION_METHOD) -> torch.tensor:
        assert fusion_method in FUSION_METHODS
        if fusion_method == "sum":
            return text_embedding + image_embedding

        elif fusion_method == "hadamard":
            return text_embedding * image_embedding

        elif fusion_method == "concat":
            combined = torch.concat((text_embedding, image_embedding),1)
            return combined

        # TODO: problem with implementing this as the fc in the vilbert relies on the input size


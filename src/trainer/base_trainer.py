from abc import ABC, abstractmethod
import typing

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

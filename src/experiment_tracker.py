import time
import os
from typing import Tuple, Optional


from config import *
from trainer import Trainer
from mm_imdb_trainer import MM_IMDB_Trainer
from vilbert import ViLBERT
import utils
import datasets
from logger import Logger
import analysis

logger = Logger()

"""
training_results = {
    "hateful_memes": {
        "alignment": {
            0: { ... },      # uninitialized
            1: {            # epoch 1
                0: {        # layer 0
                    "is_cross_attention": bool,
                    "cosine": float,
                    "cka": float,
                    "max_sim_tp": float,
                    "max_sim_pt": float,
                    "svcca": float,
                    "mknn_full_epoch": float,
                    "rank_full_epoch": float,
                    "procrustes_full_epoch": float
                },
        },
        "training": {
            1 : { ... }   # epoch 1
            2 : {
                "train_loss": float,
                "val_loss": float,
                "val_acc": float
            }
        },
        ...
    },
    "mm_imdb": { ... }
}
"""

class ExperimentTracker:
    def __init__(self,):
        self.save_dir = "res/experiments/"
        os.makedirs(self.save_dir, exist_ok=True)
        ...


    def run_single_experiment_hateful_memes(
        self,
        config: ViLBERTConfig,
        training_results: dict,
        epochs: int
    ):
        model = self.create_model(config)
        trainer = Trainer(
            model=model,
            config=config,
            gradient_accumulation=GRADIENT_ACCUMULATION
        )

        train_loader, val_loader = datasets.get_hateful_memes_datasets(
            train_test_ratio=TRAIN_TEST_RATIO,
            # train_test_ratio=0.1,
            batch_size=BATCH_SIZE_DOWNSTREAM,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            use_train_augmentation=True,
        )
        hm_dataloader, cc_dataloader = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            num_samples=2000
        )
        trainer.setup_scheduler(epochs=epochs, train_dataloader=train_loader,
                                lr=DOWNSTREAM_LR)


        for i in range(epochs):
            train_loss = self.train_model_step(
                trainer=trainer,
                train_dataloader=train_loader,
            )
            test_loss, acc = self.evaluate_model(
                trainer=trainer,
                dataloader=val_loader,
            )
            info_str = (
                f"Epoch {i+1}/{epochs}, Train Loss: {train_loss:.4f}"
                f", Val Loss: {test_loss:.4f}, Val Acc: {acc:.4f}"
            )
            print(info_str)
            logger.info(info_str)

            training_results["hateful_memes"]["training"][i+1]["train_loss"] = train_loss
            training_results["hateful_memes"]["training"][i+1]["val_loss"] = test_loss
            training_results["hateful_memes"]["training"][i+1]["val_acc"] = acc


            alignment_metrics = self.analyse_alignment(model=model, dataloader=hm_dataloader)
            training_results["hateful_memes"]["alignment"][i+1] = alignment_metrics
        return training_results

    def run_single_experiment_mm_imdb(
        self,
        config: ViLBERTConfig,
        training_results: dict,
        epochs: int
    ):
        model = self.create_model(config)
        trainer = MM_IMDB_Trainer(
            model=model,
            config=config,
            gradient_accumulation=GRADIENT_ACCUMULATION,
        )

        train_loader, val_loader = datasets.get_mmimdb_datasets(
            # train_test_ratio=TRAIN_TEST_RATIO,
            train_test_ratio=0.1,
            batch_size=BATCH_SIZE_DOWNSTREAM,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            use_train_augmentation=True,
        )
        #TODO: get alignment data loader for mmimdb
        hm_dataloader, cc_dataloader = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            num_samples=2000
        )
        trainer.setup_scheduler(epochs=epochs, train_dataloader=train_loader,
                                lr=DOWNSTREAM_LR)


        for i in range(epochs):
            train_loss = self.train_model_step(
                trainer=trainer,
                train_dataloader=train_loader,
            )
            test_loss, acc = self.evaluate_model(
                trainer=trainer,
                dataloader=val_loader,
            )
            info_str = (
                f"Epoch {i+1}/{epochs}, Train Loss: {train_loss:.4f}"
                f", Val Loss: {test_loss:.4f}, Val Acc: {acc:.4f}"
            )
            print(info_str)
            logger.info(info_str)

            training_results["mm_imdb"]["training"][i+1]["train_loss"] = train_loss
            training_results["mm_imdb"]["training"][i+1]["val_loss"] = test_loss
            training_results["mm_imdb"]["training"][i+1]["val_acc"] = acc


            alignment_metrics = self.analyse_alignment(model=model, dataloader=hm_dataloader)
            training_results["mm_imdb"]["alignment"][i+1] = alignment_metrics
        return training_results





    def run_single_experiment(self, config):
        #TODO: temp
        epochs = 4

        #TODO: proper handling of configs
        proper_config = self.create_config(config)
        config = proper_config
        #TODO: two tasks: hateful memes and mm-imdb
        utils.set_seeds(SEED)

        training_results = self._initialize_results_dict(epochs=epochs)

        training_results = self.run_single_experiment_mm_imdb(
            config=config,
            training_results=training_results,
            epochs=epochs
        )
        training_results = self.run_single_experiment_hateful_memes(
            config=config,
            training_results=training_results,
            epochs=epochs
        )







        # self.save_results()

    def _initialize_results_dict(self, epochs ):
        training_results = {
            "hateful_memes": { "alignment": {}, "training": {} },
            "mm_imdb": {  "alignment": {}, "training": {} }
        }
        training_results["hateful_memes"]["alignment"][0] = {}
        training_results["mm_imdb"]["alignment"][0] = {}
        for i in range(epochs):
            curr_epoch = i+1    # 0 is uninitialized
            training_results["hateful_memes"]["alignment"][curr_epoch] = {}
            training_results["mm_imdb"]["alignment"][curr_epoch] = {}
            training_results["hateful_memes"]["training"][curr_epoch] = {}
            training_results["mm_imdb"]["training"][curr_epoch] = {}

        return  training_results




    def train_model_step(
        self,
        trainer: Trainer,
        train_dataloader: datasets.DataLoader,
    ):
        return trainer.train_epoch(
            data_loader=train_dataloader
        )

    def evaluate_model(
        self,
        trainer: Trainer,
        dataloader: datasets.DataLoader,
        ) -> Tuple[float, float]:

        return trainer.evaluate(dataloader=dataloader)



    def analyse_alignment(
        self,
        model: ViLBERT,
        dataloader: datasets.DataLoader,
        #TODO: implemet mmimdb loader
    ):
        measures_per_layer = analysis.analyse_alignment(dataloader=dataloader, model=model)
        print(measures_per_layer)
        return measures_per_layer

    def create_model(self, config) -> ViLBERT:
        # TODO: currently config is ignored
        config = ViLBERTConfig()
        model = ViLBERT(config=config)
        return model

    def create_config(self, config):
        proper_config = ViLBERTConfig()
        return proper_config




class ExperimentResults():
    ...




def main():
    tracker = ExperimentTracker()
    config = None
    tracker.run_single_experiment(config)


if __name__ == "__main__":
    main()






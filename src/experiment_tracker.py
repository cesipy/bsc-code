import time
import os
from typing import Tuple, Optional
import dataclasses
import json



from config import *
from trainer import Trainer
from mm_imdb_trainer import MM_IMDB_Trainer
from vilbert import ViLBERT
import utils
import datasets
from logger import Logger
import analysis

logger = Logger()
EPOCHS = 1
ALIGNMENT_ANALYSIS_SIZE = 4000


@dataclasses.dataclass
class ExperimentConfig:
    cross_attention_layers: list[int]
    depth:int

    epochs: int = EPOCHS
    batch_size: int = BATCH_SIZE_DOWNSTREAM
    gradient_accumulation: int = GRADIENT_ACCUMULATION_DOWNSTREAM
    learning_rate: float = DOWNSTREAM_LR
    seed:int = SEED
    train_test_ratio: float = TRAIN_TEST_RATIO



    def __post_init__(self):
        if self.cross_attention_layers != []:
            assert self.depth >= max(self.cross_attention_layers)
            assert self.depth >= len(self.cross_attention_layers)


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
    "mm_imdb": { ... },
    "config": { ... }       #TODO: complete this
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
        hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            num_samples=ALIGNMENT_ANALYSIS_SIZE
        )
        trainer.setup_scheduler(epochs=epochs, train_dataloader=train_loader,
                                lr=config.learning_rate)


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
        hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            num_samples=ALIGNMENT_ANALYSIS_SIZE
        )
        trainer.setup_scheduler(epochs=epochs, train_dataloader=train_loader,
                                lr=config.learning_rate)


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


            alignment_metrics = self.analyse_alignment(
                model=model, dataloader=imdb_dataloader)
            training_results["mm_imdb"]["alignment"][i+1] = alignment_metrics
        return training_results





    def run_single_experiment(self, experiment_config: ExperimentConfig):

        config = self.create_config(experiment_config)
        utils.set_seeds(experiment_config.seed)

        training_results = self._initialize_results_dict(epochs=experiment_config.epochs)

        training_results = self.run_single_experiment_mm_imdb(
            config=config,
            training_results=training_results,
            epochs=experiment_config.epochs
        )
        training_results = self.run_single_experiment_hateful_memes(
            config=config,
            training_results=training_results,
            epochs=experiment_config.epochs
        )
        self.save_results(training_results=training_results, config=experiment_config)

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
        return measures_per_layer

    def create_model(self, config: ViLBERTConfig) -> ViLBERT:
        model = ViLBERT(config=config)
        return model

    def create_config(self, experiment_config: ExperimentConfig):
        config = ViLBERTConfig()
        config.cross_attention_layers = experiment_config.cross_attention_layers
        config.depth = experiment_config.depth
        if config.cross_attention_layers != []:
            assert config.depth >= max(config.cross_attention_layers)
        assert config.depth >= len(config.cross_attention_layers)
        config.epochs = experiment_config.epochs
        config.batch_size = experiment_config.batch_size
        config.gradient_accumulation = experiment_config.gradient_accumulation
        config.learning_rate = experiment_config.learning_rate
        config.seed = experiment_config.seed
        config.train_test_ratio = experiment_config.train_test_ratio
        return config


    def save_results(self, training_results: dict, config: ExperimentConfig):
        tmsp = time.strftime("%Y%m%d-%H%M%S")

        training_results["config"] = {
            "cross_attention_layers": config.cross_attention_layers,
            "depth": config.depth,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "train_test_ratio": config.train_test_ratio,
        }

        # coattn_pix: str =""
        if config.cross_attention_layers == []:
            coattn_fix = "no_coattn"
        else:
            coattn_fix = "coattn_"
            for i in config.cross_attention_layers:
                coattn_fix += str(i)
                coattn_fix += "-"
            coattn_fix = coattn_fix[:-1]

        filename = f"experiment_{coattn_fix}_{tmsp}.json"
        filename = os.path.join(self.save_dir, filename)
        with open(filename, "w") as f:
            json.dump(training_results, f, indent=4)


def get_experiments():
    from itertools import chain, combinations

    depth = 4
    arr = [i for i in range(depth)]
    all = []

    powerset = list(chain.from_iterable(combinations(arr, r) for r in range(len(arr)+1)))
    for s in powerset:
        all.append(list(s))

    print(f"total experiments: {len(all)}")
    print(all)

    exps = []

    for i in all:
        exp = ExperimentConfig(
            cross_attention_layers=i,
            depth=depth,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE_DOWNSTREAM,
            gradient_accumulation=GRADIENT_ACCUMULATION_DOWNSTREAM,
            learning_rate=DOWNSTREAM_LR,
            seed=SEED,
            train_test_ratio=TRAIN_TEST_RATIO,
        )
        exps.append(exp)

    return exps





def main():
    exps = get_experiments()
    tracker = ExperimentTracker()
    # config = ExperimentConfig(
    #     cross_attention_layers=[3],
    #     depth=4,
    #     epochs=1
    # )
    for config in exps:
        print(f"Running experiment with config: {config}")
        tracker.run_single_experiment(config)


if __name__ == "__main__":
    main()

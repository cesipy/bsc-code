import time
import os
from typing import Tuple
import dataclasses
import json
import optuna
from optuna.pruners import MedianPruner

import math

from config import *
from trainer import HatefulMemesTrainer
from mm_imdb_trainer import MM_IMDB_Trainer
from vilbert import ViLBERT
import utils
import datasets
from logger import Logger

logger = Logger()

@dataclasses.dataclass
class OptunaTuningConfig:
    """Config for hyperparameter optimization"""
    task_name: str  # "hateful_memes" or "mm_imdb"
    n_trials: int = 25
    max_epochs: int = 9
    cross_attention_layers: list[int] = None
    optimize_depth: bool = False

    def __post_init__(self):
        if self.cross_attention_layers is None:
            self.cross_attention_layers = [1, 3]


class HyperparameterOptimizer:

    def __init__(self):
        self.save_dir = "res/hyperparameter_optimization/"
        os.makedirs(self.save_dir, exist_ok=True)


    def optimize_single_task(self, tuning_config: OptunaTuningConfig, optim_obj:str = "acc"):

        assert tuning_config.task_name in ["hateful_memes", "mm_imdb"]
        assert optim_obj in ["acc", "loss"]
        def objective(trial):
            learning_rate = trial.suggest_float("learning_rate", 5e-6, 1.5e-4, log=True)
            dropout = trial.suggest_float("dropout", 0.0, 0.4)
            epochs = trial.suggest_int("epochs", 3, tuning_config.max_epochs)

            # opt: not sure if i want it right now
            if tuning_config.optimize_depth:
                depth = trial.suggest_int("depth", 3, 6)
            else:
                depth = 4


            config = self._create_trial_config(
                cross_attention_layers=tuning_config.cross_attention_layers,
                depth=depth,
                dropout=dropout,
                learning_rate=learning_rate,
                epochs=epochs
            )

            utils.set_seeds(SEED)

            try:
                val = self._run_trial(
                    trial=trial,
                    config=config,
                    task_name=tuning_config.task_name,
                    epochs=epochs,
                    optim_objective=optim_obj
                    )
                return val
            except optuna.TrialPruned:
                raise
            except Exception as e:
                print(f"Trial failed: {e}")
                return 0.0

        tmsp = time.strftime("%Y%m%d-%H%M%S")
        storage_path = f"sqlite:///{self.save_dir}optuna_study_{tmsp}.db"
        study_name = f"{tuning_config.task_name}_study_{tmsp}"

        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1, interval_steps=1)

        study = optuna.create_study(
            direction="maximize",
            pruner=pruner,
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True  # resume if exists
        )

        # print(f"Optimizing {tuning_config.task_name}: lr, dropout, epochs" +
        #       (", depth" if tuning_config.optimize_depth else ""))

        study.optimize(objective, n_trials=tuning_config.n_trials)

        best_params = study.best_params
        print(f"\nbest {tuning_config.task_name} params: {best_params}")
        print(f"best val- accuracy: {study.best_value:.4f}")

        self._save_optimization_results(study, tuning_config)

        return best_params

    def _create_trial_config(self, cross_attention_layers, depth, dropout, learning_rate, epochs):
        config = ViLBERTConfig()
        config.cross_attention_layers = cross_attention_layers
        config.depth = depth
        config.dropout_prob = dropout
        config.learning_rate = learning_rate
        config.epochs = epochs
        config.batch_size = BATCH_SIZE_DOWNSTREAM
        config.gradient_accumulation = GRADIENT_ACCUMULATION_DOWNSTREAM
        config.seed = SEED
        config.train_test_ratio = TRAIN_TEST_RATIO
        if config.cross_attention_layers != []:
            assert config.depth >= max(config.cross_attention_layers)
        assert config.depth >= len(config.cross_attention_layers)

        return config


    def _run_trial(self,trial, config: ViLBERTConfig, task_name: str, epochs: int, optim_objective:str="acc"):
        """Run single trial and return best validation accuracy"""

        assert optim_objective in ["acc", "loss"]
        model = ViLBERT(config=config)

        if task_name == "hateful_memes":
            trainer = HatefulMemesTrainer(model=model, config=config,
                            gradient_accumulation=config.gradient_accumulation)
            train_loader, val_loader = datasets.get_hateful_memes_datasets(
                train_test_ratio=config.train_test_ratio,
                batch_size=config.batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=True,
            )
        elif task_name == "mm_imdb":
            trainer = MM_IMDB_Trainer(model=model, config=config,
                                    gradient_accumulation=config.gradient_accumulation)
            train_loader, val_loader = datasets.get_mmimdb_datasets(
                train_test_ratio=config.train_test_ratio,
                batch_size=config.batch_size,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=True,
            )
        else:
            print("smth completely wrong!")
            exit()

        trainer.setup_scheduler(epochs=epochs, train_dataloader=train_loader,
                              lr=config.learning_rate)

        #track best validation accuracy
        best_val = float("-inf")
        for epoch in range(epochs):
            train_loss = trainer.train_epoch(dataloader=train_loader)
            val_loss, val_acc = trainer.evaluate(dataloader=val_loader)

            if optim_objective == "acc":
                current_val = val_acc
                best_val = max(best_val, val_acc)
            else:   # val_loss
                current_val = -val_loss
                best_val = max(best_val, current_val)
            trial.report(current_val, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
            if epoch == 0:
                print(f"  Epoch 1/{epochs} - Train: {train_loss:.4f}, "
                    f"Val: {val_loss:.4f}, Acc: {val_acc:.4f}")
        return best_val

    def _save_optimization_results(self, study, tuning_config: OptunaTuningConfig):
        """Save optimization results"""
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"optuna_{tuning_config.task_name}_{timestamp}.json"
        results = {
            "task_name": tuning_config.task_name,
            "n_trials": tuning_config.n_trials,
            "best_params": study.best_params,
            "best_value": study.best_value,
            "tuning_config": {
                "max_epochs": tuning_config.max_epochs,
                "cross_attention_layers": tuning_config.cross_attention_layers,
                "optimize_depth": tuning_config.optimize_depth,
            }
        }

        filepath = os.path.join(self.save_dir, filename)
        with open(filepath, "w") as f:
            json.dump(results, f, indent=4)

        print(f"saved to: {filepath}")


def main():
    optimizer = HyperparameterOptimizer()

    confs = [
        OptunaTuningConfig(
            task_name="hateful_memes",
            n_trials=25,
            max_epochs=9,
            cross_attention_layers=[1, 3],
            optimize_depth=False
        ),
        OptunaTuningConfig(
            task_name="mm_imdb",
            n_trials=25,
            max_epochs=9,
            cross_attention_layers=[1, 3],
            optimize_depth=False
        )
    ]

    print("------------------------------------------\nhyperparam optim")

    results = {}
    for config in confs:
        results[config.task_name] = optimizer.optimize_single_task(config, optim_obj="loss")
        print("-"*25)

    results["timestamp"] = time.strftime("%Y%m%d-%H%M%S")


    with open("res/hyperparameter_optimization/best_hyperparameters_combined.json", "w") as f:
        json.dump(results, f, indent=4)
    print(f"\nhyperparam-optim done.")


if __name__ == "__main__":
    main()
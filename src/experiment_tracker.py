import time
import os
from typing import Tuple, Optional
import dataclasses
import json
import numpy as np

import torch


import optuna; from optuna import pruners


from config import *
from trainer import HatefulMemesTrainer, PretrainingTrainer, UPMCTrainer, MM_IMDB_Trainer, VQATrainer

from vilbert import ViLBERT
import utils
import datasets
from logger import Logger
import analysis
import task as tasklib

logger = Logger()
#specific to that file
EPOCHS_ = 9
LR_ = 3.2e-5
USE_CONTRASTIVE_LOSS_ = False


ALIGNMENT_ANALYSIS_SIZE = 512
SKIP_ALIGNMENT = False


@dataclasses.dataclass
class ExperimentConfig:
    t_biattention_ids: list
    v_biattention_ids: list
    use_contrastive_loss: bool

    epochs: int = EPOCHS_
    batch_size: int = BATCH_SIZE_DOWNSTREAM
    gradient_accumulation: int = GRADIENT_ACCUMULATION_DOWNSTREAM
    learning_rate: float = DOWNSTREAM_LR
    seed:int = SEED
    train_test_ratio: float = TRAIN_TEST_RATIO
    dropout:float = DROPOUT_PROB



    def __post_init__(self):
        if self.t_biattention_ids:
            assert 12 >= max(self.t_biattention_ids)
        if self.v_biattention_ids:
            assert 12 >= max(self.v_biattention_ids)
        assert len(self.t_biattention_ids) == len(self.v_biattention_ids)

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
        self.visualization_dir = "res/experiments/visualizations/"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)


    def get_biattentions(self, num_coattn_layers:int, t_center:float,t_spread:float,v_center:float,v_spread:float):
        """ more sophisticated approach for parametrized coattention construction.

        Args:
            num_coattn_layers: int, number of coattention layers to use
            t_center: float, center of text coattention layers
            t_spread: float, spread of text coattention layers
            v_center: float, center of vision coattention layers
            v_spread: float, spread of vision coattention layers
        """
        from scipy.stats import norm

        t_biattention_ids = set()
        v_biattention_ids = set()

        for i in range(num_coattn_layers):  # if its not working, do mutliple attempts
            quantile = (i+ 1) / (num_coattn_layers + 1)

            if len(t_biattention_ids) < num_coattn_layers:
                t_layer = int(np.clip(round(t_center + t_spread * norm.ppf(quantile)), 0, 11))
                t_biattention_ids.add(t_layer)

            if len(v_biattention_ids) < num_coattn_layers:
                v_layer = int(np.clip(round(v_center + v_spread * norm.ppf(quantile)), 0, 11))
                v_biattention_ids.add(v_layer)

        t_biattention_ids = sorted(list(t_biattention_ids))
        v_biattention_ids = sorted(list(v_biattention_ids))

        min_len = min(len(t_biattention_ids), len(v_biattention_ids))
        if min_len < 2:     # TODO: make configurable
            raise optuna.TrialPruned()
        return t_biattention_ids[:min_len], v_biattention_ids[:min_len]


    def get_coattn_configs(self, trial: optuna.Trial, max_layers: int = 12
                           ) -> Tuple[list, list]:
        """ more sophisticated approach for parametrized coattention construction.
        uses continuous distrubutions to generate discrete points.

        first we sample center and spread (like mean and var in gaussian), get the quantiles from distribution and
        round/clip to the indices.
        """
        num_coattn_layers = trial.suggest_int("num_coattn_layers", 2, 8)

        t_center = trial.suggest_float("t_center", 2.0, 10.0)
        t_spread = trial.suggest_float("t_spread", 1.0, 4.0)

        v_center = trial.suggest_float("v_center", 2.0, 10.0)
        v_spread = trial.suggest_float("v_spread", 1.0, 4.0)

        return self.get_biattentions(num_coattn_layers, t_center, t_spread, v_center, v_spread)




    def construct_coattn_configs(self, strat:str) -> Tuple[list, list]:
        if strat == "early":
            return [0,1,2], [0, 1, 2]
        elif strat == "mid":
            return [4,5,6], [4,5,6]
        elif strat == "late":
            return [9,10,11], [9,10,11]
        elif strat == "early-mid":
            return [1,5,6], [1,5,6]
        elif strat == "early-late":
            return [1,10,11], [1,10,11]
        elif strat == "mid-late":
            return [5,10,11], [5,10,11]
        elif strat == "mixed":
            return [1,4,10], [2,6,11]
        else:
            raise ValueError(f"unknown fusion_strat: {strat}")

    def optimize_parameters_multi(self, n_trials, optimization_objective: str = "acc"):
        assert optimization_objective in ["acc", "loss"]
        import optuna

        def objective(trial):
            lr = LR_
            epochs = EPOCHS_
            seed = SEED + int(time.time())

            t_biattention_ids, v_biattention_ids = self.get_coattn_configs(trial)
            print("t_biattention_ids:", t_biattention_ids)
            print("v_biattention_ids:", v_biattention_ids)
            config = ExperimentConfig(
                t_biattention_ids=t_biattention_ids,
                v_biattention_ids=v_biattention_ids,
                epochs=epochs,
                learning_rate=lr,
                use_contrastive_loss=USE_CONTRASTIVE_LOSS_,
                seed=seed,
            )

            training_results = self.run_finetune(
                experiment_config=config,
                run_visualizations=False,  # is too compute intensive, not wanted here
                run_alignment_analysis=False,
            )

            if optimization_objective == "acc":
                val_accs_hm = [training_results["hateful_memes"]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]
                val_accs_imdb = [training_results["mm_imdb"]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]

                # report intermediate results for dashboard. not quite sure about it
                for epoch, (hm_acc, imdb_acc) in enumerate(zip(val_accs_hm, val_accs_imdb)):
                    combined_metric = (hm_acc + imdb_acc) / 2
                    # trial.report(combined_metric, epoch)
                result=  max(val_accs_hm), max(val_accs_imdb)
                # return val_accs_hm[-1], val_accs_imdb[-1]
            else:
                val_losses_hm = [training_results["hateful_memes"]["training"][i]["val_loss"] for i in range(1, config.epochs+1)]
                val_losses_imdb = [training_results["mm_imdb"]["training"][i]["val_loss"] for i in range(1, config.epochs+1)]


                for epoch, (hm_loss, imdb_loss) in enumerate(zip(val_losses_hm, val_losses_imdb)):
                    combined_metric = -(hm_loss + imdb_loss) / 2
                    # trial.report(combined_metric, epoch)
                result=  -min(val_losses_hm), -min(val_losses_imdb)
                # return -val_losses_hm[-1], -val_losses_imdb[-1]
            logger.info(f"completed: {str(trial)}")
            return result

        tmsp = time.strftime("%Y%m%d-%H%M%S")
        # storage_path = f"sqlite:///{self.save_dir}multi_task_optim.db"
        storage_path = f"sqlite:///{self.save_dir}multi_task_optim.db"
        study_name = f"multi_task_study_{tmsp}"
        # study_name = f"multi_task_study_20251004-131802"


        study = optuna.create_study(
            directions=["maximize", "maximize"],
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=n_trials)
        pareto_trials = study.best_trials
        print(f"{len(pareto_trials)} optimal solutions:")

        for i, trial in enumerate(pareto_trials[:5]):
            hm_val, imdb_val = trial.values
            print(f"  Solution {i+1}: HM={hm_val:.4f}, IMDB={imdb_val:.4f}")
            print(f"    Params: {trial.params}")

        # TODO: add
        # self._save_multi_optimization_results(study, optimization_objective, tmsp)
        return pareto_trials[0].params if pareto_trials else {}

    def optimize_parameters_single(self, n_trials, optimization_objective: str = "acc", task: str = "hateful_memes"):
        """ optimize parameters for one task"""
        assert optimization_objective in ["acc", "loss"]
        assert task in ["hateful_memes", "mm_imdb"]
        import optuna

        def objective(trial):
            # lr = trial.suggest_float("learning_rate", 1.5e-5, 3.2e-5, log=True)
            lr = LR_
            # analysis with optuna resulted in dropout of about 0.08.
            # this is roughly the same as in vilbert implementation of 0.1
            # therefore, no further tuning on it
            # dropout = trial.suggest_float("dropout", 0.0, 0.4)
            # epochs = trial.suggest_int("epochs", 2, 9)
            seed = SEED + int(time.time())
            # also epoch 7 is really good, like 7 - 10 based on optuna
            epochs = EPOCHS_
            # depth = trial.suggest_int("depth", 4, 8)

            t_biattention_ids, v_biattention_ids = self.get_coattn_configs(trial)

            print("t_biattention_ids:", t_biattention_ids)
            print("v_biattention_ids:", v_biattention_ids)
            config = ExperimentConfig(
                t_biattention_ids=t_biattention_ids,
                v_biattention_ids=v_biattention_ids,
                epochs=epochs,
                learning_rate=lr,
                use_contrastive_loss=USE_CONTRASTIVE_LOSS_,
                seed=seed,
            )
            training_results = self.run_finetune(
                config,
                run_visualizations=False,  # is too compute intensive, not wanted here,
                # run_alignment_analysis=True,
                tasks=[task]
            )

            if optimization_objective == "acc":
                val_accs = [training_results[task]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]

                # report intermediate results for dashboard. not quite sure about it
                for epoch, acc in enumerate(val_accs):
                    trial.report(acc, epoch)
                result =  max(val_accs)
                # return val_accs_hm[-1], val_accs_imdb[-1]
            else:
                val_losses = [training_results[task]["training"][i]["val_loss"] for i in range(1, config.epochs+1)]

                for epoch, loss in enumerate(val_losses):
                    trial.report(-loss, epoch)
                result =  -min(val_losses)
                # return -val_losses_hm[-1], -val_losses_imdb[-1]
            logger.info(f"trial {trial.number}: params={trial.params}, result={result:.4f}")
            return result

        tmsp = time.strftime("%Y%m%d-%H%M%S")
        storage_path = f"sqlite:///{self.save_dir}multi_task_optim.db"
        study_name = f"single_task_study_{task}_{tmsp}"


        study = optuna.create_study(
            direction="maximize",
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True
        )

        study.optimize(objective, n_trials=n_trials)
        pareto_trials = study.best_trials
        print(f"{len(pareto_trials)} optimal solutions:")

        print(f"Best trial: {study.best_trial.value:.4f}")
        print(f"Best params: {study.best_trial.params}")

        return study.best_trial.params




    #TODO: adapt to new architecture
    def optimize_coattn_for_accuracy(self, depth, n_trials, optimization_objective:list[str]= "acc"):
        assert optimization_objective in ["acc", "loss"]
        import optuna

        def _objective( trial):
            coattn_layers = []

            for i in range(depth):
                if trial.suggest_categorical(f"layer_{i}", [True, False]):
                    coattn_layers.append(i)

            config = ExperimentConfig(
                cross_attention_layers=coattn_layers,
                depth=depth,
            )

            # return self._run_trial(config, trial)


            training_results = self.run_finetune(
                config,
                run_visualizations=False                 # is too compute intensive, not wanted here
            )
            if optimization_objective == "acc":
                val_accs_hm = [training_results["hateful_memes"]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]
                val_accs_imdb = [training_results["mm_imdb"]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]
                return val_accs_hm[-1], val_accs_imdb[-1]
            else:
                val_losses_hm = [training_results["hateful_memes"]["training"][i]["val_loss"]  for i in range(1, config.epochs+1)]
                val_losses_imdb = [training_results["mm_imdb"]["training"][i]["val_loss"]  for i in range(1, config.epochs+1)]
                return val_losses_hm[-1], val_losses_imdb[-1]

            max_acc_hm = max(val_accs_hm)
            max_acc_imdb = max(val_accs_imdb)

            return max_acc_hm, max_acc_imdb



        storage_path = f"sqlite:///{self.save_dir}coattn_optim.db"
        study_name = "cross_attention_study"

        # pruner does not work here, as this is mutliobj.
        # surely there is a workaround, but i dont want to implement it right now lol
        # pruner = pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            directions=["maximize","maximize"],
            storage=storage_path,
            study_name=study_name,
            load_if_exists=True
        )
        study.optimize( _objective, n_trials=n_trials )

        pareto_trials = study.best_trials
        print(f"{len(pareto_trials)}optimal solutions:")

        for trial in pareto_trials:
            layers = [i for i in range(depth) if trial.params.get(f"layer_{i}", False)]
            hm_acc, imdb_acc = trial.values
            print(f"  {layers}: hm={hm_acc:.4f}, imdb={imdb_acc:.4f}")

        best_trial = max(pareto_trials, key=lambda t: sum(t.values)/2)
        best_layers = [i for i in range(depth) if best_trial.params.get(f"layer_{i}", False)]

        print(f"selected: {best_layers}")
        return best_layers



    def _get_task_trainer(self, task:str, model:ViLBERT ):
        if task == "hateful_memes":
            trainer = HatefulMemesTrainer(
                model=model,
                config=model.config,
                use_contrastive_loss=model.config.use_contrastive_loss,
                gradient_accumulation=GRADIENT_ACCUMULATION
            )
        elif task == "mm_imdb":
            trainer = MM_IMDB_Trainer(
                model=model,
                config=model.config,
                use_contrastive_loss=model.config.use_contrastive_loss,
                gradient_accumulation=GRADIENT_ACCUMULATION
            )
        elif task == "upmc_food":
            trainer = UPMCTrainer(
                model=model,
                config=model.config,
                use_contrastive_loss=model.config.use_contrastive_loss,
                gradient_accumulation=GRADIENT_ACCUMULATION
            )
        elif task =="easy_vqa":
            trainer = VQATrainer(
                model=model,
                config=model.config,
                gradient_accumulation=GRADIENT_ACCUMULATION,
            )
        else:
            raise ValueError(f"unknown task: {task}")

        return trainer



    def _get_task_dataloader(
        self,
        task:str,
        config:ViLBERTConfig,
    )-> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
        """
        Returns:
            Tuple[DataLoader, DataLoader] train_dataloader, val_dataloader
        """

        if task == "hateful_memes":
            train_loader, val_loader = datasets.get_hateful_memes_datasets(
                train_test_ratio=config.train_test_ratio,
                batch_size=BATCH_SIZE_DOWNSTREAM,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=True,
                seed=config.seed
            )
        elif task == "mm_imdb":
            train_loader, val_loader = datasets.get_mmimdb_datasets(
                train_test_ratio=config.train_test_ratio,
                batch_size=BATCH_SIZE_DOWNSTREAM,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=True,
                seed=config.seed,
            )

        elif task == "upmc_food":
            train_loader, val_loader = datasets.get_upmc_datasets(
                train_test_ratio=config.train_test_ratio,
                batch_size=BATCH_SIZE_DOWNSTREAM,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=True,
                seed=config.seed,
            )
        elif task == "easy_vqa":
            train_loader, val_loader = datasets.get_easyvqa_datasets(
                batch_size=BATCH_SIZE_DOWNSTREAM,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                seed=config.seed,
                # use_train_augmentation=True,
            )

        return train_loader, val_loader


    def _get_task_alignment_dataloader(
        self,
        task: str,
        config: ViLBERTConfig,
    ):
        if task == "hateful_memes":
            alignment_dataloader, _, _ = datasets.get_alignment_dataloaders(
                batch_size=BATCH_SIZE_ANALYSIS,
                num_workers=0,
                pin_memory=False,
                prefetch_factor=None,
                seed=config.seed,
                num_samples=ALIGNMENT_ANALYSIS_SIZE
            )
        elif task == "mm_imdb":
            _, _, alignment_dataloader = datasets.get_alignment_dataloaders(
                batch_size=BATCH_SIZE_ANALYSIS,
                num_workers=0,
                pin_memory=False,
                prefetch_factor=None,
                seed=config.seed,
                num_samples=ALIGNMENT_ANALYSIS_SIZE
            )
        # TODO: currently there is no alignment dataloader for upmc
        elif task == "upmc_food":
            _, alignment_dataloader, _ = datasets.get_alignment_dataloaders(
                batch_size=BATCH_SIZE_ANALYSIS,
                num_workers=0,
                pin_memory=False,
                prefetch_factor=None,
                seed=config.seed,
                num_samples=ALIGNMENT_ANALYSIS_SIZE
            )
        elif task == "easy_vqa":
            # TODO: also just uses cc right now, needs proper dataset!
            _, alignment_dataloader, _ = datasets.get_alignment_dataloaders(
                batch_size=BATCH_SIZE_ANALYSIS,
                num_workers=0,
                pin_memory=False,
                prefetch_factor=None,
                seed=config.seed,
                num_samples=ALIGNMENT_ANALYSIS_SIZE
            )
        else:
            raise ValueError(f"unknown task: {task}")

        return alignment_dataloader


    def run_single_experiment(
        self,
        task:str,
        config: ViLBERTConfig,
        training_results: dict,
        epochs: int,
        run_visualization: bool,
        dir_name: Optional[str] = None,
        skip_alignment_analysis: bool = False,
        tmsp: Optional[str] = None,
        pretrained_model=None
    ):
        assert task in tasklib.all_task_list

        if pretrained_model:
            model = pretrained_model
        else:
            model = self.create_model(config)
        trainer = self._get_task_trainer(task=task,model=model)

        train_loader, val_loader = self._get_task_dataloader(task=task, config=config)
        alignment_dataloader = self._get_task_alignment_dataloader(task=task, config=config)

        trainer.setup_scheduler(epochs=epochs, train_dataloader=train_loader,
                                lr=config.learning_rate)

        if not skip_alignment_analysis:
            alignment_metrics = self._analyse_alignment(model=model, dataloader=alignment_dataloader)
            training_results[task]["alignment"][0] = alignment_metrics

        if run_visualization:
            filename_extension = f"{tmsp}_e0"
            analysis.run_alignment_visualization(dataloader=alignment_dataloader, model=model,
                                                 dir_name=dir_name, filename_extension=filename_extension)

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

            training_results[task]["training"][i+1]["train_loss"] = train_loss
            training_results[task]["training"][i+1]["val_loss"] = test_loss
            training_results[task]["training"][i+1]["val_acc"] = acc

            alignment_metrics = {}
            if not skip_alignment_analysis:

                alignment_metrics = self._analyse_alignment(
                    model=model, dataloader=alignment_dataloader)

            training_results[task]["alignment"][i+1] = alignment_metrics

            if run_visualization:
                filename_extension = f"{tmsp}_e{i+1}"
                analysis.run_alignment_visualization(dataloader=alignment_dataloader, model=model,
                    dir_name=dir_name, filename_extension=filename_extension)

        if tmsp:
            save_path = f"res/checkpoints/{tmsp}_finetuned_{task}.pt"
            training_results[task]["model_path"] = save_path
            trainer.model.save_model(save_path)
            info_str = f"Saved finetuned model to {save_path}"
            logger.info(info_str)
            print(info_str)
        return training_results


    def _get_filename(self, config: ExperimentConfig):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # coattn_pix: str =""
        if config.v_biattention_ids  == []:
            coattn_fix = "no_coattn"
        else:
            coattn_fix = "coattn_"
            #TODO: only temporary, fix!
            for i in config.v_biattention_ids:
                coattn_fix += str(i)
                coattn_fix += "-"
            coattn_fix = coattn_fix[:-1]

        filename = f"{timestamp}_experiment_{coattn_fix}"

        return filename, timestamp


    def _run_task(self, training_results: dict,
        task_name: str, experiment_config:ExperimentConfig,
        filename: str, run_visualizations:bool,
        run_alignment_analysis:bool,
        tmsp: Optional[str] = None,
        pretrained_model=None
        ):
        assert task_name in tasklib.all_task_list

        config = self.create_config(experiment_config)

        print(f"Saving results to {filename}")
        logger.info(f"saving results to {filename}")
        exp_dir_name = os.path.join(self.visualization_dir, filename)
        os.makedirs(exp_dir_name, exist_ok=True)

        exp_dir_name = os.path.join(exp_dir_name, task_name)
        os.makedirs(exp_dir_name, exist_ok=True)
        training_results = self.run_single_experiment(
            task=task_name,
            config=config,
            training_results=training_results,
            epochs=experiment_config.epochs,
            dir_name=exp_dir_name,
            run_visualization=run_visualizations,
            skip_alignment_analysis=not run_alignment_analysis,
            tmsp=tmsp,
            pretrained_model=pretrained_model

        )
        return training_results




    def run_finetune(
        self,
        experiment_config: ExperimentConfig,
        run_visualizations:bool=False,
        run_alignment_analysis:bool=False,
        tasks:list[str]=["hateful_memes", "mm_imdb"],
        pretrained_model_path: Optional[str] = None,
    ) -> dict:
        print(f"seed = {experiment_config.seed}")
        logger.info(f"seed = {experiment_config.seed}")
        utils.set_seeds(experiment_config.seed)
        assert tasks != None
        for task in tasks:
            assert task in tasklib.all_task_list

        filename, tmsp = self._get_filename(config=experiment_config)

        # TODO: also include use_contrastive as param to optimize

        training_results = self._initialize_results_dict(epochs=experiment_config.epochs, tasks=tasks)

        for task in tasks:
            pretrained_model = None
            if pretrained_model_path:
                assert os.path.exists(pretrained_model_path)
                print(f"loading pretrained model from {pretrained_model_path}")
                pretrained_model = ViLBERT.load_model(pretrained_model_path, device= "cuda" if torch.cuda.is_available() else "cpu")
                info_str = f"Loaded pretrained model from {pretrained_model_path} for task {task}"
                print(info_str)
                logger.info(info_str)
                assert pretrained_model.config.text_cross_attention_layers == experiment_config.t_biattention_ids
                assert pretrained_model.config.vision_cross_attention_layers == experiment_config.v_biattention_ids

            training_results = self._run_task(
                task_name=task, experiment_config=experiment_config,
                filename=filename, run_visualizations=run_visualizations,
                training_results=training_results,
                run_alignment_analysis=run_alignment_analysis, tmsp=tmsp,
                pretrained_model=pretrained_model
            )

        self.save_results(
            training_results=training_results,
            config=experiment_config,
            filename=filename,
        )
        return training_results

    def _initialize_results_dict(self, epochs, tasks=None ):
        if tasks is None:
            tasks = ["hateful_memes", "mm_imdb", "pretraining"]
        training_results = {}

        for task in tasks:
            training_results[task] = {
                "alignment": {0: {}},  # Initial state
                "training": {}
            }

            # Initialize epochs
            for epoch in range(1, epochs + 1):
                training_results[task]["alignment"][epoch] = {}
                training_results[task]["training"][epoch] = {}

        return training_results




    def train_model_step(
        self,
        trainer: HatefulMemesTrainer,
        train_dataloader: datasets.DataLoader,
    ):
        return trainer.train_epoch(
            dataloader=train_dataloader
        )

    def evaluate_model(
        self,
        trainer: HatefulMemesTrainer,
        dataloader: datasets.DataLoader,
        ) -> Tuple[float, float]:

        return trainer.evaluate(dataloader=dataloader)



    def _analyse_alignment(
        self,
        model: ViLBERT,
        dataloader: datasets.DataLoader,
        device:str = "cuda" if torch.cuda.is_available() else "cpu",
        knn_k=KNN_K,
        verbose=True,
        #TODO: implemet mmimdb loader
    ):
        measures_per_layer = analysis.analyse_alignment(dataloader=dataloader, model=model,
            device=device, knn_k=knn_k, verbose=verbose,)
        return measures_per_layer

    def create_model(self, config: ViLBERTConfig) -> ViLBERT:
        model = ViLBERT(config=config)
        return model

    def create_config(self, experiment_config: ExperimentConfig):
        config = ViLBERTConfig()
        config.text_cross_attention_layers = experiment_config.t_biattention_ids
        config.vision_cross_attention_layers = experiment_config.v_biattention_ids

        config.epochs = experiment_config.epochs
        config.batch_size = experiment_config.batch_size
        config.gradient_accumulation = experiment_config.gradient_accumulation
        config.learning_rate = experiment_config.learning_rate
        config.seed = experiment_config.seed
        config.train_test_ratio = experiment_config.train_test_ratio
        config.dropout_prob = experiment_config.dropout
        config.use_contrastive_loss = experiment_config.use_contrastive_loss


        return config

    def train_from_config(self, config_pth:str, task:str):
        assert os.path.exists(config_pth)
        assert task in ["hateful_memes", "mm_imdb"]

        with open(config_pth, "r") as f:
            content = json.load(f)

        t_biattention_ids = content["t_biattention_ids"]
        v_biattention_ids = content["v_biattention_ids"]

        epochs = content["epochs"]
        batch_size = content["batch_size"]
        gradient_accumulation = content["gradient_accumulation"]
        learning_rate = content["learning_rate"]
        seed = content["seed"]
        train_test_ratio = content["train_test_ratio"]
        use_contrastive_loss = content.get("use_contrastive_loss", False)
        dropout = content.get("dropout", 0.1)

        exp_config = ExperimentConfig(
            t_biattention_ids=t_biattention_ids,
            v_biattention_ids=v_biattention_ids,
            epochs=epochs,
            batch_size=batch_size,
            gradient_accumulation=gradient_accumulation,
            learning_rate=learning_rate,
            seed=seed,
            train_test_ratio=train_test_ratio,
            use_contrastive_loss=use_contrastive_loss,
            dropout=dropout
        )

        config: ViLBERTConfig = self.create_config(exp_config)
        training_results = self.run_finetune(
            experiment_config=exp_config,
            run_visualizations=True,
            tasks=[task]
        )

        info_str = f"Finished training from config: {config_pth}"
        logger.info(info_str)
        print(info_str)




    def save_results(self, training_results: dict, config: ExperimentConfig, filename:str):
        def convert_to_native(obj):
            if isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        training_results["config"] = {
            "t_biattention_ids": config.t_biattention_ids,
            "v_biattention_ids": config.v_biattention_ids,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "gradient_accumulation": config.gradient_accumulation,
            "learning_rate": config.learning_rate,
            "seed": config.seed,
            "train_test_ratio": config.train_test_ratio,
            "use_contrastive_loss": config.use_contrastive_loss,
            "dropout": config.dropout,
        }
        filename += ".json"
        filename = os.path.join(self.save_dir, filename)
        with open(filename, "w") as f:
            json.dump(convert_to_native(training_results), f, indent=4)

    def _run_pretrain(
        self,
        config: ViLBERTConfig,
        train_data, val_data,
        run_visualizations:bool=False,
        run_alignment_analysis:bool=False,
    ) -> Tuple[dict, str, str]:

        task_string = ""
        tasks_vals = [task.value for task in config.pretraining_tasks]
        tasks_vals.sort()
        for val in tasks_vals:
            task_string += f"{val}"
        tmsp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"res/checkpoints/{tmsp}_pretrained_{task_string}.pt"

        training_results = self._initialize_results_dict(epochs=config.epochs)
        train_loader_ap, val_loader_ap, \
        train_loader_mlm, val_loader_mlm, \
        train_loader_mim, val_loader_mim \
            =  datasets.get_dataloaders_pretrain(
            train_data=train_data,
            val_data=val_data,
            num_workers=NUM_WORKERS,
            prefetch=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            pin_memory=PIN_MEMORY,
            batch_size=BATCH_SIZE_PRETRAIN,
            seed=config.seed,
        )

        model = self.create_model(config=config)
        if FREEZE_UNIMODAL_ENCODERS:
            utils.freeze_all_layers(model.bert)
            utils.freeze_all_layers(model.vit)
        utils.params_summary(model=model)
        trainer = PretrainingTrainer(
            model=model,
            config=config,
            use_contrastive_loss=config.use_contrastive_loss,
            tasks=config.pretraining_tasks,
            gradient_accumulation=config.gradient_accumulation,
        )
        trainer.setup_scheduler(
            epochs=config.epochs,
            total_training_steps=( len(train_loader_ap) + len(train_loader_mlm) + len(train_loader_mim)),

        )
        hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=4,
            pin_memory=False,
            prefetch_factor=4,
            num_samples=ALIGNMENT_ANALYSIS_SIZE,
            seed=config.seed
        )


        # on untrained model
        if run_alignment_analysis:
            alignment_metrics = self._analyse_alignment(model=model, dataloader=cc_dataloader)
            training_results["pretraining"]["alignment"][0] = alignment_metrics
        if run_visualizations:
            filename_extention = f"{tmsp}_e0"
            exp_dir_name = os.path.join(self.visualization_dir, f"{tmsp}_pretrained_{task_string}")
            print(f"Saving visualizations to {exp_dir_name}")
            os.makedirs(exp_dir_name, exist_ok=True)
            analysis.run_alignment_visualization(
                dataloader=cc_dataloader, model=model,
                dir_name=exp_dir_name, filename_extension=filename_extention,
            )


        for epoch in range(config.epochs):
            t_loss_ap, t_loss_mlm, t_loss_mim = trainer.train_epoch(
                dataloader_ap=train_loader_ap,
                dataloader_mlm=train_loader_mlm,
                dataloader_mim=train_loader_mim,
                tasks=config.pretraining_tasks,
            )

            v_loss_ap, acc = trainer.evaluate_ap(val_loader_ap)
            v_loss_mlm = trainer.evaluate_mlm(val_loader_mlm)
            v_loss_mim = trainer.evaluate_mim(val_loader_mim)
            info_str = (
                f"Epoch {epoch+1}/{config.epochs}, "
                f"\n\ttrain loss MLM: {t_loss_mlm:.4f}, "
                f"\n\ttest loss MLM: {v_loss_mlm:.4f}, "
                f"\n\ttrain loss AP: {t_loss_ap:.4f}, "
                f"\n\ttest loss AP: {v_loss_ap:.4f}, "
                f"\n\taccuracy AP: {acc:.4f}"
                f"\n\ttrain loss MIM: {t_loss_mim:.4f}, "
                f"\n\ttest loss MIM: {v_loss_mim:.4f}"
            )
            print(info_str)
            logger.info(info_str)

            training_results["pretraining"]["training"][epoch+1] = {
                "train_loss_ap": t_loss_ap,
                "train_loss_mlm": t_loss_mlm,
                "train_loss_mim": t_loss_mim,
                "val_loss_ap": v_loss_ap,
                "val_acc_ap": acc,
                "val_loss_mlm": v_loss_mlm,
                "val_loss_mim": v_loss_mim,
            }

            if run_alignment_analysis:
                alignment_metrics = self._analyse_alignment(model=model, dataloader=cc_dataloader)
                training_results["pretraining"]["alignment"][epoch+1] = alignment_metrics
            if run_visualizations:
                filename_extention = f"{tmsp}_e{epoch+1}"
                analysis.run_alignment_visualization(
                    dataloader=cc_dataloader, model=model,
                    dir_name=exp_dir_name, filename_extension=filename_extention
                )


        trainer.model.save_model(save_path=filename)

        info_str = f"pretraining completed; model saved: {filename}"
        logger.info(info_str)
        print(info_str)



        return training_results, task_string, tmsp, filename


    def run_pretrain(
        self,
        experiment_config: ExperimentConfig,
        tasks:Optional[list[Task]]=[Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM],
        run_visualizations:bool=False,
        run_alignment_analysis:bool=False,
        tiny_fraction:bool=False,
        num_samples:int=NUM_SAMPLES_CC,
    ) -> dict:
        """
        run pretraining with a given config.


        Args:
            experiment_config: ExperimentConfig, configuration for the experiment
            run_visualizations: bool, whether to run visualizations(CKA, mKNN, ... within model) after each epoch (+initialized only)
            run_alignment_analysis: bool, whether to run alignment analysis after each epoch
            tiny_fraction: bool, whether to use a tiny fraction of the data for quick testing/debugging. Only 800 samples
        """
        utils.set_seeds(experiment_config.seed)


        path = "res/data/conceptual-captions/train.csv"
        val_path = "res/data/conceptual-captions/validation.csv"

        if tiny_fraction:
            data_list = datasets.generate_data_list_pretrain(path=path, max_number=1_000)
            validation_list = datasets.generate_data_list_pretrain(path=val_path)
            data_list = data_list[:800]
            validation_list = validation_list[:800]
        else:
            data_list = datasets.generate_data_list_pretrain(path=path, max_number=num_samples+2)
            validation_list = datasets.generate_data_list_pretrain(path=val_path)
            data_list = data_list[:num_samples]

            assert len(data_list) > 1000


        # train_idx = int(len(data_list) * TRAIN_TEST_RATIO)
        # train_data = data_list[:train_idx]
        # val_data   = data_list[train_idx:]
        train_data = data_list
        val_data   = validation_list[:]


        info_str = (
            f"loaded from {path} and {val_path}, \n"
            f"training_data points: {len(train_data)}, val_data points: {len(val_data)}\n"
            f"seed: {experiment_config.seed}, lr: {experiment_config.learning_rate}, \n"
            f"batch_size: {BATCH_SIZE_PRETRAIN}, simulated batch_size: {BATCH_SIZE_PRETRAIN * GRADIENT_ACCUMULATION}, bs-analysis: {BATCH_SIZE_ANALYSIS}\n"
        )
        logger.info(info_str)
        print(info_str)



        config:ViLBERTConfig = self.create_config(experiment_config=experiment_config)
        #manually add the pretraining specific configs
        config.pretraining_tasks = tasks
        assert config.pretraining_tasks != None
        assert config.learning_rate == experiment_config.learning_rate

        training_results, task_string, tmsp, save_path = self._run_pretrain(config=config, train_data=train_data, val_data=val_data, run_alignment_analysis=run_alignment_analysis, run_visualizations=run_visualizations)


        self.save_results(
            training_results=training_results,
            config=experiment_config,
            filename=f"pretraining_{task_string}_{tmsp}"
            )

        training_results["model_path"] = save_path

        return training_results



    # def run_evaluation(self, pretrained_path, tasks=["hateful_memes", "mm_imdb"]):

    #     model = ViLBERT.load_model(pretrained_path, device="cuda" if torch.cuda.is_available() else "cpu")

    #     for task in tasks:
    #         # evaluate on test sets




    def run_alignment_analysis(
        self,
        model:ViLBERT,
        num_samples:int,
        task:str,
        device:str="cuda" if torch.cuda.is_available() else "cpu",
        knn_k=KNN_K,
        verbose=True
        ):
        assert task in ["hateful_memes", "mm_imdb", "cc"]       #  TODO: add more tasks
        dataloader_hm, dataloader_cc, dataloader_imdb = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=0,  pin_memory=False, prefetch_factor=None,
            num_samples=num_samples,
            seed=model.config.seed
        )


        if task == "hateful_memes":
            dataloader = dataloader_hm
        elif task == "mm_imdb":
            dataloader = dataloader_imdb
        elif task == "cc":
            dataloader = dataloader_cc
        else:
            assert False
        # print(f"len dataloader: {len(dataloader.dataset)}")

        alignment_metrics = self._analyse_alignment(model=model, dataloader=dataloader, device=device, knn_k=knn_k, verbose=verbose)
        # print(alignment_metrics)

        return alignment_metrics

    def run_visualization(self, model:ViLBERT, num_samples:int, task:str,
        device:str="cuda" if torch.cuda.is_available() else "cpu"):
        assert task in ["hateful_memes", "mm_imdb", "cc"]       #  TODO: add more tasks
        dataloader_hm, dataloader_cc, dataloader_imdb = datasets.get_alignment_dataloaders(
            batch_size=BATCH_SIZE_ANALYSIS,
            num_workers=0,  pin_memory=False, prefetch_factor=None,
            num_samples=num_samples,
            seed=model.config.seed
        )
        if task == "hateful_memes":
            dataloader = dataloader_hm
        elif task == "mm_imdb":
            dataloader = dataloader_imdb
        elif task == "cc":
            dataloader = dataloader_cc
        else:
            assert False

        analysis.run_alignment_visualization(
            dataloader=dataloader,
            model=model,
            dir_name=self.visualization_dir,
            filename_extension=f"viz_{task}"
        )



    def evaluate(self, model:ViLBERT, task:str):
        assert task in ["hateful_memes", "mm_imdb"]

        if task == "hateful_memes":
            _, val_loader = datasets.get_hateful_memes_datasets(
                train_test_ratio=TRAIN_TEST_RATIO,
                batch_size=BATCH_SIZE_DOWNSTREAM,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=False,
                seed=model.config.seed
            )
            trainer = HatefulMemesTrainer(
                model=model,
                config=model.config,
                gradient_accumulation=GRADIENT_ACCUMULATION,
                use_contrastive_loss=model.config.use_contrastive_loss,
            )
            loss, acc = trainer.evaluate(val_loader)

        elif task == "mm_imdb":
            _, val_loader = datasets.get_mmimdb_datasets(
                train_test_ratio=TRAIN_TEST_RATIO,
                batch_size=BATCH_SIZE_DOWNSTREAM,
                num_workers=NUM_WORKERS,
                pin_memory=PIN_MEMORY,
                prefetch_factor=PREFETCH,
                persistent_workers=PERSISTENT_WORKERS,
                use_train_augmentation=False,
                seed=model.config.seed,
            )
            trainer = MM_IMDB_Trainer(
                model=model,
                config=model.config,
                gradient_accumulation=GRADIENT_ACCUMULATION,
                use_contrastive_loss=model.config.use_contrastive_loss,
            )
            loss, acc = trainer.evaluate(val_loader)

        return loss, acc









def main():

    tracker = ExperimentTracker()
    tracker.optimize_parameters_multi(n_trials=100, optimization_objective="acc")
    # tracker.optimize_parameters_single(n_trials=100, optimization_objective="loss",
    #                                    #task="mm_imdb")
    #                                    task="hateful_memes")
    # best_coattn = tracker.optimize_coattn_for_accuracy(depth=5, n_trials=30)



if __name__ == "__main__":
    main()
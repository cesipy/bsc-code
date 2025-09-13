import time
import os
from typing import Tuple, Optional
import dataclasses
import json


import optuna; from optuna import pruners


from config import *
from trainer import Trainer
from mm_imdb_trainer import MM_IMDB_Trainer
from vilbert import ViLBERT
import utils
import datasets
from logger import Logger
import analysis

logger = Logger()
EPOCHS = 7
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
        self.visualization_dir = "res/experiments/visualizations/"
        os.makedirs(self.save_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True)


    def _run_trial(self, config: ExperimentConfig, trial):
        vilbert_config = self.create_config(config)
        utils.set_seeds(config.seed)
        # TODO: maybe include, but this makes it a lot slower for alignment
        training_results = self._initialize_results_dict(epochs=config.epochs)

        best_hm_acc = 0.0
        best_imdb_acc = 0.0

        #---------------------------------------------
        # hm training (with pruning)

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
        # TODO: maybe include afterwards
        # hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
        #     batch_size=BATCH_SIZE_ANALYSIS,
        #     num_workers=NUM_WORKERS,
        #     pin_memory=PIN_MEMORY,
        #     prefetch_factor=PREFETCH,
        #     num_samples=ALIGNMENT_ANALYSIS_SIZE
        # )

        model = self.create_model(vilbert_config)
        trainer = Trainer(
            model=model,
            config=vilbert_config,
            gradient_accumulation=GRADIENT_ACCUMULATION
        )
        trainer.setup_scheduler(epochs=config.epochs, train_dataloader=train_loader,
                                lr=vilbert_config.learning_rate)

        for i in range(config.epochs):
            train_loss = self.train_model_step(
                trainer=trainer,
                train_dataloader=train_loader,
            )
            test_loss, acc = self.evaluate_model(
                trainer=trainer,
                dataloader=val_loader,
            )
            info_str = (
                f"HM Epoch {i+1}/{config.epochs}, Train Loss: {train_loss:.4f}"
                f", Val Loss: {test_loss:.4f}, Val Acc: {acc:.4f}"
            )
            # print(info_str)
            # logger.info(info_str)

            intermediate_value = acc
            trial.report(intermediate_value, step=i)
            if trial.should_prune():
                raise optuna.TrialPruned()
            best_hm_acc = max(best_hm_acc, acc)

        del model, trainer, train_loader, val_loader

        #---------------------------------------------
        # imdb training (with pruning)
        model = self.create_model(vilbert_config)
        trainer = MM_IMDB_Trainer(
            model=model,
            config=vilbert_config,
            gradient_accumulation=GRADIENT_ACCUMULATION,
        )

        train_loader, val_loader = datasets.get_mmimdb_datasets(
            train_test_ratio=TRAIN_TEST_RATIO,
            # train_test_ratio=0.1,
            batch_size=BATCH_SIZE_DOWNSTREAM,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            prefetch_factor=PREFETCH,
            persistent_workers=PERSISTENT_WORKERS,
            use_train_augmentation=True,
        )
        trainer.setup_scheduler(epochs=config.epochs, train_dataloader=train_loader,
                                lr=vilbert_config.learning_rate)

        for i in range(config.epochs):
            train_loss = self.train_model_step(
                trainer=trainer,
                train_dataloader=train_loader,
            )
            test_loss, acc = self.evaluate_model(
                trainer=trainer,
                dataloader=val_loader,
            )
            info_str = (
                f"IMDB Epoch {i+1}/{config.epochs}, Train Loss: {train_loss:.4f}"
                f", Val Loss: {test_loss:.4f}, Val Acc: {acc:.4f}"
            )

            # use different step numbers to avoid overwriting hm steps
            intermediate_value = (best_hm_acc + acc) / 2  # average both tasks for pruning
            trial.report(intermediate_value, step=config.epochs + i)
            if trial.should_prune():
                raise optuna.TrialPruned()
            best_imdb_acc = max(best_imdb_acc, acc)

        return best_hm_acc, best_imdb_acc





    def optimize_coattn_for_accuracy(self, depth, n_trials):
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

            return self._run_trial(config, trial)


            # training_results = self.run_single_experiment(
            #     config,
            #     run_visualization=False                 # is too compute intensive, not wanted here
            # )

            # val_accs_hm = [training_results["hateful_memes"]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]
            # val_accs_imdb = [training_results["mm_imdb"]["training"][i]["val_acc"] for i in range(1, config.epochs+1)]

            # max_acc_hm = max(val_accs_hm)
            # max_acc_imdb = max(val_accs_imdb)

            # return max_acc_hm, max_acc_imdb



        storage_path = f"sqlite:///{self.save_dir}coattn_optim.db"
        study_name = "cross_attention_study"

        pruner = pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

        study = optuna.create_study(
            directions=["maximize","maximize"],
            pruner=pruner,
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



    def run_single_experiment_hateful_memes(
        self,
        config: ViLBERTConfig,
        training_results: dict,
        epochs: int,
        run_visualization: bool,
        dir_name: Optional[str] = None
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
            if run_visualization:
                analysis.visualize_cka(dataloader=hm_dataloader, model=model, dir_name=dir_name)
        return training_results

    def run_single_experiment_mm_imdb(
        self,
        config: ViLBERTConfig,
        training_results: dict,
        epochs: int,
        run_visualization: bool,
        dir_name: Optional[str] = None,
    ):
        model = self.create_model(config)
        trainer = MM_IMDB_Trainer(
            model=model,
            config=config,
            gradient_accumulation=GRADIENT_ACCUMULATION,
        )

        train_loader, val_loader = datasets.get_mmimdb_datasets(
            train_test_ratio=TRAIN_TEST_RATIO,
            # train_test_ratio=0.1,
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

            if run_visualization:
                analysis.visualize_cka(dataloader=imdb_dataloader, model=model, dir_name=dir_name)

        return training_results

    def _get_filename(self, config: ExperimentConfig):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        # coattn_pix: str =""
        if config.cross_attention_layers == []:
            coattn_fix = "no_coattn"
        else:
            coattn_fix = "coattn_"
            for i in config.cross_attention_layers:
                coattn_fix += str(i)
                coattn_fix += "-"
            coattn_fix = coattn_fix[:-1]

        filename = f"experiment_{coattn_fix}_{timestamp}"

        return filename




    def run_single_experiment(self, experiment_config: ExperimentConfig, run_visualization:bool=True):
        filename = self._get_filename(config=experiment_config)

        print(f"Saving results to {filename}")
        exp_dir_name = os.path.join(self.visualization_dir, filename)
        os.makedirs(exp_dir_name, exist_ok=True)
        exp_dir_name_hm = os.path.join(exp_dir_name, "hateful_memes")
        exp_dir_name_imdb = os.path.join(exp_dir_name, "mm_imdb")
        os.makedirs(exp_dir_name_hm, exist_ok=True)
        os.makedirs(exp_dir_name_imdb, exist_ok=True)


        config = self.create_config(experiment_config)
        utils.set_seeds(experiment_config.seed)

        training_results = self._initialize_results_dict(epochs=experiment_config.epochs)

        training_results = self.run_single_experiment_mm_imdb(
            config=config,
            training_results=training_results,
            epochs=experiment_config.epochs,
            dir_name=exp_dir_name_imdb,
            run_visualization=run_visualization,
        )
        training_results = self.run_single_experiment_hateful_memes(
            config=config,
            training_results=training_results,
            epochs=experiment_config.epochs,
            dir_name=exp_dir_name_hm,
            run_visualization=run_visualization,
        )
        self.save_results(
            training_results=training_results,
            config=experiment_config,
            filename=filename,

        )
        return training_results

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


    def save_results(self, training_results: dict, config: ExperimentConfig, filename:str):


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
        filename += ".json"
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
    tracker = ExperimentTracker()
    best_coattn = tracker.optimize_coattn_for_accuracy(depth=5, n_trials=30)

    # exps = get_experiments()
    # tracker = ExperimentTracker()

    # for config in exps:
    #     print(f"Running experiment with config: {config}")
    #     logger.info(f"Running experiment with config: {config}")
    #     tracker.run_single_experiment(config)

    #     info_str = "-"*25
    #     print(info_str+"\n")
    #     logger.info(info_str)


if __name__ == "__main__":
    main()

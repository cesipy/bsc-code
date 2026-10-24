"""Validation-only training must never construct or evaluate a test cohort."""
from types import SimpleNamespace

import experiment_tracker
import main as main_module
from config import Task, ViLBERTConfig


class _Dataset:
    def __len__(self):
        return 3


class _Loader:
    def __init__(self, name):
        self.name = name
        self.dataset = _Dataset()

    def __len__(self):
        return 1


def _forbid_test_access(*_args, **_kwargs):
    raise AssertionError("validation-only execution accessed a test cohort")


def test_validation_only_cli_propagates_mode_and_skips_posthoc_test_metrics(monkeypatch):
    captured = {}

    class FakeTracker:
        def run_pretrain(self, experiment_config, **kwargs):
            captured["pretrain"] = (experiment_config, kwargs)
            return {"model_path": "pretrained.pt"}

        def run_finetune(self, experiment_config, **kwargs):
            captured["finetune"] = (experiment_config, kwargs)
            return {kwargs["tasks"][0]: {"model_path": "finetuned.pt"}}

    monkeypatch.setattr(main_module.experiment_tracker, "ExperimentTracker", FakeTracker)
    monkeypatch.setattr(
        main_module.ViLBERT,
        "load_model",
        lambda _path: SimpleNamespace(
            config=SimpleNamespace(
                text_cross_attention_layers=[],
                vision_cross_attention_layers=[],
            )
        ),
    )
    monkeypatch.setattr(main_module.logger, "info", lambda _message: None)
    monkeypatch.setattr(
        main_module.performance_metric_collection,
        "collect_final_metrics",
        _forbid_test_access,
    )

    main_module.main(["--fusion", "baseline", "--validation-only"])

    assert captured["pretrain"][1]["validation_only"] is True
    _, finetune_options = captured["finetune"]
    assert finetune_options["validation_only"] is True
    assert finetune_options["run_alignment_analysis"] is False


def test_validation_evaluate_uses_explicit_loader_without_test_or_dataset_construction(monkeypatch):
    tracker = object.__new__(experiment_tracker.ExperimentTracker)
    validation_loader = _Loader("selected-validation")
    evaluated = []

    class FakeTrainer:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, dataloader):
            evaluated.append(dataloader)
            return 0.25, 0.75, 0.625

    monkeypatch.setattr(
        experiment_tracker.datasets,
        "get_task_test_dataset",
        _forbid_test_access,
    )
    monkeypatch.setattr(
        experiment_tracker.datasets,
        "get_hateful_memes_datasets",
        _forbid_test_access,
    )
    monkeypatch.setattr(experiment_tracker, "HatefulMemesTrainer", FakeTrainer)
    config = SimpleNamespace(
        batch_size=2,
        num_workers=0,
        pin_memory=False,
        prefetch=None,
        persistent_workers=False,
        seed=19,
        train_test_ratio=0.8,
        gradient_accumulation=1,
        use_contrastive_loss=False,
    )
    model = SimpleNamespace(config=config)

    metrics = tracker.evaluate(
        model=model,
        task="hateful_memes",
        dataset="val",
        validation_dataloader=validation_loader,
    )

    assert evaluated == [validation_loader]
    assert metrics == {"loss": 0.25, "acc": 0.75, "auc": 0.625}


def test_validation_finetuning_uses_same_validation_cohort_for_initial_epoch_and_restored_final(
    monkeypatch,
):
    tracker = object.__new__(experiment_tracker.ExperimentTracker)
    train_loader = _Loader("train")
    validation_loader = _Loader("validation")
    initial_and_final = []
    epoch_loaders = []

    class FakeModel:
        def __init__(self):
            self.config = None
            self.restored = False

        def state_dict(self):
            return {"weight": 1}

        def load_state_dict(self, state):
            assert state == {"weight": 1}
            self.restored = True

    class FakeTrainTrainer:
        def __init__(self, model):
            self.model = model

        def setup_scheduler(self, **_kwargs):
            pass

        def evaluate(self, dataloader):
            epoch_loaders.append(dataloader)
            return 0.4, 0.7, 0.65

    class FakeEvalTrainer:
        def __init__(self, **_kwargs):
            pass

        def evaluate(self, dataloader):
            initial_and_final.append(dataloader)
            return 0.35, 0.72, 0.68

    class FakeEarlyStopping:
        def __init__(self, **_kwargs):
            self.best_score = None
            self.best_epoch = None

        def __call__(self, value, epoch):
            if self.best_score is None or value > self.best_score:
                self.best_score = value
                self.best_epoch = epoch
            return False

    model = FakeModel()
    tracker.create_model = lambda _config: model
    tracker.get_task_trainer = lambda **_kwargs: FakeTrainTrainer(model)
    tracker.get_task_dataloader = lambda **_kwargs: (train_loader, validation_loader)
    tracker.get_task_alignment_dataloader = _forbid_test_access
    tracker.train_model_step = lambda **_kwargs: 1.25
    tracker._analyse_alignment = _forbid_test_access
    monkeypatch.setattr(experiment_tracker, "USE_EARLY_STOPPING", True)
    monkeypatch.setattr(experiment_tracker.utils, "EarlyStopping", FakeEarlyStopping)
    monkeypatch.setattr(experiment_tracker, "HatefulMemesTrainer", FakeEvalTrainer)
    monkeypatch.setattr(experiment_tracker.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(
        experiment_tracker.datasets,
        "get_task_test_dataset",
        _forbid_test_access,
    )
    monkeypatch.setattr(
        experiment_tracker.analysis,
        "run_alignment_visualization",
        _forbid_test_access,
    )

    config = ViLBERTConfig(epochs=2, testing=True)
    model.config = config
    result = tracker.run_single_experiment(
        task="hateful_memes",
        config=config,
        training_results={
            "hateful_memes": {
                "alignment": {0: {}, 1: {}, 2: {}},
                "training": {1: {}, 2: {}},
            }
        },
        epochs=2,
        run_visualization=True,
        skip_alignment_analysis=False,
        validation_only=True,
    )

    assert initial_and_final == [validation_loader, validation_loader]
    assert epoch_loaders == [validation_loader, validation_loader]
    assert model.restored is True
    assert "final_validation" in result["hateful_memes"]
    assert "final_test" not in result["hateful_memes"]


def test_validation_pretraining_skips_shared_test_alignment_loaders(monkeypatch, tmp_path):
    tracker = object.__new__(experiment_tracker.ExperimentTracker)
    tracker.save_dir = str(tmp_path)
    tracker.visualization_dir = str(tmp_path / "visualizations")
    tracker.save_results = lambda **_kwargs: None
    monkeypatch.setattr(experiment_tracker, "PRETRAIN_CHECKPOINTS_DIR", str(tmp_path / "checkpoints"))

    class FakeModel:
        def save_model(self, save_path):
            self.saved_path = save_path

    class FakePretrainingTrainer:
        def __init__(self, model, **_kwargs):
            self.model = model

        def setup_scheduler(self, **_kwargs):
            pass

        def train_epoch(self, **_kwargs):
            return 1.0, 2.0, 3.0

        def evaluate_ap(self, _loader):
            return 0.4, 0.7

        def evaluate_mlm(self, _loader):
            return 0.5

        def evaluate_mim(self, _loader):
            return 0.6

    loaders = [_Loader(name) for name in ("train-ap", "val-ap", "train-mlm", "val-mlm", "train-mim", "val-mim")]
    model = FakeModel()
    monkeypatch.setattr(tracker, "create_model", lambda **_kwargs: model)
    monkeypatch.setattr(experiment_tracker, "PretrainingTrainer", FakePretrainingTrainer)
    monkeypatch.setattr(experiment_tracker.utils, "params_summary", lambda **_kwargs: None)
    monkeypatch.setattr(experiment_tracker.mlflow, "active_run", lambda: None)
    monkeypatch.setattr(
        experiment_tracker.datasets,
        "generate_data_list_pretrain",
        lambda path, max_number=None: [path] * (1200 if "train.csv" in path else 5),
    )
    monkeypatch.setattr(
        experiment_tracker.datasets,
        "get_dataloaders_pretrain",
        lambda **_kwargs: tuple(loaders),
    )
    monkeypatch.setattr(
        experiment_tracker.datasets,
        "get_alignment_dataloaders",
        _forbid_test_access,
    )
    monkeypatch.setattr(
        experiment_tracker.analysis,
        "run_alignment_visualization",
        _forbid_test_access,
    )

    config = ViLBERTConfig(epochs=1, pretrain_batch_size=2, testing=True)
    results = tracker.run_pretrain(
        experiment_config=config,
        tasks=[Task.MASKED_LM],
        run_visualizations=True,
        run_alignment_analysis=True,
        num_samples=1001,
        validation_only=True,
    )

    assert results["evaluation_split"] == "validation"
    assert results["validation_cohort"]["pretraining"]["source"] == (
        "Conceptual Captions validation.csv"
    )
    assert model.saved_path == results["model_path"]


def test_validation_finetune_records_deterministic_trainval_holdouts_and_propagates_mode(
    monkeypatch,
):
    tracker = object.__new__(experiment_tracker.ExperimentTracker)
    tracker.save_dir = "unused"
    calls = []
    monkeypatch.setattr(experiment_tracker.utils, "set_seeds", lambda _seed: None)
    monkeypatch.setattr(tracker, "_get_filename", lambda **_kwargs: ("run", "stamp"))
    monkeypatch.setattr(tracker, "save_results", lambda **_kwargs: None)

    def fake_run_task(**kwargs):
        calls.append(kwargs)
        return kwargs["training_results"]

    monkeypatch.setattr(tracker, "_run_task", fake_run_task)
    config = ViLBERTConfig(epochs=1, testing=True)

    results = tracker.run_finetune(
        experiment_config=config,
        tasks=["hateful_memes", "mm_imdb", "upmc_food"],
        validation_only=True,
    )

    assert all(call["validation_only"] is True for call in calls)
    assert results["evaluation_split"] == "validation"
    cohorts = results["validation_cohort"]
    assert cohorts["hateful_memes"]["source"].endswith("validation.jsonl")
    assert cohorts["mm_imdb"]["kind"] == "deterministic_validation_holdout_from_trainval"
    assert cohorts["mm_imdb"]["row_shuffle_seed"] == 42
    assert cohorts["upmc_food"]["row_shuffle_seed"] == 46
    assert all("final_test" not in results[task] for task in cohorts)

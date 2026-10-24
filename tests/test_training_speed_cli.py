"""CLI and loader-routing contracts for opt-in training speed controls."""
from types import SimpleNamespace

import pytest

import experiment_tracker
import main as main_module
from config import (
    ALIGNMENT_ANALYSIS_SIZE,
    NUM_WORKERS,
    PIN_MEMORY,
    PERSISTENT_WORKERS,
    PREFETCH,
    ViLBERTConfig,
)


def _mock_main_flow(monkeypatch, argv, loaded_checkpoint_backend=None):
    captured = {"pretrain": None, "finetune": None}

    class FakeTracker:
        def run_pretrain(self, experiment_config, **kwargs):
            captured["pretrain"] = (experiment_config, kwargs)
            return {"model_path": "mock-pretrained.pt"}

        def run_finetune(self, experiment_config, **kwargs):
            captured["finetune"] = (experiment_config, kwargs)
            return {
                kwargs["tasks"][0]: {"model_path": "mock-finetuned.pt"}
            }

    monkeypatch.setattr(main_module.experiment_tracker, "ExperimentTracker", FakeTracker)

    def load_model(_path):
        if captured["pretrain"] is None:
            # Mimic an old checkpoint config that predates the backend field.
            config = SimpleNamespace(
                text_cross_attention_layers=[],
                vision_cross_attention_layers=[],
            )
            if loaded_checkpoint_backend is not None:
                config.cross_attention_backend = loaded_checkpoint_backend
        else:
            pretrain_config = captured["pretrain"][0]
            config = SimpleNamespace(
                text_cross_attention_layers=pretrain_config.text_cross_attention_layers,
                vision_cross_attention_layers=pretrain_config.vision_cross_attention_layers,
                cross_attention_backend=pretrain_config.cross_attention_backend,
            )
        return SimpleNamespace(config=config)

    monkeypatch.setattr(main_module.ViLBERT, "load_model", load_model)
    monkeypatch.setattr(
        main_module.performance_metric_collection,
        "collect_final_metrics",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(main_module.logger, "info", lambda _message: None)

    main_module.main(argv)
    return captured


def test_cli_defaults_keep_existing_training_settings():
    args = main_module.parse_args(["--fusion", "baseline"])

    assert args.cross_attention_backend is None
    assert args.mlm_selected_tokens_only is False
    assert args.num_workers is None
    assert args.persistent_workers is False
    assert args.skip_ft_alignment is False
    assert args.alignment_analysis_size is None


def test_alignment_size_help_documents_both_default_values(capsys):
    with pytest.raises(SystemExit):
        main_module.parse_args(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "128 for pretraining" in help_text
    assert f"{ALIGNMENT_ANALYSIS_SIZE} for finetuning" in help_text


@pytest.mark.parametrize(
    "extra_args",
    [
        ["--persistent-workers"],
        ["--num-workers", "0", "--persistent-workers"],
        ["--num-workers", "-1"],
        ["--cross-attention-backend", "flash"],
        ["--no-pretrain", "--cross-attention-backend", "sdpa"],
        ["--pretrain-path", "existing.pt", "--cross-attention-backend", "eager"],
    ],
)
def test_cli_rejects_ambiguous_worker_settings_and_checkpoint_backend_overrides(extra_args):
    with pytest.raises(SystemExit):
        main_module.parse_args(["--fusion", "baseline", *extra_args])


def test_mocked_main_preserves_pretrain_and_finetune_defaults(monkeypatch):
    captured = _mock_main_flow(monkeypatch, ["--fusion", "baseline"])
    pretrain_config, pretrain_options = captured["pretrain"]
    finetune_config, finetune_options = captured["finetune"]

    assert pretrain_config.cross_attention_backend == "eager"
    assert pretrain_config.mlm_selected_tokens_only is False
    assert (pretrain_config.num_workers, pretrain_config.prefetch) == (1, 2)
    assert pretrain_config.persistent_workers is False
    assert pretrain_config.pin_memory is True
    assert pretrain_options["alignment_analysis_size"] == 128

    assert finetune_config.cross_attention_backend == "eager"
    assert finetune_config.num_workers == NUM_WORKERS == 0
    assert finetune_config.prefetch == PREFETCH
    assert finetune_config.persistent_workers == PERSISTENT_WORKERS
    assert finetune_config.pin_memory == PIN_MEMORY
    assert finetune_options["run_alignment_analysis"] is True
    assert finetune_options["alignment_analysis_size"] == ALIGNMENT_ANALYSIS_SIZE


def test_loaded_checkpoint_keeps_its_backend_and_old_config_defaults_to_eager(monkeypatch):
    captured = _mock_main_flow(
        monkeypatch,
        ["--fusion", "baseline", "--pretrain-path", "existing.pt"],
        loaded_checkpoint_backend="sdpa",
    )
    assert captured["pretrain"] is None
    assert captured["finetune"][0].cross_attention_backend == "sdpa"

    legacy_captured = _mock_main_flow(
        monkeypatch,
        ["--fusion", "baseline", "--pretrain-path", "old-checkpoint.pt"],
    )
    assert legacy_captured["pretrain"] is None
    assert legacy_captured["finetune"][0].cross_attention_backend == "eager"


def test_mocked_main_routes_explicit_speed_controls_to_both_stages(monkeypatch):
    captured = _mock_main_flow(
        monkeypatch,
        [
            "--fusion", "baseline",
            "--cross-attention-backend", "sdpa",
            "--mlm-selected-tokens-only",
            "--num-workers", "3",
            "--persistent-workers",
            "--skip-ft-alignment",
            "--alignment-analysis-size", "23",
        ],
    )
    pretrain_config, pretrain_options = captured["pretrain"]
    finetune_config, finetune_options = captured["finetune"]

    assert pretrain_config.cross_attention_backend == "sdpa"
    assert pretrain_config.mlm_selected_tokens_only is True
    assert (pretrain_config.num_workers, pretrain_config.prefetch) == (3, 2)
    assert pretrain_config.persistent_workers is True
    assert pretrain_config.pin_memory is True
    assert pretrain_options["alignment_analysis_size"] == 23

    assert finetune_config.cross_attention_backend == "sdpa"
    assert (finetune_config.num_workers, finetune_config.prefetch) == (3, 2)
    assert finetune_config.persistent_workers is True
    assert finetune_config.pin_memory is True
    assert finetune_options["run_alignment_analysis"] is False
    assert finetune_options["alignment_analysis_size"] == 23


def test_zero_workers_disables_prefetch_and_persistent_workers(monkeypatch):
    captured = _mock_main_flow(
        monkeypatch, ["--fusion", "baseline", "--num-workers", "0"]
    )
    pretrain_config, _ = captured["pretrain"]
    finetune_config, _ = captured["finetune"]

    for config in (pretrain_config, finetune_config):
        assert config.num_workers == 0
        assert config.prefetch is None
        assert config.persistent_workers is False
    assert pretrain_config.pin_memory is True
    assert finetune_config.pin_memory is False


@pytest.mark.parametrize(
    ("skip_alignment_analysis", "run_visualization", "should_load"),
    [(True, False, False), (False, False, True), (True, True, True)],
)
def test_alignment_loader_is_lazy_when_analysis_and_visualization_are_skipped(
    monkeypatch, skip_alignment_analysis, run_visualization, should_load
):
    tracker = object.__new__(experiment_tracker.ExperimentTracker)
    calls = []

    class StopBeforeTraining(Exception):
        pass

    class FakeTrainer:
        def setup_scheduler(self, **_kwargs):
            raise StopBeforeTraining

    tracker.create_model = lambda _config: object()
    tracker.get_task_trainer = lambda **_kwargs: FakeTrainer()
    tracker.get_task_dataloader = lambda **_kwargs: (object(), object())

    def get_alignment_loader(**kwargs):
        calls.append(kwargs)
        return object()

    tracker.get_task_alignment_dataloader = get_alignment_loader
    tracker._analyse_alignment = lambda **_kwargs: {}
    monkeypatch.setattr(
        experiment_tracker.analysis,
        "run_alignment_visualization",
        lambda **_kwargs: None,
    )

    task_results = {"alignment": {}}
    with pytest.raises(StopBeforeTraining):
        tracker.run_single_experiment(
            task="hateful_memes",
            config=ViLBERTConfig(),
            training_results={"hateful_memes": task_results},
            epochs=1,
            run_visualization=run_visualization,
            skip_alignment_analysis=skip_alignment_analysis,
        )

    assert len(calls) == int(should_load)

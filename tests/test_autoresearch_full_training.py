"""CPU-only contract tests for the matched full-training runner."""

from __future__ import annotations

import json
import random
import signal
import csv
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import autoresearch_full_training as full


def test_repository_relative_cc_paths_use_the_separate_read_only_image_root(tmp_path):
    workspace = tmp_path / "autoresearch"
    workspace.mkdir()
    images = tmp_path / "manual" / "images"
    (images / "nested").mkdir(parents=True)
    image = images / "nested" / "sample.jpg"
    image.write_bytes(b"fixture")
    csv_path = workspace / "train.csv"
    csv_path.write_text(
        "text,filepath\ncaption,res/data/conceptual-captions/images/nested/sample.jpg\n"
    )
    pairs, identity = full.load_cc_pairs(
        csv_path, root=workspace, image_root=images, expected_count=1
    )
    assert pairs == [(str(image.resolve()), "caption")]
    assert identity["missing_image_rows_skipped"] == 0
    assert not (workspace / "res").exists()


@pytest.mark.parametrize(
    ("header", "row", "expected_caption", "expected_image"),
    [
        (("text", "filepath"), ("caption A", "one.png"), "caption A", "one.png"),
        (("filepath", "text", "unused"), ("two.png", "caption B", "ignored"), "caption B", "two.png"),
    ],
)
def test_load_cc_pairs_maps_caption_and_image_by_header_name(
    tmp_path, header, row, expected_caption, expected_image
):
    image_root = tmp_path / "images"
    image_root.mkdir()
    image_path = image_root / expected_image
    image_path.write_bytes(b"path-resolution fixture; decoding is not part of this loader test")
    csv_path = tmp_path / "cc.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerow(row)

    pairs, identity = full.load_cc_pairs(
        csv_path,
        root=tmp_path,
        image_root=image_root,
        expected_count=1,
    )

    assert pairs == [(str(image_path.resolve()), expected_caption)]
    assert identity["csv_header"] == list(header)
    assert identity["usable_pair_count"] == 1


def test_load_cc_pairs_rejects_unrecognized_column_names(tmp_path):
    csv_path = tmp_path / "cc.csv"
    csv_path.write_text("caption,image\ntext,one.png\n", encoding="utf-8")
    image_root = tmp_path / "images"
    image_root.mkdir()

    with pytest.raises(full.FullTrainingError, match="named 'text' and 'filepath'"):
        full.load_cc_pairs(
            csv_path,
            root=tmp_path,
            image_root=image_root,
            expected_count=1,
        )


@pytest.mark.parametrize("argument", ["cuda", "cuda:0"])
def test_cuda_cli_forms_resolve_to_explicit_visible_device_zero(argument):
    device = torch.device(full._logical_cuda_device(argument))
    assert device.type == "cuda"
    assert device.index == 0


@pytest.mark.parametrize("argument", ["cpu", "cuda:1", "cuda:-1", "mps"])
def test_cuda_device_selection_cannot_escape_visible_device_zero(argument):
    with pytest.raises(full.FullTrainingError, match="CUDA_VISIBLE_DEVICES"):
        full._logical_cuda_device(argument)


class _ToyConfig(SimpleNamespace):
    pass


class _ToyTask:
    ALIGNMENT_PREDICTION = "ap"
    MASKED_LM = "mlm"
    MASKED_IM = "mim"


class _ToyModel(torch.nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.25, -0.5], dtype=torch.float32))
        self.config = config or SimpleNamespace(toy=True)


class _ToyScheduler:
    def __init__(self):
        self.iteration = 0
        self.total_steps = 4
        self.base_lr = 0.025

    def get_lr(self):
        self.iteration += 1
        return self.base_lr / self.iteration


class _ToyDataset(torch.utils.data.Dataset):
    def __init__(self, split: str, task: str, size: int):
        self.split = split
        self.task = task
        self.size = size
        self.epoch = 0

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        value = torch.tensor([float(index + 1), float(self.epoch + 1)], dtype=torch.float32)
        batch = {"value": value}
        if self.task == "mlm":
            # Keep at least one supervised token in each sample for this fixture.
            batch["label"] = torch.tensor([index % 3, -100], dtype=torch.long)
        return batch


class _ToyTrainer:
    def __init__(self, model, *, on_first_mim=None):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=0.025)
        self.scaler = torch.amp.GradScaler("cpu", enabled=False)
        self.scheduler = _ToyScheduler()
        self.on_first_mim = on_first_mim
        self._mim_calls = 0
        self.validation_calls = []

    def _train(self, batch, task_scale):
        random_factor = (
            torch.rand((), dtype=torch.float32)
            + random.random() * 0.1
            + float(np.random.random()) * 0.1
        )
        loss = (self.model.weight * batch["value"]).sum() * random_factor * task_scale
        loss.backward()
        return float(loss.detach())

    def train_alignment_prediction_batch(self, batch, *, flag_optimizer):
        assert flag_optimizer is False
        return self._train(batch, 0.5)

    def train_mlm_batch(self, batch, *, flag_optimizer):
        assert flag_optimizer is False
        return self._train(batch, 0.25)

    def train_mim_batch(self, batch, *, flag_optimizer):
        assert flag_optimizer is False
        self._mim_calls += 1
        if self.on_first_mim is not None:
            self.on_first_mim()
        return self._train(batch, 0.75)

    def _evaluate(self, loader, name, result):
        self.validation_calls.append((name, loader.dataset.task, len(loader.dataset)))
        seen = 0
        for batch in loader:
            seen += int(batch["value"].shape[0])
        assert seen == len(loader.dataset)
        return result

    def evaluate_ap(self, loader):
        return self._evaluate(loader, "ap", (0.7, 0.6))

    def evaluate_mlm(self, loader):
        return self._evaluate(loader, "mlm", 0.8)

    def evaluate_mim(self, loader):
        return self._evaluate(loader, "mim", 0.9)


def _injected_run(
    *, output_root: Path, run_id: str, interrupt_state=None, train_size: int = 8,
    interrupt_after_mim_call: int | None = None,
):
    signal_callback = None
    if interrupt_state is not None and interrupt_after_mim_call is not None:
        def signal_callback():
            if trainer._mim_calls == interrupt_after_mim_call:
                interrupt_state["requested"] = signal.SIGTERM
    elif interrupt_state is not None:
        signal_callback = lambda: interrupt_state.__setitem__("requested", signal.SIGTERM)
    model = _ToyModel()
    trainer = _ToyTrainer(model, on_first_mim=signal_callback)
    datasets_train = {task: _ToyDataset("train", task, train_size) for task in full.TASK_KEYS}
    datasets_val = {task: _ToyDataset("validation", task, 4) for task in full.TASK_KEYS}
    injected = {
        "modules": {"torch": torch, "np": np},
        "source_hashes": {"toy_source": "fixed"},
        "inputs": {"toy": "cpu-only"},
        "anchor": {
            "initial_model_state_sha256_by_arm": {"df2": "toy-df2", "rev": "toy-rev"}
        },
        "datasets_train": datasets_train,
        "datasets_val": datasets_val,
        "data_identity": {"ordered_rows": [0, 1, 2, 3, 4, 5, 6, 7]},
        "model": model,
        "trainer": trainer,
    }
    args = SimpleNamespace(
        arm="df2",
        seed=full.EXPECTED_SEED,
        workers=0,
        device="cpu",
        run_id=run_id,
        output_root=str(output_root),
        resume=False,
        initialization_anchor="unused-by-injected-path",
    )
    return args, injected, trainer


@pytest.fixture
def tiny_training_contract(monkeypatch, tmp_path):
    research_dir = tmp_path / "autoresearch"
    research_dir.mkdir()
    monkeypatch.setattr(full, "AUTORESEARCH", research_dir)
    monkeypatch.setattr(full, "EPOCHS", 1)
    monkeypatch.setattr(full, "EXPECTED_TRAIN_ROWS", 8)
    monkeypatch.setattr(full, "EXPECTED_VALIDATION_ROWS", 4)
    monkeypatch.setattr(full, "BATCH_SIZE", 2)
    monkeypatch.setattr(full, "GRADIENT_ACCUMULATION", 2)
    monkeypatch.setattr(full, "CHECKPOINT_EVERY_UPDATES", 1)
    monkeypatch.setattr(full, "MAX_WORKERS", 0)
    return research_dir / "runs"


def _install_signal_state(monkeypatch, requested=None):
    state = {"requested": requested}
    monkeypatch.setattr(full, "_install_signal_handlers", lambda: ({}, state))
    monkeypatch.setattr(full, "_restore_signal_handlers", lambda _previous: None)
    return state


def _checkpoint(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def test_interrupted_resume_matches_uninterrupted_and_reconciles_log_suffixes(
    tiny_training_contract, monkeypatch
):
    root = tiny_training_contract

    # Reference trajectory: four triads, two optimizer updates, one validation pass.
    uninterrupted_args, uninterrupted_injected, uninterrupted_trainer = _injected_run(
        output_root=root, run_id="uninterrupted"
    )
    _install_signal_state(monkeypatch)
    uninterrupted = full.run_training(uninterrupted_args, injected=uninterrupted_injected)
    assert uninterrupted["status"] == "completed"
    assert uninterrupted["training_totals"]["optimizer_attempts"] == 2
    assert len(uninterrupted_trainer.validation_calls) == 3

    # Signal during the first accumulation window. The runner must save after the
    # second triad, then continue with the exact remaining sampler/RNG state.
    interrupt_state = _install_signal_state(monkeypatch)
    interrupted_args, interrupted_injected, _ = _injected_run(
        output_root=root,
        run_id="resumed",
        interrupt_state=interrupt_state,
    )
    interrupted = full.run_training(interrupted_args, injected=interrupted_injected)
    assert interrupted["status"] == "interrupted"
    assert interrupted["progress"] == {
        "epoch_index": 0,
        "next_batch_index": 2,
        "validation_pending": False,
    }
    assert interrupted["training_totals"]["optimizer_attempts"] == 1

    run_dir = root / "resumed"
    # Simulate append-before-checkpoint crash tails. They are archived verbatim,
    # then removed from the authoritative log before real continuation appends.
    attempt_tail = b'{"global_attempt":2,"orphan":"simulated crash tail"}\n'
    epoch_tail = b'{"epoch":1,"orphan":"simulated validation tail"}\n'
    with (run_dir / "optimizer_attempts.jsonl").open("ab") as stream:
        stream.write(attempt_tail)
    with (run_dir / "epochs.jsonl").open("ab") as stream:
        stream.write(epoch_tail)

    resumed_args, resumed_injected, resumed_trainer = _injected_run(
        output_root=root, run_id="resumed"
    )
    resumed_args.resume = True
    interrupt_state["requested"] = None
    resumed = full.run_training(resumed_args, injected=resumed_injected)
    assert resumed["status"] == "completed"
    assert resumed["training_totals"] == uninterrupted["training_totals"]
    assert len(resumed_trainer.validation_calls) == 3

    reference_state = _checkpoint(root / "uninterrupted" / "resume_latest.pt")
    resumed_state = _checkpoint(root / "resumed" / "resume_latest.pt")
    for key in reference_state["model_state_dict"]:
        assert torch.equal(reference_state["model_state_dict"][key], resumed_state["model_state_dict"][key])
    for key in reference_state["optimizer_state_dict"]["state"]:
        reference_optimizer = reference_state["optimizer_state_dict"]["state"][key]
        resumed_optimizer = resumed_state["optimizer_state_dict"]["state"][key]
        for name, reference_value in reference_optimizer.items():
            value = resumed_optimizer[name]
            if torch.is_tensor(reference_value):
                assert torch.equal(reference_value, value)
            else:
                assert reference_value == value
    assert reference_state["scheduler_state"] == resumed_state["scheduler_state"]
    assert reference_state["rng_state"]["python"] == resumed_state["rng_state"]["python"]
    assert np.array_equal(reference_state["rng_state"]["numpy"][1], resumed_state["rng_state"]["numpy"][1])
    assert torch.equal(reference_state["rng_state"]["torch_cpu"], resumed_state["rng_state"]["torch_cpu"])

    # Actual durable records are contiguous; simulated tails survive only in
    # content-addressed audit files and the reconciliation index.
    attempts = [json.loads(line) for line in (run_dir / "optimizer_attempts.jsonl").read_text().splitlines()]
    epochs = [json.loads(line) for line in (run_dir / "epochs.jsonl").read_text().splitlines()]
    assert [row["global_attempt"] for row in attempts] == [1, 2]
    assert [row["epoch"] for row in epochs] == [1]
    orphan_attempt = list(run_dir.glob("orphaned_optimizer_attempts_*.jsonl"))
    orphan_epoch = list(run_dir.glob("orphaned_epochs_*.jsonl"))
    assert len(orphan_attempt) == len(orphan_epoch) == 1
    assert orphan_attempt[0].read_bytes() == attempt_tail
    assert orphan_epoch[0].read_bytes() == epoch_tail
    audit = [json.loads(line) for line in (run_dir / "resume_reconciliation.jsonl").read_text().splitlines()]
    assert {row["log"] for row in audit} == {"optimizer_attempts.jsonl", "epochs.jsonl"}


def test_first_optimizer_boundary_is_checkpointed_before_the_periodic_interval(
    tiny_training_contract, monkeypatch
):
    root = tiny_training_contract
    state = _install_signal_state(monkeypatch)
    args, injected, trainer = _injected_run(output_root=root, run_id="early-recovery")
    original_step = trainer.scaler.step
    call_count = {"value": 0}

    def fail_on_second_step(optimizer):
        call_count["value"] += 1
        if call_count["value"] == 2:
            raise RuntimeError("simulated hard failure after first checkpoint")
        return original_step(optimizer)

    trainer.scaler.step = fail_on_second_step
    with pytest.raises(RuntimeError, match="simulated hard failure"):
        full.run_training(args, injected=injected)

    run_dir = root / "early-recovery"
    checkpoint = _checkpoint(run_dir / "resume_latest.pt")
    assert checkpoint["totals"]["optimizer_attempts"] == 1
    assert checkpoint["progress"]["next_batch_index"] == 2
    assert json.loads((run_dir / "result.json").read_text())["status"] == "failed"

    resume_args, resume_injected, _ = _injected_run(output_root=root, run_id="early-recovery")
    resume_args.resume = True
    state["requested"] = None
    resumed = full.run_training(resume_args, injected=resume_injected)
    assert resumed["status"] == "completed"
    assert resumed["training_totals"]["optimizer_attempts"] == 2
    rows = [json.loads(line) for line in (run_dir / "optimizer_attempts.jsonl").read_text().splitlines()]
    assert [row["global_attempt"] for row in rows] == [1, 2]


def test_consecutive_skipped_updates_survive_resume_and_stop_without_retry(
    tiny_training_contract, monkeypatch
):
    root = tiny_training_contract
    monkeypatch.setattr(full, "EPOCHS", 2)
    monkeypatch.setattr(full, "EXPECTED_TRAIN_ROWS", 40)
    signal_state = _install_signal_state(monkeypatch)
    args, injected, trainer = _injected_run(
        output_root=root,
        run_id="skip-streak",
        interrupt_state=signal_state,
        train_size=40,
        interrupt_after_mim_call=24,
    )
    # Model gradients are intentionally discarded at each boundary so the
    # optimizer post-hook records every AMP attempt as skipped.
    trainer.scaler.step = lambda _optimizer: None
    interrupted = full.run_training(args, injected=injected)
    assert interrupted["status"] == "interrupted"
    assert interrupted["training_totals"]["optimizer_attempts"] == 12
    assert interrupted["training_totals"]["consecutive_skipped_updates"] == 12

    resume_args, resume_injected, resume_trainer = _injected_run(
        output_root=root, run_id="skip-streak", train_size=40
    )
    resume_args.resume = True
    resume_trainer.scaler.step = lambda _optimizer: None
    signal_state["requested"] = None
    stopped = full.run_training(resume_args, injected=resume_injected)
    assert stopped["status"] == "persistent_skipped_updates"
    assert stopped["training_totals"]["optimizer_attempts"] == 20
    assert stopped["training_totals"]["consecutive_skipped_updates"] == 20
    run_dir = root / "skip-streak"
    checkpoint = _checkpoint(run_dir / "resume_latest.pt")
    assert checkpoint["totals"]["consecutive_skipped_updates"] == 20
    gate = json.loads((run_dir / "first20_gate.json").read_text())
    assert gate["status"] == "zero_applied_updates"
    assert (gate["applied_updates"], gate["skipped_updates"]) == (0, 20)

    retry_args, retry_injected, _ = _injected_run(
        output_root=root, run_id="skip-streak", train_size=40
    )
    retry_args.resume = True
    with pytest.raises(full.FullTrainingError, match="persistent skipped"):
        full.run_training(retry_args, injected=retry_injected)


def test_initialization_preflight_checks_paired_model_hash_and_create_only(monkeypatch, tmp_path):
    research_dir = tmp_path / "autoresearch"
    research_dir.mkdir()
    monkeypatch.setattr(full, "AUTORESEARCH", research_dir)
    modules = {"torch": torch, "np": np, "ViLBERTConfig": _ToyConfig, "Task": _ToyTask}

    def same_model(_config, _arm, _assets):
        return _ToyModel()

    output = research_dir / "anchor.json"
    artifact = full.create_initialization_preflight(
        output_path=output,
        modules=modules,
        model_factory=same_model,
        asset_record={"local_assets": True},
        source_hashes={"toy": "sha"},
    )
    assert artifact["status"] == "passed"
    assert artifact["initial_model_state_sha256_by_arm"]["df2"] == artifact[
        "initial_model_state_sha256_by_arm"
    ]["rev"]
    with pytest.raises(full.FullTrainingError, match="overwrite"):
        full.create_initialization_preflight(
            output_path=output,
            modules=modules,
            model_factory=same_model,
            asset_record={"local_assets": True},
            source_hashes={"toy": "sha"},
        )

    def arm_dependent_model(config, arm, _assets):
        model = _ToyModel()
        with torch.no_grad():
            model.weight[0] += 1.0 if arm == "rev" else 0.0
        return model

    with pytest.raises(full.FullTrainingError, match="initial model tensors differ"):
        full.create_initialization_preflight(
            output_path=research_dir / "bad-anchor.json",
            modules=modules,
            model_factory=arm_dependent_model,
            asset_record={"local_assets": True},
            source_hashes={"toy": "sha"},
        )


def test_isolated_randomness_dataset_is_repeatable_and_restores_process_rng():
    class RandomDataset:
        def __len__(self):
            return 4

        def __getitem__(self, index):
            return (
                index,
                random.random(),
                float(np.random.random()),
                float(torch.rand(())),
            )

    random.seed(7)
    np.random.seed(8)
    torch.manual_seed(9)
    py_before = random.getstate()
    np_before = np.random.get_state()
    torch_before = torch.get_rng_state().clone()
    wrapped = full.IsolatedRandomnessDataset(RandomDataset(), seed=1804, split="train", task="mlm")
    first = wrapped[2]
    second = wrapped[2]
    assert first == second
    assert random.getstate() == py_before
    np_after = np.random.get_state()
    assert np_after[0] == np_before[0]
    assert np.array_equal(np_after[1], np_before[1])
    assert np_after[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)
    wrapped.epoch = 1
    assert wrapped[2] != first


def test_autoresearch_path_bounds_and_symlink_rejection(monkeypatch, tmp_path):
    research_dir = tmp_path / "autoresearch"
    research_dir.mkdir()
    monkeypatch.setattr(full, "AUTORESEARCH", research_dir)
    assert full._safe_autoresearch_path(research_dir / "ok" / "file.json").is_relative_to(research_dir)
    with pytest.raises(full.FullTrainingError, match="under autoresearch"):
        full._safe_autoresearch_path(tmp_path / "outside.json")
    symlink = research_dir / "link"
    symlink.symlink_to(tmp_path / "target")
    with pytest.raises(full.FullTrainingError, match="symlinks"):
        full._safe_autoresearch_path(symlink)


def test_tiny_run_validates_only_the_three_injected_cc_task_loaders(tiny_training_contract, monkeypatch):
    args, injected, trainer = _injected_run(output_root=tiny_training_contract, run_id="validation")
    _install_signal_state(monkeypatch)
    result = full.run_training(args, injected=injected)
    assert result["status"] == "completed"
    assert trainer.validation_calls == [
        ("ap", "ap", 4),
        ("mlm", "mlm", 4),
        ("mim", "mim", 4),
    ]
    epoch_record = json.loads((tiny_training_contract / "validation" / "epochs.jsonl").read_text())
    assert set(epoch_record["validation"]) >= {"ap_loss", "ap_accuracy", "mlm_loss", "mim_loss"}

    # Completed run IDs are immutable and symlinked run directories are rejected.
    retry_args, retry_injected, _ = _injected_run(
        output_root=tiny_training_contract, run_id="validation"
    )
    retry_args.resume = True
    with pytest.raises(full.FullTrainingError, match="completed run IDs"):
        full.run_training(retry_args, injected=retry_injected)

    symlink = tiny_training_contract / "alias"
    symlink.symlink_to(tiny_training_contract / "validation", target_is_directory=True)
    alias_args, alias_injected, _ = _injected_run(
        output_root=tiny_training_contract, run_id="alias"
    )
    with pytest.raises(full.FullTrainingError, match="run directory may not be a symlink"):
        full.run_training(alias_args, injected=alias_injected)


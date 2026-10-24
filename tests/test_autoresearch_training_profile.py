import csv
import hashlib
import json
from pathlib import Path

import pytest

from src import autoresearch_training_profile as profile


def _write_csv(path: Path, rows: list[tuple[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["text", "filepath"])
        writer.writerows(rows)


def _make_manifest(tmp_path: Path) -> Path:
    image_root = tmp_path / "images"
    tokenizer_path = tmp_path / "tokenizer"
    image_root.mkdir()
    tokenizer_path.mkdir()
    tokenizer_files = {
        "tokenizer.json": b"tokenizer",
        "tokenizer_config.json": b"config",
        "vocab.txt": b"vocab",
    }
    tokenizer_hashes = {}
    for name, content in tokenizer_files.items():
        (tokenizer_path / name).write_bytes(content)
        tokenizer_hashes[name] = hashlib.sha256(content).hexdigest()

    source_rows = [("filler", "filler.jpg") for _ in range(8)]
    selection_rows = []
    samples = []
    for index in range(20):
        caption = f"caption {index}"
        relpath = f"res/data/conceptual-captions/images/{index}.jpg"
        image_path = image_root / f"{index}.jpg"
        image_bytes = f"image-{index}".encode()
        image_path.write_bytes(image_bytes)
        source_index = len(source_rows)
        source_rows.append((caption, relpath))
        selection_rows.append((caption, relpath))
        caption_sha = hashlib.sha256(caption.encode()).hexdigest()
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        samples.append({
            "sample_id": f"sample-{index:02d}",
            "caption": caption,
            "caption_sha256": caption_sha,
            "image_path_abs": str(image_path),
            "image_path_absolute": str(image_path),
            "image_path_rel": relpath,
            "image_path_relative_to_manual_root": relpath,
            "image_sha256": image_sha,
            "image_size_bytes": len(image_bytes),
            "image_dimensions": [224, 224],
            "train_source_row_index_zero_based": source_index,
            "train_local_row_index_zero_based": index,
        })

    source_csv = tmp_path / "train.csv"
    selection_csv = tmp_path / "train-local.csv"
    _write_csv(source_csv, source_rows)
    _write_csv(selection_csv, selection_rows)
    bert_config = tmp_path / "bert-config.json"
    bert_config.write_text("{}", encoding="utf-8")
    vision_weights = tmp_path / "unused-vit-weights"
    vision_weights.write_bytes(b"not loaded by profile")
    manifest = {
        "schema_version": "profile-cc-train20-v1",
        "split": "training-only",
        "sample_count": 20,
        "samples": samples,
        "train_csv_row_indexing": "1-based physical CSV line number including the header line",
        "source_csv_path": str(source_csv),
        "source_csv_sha256": hashlib.sha256(source_csv.read_bytes()).hexdigest(),
        "selection_csv_path": str(selection_csv),
        "selection_csv_sha256": hashlib.sha256(selection_csv.read_bytes()).hexdigest(),
        "image_root": str(image_root),
        "tokenizer_path": str(tokenizer_path),
        "tokenizer_source_files_sha256": tokenizer_hashes,
        "tokenizer": {
            "local_path": str(tokenizer_path),
            "max_length": 197,
            "source_file_sha256": tokenizer_hashes,
        },
        "bert_model_config_path": str(bert_config),
        "vit_model_weights_path": str(vision_weights),
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path


def test_variant_matrix_is_fixed_and_validates_checkpoint_fusion():
    assert profile.variant_settings("eager_dense") == {
        "cross_attention_backend": "eager",
        "mlm_selected_tokens_only": False,
    }
    assert profile.variant_settings("eager_selected") == {
        "cross_attention_backend": "eager",
        "mlm_selected_tokens_only": True,
    }
    assert profile.variant_settings("sdpa_selected") == {
        "cross_attention_backend": "sdpa",
        "mlm_selected_tokens_only": True,
    }
    profile.validate_fusion_config({
        "text_cross_attention_layers": [3, 4, 5],
        "vision_cross_attention_layers": [9, 10, 11],
    })
    with pytest.raises(profile.ProfileError, match="text fusion"):
        profile.validate_fusion_config({
            "text_cross_attention_layers": [0, 1, 2],
            "vision_cross_attention_layers": [9, 10, 11],
        })


def test_manifest_validation_binds_csv_rows_images_and_tokenizer_files(tmp_path):
    path = _make_manifest(tmp_path)
    manifest, digest = profile.validate_manifest(path, expected_sha256=None)
    assert len(manifest["samples"]) == 20
    assert manifest["_identity"]["manifest_sha256"] == digest
    assert manifest["_identity"]["sample_ids"][0] == "sample-00"

    broken = json.loads(path.read_text(encoding="utf-8"))
    broken["samples"][0]["caption"] = "changed caption"
    path.write_text(json.dumps(broken), encoding="utf-8")
    with pytest.raises(profile.ProfileError, match="caption hash"):
        profile.validate_manifest(path, expected_sha256=None)


class _FakeParameter:
    requires_grad = True
    grad = object()


class _FakeModel:
    def __init__(self):
        self.parameter = _FakeParameter()

    def parameters(self):
        return iter([self.parameter])


class _FakeOptimizer:
    def __init__(self, parameter):
        self.parameter = parameter
        self.state = {}
        self.zero_calls = 0

    def step(self):
        previous = self.state.get(self.parameter, {}).get("step", 0)
        self.state[self.parameter] = {"step": previous + 1}

    def zero_grad(self):
        self.zero_calls += 1
        self.parameter.grad = None


class _FakeScaler:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.step_calls = 0
        self.update_calls = 0

    def step(self, optimizer):
        self.step_calls += 1
        optimizer.step()

    def update(self):
        self.update_calls += 1


class _FakeTrainer:
    def __init__(self):
        self.model = _FakeModel()
        self.optimizer = _FakeOptimizer(self.model.parameter)
        self.scaler = _FakeScaler(self.optimizer)
        self.calls = []

    def _task(self, name, batch, flag_optimizer):
        assert flag_optimizer is False
        self.calls.append((name, batch))
        return {"ap": 0.7, "mlm": 2.1, "mim": 0.3}[name]

    def train_alignment_prediction_batch(self, batch, flag_optimizer=True):
        return self._task("ap", batch, flag_optimizer)

    def train_mlm_batch(self, batch, flag_optimizer=True):
        return self._task("mlm", batch, flag_optimizer)

    def train_mim_batch(self, batch, flag_optimizer=True):
        return self._task("mim", batch, flag_optimizer)


def test_joint_update_calls_three_objectives_then_one_optimizer_step():
    trainer = _FakeTrainer()
    syncs = []
    triad = {"ap": "a", "mlm": "l", "mim": "i"}
    result = profile.run_joint_update(
        trainer,
        triad,
        synchronize=lambda: syncs.append("sync"),
    )
    assert trainer.calls == [("ap", "a"), ("mlm", "l"), ("mim", "i")]
    assert trainer.scaler.step_calls == trainer.scaler.update_calls == 1
    assert trainer.optimizer.zero_calls == 1
    assert len(syncs) == 2
    assert result["task_losses"] == {"ap": 0.7, "mlm": 2.1, "mim": 0.3}
    assert result["actual_optimizer_step"] is True
    assert result["tracked_parameter_optimizer_step"] == 1


def test_json_result_is_immutable_and_nonfinite_values_are_rejected(tmp_path):
    result_path = tmp_path / "variant.json"
    profile.write_result_once(result_path, {"status": "completed", "count": 8})
    assert json.loads(result_path.read_text())["count"] == 8
    with pytest.raises(FileExistsError):
        profile.write_result_once(result_path, {"status": "replacement"})
    with pytest.raises(ValueError):
        profile.write_result_once(tmp_path / "bad.json", {"metric": float("nan")})

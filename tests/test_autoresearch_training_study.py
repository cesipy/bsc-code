"""CPU-only invariant tests for the staged fresh-training runner."""

from __future__ import annotations

import hashlib
import copy
import importlib.util
import json
import os
from pathlib import Path
import random
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn


STUDY_PATH = Path(__file__).with_name("autoresearch_training_study.py")
if not STUDY_PATH.is_file():
    STUDY_PATH = Path(__file__).parents[1] / "src" / "autoresearch_training_study.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_training_study_staged", STUDY_PATH)
study = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(study)


def test_run_bounds_are_hard_and_finite():
    study.validate_run_bounds(
        attempts=100,
        max_wall_seconds=600,
        seed=1801,
        arm="DF2",
        backend="eager",
        mlm_mode="dense",
        mim_policy="unit_scale",
        empty_mlm_policy="zero_gradient",
    )
    for kwargs in (
        {"attempts": 0, "max_wall_seconds": 600},
        {"attempts": 101, "max_wall_seconds": 600},
        {"attempts": 1, "max_wall_seconds": float("inf")},
        {"attempts": 1, "max_wall_seconds": 601},
    ):
        with pytest.raises(study.StudyError):
            study.validate_run_bounds(**kwargs)
    assert study.derive_training_seed(1801) == 101801
    with pytest.raises(study.StudyError, match="fixed to the profiled"):
        study.validate_run_bounds(
            attempts=100,
            max_wall_seconds=600,
            arm="DF2",
            backend="sdpa",
            mlm_mode="dense",
        )


def test_test_split_is_rejected_before_any_source_access(tmp_path: Path):
    manifest_path = tmp_path / "test-only.json"
    manifest_path.write_text(json.dumps({"dataset": "Conceptual Captions", "split": "test-only"}))
    digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    with pytest.raises(study.StudyError, match="split"):
        study.validate_manifest(
            manifest_path,
            expected_sha256=digest,
            split="training-only",
            sample_count=2,
        )


def _seed_sample(sample_id: str, index: int) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "image_path_abs": f"/fake/{index}.jpg",
        "caption": f"caption {index}",
    }


class _TinyDataset:
    kind = "base"

    def __init__(self, data, _tokenizer, _processor, **_kwargs):
        rows = list(data)
        random.shuffle(rows)
        # Match the production dataset's ``(task, image_path, text, label)``
        # record shape, which the materializer hashes to recover sample IDs.
        self.data = [(None, path, caption, 0) for path, caption in rows]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        path = self.data[index][1]
        row = int(Path(path).stem)
        text = {
            "input_ids": torch.tensor([[row % 7 + 1, 2, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.long),
            "token_type_ids": torch.zeros((1, 3), dtype=torch.long),
        }
        image = {"pixel_values": torch.full((1, 3, 2, 2), float(row + 1))}
        if self.kind == "ap":
            return {"img": image, "text": text, "label": torch.tensor(row % 2)}
        if self.kind == "mlm":
            label = torch.full((1, 3), -100, dtype=torch.long)
            return {"img": image, "text": text, "label": label}
        mask = torch.zeros(196, dtype=torch.long)
        mask[0] = 1
        return {
            "img": image,
            "masked_img": {"pixel_values": torch.full((1, 3, 2, 2), 255.0)},
            "masked_patches_idxs": mask,
            "text": text,
        }


class _TinyAP(_TinyDataset):
    kind = "ap"


class _TinyMLM(_TinyDataset):
    kind = "mlm"


class _TinyMIM(_TinyDataset):
    kind = "mim"


def test_materialization_pairs_same_order_and_unit_scale_changes_only_masked_view():
    rows = [_seed_sample(f"s{index}", index) for index in range(8)]
    manifest = {"samples": rows}
    dataset_types = {"ap": _TinyAP, "mlm": _TinyMLM, "mim": _TinyMIM}

    legacy, legacy_report = study.materialize_training_batches(
        manifest,
        tokenizer=object(),
        torch=torch,
        np=np,
        mim_policy="legacy",
        expected_sample_count=8,
        planned_batches=4,
        dataset_types=dataset_types,
    )
    unit, unit_report = study.materialize_training_batches(
        manifest,
        tokenizer=object(),
        torch=torch,
        np=np,
        mim_policy="unit_scale",
        expected_sample_count=8,
        planned_batches=4,
        dataset_types=dataset_types,
    )
    assert legacy_report["sample_order_ids"] == unit_report["sample_order_ids"]
    assert legacy_report["policy_invariant_batch_sha256"] == unit_report["policy_invariant_batch_sha256"]
    assert legacy_report["mim_patch_mask_sha256"] == unit_report["mim_patch_mask_sha256"]
    for old, scaled in zip(legacy, unit):
        assert torch.equal(old["mim"]["masked_img"]["pixel_values"] / 255.0,
                           scaled["mim"]["masked_img"]["pixel_values"])
        assert torch.equal(old["mim"]["masked_patches_idxs"], scaled["mim"]["masked_patches_idxs"])
        assert torch.equal(old["mim"]["img"]["pixel_values"], scaled["mim"]["img"]["pixel_values"])
        assert torch.equal(old["ap"]["img"]["pixel_values"], scaled["ap"]["img"]["pixel_values"])
        assert torch.equal(old["mlm"]["label"], scaled["mlm"]["label"])
    assert legacy_report["empty_mlm_batch_count"] == 4
    assert legacy_report["empty_mlm_batch_indices_zero_based"] == [0, 1, 2, 3]
    paired = study.build_paired_triad_identity_records(
        legacy,
        unit,
        legacy_report,
        unit_report,
        torch,
        expected_batch_count=4,
    )
    assert paired["batch_count"] == 4
    assert len(paired["triads"]) == 4
    assert all(row["mim"]["unit_scale_equals_legacy_divided_by_255"] for row in paired["triads"])
    assert all(
        row["invariant_inputs_legacy_sha256"] == row["invariant_inputs_unit_scale_sha256"]
        for row in paired["triads"]
    )


def test_unit_scale_policy_is_out_of_place_and_preserves_masked_zeros():
    original = torch.tensor([[[[[0.0, 255.0]]]]])
    batch = {
        "img": {"pixel_values": torch.tensor([[[[[4.0, 5.0]]]]])},
        "masked_img": {"pixel_values": original},
        "masked_patches_idxs": torch.tensor([[1, 0]]),
        "label": torch.tensor([1]),
    }
    before = {key: value.clone() for key, value in batch["img"].items()}
    scaled = study.apply_mim_input_policy(batch, "unit_scale")
    assert torch.equal(batch["masked_img"]["pixel_values"], original)
    assert torch.equal(scaled["masked_img"]["pixel_values"], original / 255.0)
    assert scaled["masked_img"]["pixel_values"].flatten()[0].item() == 0.0
    assert torch.equal(scaled["img"]["pixel_values"], before["pixel_values"])
    assert torch.equal(scaled["masked_patches_idxs"], batch["masked_patches_idxs"])
    assert torch.equal(scaled["label"], batch["label"])


def test_empty_mlm_policy_uses_graph_connected_zero_gradients():
    for selected in (False, True):
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        head = nn.Linear(4, 7)
        labels = torch.full((2, 3), -100, dtype=torch.long)
        loss, status = study.safe_mlm_projection_loss(
            hidden,
            labels,
            head,
            nn.CrossEntropyLoss(ignore_index=-100),
            selected_tokens_only=selected,
            empty_policy="zero_gradient",
        )
        assert status == "empty_zero_gradient"
        assert loss.item() == 0.0
        loss.backward()
        assert hidden.grad is not None and torch.equal(hidden.grad, torch.zeros_like(hidden))
        assert head.weight.grad is not None and torch.equal(head.weight.grad, torch.zeros_like(head.weight))
        assert head.bias.grad is not None and torch.equal(head.bias.grad, torch.zeros_like(head.bias))
        with pytest.raises(study.StudyError, match="no supervised"):
            study.safe_mlm_projection_loss(
                hidden.detach(),
                labels,
                head,
                nn.CrossEntropyLoss(ignore_index=-100),
                selected_tokens_only=selected,
                empty_policy="fail",
            )


def test_offline_constructor_rewrites_every_pretrained_request_to_local(tmp_path: Path):
    bert_dir = tmp_path / "bert-snapshot"
    bert_dir.mkdir()
    vit_file = tmp_path / "vit.safetensors"
    vit_file.write_bytes(b"local-only")
    calls = {}

    class FakeBert:
        @classmethod
        def from_pretrained(cls, model_id, **kwargs):
            assert Path(model_id).is_absolute()
            assert kwargs["local_files_only"] is True
            assert kwargs["use_safetensors"] is True
            assert os.environ["HF_HUB_OFFLINE"] == "1"
            calls["bert"] = (model_id, kwargs)
            return "local-bert"

    class FakeVision:
        def state_dict(self):
            return {"body": torch.zeros(1)}

        def load_state_dict(self, state, strict):
            calls["loaded_vision_state"] = (state, strict)
            assert strict is True
            return SimpleNamespace(missing_keys=[], unexpected_keys=[])

    timm_fake = SimpleNamespace()

    def fake_timm_factory(name, **kwargs):
        assert name == "vit_base_patch16_224"
        assert kwargs["pretrained"] is False
        calls["timm"] = (name, kwargs)
        return FakeVision()

    timm_fake.create_model = fake_timm_factory

    class FakeModel:
        def __init__(self, _config):
            self.bert = FakeBert.from_pretrained("google-bert/bert-base-uncased")
            self.vision = timm_fake.create_model(
                "vit_base_patch16_224", pretrained=True, num_classes=0, global_pool=""
            )

    model = study.construct_fresh_model(
        object(),
        bert_snapshot=bert_dir,
        vit_weights=vit_file,
        vit_model_name="vit_base_patch16_224",
        model_class=FakeModel,
        bert_model_class=FakeBert,
        timm_module=timm_fake,
        load_vit_state=lambda path: {
            "body": torch.ones(1),
            "head.weight": torch.zeros(1000, 768),
            "head.bias": torch.zeros(1000),
        },
    )
    assert model.bert == "local-bert"
    loaded_state, strict = calls["loaded_vision_state"]
    assert set(loaded_state) == {"body"}
    assert strict is True
    assert model.vision._autoresearch_pretrained_load_info == {
        "strict_backbone_load": True,
        "classifier_keys_dropped": ["head.bias", "head.weight"],
        "classifier_shapes_dropped": {
            "head.weight": [1000, 768],
            "head.bias": [1000],
        },
        "classifier_reason": "feature extractor constructed with num_classes=0; local source checkpoint includes its ImageNet-1k classifier",
    }
    assert os.environ.get("HF_HUB_OFFLINE") is None
    assert FakeBert.from_pretrained.__func__.__name__ == "from_pretrained"
    assert timm_fake.create_model is fake_timm_factory


def test_vit_loader_rejects_anything_beyond_the_exact_classifier_head():
    class TinyVision:
        def state_dict(self):
            return {"body": torch.zeros(1)}

        def load_state_dict(self, _state, strict):
            assert strict is True

    good_shape_except_key = {
        "body": torch.ones(1),
        "head.weight": torch.zeros(1000, 768),
        "head.bias": torch.zeros(1000),
        "unexpected": torch.zeros(1),
    }
    with pytest.raises(study.StudyError, match="unexpected keys"):
        study._load_pretrained_vit_body_without_classifier(TinyVision(), good_shape_except_key)

    bad_classifier_shape = {
        "body": torch.ones(1),
        "head.weight": torch.zeros(1001, 768),
        "head.bias": torch.zeros(1001),
    }
    with pytest.raises(study.StudyError, match="classifier shape"):
        study._load_pretrained_vit_body_without_classifier(TinyVision(), bad_classifier_shape)


def test_local_asset_validation_keeps_snapshot_parent_when_weights_are_symlinks(tmp_path: Path):
    blobs = tmp_path / "hub" / "blobs"
    snapshot = tmp_path / "hub" / "snapshots" / "revision"
    tokenizer = tmp_path / "tokenizer"
    blobs.mkdir(parents=True)
    snapshot.mkdir(parents=True)
    tokenizer.mkdir()
    for name, content in (
        ("config.json", b"{}"),
        ("bert.safetensors", b"bert-local"),
        ("vit.safetensors", b"vit-local"),
    ):
        (blobs / name).write_bytes(content)
    (snapshot / "config.json").symlink_to(blobs / "config.json")
    (snapshot / "model.safetensors").symlink_to(blobs / "bert.safetensors")
    for name, content in (("tokenizer.json", b"tokenizer"), ("tokenizer_config.json", b"config"), ("vocab.txt", b"[PAD]")):
        (tokenizer / name).write_bytes(content)
    feasibility = {
        "data": {
            "offline_initialization": {
                "bert": {
                    "weights_path": str(snapshot / "model.safetensors"),
                    "config_path": str(snapshot / "config.json"),
                    "model_id": "local/bert",
                    "revision": "revision",
                },
                "vit": {
                    "weights_path": str(blobs / "vit.safetensors"),
                    "cached_model_id": "local/vit",
                    "timm_model_name": "vit_base_patch16_224",
                    "revision": "revision",
                },
                "tokenizer": {
                    "path": str(tokenizer),
                    "model_id": "local/tokenizer",
                    "source_file_sha256": {
                        name: hashlib.sha256((tokenizer / name).read_bytes()).hexdigest()
                        for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt")
                    },
                },
            }
        }
    }
    assets = study.validate_local_assets(feasibility)
    assert assets["bert_snapshot"] == str(snapshot.resolve())
    assert Path(assets["bert_weights_path"]).resolve() == (blobs / "bert.safetensors").resolve()


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(12, 4)
        self.image_projection = nn.Linear(3, 4)
        self.alignment_fc = nn.Linear(4, 1)
        self.mlm = nn.Linear(4, 12)
        self.call_counts = {"alignment_prediction": 0, "mlm": 0, "mim": 0}

    def forward_pretrain(
        self,
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None,
        tasks=None,
    ):
        task = tasks[0]
        self.call_counts[task] += 1
        text = self.embedding(text_input_ids)
        pooled = image_pixel_values.mean(dim=(-1, -2))
        vision_cls = self.image_projection(pooled)
        text = text + vision_cls[:, None, :]
        vision = vision_cls[:, None, :].expand(-1, 4, -1)
        if task == "alignment_prediction":
            return text[:, 0, :], vision[:, 0, :]
        return text, vision


def _tiny_triad():
    ids = torch.tensor([[[1, 2, 3, 0]], [[4, 5, 6, 0]]], dtype=torch.long)
    text = {
        "input_ids": ids,
        "attention_mask": torch.tensor([[[1, 1, 1, 0]], [[1, 1, 1, 0]]]),
        "token_type_ids": torch.zeros_like(ids),
    }
    image = {"pixel_values": torch.randn(2, 1, 3, 4, 4)}
    mlm_labels = torch.full((2, 1, 4), -100, dtype=torch.long)
    mlm_labels[0, 0, 1] = 2
    mlm_labels[1, 0, 0] = 4
    patch_mask = torch.tensor([[0, 1, 0], [1, 0, 0]], dtype=torch.long)
    return {
        "ap": {"text": text, "img": image, "label": torch.tensor([1, 0])},
        "mlm": {"text": text, "img": image, "label": mlm_labels},
        "mim": {
            "text": text,
            "img": image,
            "masked_img": {"pixel_values": image["pixel_values"].clone() * 0.5},
            "masked_patches_idxs": patch_mask,
        },
    }


def _tiny_mim_loss(masked, unmasked):
    return (masked - unmasked).square().mean() + 0.1 * masked.square().mean()


def _manual_summed_loss_update(model, triad):
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    ap_text, ap_img = study._squeeze_inputs(triad["ap"], torch, torch.device("cpu"))
    ap_text_features, _ = study._call_model(model, ap_text, ap_img, "alignment_prediction")
    ap_loss = nn.BCEWithLogitsLoss()(
        model.alignment_fc(ap_text_features).squeeze(-1),
        triad["ap"]["label"].float(),
    )

    mlm_text, mlm_img = study._squeeze_inputs(triad["mlm"], torch, torch.device("cpu"))
    mlm_hidden, _ = study._call_model(model, mlm_text, mlm_img, "mlm")
    mlm_loss, _ = study.safe_mlm_projection_loss(
        mlm_hidden,
        triad["mlm"]["label"],
        model.mlm,
        nn.CrossEntropyLoss(ignore_index=-100),
        selected_tokens_only=True,
        empty_policy="zero_gradient",
    )

    mim_text, mim_img = study._squeeze_inputs(triad["mim"], torch, torch.device("cpu"))
    masked_img = {key: value.squeeze(1) for key, value in triad["mim"]["masked_img"].items()}
    _, masked_features = study._call_model(model, mim_text, masked_img, "mim")
    _, unmasked_features = study._call_model(model, mim_text, mim_img, "mim")
    patch_mask = triad["mim"]["masked_patches_idxs"]
    full_mask = torch.cat((torch.ones(2, 1, dtype=torch.bool), patch_mask.eq(0)), dim=1)
    mim_loss = _tiny_mim_loss(masked_features[full_mask], unmasked_features[full_mask])

    losses = {"ap": ap_loss.item(), "mlm": mlm_loss.item(), "mim": mim_loss.item()}
    (ap_loss + mlm_loss + mim_loss).backward()
    optimizer.step()
    return optimizer, losses


def test_mocked_joint_cpu_update_reports_actual_step_and_full_objectives():
    torch.set_num_threads(4)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    result = study.joint_optimizer_step(
        model,
        optimizer,
        scaler,
        _tiny_triad(),
        device=torch.device("cpu"),
        selected_tokens_only=True,
        empty_mlm_policy="zero_gradient",
        loss_fn_mim=_tiny_mim_loss,
        torch_module=torch,
    )
    assert result["actual_optimizer_step"] is True
    assert result["optimizer_step_status"] == "applied"
    assert result["gradients_finite_after_unscale"] is True
    assert result["parameters_finite_after_step"] is True
    assert set(result["losses"]) == {"ap", "mlm", "mim"}
    assert all(np.isfinite(value) for value in result["losses"].values())
    assert model.call_counts == {"alignment_prediction": 1, "mlm": 1, "mim": 2}
    assert all(parameter.grad is None for parameter in model.parameters())


def test_sequential_objective_backwards_match_summed_loss_update():
    torch.set_num_threads(4)
    torch.manual_seed(44)
    summed_model = _TinyModel()
    sequential_model = copy.deepcopy(summed_model)
    triad = _tiny_triad()
    reference_optimizer, expected_losses = _manual_summed_loss_update(summed_model, triad)
    sequential_optimizer = torch.optim.AdamW(
        sequential_model.parameters(), lr=1e-4, weight_decay=0.01
    )
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    result = study.joint_optimizer_step(
        sequential_model,
        sequential_optimizer,
        scaler,
        triad,
        device=torch.device("cpu"),
        selected_tokens_only=True,
        empty_mlm_policy="zero_gradient",
        loss_fn_mim=_tiny_mim_loss,
        torch_module=torch,
    )
    assert result["losses"] == pytest.approx(expected_losses, abs=1e-7)
    assert result["actual_optimizer_step"] is True
    for expected, actual in zip(summed_model.parameters(), sequential_model.parameters()):
        assert torch.allclose(expected, actual, atol=2e-6, rtol=1e-5)
    for expected, actual in zip(reference_optimizer.state.values(), sequential_optimizer.state.values()):
        for key in ("step", "exp_avg", "exp_avg_sq"):
            if key in expected:
                assert torch.allclose(expected[key], actual[key], atol=2e-6, rtol=1e-5)


def test_run_loop_records_skipped_step_without_extending_attempt_budget():
    torch.set_num_threads(4)
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    class SkipScaler:
        def scale(self, loss):
            return loss

        def unscale_(self, _optimizer):
            model.embedding.weight.grad[0, 0] = float("inf")

        def step(self, _optimizer):
            return None

        def update(self):
            return None

        def get_scale(self):
            return 1.0

    logged = []
    results, reason = study.run_training_attempts(
        model,
        optimizer,
        SkipScaler(),
        [_tiny_triad(), _tiny_triad()],
        attempts=1,
        max_wall_seconds=60,
        started_monotonic=time.monotonic(),
        device=torch.device("cpu"),
        selected_tokens_only=False,
        empty_mlm_policy="zero_gradient",
        rss_bytes=lambda: 1024,
        max_rss_bytes=1024 * 1024,
        loss_fn_mim=_tiny_mim_loss,
        torch_module=torch,
        on_attempt=lambda row: logged.append(dict(row)),
    )
    assert len(results) == 1
    assert reason == "attempt_budget_reached"
    assert results[0]["attempt"] == 1
    assert results[0]["actual_optimizer_step"] is False
    assert results[0]["optimizer_step_status"] == "skipped_by_grad_scaler"
    assert results[0]["gradients_finite_after_unscale"] is False
    assert results[0]["parameters_finite_after_step"] is True
    assert results[0]["optimizer_skip_reason"] == "nonfinite_gradients_grad_scaler_skip"
    assert logged == results


def test_run_loop_durably_records_failed_attempt_before_propagating():
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    logged = []

    def fail_before_step(*_args, **_kwargs):
        raise study.StudyError("synthetic nonfinite loss")

    with pytest.raises(study.StudyError, match="synthetic nonfinite loss"):
        study.run_training_attempts(
            model,
            optimizer,
            scaler,
            [_tiny_triad()],
            attempts=1,
            max_wall_seconds=60,
            started_monotonic=time.monotonic(),
            device=torch.device("cpu"),
            selected_tokens_only=False,
            empty_mlm_policy="zero_gradient",
            rss_bytes=lambda: 1024,
            max_rss_bytes=1024 * 1024,
            torch_module=torch,
            update_fn=fail_before_step,
            on_attempt=lambda row: logged.append(dict(row)),
        )
    assert len(logged) == 1
    assert logged[0]["attempt_record_status"] == "failed_before_step_result"
    assert logged[0]["actual_optimizer_step"] is None
    assert logged[0]["optimizer_step_status"] == "optimizer_step_outcome_unknown"
    assert logged[0]["error"] == "synthetic nonfinite loss"


def test_run_loop_marks_exception_after_optimizer_update_as_applied():
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    logged = []

    def step_then_fail(_model, optimizer, _scaler, *_args, **_kwargs):
        tracked = next(parameter for parameter in model.parameters() if parameter.requires_grad)
        tracked.grad = torch.ones_like(tracked)
        optimizer._autoresearch_step_counter_before = study._tracked_optimizer_step_value(optimizer, model)
        optimizer._autoresearch_step_phase = "calling_optimizer_step"
        optimizer.step()
        optimizer._autoresearch_step_phase = "optimizer_step_returned"
        raise study.StudyError("synthetic post-step diagnostic failure")

    with pytest.raises(study.StudyError, match="post-step diagnostic failure"):
        study.run_training_attempts(
            model,
            optimizer,
            scaler,
            [_tiny_triad()],
            attempts=1,
            max_wall_seconds=60,
            started_monotonic=time.monotonic(),
            device=torch.device("cpu"),
            selected_tokens_only=False,
            empty_mlm_policy="zero_gradient",
            rss_bytes=lambda: 1024,
            max_rss_bytes=1024 * 1024,
            torch_module=torch,
            update_fn=step_then_fail,
            on_attempt=lambda row: logged.append(dict(row)),
        )
    assert len(logged) == 1
    assert logged[0]["actual_optimizer_step"] is True
    assert logged[0]["optimizer_step_status"] == "optimizer_step_advanced_before_exception"
    assert logged[0]["tracked_parameter_step_before"] == 0.0
    assert logged[0]["tracked_parameter_step_after"] == 1.0


def test_run_loop_can_prove_a_pre_optimizer_failure():
    model = _TinyModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = torch.amp.GradScaler(device="cpu", enabled=False)
    logged = []

    def fail_before_step(_model, optimizer, *_args, **_kwargs):
        optimizer._autoresearch_step_phase = "before_optimizer_step"
        raise study.StudyError("synthetic forward failure")

    with pytest.raises(study.StudyError, match="forward failure"):
        study.run_training_attempts(
            model,
            optimizer,
            scaler,
            [_tiny_triad()],
            attempts=1,
            max_wall_seconds=60,
            started_monotonic=time.monotonic(),
            device=torch.device("cpu"),
            selected_tokens_only=False,
            empty_mlm_policy="zero_gradient",
            rss_bytes=lambda: 1024,
            max_rss_bytes=1024 * 1024,
            torch_module=torch,
            update_fn=fail_before_step,
            on_attempt=lambda row: logged.append(dict(row)),
        )
    assert logged[0]["actual_optimizer_step"] is False
    assert logged[0]["optimizer_step_status"] == "failed_before_optimizer_step"


def test_durable_jsonl_log_is_create_only_and_readable(tmp_path: Path):
    path = tmp_path / "steps.jsonl"
    study.create_jsonl_log(path)
    study.append_jsonl_durable(path, {"attempt": 1, "actual_optimizer_step": True})
    study.append_jsonl_durable(path, {"attempt": 2, "actual_optimizer_step": False})
    rows, corrupt_tail = study.read_jsonl_steps(path)
    assert [row["attempt"] for row in rows] == [1, 2]
    assert corrupt_tail is False
    with pytest.raises(study.StudyError, match="overwrite"):
        study.create_jsonl_log(path)


def test_seed_anchor_pairs_initial_state_hash_across_arms_and_policies(tmp_path: Path):
    root = tmp_path / "autoresearch-output"
    root.mkdir()
    source = {"src/vilbert.py": "source-hash"}
    backbones = {"bert_weights_sha256": "bert", "vision_weights_sha256": "vit"}
    first = study.register_seed_initialization(
        root,
        seed=1801,
        state_sha256="identical-initial-state",
        backbone_hashes=backbones,
        source_hashes=source,
    )
    second = study.register_seed_initialization(
        root,
        seed=1801,
        state_sha256="identical-initial-state",
        backbone_hashes=backbones,
        source_hashes=source,
    )
    assert first == second
    with pytest.raises(study.StudyError, match="mismatch"):
        study.register_seed_initialization(
            root,
            seed=1801,
            state_sha256="different-state",
            backbone_hashes=backbones,
            source_hashes=source,
        )


def test_seed_anchor_can_live_in_shared_global_root(tmp_path: Path):
    run_root = tmp_path / "artifact-dir"
    run_root.mkdir()
    shared = tmp_path / "shared" / "seed-initializations"
    anchor = study.register_seed_initialization(
        run_root,
        seed=1802,
        state_sha256="seed-state",
        backbone_hashes={"bert": "b", "vision": "v"},
        source_hashes={"driver": "d"},
        anchor_root=shared,
    )
    path = shared / "seed-1802.json"
    assert path.is_file()
    assert json.loads(path.read_text()) == anchor
    assert list(run_root.iterdir()) == []


def test_create_only_json_and_storage_bounds(tmp_path: Path):
    output = tmp_path / "artifact.json"
    study.write_json_create_only(output, {"status": "ok"})
    with pytest.raises(study.StudyError, match="overwrite"):
        study.write_json_create_only(output, {"status": "different"})
    with pytest.raises(study.StudyError, match="planned run"):
        study.check_storage_budget(tmp_path, 1024, planned_runs=10)
    with pytest.raises(study.StudyError, match="per-run"):
        study.check_storage_budget(tmp_path, study.MAX_RUN_ARTIFACT_BYTES, planned_runs=1)
    measured = study.check_storage_budget(
        tmp_path,
        model_state_bytes=1_531_390_740,
        planned_runs=6,
    )
    assert study.MAX_TOTAL_ARTIFACT_BYTES == 10 * 1024**3
    assert measured["estimated_total_artifact_bytes"] <= 10 * 1024**3
    assert measured["per_run_artifact_limit_bytes"] == int(1.5 * 1024**3)

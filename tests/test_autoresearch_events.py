from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

import autoresearch_events as events


class AddBlock(torch.nn.Module):
    def __init__(self, amount: float):
        super().__init__()
        self.amount = amount

    def forward(self, text, vision):
        # In-place updates make the snapshot test catch tensor aliasing.
        text.add_(self.amount)
        vision.add_(10 * self.amount)
        return text, vision


class ThreeBlockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.c_layers = torch.nn.ModuleList(AddBlock(value) for value in (1, 2, 3))

    def run_states(self, text, vision):
        for layer in self.c_layers:
            text, vision = layer(text, vision)
        return text, vision


def _specs(dim=4):
    del dim
    return (
        events.EventSpec(0, 3, 5),
        events.EventSpec(1, 4, 6),
        events.EventSpec(2, 5, 7),
    )


def _assert_no_hooks(model):
    for module in model.c_layers:
        assert not module._forward_pre_hooks
        assert not module._forward_hooks


def test_event_specs_keep_c_layer_ordinals_separate_from_backbone_ids():
    specs = events.event_specs_from_architecture(
        {
            "text_cross_attention_layers_zero_based": [3, 4, 5],
            "vision_cross_attention_layers_zero_based": [5, 6, 7],
        },
        cross_attention_block_count=3,
    )
    assert specs == _specs()
    with pytest.raises(events.EventExtractionError, match="exactly three"):
        events.event_specs_from_architecture(
            {
                "text_cross_attention_layers_zero_based": [3, 4],
                "vision_cross_attention_layers_zero_based": [5, 6],
            },
            cross_attention_block_count=2,
        )


def test_hooks_snapshot_full_block_cls_without_changing_outputs_or_aliasing():
    baseline = ThreeBlockModel().eval()
    observed = ThreeBlockModel().eval()
    text = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    vision = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4) + 100
    with torch.inference_mode():
        expected = baseline.run_states(text.clone(), vision.clone())

    capture = events.CrossAttentionEventCapture(observed.c_layers, _specs(), embedding_dim=4)
    with capture:
        capture.begin_batch(2)
        with torch.inference_mode():
            actual = observed.run_states(text.clone(), vision.clone())
        pre, post = capture.finish_batch()

    assert torch.equal(actual[0], expected[0])
    assert torch.equal(actual[1], expected[1])
    assert torch.equal(pre[:, 0, 0], text[:, 0])
    assert torch.equal(pre[:, 0, 1], vision[:, 0])
    assert torch.equal(post[:, 0, 0], text[:, 0] + 1)
    assert torch.equal(post[:, 1, 0], text[:, 0] + 3)
    assert torch.equal(post[:, 2, 1], vision[:, 0] + 60)
    # Later in-place block updates cannot mutate event 0's already copied output.
    assert torch.equal(post[:, 0, 0], text[:, 0] + 1)
    assert capture.pre_call_counts == [1, 1, 1]
    assert capture.post_call_counts == [1, 1, 1]
    _assert_no_hooks(observed)


def test_hooks_reject_skipped_repeated_or_reordered_calls_and_cleanup():
    model = ThreeBlockModel().eval()
    capture = events.CrossAttentionEventCapture(model.c_layers, _specs(), embedding_dim=4)
    with capture:
        capture.begin_batch(1)
        one = torch.zeros((1, 1, 4))
        with pytest.raises(events.EventExtractionError, match="call order"):
            model.c_layers[1](one.clone(), one.clone())
    _assert_no_hooks(model)

    with events.CrossAttentionEventCapture(model.c_layers, _specs(), embedding_dim=4) as capture:
        capture.begin_batch(1)
        one = torch.zeros((1, 1, 4))
        model.c_layers[0](one.clone(), one.clone())
        with pytest.raises(events.EventExtractionError, match="call order"):
            model.c_layers[0](one.clone(), one.clone())
    _assert_no_hooks(model)

    with events.CrossAttentionEventCapture(model.c_layers, _specs(), embedding_dim=4) as capture:
        capture.begin_batch(1)
        one = torch.zeros((1, 1, 4))
        model.c_layers[0](one.clone(), one.clone())
        with pytest.raises(events.EventExtractionError, match="once in c_layers order"):
            capture.finish_batch()


def test_hooks_validate_tensor_shapes_and_tuple_outputs():
    model = ThreeBlockModel().eval()
    capture = events.CrossAttentionEventCapture(model.c_layers, _specs(), embedding_dim=4)
    with capture:
        capture.begin_batch(2)
        with pytest.raises(events.EventExtractionError, match="batch size"):
            model.c_layers[0](torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    _assert_no_hooks(model)

    class BadOutput(torch.nn.Module):
        def forward(self, text, vision):
            return text

    bad_layers = torch.nn.ModuleList([BadOutput(), AddBlock(1), AddBlock(1)])
    capture = events.CrossAttentionEventCapture(bad_layers, _specs(), embedding_dim=4)
    with capture:
        capture.begin_batch(1)
        with pytest.raises(events.EventExtractionError, match="tensor tuple"):
            bad_layers[0](torch.zeros(1, 1, 4), torch.zeros(1, 1, 4))
    assert not bad_layers[0]._forward_pre_hooks
    assert not bad_layers[0]._forward_hooks


def test_hooks_are_removed_when_model_forward_raises():
    class FailureBlock(torch.nn.Module):
        def forward(self, text, vision):
            raise RuntimeError("intentional test failure")

    model = ThreeBlockModel().eval()
    model.c_layers[1] = FailureBlock()
    capture = events.CrossAttentionEventCapture(model.c_layers, _specs(), embedding_dim=4)
    with pytest.raises(RuntimeError, match="intentional"):
        with capture:
            capture.begin_batch(1)
            one = torch.zeros((1, 1, 4))
            model.run_states(one, one.clone())
    _assert_no_hooks(model)


class TinyTokenizer:
    def __call__(self, texts, **_kwargs):
        values = torch.tensor([int(value) for value in texts], dtype=torch.long)
        ids = values[:, None].repeat(1, 2)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class TinyValidationModel(ThreeBlockModel):
    def forward(
        self,
        *,
        text_input_ids,
        image_pixel_values,
        save_intermediate_representations,
        **_kwargs,
    ):
        assert save_intermediate_representations is True
        text = text_input_ids[:, :1].float().unsqueeze(-1).expand(-1, 1, 4).clone()
        vision = image_pixel_values[:, :1].float().unsqueeze(-1).expand(-1, 1, 4).clone()
        text, vision = self.run_states(text, vision)
        return text[:, 0], vision[:, 0], [None]


def test_extraction_batches_500_rows_in_eval_mode_and_counts_each_hook_once():
    model = TinyValidationModel().eval()
    records = [SimpleNamespace(row=str(index)) for index in range(500)]
    extracted = events.extract_event_features(
        model=model,
        tokenizer=TinyTokenizer(),
        records=records,
        sample_ids=[f"sample-{index}" for index in range(500)],
        text_loader=lambda record: record.row,
        image_loader=lambda record: torch.tensor([int(record.row)]),
        batch_size=2,
        device="cpu",
        max_length=4,
        event_specs=_specs(),
        embedding_dim=4,
    )
    assert model.training is False
    assert extracted["pre_cls"].shape == (500, 3, 2, 4)
    assert extracted["post_cls"].shape == (500, 3, 2, 4)
    assert extracted["final_text_cls"].shape == (500, 4)
    assert extracted["final_vision_cls"].shape == (500, 4)
    assert extracted["hook_call_counts"]["2"] == {
        "pre_hook_calls": 250,
        "post_hook_calls": 250,
        "expected_calls": 250,
    }
    _assert_no_hooks(model)


def test_control_compares_event_pre_states_to_preceding_cached_layer():
    specs = _specs()
    rows, width = 2, 4
    text_cache = np.zeros((rows, 12, width), dtype=np.float32)
    vision_cache = np.zeros_like(text_cache)
    for layer in range(12):
        text_cache[:, layer] = layer + 0.25
        vision_cache[:, layer] = layer + 0.5
    pre = np.zeros((rows, 3, 2, width), dtype=np.float32)
    for spec in specs:
        pre[:, spec.ordinal_id, 0] = text_cache[:, spec.text_insertion_layer_id - 1]
        pre[:, spec.ordinal_id, 1] = vision_cache[:, spec.vision_insertion_layer_id - 1]
    checked = events.validate_capture_controls(
        event_pre_cls=pre,
        final_text_cls=text_cache[:, -1],
        final_vision_cls=vision_cache[:, -1],
        reference_text_features=text_cache,
        reference_vision_features=vision_cache,
        event_specs=specs,
    )
    assert checked["all_controls_passed"]
    assert checked["pre_event_inputs_vs_backbone_cache"]["0"]["text"]["reference_backbone_layer_id"] == 2
    assert checked["pre_event_inputs_vs_backbone_cache"]["0"]["vision"]["reference_backbone_layer_id"] == 4
    pre[0, 0, 0, 0] += 1
    with pytest.raises(events.EventExtractionError, match="did not match"):
        events.validate_capture_controls(
            event_pre_cls=pre,
            final_text_cls=text_cache[:, -1],
            final_vision_cls=vision_cache[:, -1],
            reference_text_features=text_cache,
            reference_vision_features=vision_cache,
            event_specs=specs,
        )


def test_manifest_protocol_rejects_test_or_nonfrozen_rows():
    ids = [f"hm:{index}" for index in range(500)]
    manifest = {
        "task": "hateful_memes",
        "split": "dev_seen",
        "validation_role": "validation_discovery_seen_historical",
        "ordered_sample_ids": ids,
        "selection": {"n_samples": 500},
        "samples": [{"id": sample_id} for sample_id in ids],
    }
    assert events.validate_hm_manifest_protocol(manifest) == ids
    with pytest.raises(events.EventExtractionError, match="dev_seen"):
        events.validate_hm_manifest_protocol({**manifest, "split": "test"})
    with pytest.raises(events.EventExtractionError, match="frozen 500-row"):
        events.validate_hm_manifest_protocol({**manifest, "ordered_sample_ids": ids[:-1]})


def test_output_destination_allows_only_runner_stdout_and_rejects_result_or_symlink(tmp_path):
    authorized = tmp_path / "autoresearch"
    authorized.mkdir()
    data_root = tmp_path / "dataset"
    data_root.mkdir()
    target = authorized / "event-analysis" / "df1"
    target.mkdir(parents=True)
    (target / "stdout.log").write_text("runner log\n")
    assert events.validate_output_destination(
        target, authorized_root=authorized, data_root=data_root
    ) == target.resolve()
    (target / "events.npz").write_bytes(b"old")
    with pytest.raises(events.EventExtractionError, match="only the runner"):
        events.validate_output_destination(target, authorized_root=authorized, data_root=data_root)

    outside = tmp_path / "outside"
    outside.mkdir()
    linked = authorized / "event-analysis" / "df2"
    linked.symlink_to(outside, target_is_directory=True)
    with pytest.raises(events.EventExtractionError):
        events.validate_output_destination(linked, authorized_root=authorized, data_root=data_root)


def _canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode()
    ).hexdigest()


def _write_reference_bundle(root: Path, *, corrupt_identity_hash=False):
    import autoresearch_events as event_module

    auto_root = root / "autoresearch"
    result_dir = auto_root / "trials" / "df1" / "results"
    cache_dir = auto_root / "trials" / "df1" / "features"
    result_dir.mkdir(parents=True)
    cache_dir.mkdir(parents=True)
    data_root = root / "data"
    data_root.mkdir()
    checkpoint = root / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint identity")
    checkpoint_hash = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    ids = [f"hm:{index}" for index in range(500)]
    pair_hashes = {sample_id: f"pair-{index}" for index, sample_id in enumerate(ids)}
    source_hashes = {"dev_seen.jsonl": "source-hash"}
    manifest_hash = "manifest-hash"
    tokenizer = {"requested_path": "/fixed/tokenizer", "vocab_sha256": "vocab"}
    code_hashes = {relative: f"hash-{index}" for index, relative in enumerate(event_module.REFERENCE_CODE_FILES)}
    identity = {
        "checkpoint_sha256": checkpoint_hash,
        "manifest_sha256": manifest_hash,
        "tokenizer_identity": tokenizer,
        "code_sha256": code_hashes,
    }
    identity_hash = _canonical_hash(identity)
    metadata = {
        "cache_identity": identity,
        "cache_identity_sha256": "wrong-hash" if corrupt_identity_hash else identity_hash,
        "ordered_sample_ids": ids,
        "sample_pair_hashes": pair_hashes,
        "tokenizer_identity": tokenizer,
        "data_manifest_sha256": manifest_hash,
        "source_data_sha256": source_hashes,
        "data_root": str(data_root),
        "task": "hateful_memes",
        "split": "dev_seen",
        "validation_role": "validation_discovery_seen_historical",
        "image_preprocessing": event_module.IMAGE_PREPROCESSING_ID,
        "text_max_length": 197,
        "code_sha256": code_hashes,
        "extraction_version": "vilbert-cls-layer-pair-v1",
    }
    cache_path = cache_dir / "features-test.npz"
    np.savez(
        cache_path,
        text_features=np.zeros((500, 12, 768), dtype=np.float32),
        vision_features=np.zeros((500, 12, 768), dtype=np.float32),
        layer_ids=np.arange(12, dtype=np.int64),
        sample_ids=np.asarray(ids, dtype="U16"),
        metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
    )
    cache_hash = hashlib.sha256(cache_path.read_bytes()).hexdigest()
    result_path = result_dir / "result-test.json"
    result = {
        "format": "autoresearch-checkpoint-validation-result",
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_hash},
        "data": {
            "task": "hateful_memes",
            "split": "dev_seen",
            "validation_role": "validation_discovery_seen_historical",
            "data_root": str(data_root),
            "sample_manifest_sha256": manifest_hash,
            "source_data_sha256": source_hashes,
        },
        "ordered_sample_ids": ids,
        "analysis_identity": {
            "checkpoint_sha256": checkpoint_hash,
            "manifest_sha256": manifest_hash,
            "tokenizer_identity": tokenizer,
            "code_sha256": code_hashes,
        },
        "tokenizer": tokenizer,
        "features": {"path": str(cache_path), "sha256": cache_hash},
    }
    result_path.write_text(json.dumps(result), encoding="utf-8")
    return {
        "result_path": result_path,
        "auto_root": auto_root,
        "data_root": data_root,
        "checkpoint": checkpoint,
        "checkpoint_hash": checkpoint_hash,
        "ids": ids,
        "pair_hashes": pair_hashes,
        "source_hashes": source_hashes,
        "manifest_hash": manifest_hash,
        "tokenizer": tokenizer,
        "code_hashes": code_hashes,
    }


def test_reference_bundle_validates_canonical_cache_identity_hash(tmp_path):
    bundle = _write_reference_bundle(tmp_path)
    result = events.validate_reference_bundle(
        bundle["result_path"],
        project_root=tmp_path,
        authorized_root=bundle["auto_root"],
        data_root=bundle["data_root"],
        expected_model="df1",
        checkpoint_path=bundle["checkpoint"],
        checkpoint_sha256=bundle["checkpoint_hash"],
        manifest_sha256=bundle["manifest_hash"],
        source_data_sha256=bundle["source_hashes"],
        ordered_sample_ids=bundle["ids"],
        sample_pair_hashes=bundle["pair_hashes"],
        tokenizer_identity=bundle["tokenizer"],
        image_preprocessing=events.IMAGE_PREPROCESSING_ID,
        text_max_length=197,
        current_code_sha256=bundle["code_hashes"],
    )
    assert result["text_features"].shape == (500, 12, 768)
    assert result["cache_identity_sha256"]


def test_reference_bundle_rejects_tampered_canonical_cache_identity_hash(tmp_path):
    bundle = _write_reference_bundle(tmp_path, corrupt_identity_hash=True)
    with pytest.raises(events.EventExtractionError, match="identity hash"):
        events.validate_reference_bundle(
            bundle["result_path"],
            project_root=tmp_path,
            authorized_root=bundle["auto_root"],
            data_root=bundle["data_root"],
            expected_model="df1",
            checkpoint_path=bundle["checkpoint"],
            checkpoint_sha256=bundle["checkpoint_hash"],
            manifest_sha256=bundle["manifest_hash"],
            source_data_sha256=bundle["source_hashes"],
            ordered_sample_ids=bundle["ids"],
            sample_pair_hashes=bundle["pair_hashes"],
            tokenizer_identity=bundle["tokenizer"],
            image_preprocessing=events.IMAGE_PREPROCESSING_ID,
            text_max_length=197,
            current_code_sha256=bundle["code_hashes"],
        )


def test_cli_requires_all_boundaries_and_explicit_device_batch():
    parser = events.build_parser()
    required = [
        "--checkpoint",
        "--data-root",
        "--manifest",
        "--output-dir",
        "--tokenizer-path",
        "--reference-result",
        "--device",
        "--batch-size",
    ]
    with pytest.raises(SystemExit):
        parser.parse_args([])
    parsed = parser.parse_args(
        [
            "--checkpoint", "/checkpoint",
            "--data-root", "/data",
            "--manifest", "/manifest",
            "--output-dir", "/output/df1",
            "--tokenizer-path", "/tokenizer",
            "--reference-result", "/reference.json",
            "--device", "cuda:0",
            "--batch-size", "2",
        ]
    )
    assert parsed.device == "cuda:0"
    assert parsed.batch_size == 2
    assert len(required) == 8

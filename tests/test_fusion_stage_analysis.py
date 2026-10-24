"""Focused CPU controls for the saved fusion-stage analysis."""
from __future__ import annotations

import importlib.util
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest


SOURCE = Path(__file__).resolve().parents[1] / "autoresearch/analyze_fusion_stages.py"
SPEC = importlib.util.spec_from_file_location("autoresearch_fusion_stage_analysis", SOURCE)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)

LEGACY_SOURCE = Path(__file__).resolve().parents[1] / "autoresearch/event_summary.py"
LEGACY_SPEC = importlib.util.spec_from_file_location("legacy_autoresearch_event_summary", LEGACY_SOURCE)
assert LEGACY_SPEC is not None and LEGACY_SPEC.loader is not None
LEGACY = importlib.util.module_from_spec(LEGACY_SPEC)
sys.modules[LEGACY_SPEC.name] = LEGACY
LEGACY_SPEC.loader.exec_module(LEGACY)


def test_centered_participation_ratio_has_known_rank_and_explicit_degeneracy():
    equal_axes = np.asarray(
        [[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]],
        dtype=np.float64,
    )
    assert MODULE.participation_ratio(equal_axes) == {"value": pytest.approx(2.0), "status": "ok"}

    constant = np.ones((8, 4), dtype=np.float64)
    result = MODULE.participation_ratio(constant)
    assert result["value"] is None
    assert result["status"] == "zero_variance"


def test_row_normalized_linear_cka_and_shared_permutation_schedule_are_deterministic():
    rng = np.random.default_rng(1804)
    features = rng.normal(size=(24, 7))
    first_permutations = MODULE.make_permutations(24, 37, seed=1843)
    second_permutations = MODULE.make_permutations(24, 37, seed=1843)
    np.testing.assert_array_equal(first_permutations, second_permutations)

    first = MODULE.calibrated_linear_cka(features, features.copy(), first_permutations)
    second = MODULE.calibrated_linear_cka(features, features.copy(), second_permutations)
    assert first == second
    assert first["status"] == "ok"
    assert first["raw"] == pytest.approx(1.0, abs=1e-12)
    assert first["null_mean"] is not None
    assert first["null_sd"] is not None
    assert first["null_q025"] is not None
    assert first["null_q975"] is not None
    assert first["corrected"] == pytest.approx(1.0, abs=1e-12)


def test_ordered_sample_id_hashes_identify_the_encoding_convention():
    sample_ids = ["hm-0001", "hm-0002", "hm-0003"]
    hashes = MODULE._ordered_sample_id_hashes(sample_ids)
    expected_canonical = MODULE._canonical_sha256(sample_ids)
    expected_newline = hashlib.sha256("hm-0001\nhm-0002\nhm-0003\n".encode("utf-8")).hexdigest()
    assert hashes == {
        "canonical_json_sha256": expected_canonical,
        "newline_delimited_trailing_lf_sha256": expected_newline,
    }
    assert hashes["canonical_json_sha256"] != hashes["newline_delimited_trailing_lf_sha256"]


def test_zero_variance_cka_is_kept_as_invalid_instead_of_fabricating_a_score():
    permutations = MODULE.make_permutations(9, 11, seed=3)
    constant = np.ones((9, 4), dtype=np.float64)
    result = MODULE.calibrated_linear_cka(constant, constant, permutations)
    assert result["raw"] is None
    assert result["corrected"] is None
    assert result["status"] == "zero_variance"


def test_extractor_source_hashes_must_match_the_local_frozen_code(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    sources = (
        "src/autoresearch_fusion_stages.py",
        "src/autoresearch_task_reliance.py",
        "autoresearch/fusion_stages_protocol.md",
    )
    hashes = {}
    for index, relative in enumerate(sources):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"source {index}", encoding="utf-8")
        hashes[relative] = MODULE._file_sha256(path)
    result_path = tmp_path / "result.json"
    assert MODULE._validate_source_hashes({"code_sha256": hashes}, result_path) == hashes

    (tmp_path / sources[0]).write_text("changed source", encoding="utf-8")
    with pytest.raises(ValueError, match="source-code hash mismatch"):
        MODULE._validate_source_hashes({"code_sha256": hashes}, result_path)


def test_legacy_raw_parity_metrics_match_the_previous_event_summary_formulas():
    rng = np.random.default_rng(418)
    text = rng.normal(size=(31, 12)).astype(np.float32)
    vision = rng.normal(size=(31, 12)).astype(np.float32)
    assert MODULE._legacy_raw_participation_ratio(text) == LEGACY._participation_ratio(text)
    assert MODULE._legacy_raw_participation_ratio(vision) == LEGACY._participation_ratio(vision)
    assert MODULE._legacy_raw_linear_cka(text, vision) == pytest.approx(
        LEGACY._cka(text, vision)["raw"], abs=1e-15
    )


def test_stage_identity_and_projected_update_are_separate_from_main_trajectory():
    rng = np.random.default_rng(71)
    n_samples, n_features = 20, 6
    text = rng.normal(size=(3, 5, n_samples, n_features)).astype(np.float32)
    vision = rng.normal(size=(3, 5, n_samples, n_features)).astype(np.float32)
    # Give the event-1 projected update recognizable values; it is summarized
    # for scale/cosine but excluded from the four-state representation plot.
    text[:, 1] = 0.25
    vision[:, 1] = 0.5
    update_summaries = {
        "projected_update_full_token_rms": np.full((n_samples, 3, 2), 0.4, dtype=np.float32),
        "projected_update_to_pre_event_norm_ratio": np.full((n_samples, 3, 2), 0.2, dtype=np.float32),
        "projected_update_cls_to_pre_event_cls_norm_ratio": np.full((n_samples, 3, 2), 0.3, dtype=np.float32),
        "projected_update_cls_cosine_to_pre_event_cls": np.full((n_samples, 3, 2), 0.1, dtype=np.float32),
    }
    rows = MODULE.analyze_feature_stack(
        text,
        vision,
        MODULE.make_permutations(n_samples, 17, seed=9),
        architecture="df1",
        training_status="pretrained",
        text_layer_ids=(3, 4, 5),
        vision_layer_ids=(5, 6, 7),
        update_summaries=update_summaries,
    )

    assert len(rows) == 15
    first_event = rows[:5]
    assert [row["state_name"] for row in first_event] == list(MODULE.STATE_NAMES)
    assert [row["main_plot_state"] for row in first_event] == [True, False, True, True, True]
    update = first_event[1]
    assert update["cka_status"] == "not_scored_projected_update"
    assert update["text_pr_status"] == "not_scored_projected_update"
    assert update["vision_update_full_token_rms_mean"] == pytest.approx(0.4)
    assert update["text_update_norm_ratio_mean"] == pytest.approx(0.2)
    assert update["vision_update_cls_cosine_mean"] == pytest.approx(0.1)
    assert update["text_update_derived_cls_cosine_valid_n"] == 20
    assert first_event[0]["text_layer_id"] == 3
    assert first_event[0]["vision_layer_id"] == 5
    assert first_event[0]["event_ordinal_id"] == 0


def test_extractor_axis_conversion_preserves_sample_event_and_stream_identity():
    n_samples, n_events, n_features = 4, 3, 2
    arrays = {
        name: np.empty((n_samples, n_events, 2, n_features), dtype=np.float32)
        for name in MODULE.STAGE_ARRAY_NAMES
    }
    for stage_index, name in enumerate(MODULE.STAGE_ARRAY_NAMES):
        for sample in range(n_samples):
            for event in range(n_events):
                for stream in range(2):
                    arrays[name][sample, event, stream] = (
                        1000 * stage_index + 100 * event + 10 * stream + sample
                    )

    text, vision = MODULE._stage_arrays_to_features(arrays)
    assert text.shape == vision.shape == (3, 5, n_samples, n_features)
    assert text[2, 4, 3, 0] == pytest.approx(4203.0)
    assert vision[1, 2, 0, 0] == pytest.approx(2110.0)


def test_wrong_schedule_or_nonfinite_features_fail_closed():
    features = np.ones((3, 5, 5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="text layer IDs"):
        MODULE.analyze_feature_stack(
            features,
            features,
            MODULE.make_permutations(5, 3, seed=1),
            architecture="df1",
            training_status="pretrained",
            text_layer_ids=(2, 3, 4),
            vision_layer_ids=(5, 6, 7),
        )

    invalid = features.copy()
    invalid[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite real values"):
        MODULE.analyze_feature_stack(
            invalid,
            features,
            MODULE.make_permutations(5, 3, seed=1),
            architecture="df1",
            training_status="pretrained",
            text_layer_ids=(3, 4, 5),
            vision_layer_ids=(5, 6, 7),
        )


def test_frozen_extractor_contract_writes_all_cells_reports_and_separate_pt_ft_figures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)
    input_dir = tmp_path / "autoresearch" / "fusion-stage-analysis" / "synthetic"
    input_dir.mkdir(parents=True)
    source_hashes = {}
    for relative, content in (
        ("src/autoresearch_fusion_stages.py", b"synthetic extractor source"),
        ("src/autoresearch_task_reliance.py", b"synthetic task source"),
        ("autoresearch/fusion_stages_protocol.md", b"synthetic protocol"),
    ):
        source_path = tmp_path / relative
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_bytes(content)
        source_hashes[relative] = MODULE._file_sha256(source_path)
    sample_ids = [f"hm-{index:03d}" for index in range(500)]
    row_labels = np.arange(500, dtype=np.int64) % 2
    image_groups = np.asarray([f"image-{index:03d}" for index in range(500)], dtype="U80")
    text_groups = np.asarray([f"caption-{index:03d}" for index in range(500)], dtype="U80")
    manifest_hash = "fixed-manifest-hash"
    event_records = []
    legacy_features = None
    for ordinal in range(3):
        event_records.append(
            {
                "ordinal_id": ordinal,
                "text_insertion_layer_id": 3 + ordinal,
                "vision_insertion_layer_id": 5 + ordinal,
            }
        )
    manifest = {
        "schema": MODULE.SCHEMA,
        "cohort": {"dataset": "hateful_memes", "split": "dev_seen", "sample_count": 500},
        "checkpoints": [],
    }
    for architecture in MODULE.ARCHITECTURE_ORDER:
        architecture_ids = MODULE.ARCHITECTURES[architecture]
        for status in MODULE.TRAINING_STATUS_ORDER:
            if architecture == "rev":
                text_ids, vision_ids = (9, 10, 11), (3, 4, 5)
            else:
                text_ids = (3, 4, 5)
                vision_ids = architecture_ids["vision_layers"]
            expected_events = [
                {
                    "ordinal_id": ordinal,
                    "text_insertion_layer_id": int(text_ids[ordinal]),
                    "vision_insertion_layer_id": int(vision_ids[ordinal]),
                }
                for ordinal in range(3)
            ]
            metadata = {
                "format": "autoresearch-hm-fusion-stages-v1",
                "axis_order": ["sample", "event_ordinal", "stream_text_then_vision", "feature"],
                "event_mapping": expected_events,
                "ordered_sample_ids": sample_ids,
                "checkpoint_sha256": f"checkpoint-{architecture}-{status}",
                "checkpoint_mode": "pretrain" if status == "pretrained" else "hm-finetuned",
                "model_id": architecture,
                "manifest_sha256": manifest_hash,
            }
            npz_path = input_dir / f"{architecture}-{status}.npz"
            arrays = {
                name: np.ones((500, 3, 2, 768), dtype=np.float32)
                for name in MODULE.STAGE_ARRAY_NAMES
            }
            arrays.update(
                {
                    "projected_update_full_token_rms": np.full((500, 3, 2), 0.2, dtype=np.float32),
                    "projected_update_to_pre_event_norm_ratio": np.full((500, 3, 2), 0.1, dtype=np.float32),
                    "projected_update_cls_to_pre_event_cls_norm_ratio": np.full((500, 3, 2), 0.3, dtype=np.float32),
                    "projected_update_cls_cosine_to_pre_event_cls": np.full((500, 3, 2), 0.4, dtype=np.float32),
                    "ordered_sample_ids": np.asarray(sample_ids, dtype="U64"),
                    "row_labels": row_labels,
                    "image_group_ids": image_groups,
                    "text_group_ids": text_groups,
                    "event_ordinal_ids": np.arange(3, dtype=np.int64),
                    "text_insertion_layer_ids": np.asarray(text_ids, dtype=np.int64),
                    "vision_insertion_layer_ids": np.asarray(vision_ids, dtype=np.int64),
                    "event_stage_names": np.asarray(MODULE.STAGE_ARRAY_NAMES, dtype="U32"),
                    "stream_names": np.asarray(("text", "vision"), dtype="U8"),
                    "metadata_json": np.asarray(json.dumps(metadata)),
                }
            )
            if architecture == "df1" and status == "pretrained":
                legacy_features = {
                    name: arrays[name].copy()
                    for name in ("pre_cls", "post_cls", "ordered_sample_ids")
                }
            np.savez_compressed(npz_path, **arrays)
            result_path = input_dir / f"{architecture}-{status}-result.json"
            result = {
                "status": "complete",
                "model_id": architecture,
                "checkpoint_mode": "pretrain" if status == "pretrained" else "hm-finetuned",
                "code_sha256": source_hashes,
                "checkpoint": {"sha256": metadata["checkpoint_sha256"]},
                "cohort": {
                    "task": "hateful_memes",
                    "split": "dev_seen",
                    "n": 500,
                    "manifest_sha256": manifest_hash,
                    "ordered_sample_ids": sample_ids,
                },
                "event_mapping": expected_events,
                "npz": {
                    "path": str(npz_path),
                    "sha256": MODULE._file_sha256(npz_path),
                },
            }
            result_path.write_text(json.dumps(result), encoding="utf-8")
            manifest["checkpoints"].append(
                {
                    "architecture": architecture,
                    "training_status": status,
                    "path": str(npz_path),
                    "result_path": str(result_path),
                }
            )

    assert legacy_features is not None
    legacy_npz_path = tmp_path / "autoresearch" / "event-analysis" / "df1" / "events.npz"
    legacy_npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(legacy_npz_path, **legacy_features)
    legacy_events = []
    for ordinal in range(3):
        stage_metrics = {}
        for stage_key, summary_key in (("pre_cls", "pre"), ("post_cls", "post")):
            values = legacy_features[stage_key][:, ordinal, :, :]
            stage_metrics[summary_key] = {
                "text_participation_ratio": MODULE._legacy_raw_participation_ratio(values[:, 0, :]),
                "vision_participation_ratio": MODULE._legacy_raw_participation_ratio(values[:, 1, :]),
                "linear_cka": {"raw": MODULE._legacy_raw_linear_cka(values[:, 0, :], values[:, 1, :])},
            }
        legacy_events.append(
            {
                "ordinal_id": ordinal,
                "text_insertion_layer_id": 3 + ordinal,
                "vision_insertion_layer_id": 5 + ordinal,
                **stage_metrics,
            }
        )
    legacy_summary_path = tmp_path / "autoresearch" / "event_summary.json"
    legacy_summary_path.write_text(
        json.dumps(
            {
                "status": "complete",
                "models": {
                    "df1": {
                        "status": "complete",
                        "checkpoint_sha256": "checkpoint-df1-pretrained",
                        "event_npz_path": str(legacy_npz_path.relative_to(tmp_path)),
                        "event_npz_sha256": MODULE._file_sha256(legacy_npz_path),
                        "events": legacy_events,
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    manifest_path = tmp_path / "autoresearch" / "fusion-stage-analysis" / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    output_dir = tmp_path / "autoresearch" / "fusion-stages-analysis"
    report = MODULE.analyze_manifest(
        manifest_path,
        output_dir,
        n_permutations=13,
        permutation_seed=1843,
    )

    assert len(report["cells"]) == 8 * 3 * 5
    assert report["analysis_code"]["path"] == "autoresearch/analyze_fusion_stages.py"
    assert len(report["analysis_code"]["sha256"]) == 64
    assert report["measurement_definitions"]["null_count"] == 13
    assert report["cells"][0]["cka_status"] == "zero_variance"
    assert report["cells"][1]["cka_status"] == "not_scored_projected_update"
    assert report["cells"][1]["text_update_full_token_rms_mean"] == pytest.approx(0.2)
    assert report["inputs"]["checkpoints"][0]["code_sha256"] == source_hashes
    assert report["inputs"]["ordered_sample_ids_hashes"] == MODULE._ordered_sample_id_hashes(sample_ids)
    assert "trailing LF" in report["inputs"]["ordered_sample_ids_hash_conventions"]["newline_delimited_trailing_lf_sha256"]
    assert report["inputs"]["legacy_df1_pretrain_parity"]["status"] == "passed"
    assert len(report["outputs"]) == 12
    assert all((tmp_path / path).is_file() for path in report["outputs"])
    with (output_dir / "fusion_stage_comparison_table.csv").open(encoding="utf-8") as stream:
        comparison_rows = list(csv.DictReader(stream))
    assert len(comparison_rows) == 8 * 3
    assert "cka_corrected_full_block_minus_pre" in comparison_rows[0]
    assert "text_pr_full_block_minus_pre" in comparison_rows[0]
    assert "vision_pr_full_block_minus_pre" in comparison_rows[0]
    summary = (output_dir / "fusion_stage_summary.md").read_text(encoding="utf-8")
    assert "not a fresh holdout" in summary
    assert "DF1 (delayed vision)" in summary
    assert "REV (delayed text)" in summary
    assert "Δ full−pre" in summary

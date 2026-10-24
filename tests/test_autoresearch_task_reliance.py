from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from autoresearch_task_reliance import (
    DEFAULT_SEEDS,
    RelianceError,
    binary_metrics,
    evaluate_input_shuffles,
    fuse_cls_embeddings,
    make_derangement,
    validate_head_dimensions,
    validate_hm_manifest_protocol,
    validate_output_destination,
)


class FakeTokenizer:
    def __call__(self, texts, **_kwargs):
        ids = [int(text.rsplit("-", 1)[1]) for text in texts]
        tensor = torch.tensor(ids, dtype=torch.long).unsqueeze(1)
        return {
            "input_ids": tensor,
            "attention_mask": torch.ones_like(tensor),
            "token_type_ids": torch.zeros_like(tensor),
        }


class FakeModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(embedding_dim=1)
        self.fc = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.fc.weight.copy_(torch.tensor([[1.0, 10.0]]))
        self.seen_pairs = []
        self.train_modes = []

    def train(self, mode=True):
        self.train_modes.append(mode)
        return super().train(mode)

    def forward(self, *, text_input_ids, image_pixel_values, **_kwargs):
        text_cls = text_input_ids[:, :1].float()
        image_cls = image_pixel_values.reshape(len(image_pixel_values), -1)[:, :1]
        self.seen_pairs.extend(
            zip(text_cls[:, 0].tolist(), image_cls[:, 0].tolist())
        )
        return text_cls, image_cls


def _records(n=6):
    return [
        SimpleNamespace(
            sample_id=f"hm:{row}",
            payload={"text": f"text-{row}", "label": row % 2},
        )
        for row in range(n)
    ]


def _evaluate(model):
    records = _records()
    sample_ids = [record.sample_id for record in records]
    labels = [record.payload["label"] for record in records]
    return evaluate_input_shuffles(
        model=model,
        tokenizer=FakeTokenizer(),
        records=records,
        sample_ids=sample_ids,
        row_labels=labels,
        text_loader=lambda record: record.payload["text"],
        image_loader=lambda record: torch.tensor([int(record.sample_id.split(":")[1])]).float(),
        batch_size=2,
        device="cpu",
        max_length=4,
        fusion_method="concat",
        embedding_dim=1,
        seeds=DEFAULT_SEEDS,
    )


def test_derangements_have_no_fixed_points_and_are_seed_stable():
    for seed in DEFAULT_SEEDS:
        first, _ = make_derangement(50, seed)
        second, _ = make_derangement(50, seed)
        assert first == second
        assert all(row != donor for row, donor in enumerate(first))


def test_fake_model_uses_donors_and_original_row_labels():
    first_model = FakeModel()
    result = _evaluate(first_model)
    second_result = _evaluate(FakeModel())
    ids = [f"hm:{row}" for row in range(6)]
    labels = [row % 2 for row in range(6)]

    assert first_model.training is False
    assert first_model.train_modes and all(mode is False for mode in first_model.train_modes)
    assert result["derangements"] == second_result["derangements"]
    assert result["predictions"]["matched_input"]["row_labels"] == labels

    donor_rows = result["derangements"]["101"]["donor_row_indices"]
    image = result["predictions"]["image_shuffled_seed_101"]
    text = result["predictions"]["text_shuffled_seed_101"]
    joint = result["predictions"]["joint_pair_shuffle_seed_101"]
    assert image["sample_ids"] == ids
    assert image["row_labels"] == labels
    assert image["image_partner_ids"] == [ids[row] for row in donor_rows]
    assert image["text_partner_ids"] == ids
    assert text["image_partner_ids"] == ids
    assert text["text_partner_ids"] == [ids[row] for row in donor_rows]
    assert image["logits"] == [float(row + 10 * donor_rows[row]) for row in range(6)]
    assert text["logits"] == [float(donor_rows[row] + 10 * row) for row in range(6)]
    baseline = result["predictions"]["matched_input"]["logits"]
    assert joint["logits"] == [baseline[row] for row in donor_rows]
    assert joint["row_labels"] == labels


def test_cls_fusion_order_and_head_dimensions_are_checked():
    text = torch.tensor([[1.0, 2.0]])
    image = torch.tensor([[3.0, 4.0]])
    assert fuse_cls_embeddings(text, image, "concat").tolist() == [[1.0, 2.0, 3.0, 4.0]]
    assert fuse_cls_embeddings(text, image, "sum").tolist() == [[4.0, 6.0]]
    assert fuse_cls_embeddings(text, image, "hadamard").tolist() == [[3.0, 8.0]]
    assert validate_head_dimensions(FakeModel().fc, "concat", 1)["input_features"] == 2
    with pytest.raises(RelianceError, match="input width"):
        validate_head_dimensions(FakeModel().fc, "sum", 1)
    with pytest.raises(RelianceError, match="shapes must match"):
        fuse_cls_embeddings(text, image[:, :1], "concat")


def test_binary_metrics_use_fixed_probability_threshold():
    values = binary_metrics([0, 0, 1, 1], [-2.0, 1.0, -1.0, 3.0])
    assert values["auroc"] == pytest.approx(0.75)
    assert values["accuracy_at_0_5"] == pytest.approx(0.5)
    assert values["bce"] == pytest.approx(
        sum(max(logit, 0.0) - logit * label + torch.nn.functional.softplus(torch.tensor(-abs(logit))).item()
            for label, logit in zip([0, 0, 1, 1], [-2.0, 1.0, -1.0, 3.0])) / 4
    )


def test_manifest_rejects_test_split_and_fallback():
    base = {
        "task": "hateful_memes",
        "split": "dev_seen",
        "validation_role": "validation_discovery_seen_historical",
        "selection": {"n_samples": 2},
        "ordered_sample_ids": ["hm:one", "hm:two"],
    }
    assert validate_hm_manifest_protocol(base, requested_n=2) == ["hm:one", "hm:two"]
    for split in ("test", "train", "validation"):
        with pytest.raises(RelianceError, match="dev_seen"):
            validate_hm_manifest_protocol({**base, "split": split}, requested_n=2)


def test_output_validation_rejects_escape_and_input_overlap(tmp_path: Path):
    authorized = tmp_path / "autoresearch"
    authorized.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    checkpoint = authorized / "checkpoint.pt"
    checkpoint.write_bytes(b"checkpoint")

    with pytest.raises(RelianceError, match="stay below"):
        validate_output_destination(
            tmp_path / "outside",
            authorized_root=authorized,
            data_root=data_root,
        )
    with pytest.raises(RelianceError, match="overlaps an input"):
        validate_output_destination(
            authorized,
            authorized_root=authorized,
            data_root=data_root,
            input_paths=(checkpoint,),
        )


def test_output_validation_accepts_runner_stdout_only_directory(tmp_path: Path):
    authorized = tmp_path / "autoresearch"
    authorized.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner_dir = authorized / "model-run"
    runner_dir.mkdir()
    (runner_dir / "stdout.log").write_text("runner output\n", encoding="utf-8")

    assert validate_output_destination(
        runner_dir,
        authorized_root=authorized,
        data_root=data_root,
    ) == runner_dir.resolve()


def test_output_validation_rejects_preexisting_result(tmp_path: Path):
    authorized = tmp_path / "autoresearch"
    authorized.mkdir()
    data_root = tmp_path / "data"
    data_root.mkdir()
    runner_dir = authorized / "model-run"
    runner_dir.mkdir()
    (runner_dir / "stdout.log").write_text("runner output\n", encoding="utf-8")
    (runner_dir / "input-shuffle-diagnostic.json").write_text("{}\n", encoding="utf-8")

    with pytest.raises(RelianceError, match="only the runner's regular stdout.log"):
        validate_output_destination(
            runner_dir,
            authorized_root=authorized,
            data_root=data_root,
        )

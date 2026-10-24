from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from torch import nn

from trainer import PretrainingTrainer
from trainer.mlm_loss import mlm_projection_loss


def _dense_mlm_loss(hidden, labels, projection, loss_fn):
    logits = projection(hidden)
    return loss_fn(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))


def test_selected_projection_matches_dense_loss_and_gradients():
    torch.manual_seed(17)
    hidden = torch.randn(2, 5, 4)
    labels = torch.tensor([
        [1, -100, 3, -100, 0],
        [-100, 2, -100, 1, -100],
    ])
    dense_head = nn.Linear(4, 7)
    selected_head = nn.Linear(4, 7)
    selected_head.load_state_dict(dense_head.state_dict())
    dense_hidden = hidden.clone().requires_grad_()
    selected_hidden = hidden.clone().requires_grad_()
    loss_fn = nn.CrossEntropyLoss()

    dense_loss = _dense_mlm_loss(dense_hidden, labels, dense_head, loss_fn)
    selected_loss = mlm_projection_loss(
        selected_hidden,
        labels,
        selected_head,
        loss_fn,
        selected_tokens_only=True,
    )
    dense_loss.backward()
    selected_loss.backward()

    torch.testing.assert_close(selected_loss, dense_loss)
    torch.testing.assert_close(selected_hidden.grad, dense_hidden.grad)
    for dense_param, selected_param in zip(dense_head.parameters(), selected_head.parameters()):
        torch.testing.assert_close(selected_param.grad, dense_param.grad)


def test_selected_projection_handles_noncontiguous_hidden_and_labels():
    torch.manual_seed(23)
    hidden = torch.randn(5, 2, 4).transpose(0, 1)
    labels = torch.tensor([
        [1, -100],
        [2, 3],
        [-100, 4],
        [5, -100],
        [6, 1],
    ]).transpose(0, 1)
    dense_head = nn.Linear(4, 8)
    selected_head = nn.Linear(4, 8)
    selected_head.load_state_dict(dense_head.state_dict())
    dense_hidden = hidden.clone().requires_grad_()
    selected_hidden = hidden.clone().requires_grad_()
    loss_fn = nn.CrossEntropyLoss()

    dense_loss = _dense_mlm_loss(dense_hidden, labels, dense_head, loss_fn)
    selected_loss = mlm_projection_loss(
        selected_hidden,
        labels,
        selected_head,
        loss_fn,
        selected_tokens_only=True,
    )
    dense_loss.backward()
    selected_loss.backward()

    torch.testing.assert_close(selected_loss, dense_loss)
    torch.testing.assert_close(selected_hidden.grad, dense_hidden.grad)
    for dense_param, selected_param in zip(dense_head.parameters(), selected_head.parameters()):
        torch.testing.assert_close(selected_param.grad, dense_param.grad)


def test_all_ignored_labels_keep_dense_nan_and_zero_gradient_behavior():
    hidden = torch.randn(2, 3, 4)
    labels = torch.full((2, 3), -100)
    dense_head = nn.Linear(4, 5)
    selected_head = nn.Linear(4, 5)
    selected_head.load_state_dict(dense_head.state_dict())
    dense_hidden = hidden.clone().requires_grad_()
    selected_hidden = hidden.clone().requires_grad_()
    loss_fn = nn.CrossEntropyLoss()

    dense_loss = _dense_mlm_loss(dense_hidden, labels, dense_head, loss_fn)
    selected_loss = mlm_projection_loss(
        selected_hidden,
        labels,
        selected_head,
        loss_fn,
        selected_tokens_only=True,
    )
    dense_loss.backward()
    selected_loss.backward()

    assert torch.isnan(dense_loss)
    assert torch.isnan(selected_loss)
    torch.testing.assert_close(selected_hidden.grad, dense_hidden.grad)
    for dense_param, selected_param in zip(dense_head.parameters(), selected_head.parameters()):
        torch.testing.assert_close(selected_param.grad, dense_param.grad)
        assert torch.count_nonzero(selected_param.grad) == 0


class _RecordingLinear(nn.Linear):
    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.rows_seen = []

    def forward(self, inputs):
        self.rows_seen.append(inputs.numel() // inputs.shape[-1])
        return super().forward(inputs)


class _TinyPretrainingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(16, 4)
        self.mlm = _RecordingLinear(4, 9)

    def forward_pretrain(self, *, text_input_ids, **kwargs):
        return self.embedding(text_input_ids), None


def _mlm_batch():
    return {
        "task": torch.tensor([3, 3]),
        "text": {
            "input_ids": torch.tensor([[[1, 2, 3, 4]], [[5, 6, 7, 8]]]),
            "attention_mask": torch.ones(2, 1, 4, dtype=torch.long),
            "token_type_ids": torch.zeros(2, 1, 4, dtype=torch.long),
        },
        "img": {"pixel_values": torch.zeros(2, 1, 3, 2, 2)},
        "label": torch.tensor([[1, -100, 2, -100], [-100, 3, -100, 4]]),
    }


@pytest.mark.parametrize("selected_only, expected_rows", [(False, 8), (True, 4)])
@pytest.mark.parametrize("method", ["train", "evaluate"])
def test_trainer_routes_train_and_eval_through_selected_projection(
    selected_only, expected_rows, method
):
    model = _TinyPretrainingModel()
    config = SimpleNamespace(
        learning_rate=1e-3,
        mlm_selected_tokens_only=selected_only,
    )
    batch = _mlm_batch()

    # Constructing the trainer in this test is CPU-only even on GPU hosts.
    with patch("trainer.trainers.torch.cuda.is_available", return_value=False):
        trainer = PretrainingTrainer(
            model=model,
            config=config,
            use_contrastive_loss=False,
            gradient_accumulation=1,
        )

    if method == "train":
        loss = trainer.train_mlm_batch(batch)
    else:
        loss = trainer.evaluate_mlm([batch])

    assert torch.isfinite(torch.as_tensor(loss))
    assert model.mlm.rows_seen == [expected_rows]


@pytest.mark.parametrize("method", ["train", "evaluate"])
def test_trainer_keeps_dense_projection_when_flag_is_absent(method):
    model = _TinyPretrainingModel()
    config = SimpleNamespace(learning_rate=1e-3)
    with patch("trainer.trainers.torch.cuda.is_available", return_value=False):
        trainer = PretrainingTrainer(
            model=model,
            config=config,
            use_contrastive_loss=False,
            gradient_accumulation=1,
        )

    batch = _mlm_batch()
    if method == "train":
        with patch("trainer.trainers.torch.amp.autocast", side_effect=lambda **_: nullcontext()):
            trainer.train_mlm_batch(batch)
    else:
        trainer.evaluate_mlm([batch])

    assert model.mlm.rows_seen == [8]

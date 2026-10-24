import json
import statistics

import pytest
import torch

import scripts.benchmark_training_kernels as benchmark_module
from scripts.benchmark_training_kernels import (
    KernelShape,
    benchmark_attention,
    benchmark_mlm_projection,
    run_benchmark,
    validate_resources,
    write_output_create_only,
)


def _tiny_shape():
    return KernelShape(
        batch_size=2,
        attention_sequence_length=8,
        real_text_tokens=3,
        model_dimension=8,
        projected_dimension=12,
        heads=2,
        mlm_sequence_length=8,
        mlm_real_tokens=3,
        mlm_hidden_dimension=8,
        vocabulary_size=17,
    )


def _assert_timing_record(record):
    assert record["status"] == "complete"
    assert record["measured_iterations"] == 1
    assert record["warmup_iterations_excluded"] == 0
    assert len(record["iteration_seconds"]) == 1
    assert record["iteration_seconds"][0] >= 0
    assert record["median_seconds"] == statistics.median(record["iteration_seconds"])
    assert record["peak_allocated_bytes"] is None


def test_attention_eager_sdpa_smoke_is_finite_and_close():
    torch.set_num_threads(4)
    result = benchmark_attention(
        shape=_tiny_shape(),
        device=torch.device("cpu"),
        iterations=1,
        warmup=0,
        seed=19,
    )

    _assert_timing_record(result["timings"]["eager"])
    _assert_timing_record(result["timings"]["sdpa"])
    assert result["eval_parity"]["status"] == "passed"
    assert result["eval_parity"]["outputs_and_gradients_finite"] is True
    assert result["method"]["text_padding_mask"].startswith("additive float32 mask")
    assert result["method"]["shared_inputs"] is True
    assert result["method"]["shared_initial_weights"] is True


def test_dense_and_selected_mlm_smoke_is_finite_and_gradient_equivalent():
    torch.set_num_threads(4)
    result = benchmark_mlm_projection(
        shape=_tiny_shape(),
        device=torch.device("cpu"),
        iterations=1,
        warmup=0,
        seed=23,
    )

    _assert_timing_record(result["timings"]["dense"])
    _assert_timing_record(result["timings"]["selected_tokens"])
    assert result["loss_gradient_parity"]["status"] == "passed"
    assert result["loss_gradient_parity"]["outputs_and_gradients_finite"] is True
    assert result["method"]["selected_projection_rows"] < result["method"]["dense_projection_rows"]


def test_cpu_default_run_uses_smoke_shape_and_complete_metadata():
    torch.set_num_threads(4)
    result = run_benchmark(device_name="cpu", iterations=1, warmup=0, batch_size=2)

    assert result["status"] == "complete"
    assert result["smoke_shape_used"] is True
    assert result["shape"]["model_dimension"] < 768
    assert result["versions"]["torch_cpu_threads"] == 4
    assert result["source_sha256"]["src/attention.py"]
    assert result["source_sha256"]["src/trainer/mlm_loss.py"]
    assert result["benchmarks"]["cross_attention"]["eval_parity"]["status"] == "passed"


@pytest.mark.parametrize(
    ("iterations", "warmup", "batch_size", "timeout_seconds"),
    [
        (31, 0, 1, 180),
        (1, 31, 1, 180),
        (1, 0, 21, 180),
        (1, 0, 1, 181),
        (0, 0, 1, 180),
    ],
)
def test_resource_bounds_reject_oversized_runs(iterations, warmup, batch_size, timeout_seconds):
    with pytest.raises(ValueError):
        validate_resources(
            iterations=iterations,
            warmup=warmup,
            batch_size=batch_size,
            timeout_seconds=timeout_seconds,
        )


def test_output_is_bounded_and_create_only(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark_module, "ROOT", tmp_path)
    path = tmp_path / "benchmark.json"
    result = {"status": "complete", "value": 1}
    created = write_output_create_only(str(path), result)
    assert created == path
    assert path.stat().st_size < 1_000_000
    assert json.loads(path.read_text()) == result
    with pytest.raises(FileExistsError):
        write_output_create_only(str(path), result)


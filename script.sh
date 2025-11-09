#!/usr/bin/env bash
source venv310/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source venv310/bin/activate

# python src/experiment_tracker.py
# python src/metric_analysis.py
# python src/correlation_analysis2.py
# python src/pretraining_experiments.py
# python src/main.py
# python src/finetune_experiments.py
# python src/finetune_only_baseline.py
# python src/performance_metric_collection.py
python src/correlation_analysis3.py

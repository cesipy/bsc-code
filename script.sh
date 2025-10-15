#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
source venv310/bin/activate
# python src/main.py
# python src/experiment_tracker.py
# python src/metric_analysis.py
# python src/correlation_analysis.py
# python src/pretraining_experiments.py
python src/finetune_experiments.py
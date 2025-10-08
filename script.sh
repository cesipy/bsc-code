#!/usr/bin/env bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
python src/main.py
# python src/experiment_tracker.py
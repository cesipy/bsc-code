# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Bachelor thesis code: **Probing Layer-wise Alignment Depth in Multimodal Transformers**. A PyTorch implementation of ViLBERT that investigates representational alignment in a dual-stream (BERT + ViT) encoder-only architecture, focusing on cross-attention placement (Early / Middle / Late / Hybrid Fusion).

## Setup

```bash
pip install -r requirements.txt          # Python 3.10 recommended
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
activate-global-python-argcomplete       # tab completion for evaluate.py
```

## Key Commands

```bash
# Pretraining
python src/pretraining_experiments.py         # all fusion configs
python src/pretrain.py                         # single run (all tasks)
python src/pretrain.py --no-mim               # skip Masked Image Modeling
python src/pretrain.py --no-mlm               # skip Masked Language Modeling
python src/pretrain.py --no-ap                # skip Alignment Prediction

# Finetuning
python src/finetune_experiments.py            # multi-seed, all tasks
python src/finetune.py --task hateful_memes
python src/finetune.py --task hateful_memes --path res/checkpoints/pretrains/<ckpt>.pt

# Hyperparameter / NAS optimization
python src/hyperparameter_optimizer.py
optuna-dashboard sqlite:///res/hyperparameter_optimization/optuna_study.db

# Dataset download
python src/download_cc.py                     # downloads Conceptual Captions
```

## Architecture

**Dual-stream transformer** — two independent 12-layer transformer stacks that communicate via bidirectional cross-attention layers inserted at configurable positions:

| File | Role |
|---|---|
| `src/vilbert.py` | Main `ViLBERT` model; coordinates BERT (text) + timm ViT (vision) streams and injects cross-attention blocks |
| `src/attention.py` | Custom `Attention_Block`, `CrossAttention`, `CrossAttentionBlock`, `DualAttention_Block`, `FeedForward_Block` |
| `src/config.py` | All hyperparameters and `ViLBERTConfig` dataclass; detects `MACHINE_TYPE` env var (`local` vs `remote`) for batch sizes |
| `src/task.py` | `Task` enum: `ALIGNMENT_PREDICTION`, `MASKED_LM`, `MASKED_IM` |

**Cross-attention placement** is controlled by two index lists in `ViLBERTConfig`:
- `text_cross_attention_layers` (default `[6,7,8,9,10,11]`)
- `vision_cross_attention_layers` (default `[0,1,2,3,4,5]`)

Moving these indices earlier/later corresponds to Early/Middle/Late fusion.

**Training pipeline:**
- `src/pretrain.py` — pretraining loop (MLM + MIM + Alignment Prediction jointly)
- `src/finetune.py` — finetuning entry point; dispatches to task trainers
- `src/pretraining_experiments.py` / `src/finetune_experiments.py` — orchestrate full multi-seed experiment sweeps via `ExperimentTracker`
- `src/trainer/base_trainer.py` — abstract `BaseTrainer`; concrete implementations in `hm_trainer.py`, `mm_imdb_trainer.py`, `upmc_trainer.py`, `vqa_trainer.py`

**Analysis:**
- `src/analysis.py` — computes CKA, SVCCA, mkNN, Orthogonal Procrustes across layers; generates heatmaps
- `src/analyses/` — specialised correlation, dimensionality reduction, and metric evolution scripts
- `src/metrics.py` / `src/measures.py` — alignment metric implementations
- `src/probing.py` / `src/probing_analysis.py` — linear probing of intermediate representations

**Datasets** (all under `res/data/`):

| Task | Dataset | Path |
|---|---|---|
| Pretraining | Conceptual Captions | `res/data/conceptual-captions/` |
| Binary classification | Hateful Memes | `res/data/hateful_memes_data/` |
| Multi-label (23 genres) | MM-IMDB | `res/data/mm-imdb/` |
| 101-class | UPMC Food-101 | `res/data/UPMC_Food-101/` |

Dataset classes live in `src/datasets/`; they return dicts with `img`, `text`, `label` (and for pretraining also `masked_img`, `masked_patches_idxs`, `task`).

**Checkpoints** saved to `res/checkpoints/pretrains/`. `FINETUNE_CHECKPOINTS_DIR` in `config.py` points to the pretrained weights used for downstream evaluation.

## Environment

Set `MACHINE_TYPE=remote` on university GPU servers to use larger batch sizes (`BATCH_SIZE_PRETRAIN=20`, `BATCH_SIZE_DOWNSTREAM=24`). Locally it defaults to smaller batches.

The code auto-detects the GPU hostname (`c703i-gpuN`) to select appropriate batch sizes.

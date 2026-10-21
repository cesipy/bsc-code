#!/usr/bin/env bash
source .venv/bin/activate

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


# python src/main.py --fusion late --num-samples 500000  --pt-epochs 7 --ft-epochs 15 --alignment-analysis-size 1024

# launcher for pretraining + finetune with delayed fusion for vision.
# python src/main.py --t-ids 3 4 5 --v-ids 5 6 7 \
#   --num-samples 500000  --pt-epochs 7 --ft-epochs 15 \
#   --alignment-analysis-size 1024 \
#   --tasks "hateful_memes" "upmc_food" "mm_imdb"


# python src/main.py --t-ids 3 4 5 --v-ids 5 6 7 \
#     --pretrain-path res/checkpoints/pretrains/20261004-180351_pretrained_123.pt \
#     --name "finetune_delayed_fusion_1_seed_1567" \
#     --ft-epochs 15 \
#     --alignment-analysis-size 1024 \
#     --seed 1567 \
#     --tasks hateful_memes upmc_food mm_imdb

# python src/main.py --t-ids 3 4 5 --v-ids 5 6 7 \
#     --pretrain-path res/checkpoints/pretrains/20261004-180351_pretrained_123.pt \
#     --name "finetune_delayed_fusion_1_seed_1568" \
#     --ft-epochs 15 \
#     --alignment-analysis-size 1024 \
#     --seed 1568 \
#     --tasks hateful_memes upmc_food mm_imdb

# python src/main.py --t-ids 3 4 5 --v-ids 5 6 7 \
#     --pretrain-path res/checkpoints/pretrains/20261004-180351_pretrained_123.pt \
#     --name "finetune_delayed_fusion_1_seed_1569" \
#     --ft-epochs 15 \
#     --alignment-analysis-size 1024 \
#     --seed 1569 \
#     --tasks hateful_memes upmc_food mm_imdb


python src/main.py --v-ids 3 4 5 --t-ids 9 10 11 \
  --name "pt+ft_delayed-fusion-text" \
  --num-samples 500000  --pt-epochs 7 --ft-epochs 15 \
  --alignment-analysis-size 1024 \
  --tasks "hateful_memes" "upmc_food" "mm_imdb"

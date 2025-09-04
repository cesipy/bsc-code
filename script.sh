#!/usr/bin/env bash

python src/evaluate.py # path is none per default.
sleep 5s
python src/main.py       # all tasks
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch1_task123.pt
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch4_task123.pt

python src/evaluate.py --path res/checkpoints/pretrained_epoch1_task123.pt --use-constrastive
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch4_task123.pt --use-constrastive


# python src/evaluate.py # path is none per default.
sleep 5s
python src/main.py  --no-mim
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch1_task12.pt
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch4_task12.pt



# python src/evaluate.py # path is none per default.
sleep 5s
python src/main.py --no-mlm --no-mim      # all tasks
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch1_task1.pt
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_epoch4_task1.pt

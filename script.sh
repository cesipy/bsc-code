#!/usr/bin/env bash

python src/evaluate.py # path is none per default. 
sleep 5s
python src/main.py      # do the pretraining here
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_1.pt
sleep 5s
python src/evaluate.py --path res/checkpoints/pretrained_4.pt

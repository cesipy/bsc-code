# Code for bachelor thesis

This repository contains the code for my bachelor thesis. It is a PyTorch implementation of ViLBERT, a model that combines visual and language understanding.
I'm currently working on it, so it is not yet complete. 


## TODO
- [x] Tokenizer for text dependency injected
- [ ] caching , [mmap](https://github.com/DACUS1995/pytorch-mmap-dataset/blob/main/pytorch_mmap_dataset/dataset.py)
- [x] torch compile
- [ ] add dropout in attention
- [x] pretrain dataset fix: filter out images that are not working
- [ ] pretrain dataset mlm task
- [ ] pretrain dataset image task
- [ ] apparently there is a problem with the `transformers` library, where ViT implementation causes 10x-40x? https://x.com/jbohnslav/status/1950550831381782798, => own implementation of ViT (maybe adapt from dl VU, assignment 03)
- [ ] logging to keep track of my development
- [ ] visualization of pretraining tasks - like acc, loss, etc
    - [ ] implement accuracy for pretraining - handle specific tasks
- [x] fix problem with compile and saving


## ViLBERT
original [vilbert](https://github.com/facebookresearch/vilbert-multi-task) under `vilbert/vilbert.py`.

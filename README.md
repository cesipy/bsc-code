# Code for bachelor thesis


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

## ViLBERT
original [vilbert](https://github.com/facebookresearch/vilbert-multi-task) under `vilbert/vilbert.py`.

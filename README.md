# Code for bachelor thesis

This repository contains the code for my bachelor thesis. It is a PyTorch implementation of ViLBERT, a model that combines visual and language understanding.
I'm currently working on it, so it is not yet complete. 


## TODO
- [x] Tokenizer for text dependency injected
- [ ] caching , [mmap](https://github.com/DACUS1995/pytorch-mmap-dataset/blob/main/pytorch_mmap_dataset/dataset.py)
- [x] torch compile
- [ ] add dropout in attention
- [x] pretrain dataset fix: filter out images that are not working
- [x] pretrain dataset mlm task
- [x] pretrain dataset image task
- [ ] apparently there is a problem with the `transformers` library, where ViT implementation causes 10x-40x? https://x.com/jbohnslav/status/1950550831381782798, => own implementation of ViT (maybe adapt from dl VU, assignment 03)
- [x] logging to keep track of my development
- [ ] visualization of pretraining tasks - like acc, loss, etc
    - [x] implement accuracy for pretraining - handle specific tasks
- [x] fix problem with compile and saving


## ViLBERT
original [vilbert](https://github.com/facebookresearch/vilbert-multi-task) under `vilbert/vilbert.py`.


## Pretraining

There are three pretraining tasks in ViLBERT: Masked Language Modelling, Masked Image Modelling, Alignment Prediction

### MLM 
for 15% of all tokens: 
- 80% replaced with [MASK]
- 10% replaced with random token
- 10% unchanged

array of length of tokens is returned. if masked: masked token, if not `token[i] = -100`(value for lossfunction to ignore it)

### Alignment Prediction
predict if images and caption are aligned. Is a dataset of 50/50 balance. 
- "research has been focused on two main schemes, either reconstructing the masked signal, or comparing two latent representations, one for the unaltered input signal and one for the masked input."




### MIM

Two options: reconstruct masked patches, contrastive comparision of hidden representations. 

Basd on my research I go for the contrastice approach, as this seems more interesting for me to implement .

Mask patches of image

## Results
Running `train_and_eval_on_downstream_task` with randomly initialized cross-attentions and 4 epochs gives the following results. 
```
Epoch 1/4, train loss: 0.6354, test loss: 0.6107,  accuracy: 0.6829
Epoch 2/4, train loss: 0.5859, test loss: 0.5861,  accuracy: 0.6994
Epoch 3/4, train loss: 0.5509, test loss: 0.5803,  accuracy: 0.7076
Epoch 4/4, train loss: 0.5324, test loss: 0.5775,  accuracy: 0.7100
```


Then I pretrained on 200k subset from CC for 4 epochs. 
Afterwards I trained again for 4 epochs on the downstream task, which gives the following results:
```
# pretrain
Epoch 4/4, 
        train loss AP: 0.2628, 
        train loss MLM: 3.2051, 
        test loss AP: 0.2891, 
        test loss MLM: 3.1713, 
        accuracy AP: 0.8816
# finetune
Epoch 1/4, train loss: 0.6208, test loss: 0.5848,  accuracy: 0.6935
Epoch 2/4, train loss: 0.5514, test loss: 0.5547,  accuracy: 0.7118
Epoch 3/4, train loss: 0.5058, test loss: 0.5308,  accuracy: 0.7335
Epoch 4/4, train loss: 0.4704, test loss: 0.5292,  accuracy: 0.7371
```



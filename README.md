# Code for bachelor thesis

This repository contains the code for my bachelor thesis. It is a PyTorch implementation of ViLBERT, a model that combines visual and language understanding.
I'm currently working on it, so it is not yet complete.


## TODO
- [ ] add dropout in attention
- [ ] caching , [mmap](https://github.com/DACUS1995/pytorch-mmap-dataset/blob/main/pytorch_mmap_dataset/dataset.py)
- [ ] visualization of pretraining tasks - like acc, loss, etc
- [ ] different batchsizes for tasks
	- maybe too difficult to implement!
- [ ] is residual handling in crossattention correct?
- [ ] other datasets implement
	- [ ] find alignment datasets in literature

- [ ] data augmentation for AP pretraining
- [ ] implement further alignment measures
	- [x] cca
	- [ ] wasserstein distance
	- [x] svcca
	- [ ] sae (maybe)

- [ ] pytorch hooks for intermediate layers
	- quite hard to implement, plus there is not much documentation on this topic.

- [ ] fix problem with ap pretraining only - has really bad performance, slightly worse than guessing!
	- 2025-08-23 22:35:30 - INFO  - trainer.py:train:691 - Epoch 4/4,
	```
	train loss MLM: 0.0000,
	test loss MLM: 10.5104,
	train loss AP: 0.6946,
	test loss AP: 0.6946,
	accuracy AP: 0.4986
	train loss MIM: 0.0000,
	test loss MIM: 8.8669
	```

- [ ] investigating platonic representation hypothesis:
	- simply concat represetnations of bert + vit: use as baseline.
	



### past TODOs
- [x] Tokenizer for text dependency injected
- [x] pretrain dataset fix: filter out images that are not working
- [x] pretrain dataset mlm task
- [x] apparently there is a problem with the `transformers` library, where ViT implementation causes 10x-40x? https://x.com/jbohnslav/status/1950550831381782798, => own implementation of ViT (maybe adapt from dl VU, assignment 03)
- [x] fix problem with compile and saving

- [x] log everything
- [x] complete mim
    - [x] data augmentation pipeline.
    - [x] teacher, student ? this is to avoid moving target problem, but is it necessary? - not using this
    - [x] gradient stopping - not used, would require teacher-student setup

- [x] better config handling
- [x] infonce review

- [x] evaluate functino for measuring avg alignment measures.
- [x] complete pipeline for running experiments
- [x] hateful memes downsize to 224
- [x] unify the alignment measurements

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

Basd on my research I go for the contrastice approach, as this seems more interesting for me to implement.

<figure>
    <img src="./res/markdown_res/contrastive_mim.png" width=400>
</figure>

the workflow is the following:
1) augment the data. Not yes timplemented
2) mask image => (image, masked_image)
3) encode(masked_image); encode(image)
4) compute infoNCE on the representations, ONLY FOR UNMASKED tokens.


dataset returned from dataloader/dataset:
```python
{
    "task": task.value,
    "img" : img_embedding,      # og img, as tensor
    "masked_img": masked_img, # masked image as tensor
    "masked_patches_idxs": masked_patches_idxs, # indices of the masked patches,
    "text": text_embeddings,
}
```




## Results

## 24.08
pretraining with ap alone does not work. over 300k samples and 4 epochs it does not learn anything:

```bash
2025-08-23 22:35:30 - INFO  - trainer.py:train:691 - Epoch 4/4,
	train loss MLM: 0.0000,
	test loss MLM: 10.5104,
	train loss AP: 0.6946,
	test loss AP: 0.6946,
	accuracy AP: 0.4986
	train loss MIM: 0.0000,
	test loss MIM: 8.8669
```

Suspicion: AP alone is too hard to train for. needs the other pretraining tasks!

## 22.08
```bash
(venv310) cedric.sillaber@c703i-gpu10:~/coding/bsc-code$ python src/evaluate.py
Pretrained model path None does not exist, using fresh model.
trainable params: 280982844/280982844
dirname:  res/data/hateful_memes_data
32
bs_alignment_analysis: 100
dirname:  res/data/hateful_memes_data
Epoch 1/5, train loss: 0.6454, test loss: 0.6212,  accuracy: 0.6535
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:12<00:00,  4.07s/it]
Layer 0 CKA alignment score: 0.0704
Layer 0 (cross-attn: False): CKA = 0.0704
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 1 CKA alignment score: 0.0727
Layer 1 (cross-attn: True): CKA = 0.0727
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 2 CKA alignment score: 0.0731
Layer 2 (cross-attn: False): CKA = 0.0731
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 3 CKA alignment score: 0.0720
Layer 3 (cross-attn: True): CKA = 0.0720
Epoch 2/5, train loss: 0.5965, test loss: 0.5645,  accuracy: 0.7229
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 0 CKA alignment score: 0.0759
Layer 0 (cross-attn: False): CKA = 0.0759
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 1 CKA alignment score: 0.0746
Layer 1 (cross-attn: True): CKA = 0.0746
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 2 CKA alignment score: 0.0779
Layer 2 (cross-attn: False): CKA = 0.0779
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 3 CKA alignment score: 0.0776
Layer 3 (cross-attn: True): CKA = 0.0776
Epoch 3/5, train loss: 0.5528, test loss: 0.5626,  accuracy: 0.7412
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 0 CKA alignment score: 0.0787
Layer 0 (cross-attn: False): CKA = 0.0787
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 1 CKA alignment score: 0.0817
Layer 1 (cross-attn: True): CKA = 0.0817
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 2 CKA alignment score: 0.0811
Layer 2 (cross-attn: False): CKA = 0.0811
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 3 CKA alignment score: 0.0760
Layer 3 (cross-attn: True): CKA = 0.0760
Epoch 4/5, train loss: 0.5223, test loss: 0.5519,  accuracy: 0.7353
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 0 CKA alignment score: 0.0811
Layer 0 (cross-attn: False): CKA = 0.0811
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 1 CKA alignment score: 0.0862
Layer 1 (cross-attn: True): CKA = 0.0862
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 2 CKA alignment score: 0.0805
Layer 2 (cross-attn: False): CKA = 0.0805
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 3 CKA alignment score: 0.0831
Layer 3 (cross-attn: True): CKA = 0.0831
Epoch 5/5, train loss: 0.4917, test loss: 0.5586,  accuracy: 0.7359
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 0 CKA alignment score: 0.0868
Layer 0 (cross-attn: False): CKA = 0.0868
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 1 CKA alignment score: 0.0841
Layer 1 (cross-attn: True): CKA = 0.0841
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 2 CKA alignment score: 0.0875
Layer 2 (cross-attn: False): CKA = 0.0875
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.97s/it]
Layer 3 CKA alignment score: 0.0848
Layer 3 (cross-attn: True): CKA = 0.0848
(venv310) cedric.sillaber@c703i-gpu10:~/coding/bsc-code$ python src/evaluate.py --path res/checkpoints/pretrained_epoch2_task123.pt
Model loaded from res/checkpoints/pretrained_epoch2_task123.pt, epoch 1
Loaded model from res/checkpoints/pretrained_epoch2_task123.pt with config: {'embedding_dim': 768, 'vocab_size': 30522, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'dropout_prob': 0.4, 'learning_rate': 3e-05, 'img_size': (224, 224), 'preprocessed_path': 'res/preprocessed.pkl', 'train_test_ratio': 0.8, 'batch_size': 40, 'depth': 4, 'pretraining_tasks': [3, 2, 1], 'cross_attention_layers': [1, 3]}
trainable params: 280982844/280982844
dirname:  res/data/hateful_memes_data
48
bs_alignment_analysis: 100
dirname:  res/data/hateful_memes_data
Epoch 1/5, train loss: 0.6515, test loss: 0.6571,  accuracy: 0.6312
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:12<00:00,  4.02s/it]
Layer 0 CKA alignment score: 0.0153
Layer 0 (cross-attn: False): CKA = 0.0153
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 1 CKA alignment score: 0.0183
Layer 1 (cross-attn: True): CKA = 0.0183
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 2 CKA alignment score: 0.0179
Layer 2 (cross-attn: False): CKA = 0.0179
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 3 CKA alignment score: 0.0157
Layer 3 (cross-attn: True): CKA = 0.0157
Epoch 2/5, train loss: 0.6348, test loss: 0.6428,  accuracy: 0.6359
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |:  67%|██████████████████████████████████████████████                       | 2/3 [00:07<00:03,  3.98s/it]
| Computing CKA epoch 2 |:  60%|████████████████████████████████████▌                        | 3/5
| Computing CKA epoch 2 |:  80%|████████████████████████████████████████████████▊            | 4/5
| Computing CKA epoch 2 |: 100%|█████████████████████████████████████████████████████████████| 5/5
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 | Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 0 CKA alignment score: 0.0162
Layer 0 (cross-attn: False): CKA = 0.0162
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 1 CKA alignment score: 0.0186
Layer 1 (cross-attn: True): CKA = 0.0186
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 2 CKA alignment score: 0.0154
Layer 2 (cross-attn: False): CKA = 0.0154
| Computing CKA |: 100%|█████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 3 CKA alignment score: 0.0159
Layer 3 (cross-attn: True): CKA = 0.0159
Epoch 3/5, train loss: 0.6070, test loss: 0.6238,  accuracy: 0.6647
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 0 CKA alignment score: 0.0162
Layer 0 (cross-attn: False): CKA = 0.0162
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.98s/it]
Layer 1 CKA alignment score: 0.0170
Layer 1 (cross-attn: True): CKA = 0.0170
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 2 CKA alignment score: 0.0173
Layer 2 (cross-attn: False): CKA = 0.0173
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 3 CKA alignment score: 0.0142
Layer 3 (cross-attn: True): CKA = 0.0142
Epoch 4/5, train loss: 0.5768, test loss: 0.6099,  accuracy: 0.6912
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 0 CKA alignment score: 0.0148
Layer 0 (cross-attn: False): CKA = 0.0148
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  4.00s/it]
Layer 1 CKA alignment score: 0.0190
Layer 1 (cross-attn: True): CKA = 0.0190
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 2 CKA alignment score: 0.0164
Layer 2 (cross-attn: False): CKA = 0.0164
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 3 CKA alignment score: 0.0150
Layer 3 (cross-attn: True): CKA = 0.0150
Epoch 5/5, train loss: 0.5560, test loss: 0.5981,  accuracy: 0.7018
CKA alignment analysis on CC dataset:
Analyzing layer-wise alignment with ckatorch...
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 0 CKA alignment score: 0.0166
Layer 0 (cross-attn: False): CKA = 0.0166
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 1 CKA alignment score: 0.0181
Layer 1 (cross-attn: True): CKA = 0.0181
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 2 CKA alignment score: 0.0164
Layer 2 (cross-attn: False): CKA = 0.0164
| Computing CKA |: 100%|█████████████████████████████████████████████████████████████████████| 3/3 [00:11<00:00,  3.99s/it]
Layer 3 CKA alignment score: 0.0160
Layer 3 (cross-attn: True): CKA = 0.0160
```

lower alignment with pretraining! cosine similarity is higher after pretraining, also for hateful memes, where it was not pretrained, but cka is really lower.

Also implemented ckatorch integration.
the modules in `cka_wrapper.py`  wrap vilbert to one modality, so i can use ckatorch library. this library takes two models as input and calculates the batch cka for them. however, my tests found, that this is more or less the same as using cka_batch in my own implementation. the only point: they efficently use pytorch hooks to save memory.

## 18.08
easy similarity implementation with cosine-similarity lead to poor results on hatefulmemes:
```bash
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer0 (cross attention): avg cosine similarity: -0.0169
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer1 (cross attention): avg cosine similarity: -0.0050
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer2 (cross attention): avg cosine similarity: 0.0123
2025-08-18 18:40:27 - INFO  - analysis.py:analyse:124 - Layer layer3 (cross attention): avg cosine similarity: 0.0212
2025-08-18 18:40:27 - INFO  - trainer.py:train:87 - Epoch 4/4, train loss: 0.4890, test loss: 0.5661,  accuracy: 0.7224
2025-08-18 18:40:27 - INFO  - evaluate.py:train_and_eval_on_downstream_task:118 - Training and evaluation on downstream task finished, cleaning up memory

```
- nearly no similarity, layers 0 and 2 are coattentions, the other dualselfattention
- tried comparison of cls and global avg pool => same bad results

=> problem of dataset? hateful memes might have complex alignment, more complex than conceptual captions?
- also: only small pretraining on home gpu, might need longer runs for real results; alignment might not happened yet?



### 17.08 - comparing different pretraining tasks
with frozen encoders in pretraining.


**Task Combinations Tested:**
1. **Baseline**: No pretraining
2. **All Tasks**: MIM + MLM + AP
3. **Two Tasks**: MLM + AP (no MIM)
4. **Single Task**: AP only

**Findings:**

| Pretraining Tasks | Downstream Accuracy (Final) | Improvement over Baseline |
|-------------------|----------------------------|--------------------------|
| **None (Baseline)** | 67.5% ± 0.7% | - |
| **All (MIM+MLM+AP)** | **69.7%** | **+2.2%** |
| **MLM + AP** | **71.0%** | **+3.5%** |
| **AP Only** | **70.8%** | **+3.3%** |


**Pretraining Task Analysis:**

**Alignment Prediction (AP) Performance:**
- All tasks: 80% → 86% accuracy
- MLM + AP: 82% → 87% accuracy
- AP only: 83% → **88%** accuracy (best)


running the pretraining on 125k images with all three pretraining tasks resulted in this pretraining loss:

<figure>
<img src="res/markdown_res/training_losses-1755120573.png" width=400>
</figure>

```
2025-08-13 15:45:10 - INFO  - trainer.py:train:518 - training with tasks: [<Task.ALIGNMENT_PREDICTION: 1>, <Task.MASKED_LM: 2>, <Task.MASKED_IM: 3>]
2025-08-13 17:41:13 - INFO  - trainer.py:train:566 - Epoch 1/4,
	train loss MLM: 4.0114,
	test loss MLM: 3.4809,
	train loss AP: 0.4748,
	test loss AP: 0.3476,
	accuracy AP: 0.8484
	train loss MIM: 3.1180,
	test loss MIM: 1.0458
2025-08-13 17:41:16 - INFO  - trainer.py:__save_checkpoint:612 - Checkpoint saved to res/checkpoints/pretrained_1.pt
2025-08-13 19:37:17 - INFO  - trainer.py:train:566 - Epoch 2/4,
	train loss MLM: 3.1049,
	test loss MLM: 2.9979,
	train loss AP: 0.3050,
	test loss AP: 0.3105,
	accuracy AP: 0.8682
	train loss MIM: 0.8290,
	test loss MIM: 0.2725
2025-08-13 19:37:20 - INFO  - trainer.py:__save_checkpoint:612 - Checkpoint saved to res/checkpoints/pretrained_2.pt
2025-08-13 21:33:23 - INFO  - trainer.py:train:566 - Epoch 3/4,
	train loss MLM: 2.7849,
	test loss MLM: 2.8589,
	train loss AP: 0.2659,
	test loss AP: 0.3059,
	accuracy AP: 0.8666
	train loss MIM: 0.4698,
	test loss MIM: 1.0823
2025-08-13 21:33:26 - INFO  - trainer.py:__save_checkpoint:612 - Checkpoint saved to res/checkpoints/pretrained_3.pt
2025-08-13 23:29:30 - INFO  - trainer.py:train:566 - Epoch 4/4,
	train loss MLM: 2.6156,
	test loss MLM: 2.7057,
	train loss AP: 0.2426,
	test loss AP: 0.2943,
	accuracy AP: 0.8787
	train loss MIM: 0.3226,
	test loss MIM: 0.0857
```


on the downstream task it achieved, with encoders frozen:
```
❯ python src/evaluate.py --path res/checkpoints/pretrained_4.pt
Model loaded from res/checkpoints/pretrained_4.pt, epoch 3
Loaded model from res/checkpoints/pretrained_4.pt with config: {'embedding_dim': 768, 'vocab_size': 30522, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'dropout_prob': 0.1, 'learning_rate': 3e-05, 'img_size': (224, 224), 'preprocessed_path': 'res/preprocessed.pkl', 'train_test_ratio': 0.8, 'batch_size': 32}
trainable params: 42705468/237986364
Epoch 1/4, train loss: 0.6365, test loss: 0.6183,  accuracy: 0.6576
Epoch 2/4, train loss: 0.5943, test loss: 0.5932,  accuracy: 0.6753
Epoch 3/4, train loss: 0.5659, test loss: 0.5823,  accuracy: 0.6818
Epoch 4/4, train loss: 0.5404, test loss: 0.5725,  accuracy: 0.7035
❯ python src/evaluate.py --path res/checkpoints/pretrained_1.pt
Model loaded from res/checkpoints/pretrained_1.pt, epoch 0
Loaded model from res/checkpoints/pretrained_1.pt with config: {'embedding_dim': 768, 'vocab_size': 30522, 'num_hidden_layers': 12, 'num_attention_heads': 12, 'dropout_prob': 0.1, 'learning_rate': 3e-05, 'img_size': (224, 224), 'preprocessed_path': 'res/preprocessed.pkl', 'train_test_ratio': 0.8, 'batch_size': 32}
trainable params: 42705468/237986364
dirname:  res/data/hateful_memes_data
Epoch 1/4, train loss: 0.6353, test loss: 0.6286,  accuracy: 0.6535
Epoch 2/4, train loss: 0.5980, test loss: 0.6170,  accuracy: 0.6665
Epoch 3/4, train loss: 0.5668, test loss: 0.5937,  accuracy: 0.7000
Epoch 4/4, train loss: 0.5389, test loss: 0.5853,  accuracy: 0.7041
```


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




## Remarks

This section covers implementation decisions.

### Datasets

the pretraining dataset is downloaded using `src/download_cc.py`. It tries to download and open pictures from the conceptual captions dataset (`res/data/conceptual-captions/Train_GCC-training.tsv`). Some links are invalid and some images not openable, those are not saved.
In the downloading step, I already resize to 224x224, in order to save memory.

In the dataset handling in `src/datasets.py`, transformations for the timm-vit are applied.
```python
vit_transform = create_transform(**config)        # this was used before
vit_transform: Compose(
    Resize(size=256, interpolation=bicubic, max_size=None, antialias=True)
    CenterCrop(size=(224, 224))
    MaybeToTensor()
    Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
)
```

But as I already resize the images to 224x224, I don't need the resizing and cropping anymore.
=>
```python
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
vit_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
])
```

This generates the correct input for ViT. In the dataset, then the transformations are done. for pretraining, masking the language tokens, masking the vision tokens and creating the correct task for alignment prediction.


The custom datasets inherit from `torch.utils.data.dataset` and return the following dictionary:
```python
{
	"task": task.value,  # Task enum value
	"img": img_embedding,  # Image embedding as tensor
	"masked_img": masked_img,  # Masked image as tensor
	"masked_patches_idxs": masked_patches_idxs,  # Indices of the masked patches
	"text": text_embeddings,  # Text embeddings as tensor
}
```

If alignment on new datasets should be tested, it should have the form of:
```python
{"img": ..., "text": ..., "label": ...}
```
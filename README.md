# Code for bachelor thesis

This repository contains the code for my bachelor thesis. It is a PyTorch implementation of ViLBERT, a model that combines visual and language understanding.
I'm currently working on it, so it is not yet complete.

## Installation & Usage

To install:
```bash
git clone https://github.com/cesipy/bsc-code
cd bsc-code

# in virtual env:
pip install -r requirements.txt
```

To run the code, you have to set the following environment variable:
```bash
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```


For the basic evaluation program `argcomplete` is installed. This is for tab completion.  Its included in the `requirements.txt` file, to install run:
```bash
activate-global-python-argcomplete
```


## Optuna parameter tuning
To perform either hyperparam-optimization or neural architecture search (NAS), simply run:
```bash
python src/hyperparameter_optimizer.py

#opt: to see visualizations via dashboard:
optuna-dashboard sqlite:///res/hyperparameter_optimization/optuna_study.db
```




## Implementation Decisions

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


### Optimization
The optimization for this thesis consists of two task. i) to get the best hyperparams for a given depth (lr, epochs, dropout_pro) and ii) to optimize with those hyperparams using neural architecture search for the best coattn configuration.

currently those parts are seperated by modules (might be different in newer implementations). `hyperparameter_optimizer.py` is used for hyperparam optimization and `experiment_tracker.py` for neural architecture search.




### Pretraining

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

## ViLBERT
original [vilbert](https://github.com/facebookresearch/vilbert-multi-task) under `vilbert/vilbert.py`.


---



## TODO
**immediate:**
- [ ] all the same architectures do finetune only, not pretrain
- [ ] baseline where cka is optimized!
    - [ ] currently with late fusion, but here the values are pretty high anyways, maybe implement with baseline no contn
    - [ ] no coattn baseline cka optim
- [ ] accuracy on each layer, similar plot to hintons paper from 2019


- [ ] pretrain vs non-pretrain: run the same experiments on finetune-only!
- [ ] update `all_finetune_logs` with new finetunes. currently includes up to late fusion
- [ ] test acc und loss für finetunes, wo noch nicht geändert und in `all_finetune_logs.txt` inkludieren. Nur für `upmc_food, mm_imdb` Für
    - [ ] pretrain_early_fusion
    - [ ] pretrain_middle_fusion

- [ ] pretrain für starke MM-imdb. optuna 1 ist stark für hateful_memes, optuna2 ist guter kompromiss => optuna 3 fehlt!
- [ ] still optuna 3 needed! good archicture for mm_imdb alone!

- [ ] correlation analysis of metrics on all new architectures
- [ ] is finetuned good performance indicator for pretraining good performance?
    => no it was not!
- [ ] convert all torchvision to albuminations + seeding

- [ ] https://docs.pytorch.org/tutorials/recipes/recipes/tuning_guide.html
- [ ] analysis of pretrained models: discrepancies in end representation of streams


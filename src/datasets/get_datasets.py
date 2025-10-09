
import sys, os; sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'src'))
import torch; from torch.utils.data import Dataset, DataLoader
import typing

from transformers import (
     # ViT stuff
    BaseImageProcessor,
    ViTImageProcessor,

    # type hinting stuff
    PreTrainedTokenizerFast,
    BertTokenizerFast
)

import utils

import random

from logger import Logger
from config import *; import config
from .dataset_utils import get_image_embedding, get_text_embedding
import augments_transforms

from .dataset_hateful_memes import HM_Dataset
from .dataset_pretrain import *
from .dataset_mm_imdb import *
from .dataset_vqa import *
from .dataset_upmc import *


from .dataset_utils import generate_data_list, generate_data_list_pretrain


def get_alignment_dataloaders(
    batch_size,
    num_workers: int,
    pin_memory: bool,
    prefetch_factor: int,
    seed: int,
    num_samples:int = 1000,
    )-> typing.Tuple[DataLoader, DataLoader, DataLoader]:
    """
    returns tuple of dataloader in the following order:
    dataloader-hateful-memes, dataloader-conceputal-captions, dataloader-mmimdb
    """
    utils.set_seeds(seed)

    path_cc       = "res/data/conceptual-captions/validation.csv"
    path_hm       = "res/data/hateful_memes_data/train.jsonl"
    path_imdb     = "res/data/mm-imdb/images.h5"
    csv_path_imdb = "res/data/mm-imdb/mmimdb_test.csv"

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    data_list_hm = generate_data_list(path_hm)
    train_idx = int(len(data_list_hm) * TRAIN_TEST_RATIO)

    # has len 1700
    data_list_hm = data_list_hm[train_idx:]
    # random.shuffle(data_list_hm)    # validation set is shuffled

    assert num_samples <= len(data_list_hm)

    #TODO: it now uses the complete dataset, not the train-test split here for alignment analysis
    data_list_cc = generate_data_list_pretrain(path=path_cc, max_number=None)
    random.shuffle(data_list_cc)
    data_list_hm = data_list_hm[:num_samples]
    data_list_cc = data_list_cc[:num_samples]


    dataset_imdb = MM_IMDB_Dataset(
        csv_path=csv_path_imdb,
        img_path=path_imdb,
        tokenizer=tokenizer,
        image_processor=image_processor,
        train_test_ratio=0.8,
        is_train=False,  # is ignored,
        max_samples=num_samples
    )

    dataset_hm = HM_Dataset(
        data=data_list_hm,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    dataset_cc = ConceptualCaptionsDataset(
        data=data_list_cc, tokenizer=tokenizer, image_processor=image_processor,
    )

    dataloader_hm = DataLoader(
        dataset=dataset_hm,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    dataloader_cc = DataLoader(
        dataset=dataset_cc,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )
    dataloader_imdb = DataLoader(
        dataset=dataset_imdb,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )


    return dataloader_hm, dataloader_cc, dataloader_imdb




def get_dataloaders_pretrain(
    train_data,
    val_data,
    batch_size: int,
    num_workers,
    prefetch,
    persistent_workers,
    seed:int,
    pin_memory=True,
    ):
    """
    Returns: (
        train_loader_ap,
        val_loader_ap,
        train_loader_mlm,
        val_loader_mlm,
        train_loader_mim,
        val_loader_mim
    )
    """
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    print(f"dataset lengths: train: {len(train_data)}; val: {len(val_data)}")

    train_dataset_mim = PretrainDatasetMIM(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms_weak=augments_transforms.get_transform_unmasked(),
        transforms_strong=augments_transforms.get_transform_masked()
    )
    val_dataset_mim   = PretrainDatasetMIM(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms_weak=augments_transforms.get_transform_unmasked(),
        transforms_strong=augments_transforms.get_transform_masked(),
    )

    train_dataset_ap = PretrainDatasetAP(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=True
    )


    val_dataset_ap   = PretrainDatasetAP(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    train_dataset_mlm = PretrainDatasetMLM(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    val_dataset_mlm   = PretrainDatasetMLM(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    train_loader_mim = DataLoader(
        dataset=train_dataset_mim,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_loader_mim = DataLoader(
        dataset=val_dataset_mim,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    train_loader_ap = DataLoader(
        dataset=train_dataset_ap,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_loader_ap = DataLoader(
        dataset=val_dataset_ap,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    train_loader_mlm = DataLoader(
        dataset=train_dataset_mlm,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_loader_mlm = DataLoader(
        dataset=val_dataset_mlm,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    return (
        train_loader_ap,
        val_loader_ap,
        train_loader_mlm,
        val_loader_mlm,
        train_loader_mim,
        val_loader_mim,
    )

def get_mmimdb_datasets(
    train_test_ratio: float,
    batch_size: int,
    seed: int,
    num_workers: int=0,
    pin_memory: bool = False,
    prefetch_factor: int = None,
    persistent_workers: bool = False,
    use_train_augmentation:bool=True,
) -> typing.Tuple[DataLoader, DataLoader]:
    """
    get the mmimdb dataset. per default enables data augmentation on testset

    parameters:
        train_test_ratio: float
        batch_size: int
        num_workers: int
        pin_memory: bool
        prefetch_factor: int
        persistent_workers: bool
        use_train_augmentation: bool, whether to use data augmentation on the training set. defaults to True
    """

    assert 0< train_test_ratio <1

    if use_train_augmentation:
        transform = augments_transforms.get_mm_imdb_train_augmentation(seed=seed)
    else: transform = None

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    path = "res/data/mm-imdb/images.h5"
    csv_path = "res/data/mm-imdb/mmimdb_trainval.csv"

    train_dataset = MM_IMDB_Dataset(
        csv_path=csv_path,
        img_path=path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        train_test_ratio=train_test_ratio,
        is_train=True,
        transform=transform
    )

    val_dataset = MM_IMDB_Dataset(
        csv_path=csv_path,
        img_path=path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    return train_dataloader, val_dataloader


def get_hateful_memes_datasets(
    train_test_ratio: float,
    batch_size: int,
    num_workers: int,
    seed:int,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    use_train_augmentation:bool=True,
    limit_total_dataset:bool=False,
) -> typing.Tuple[DataLoader, DataLoader]:
    """
    get the hateful memes dataset. per default enables data augmentation on testset
    parameters:
        train_test_ratio: float
        batch_size: int
        num_workers: int
        pin_memory: bool
        prefetch_factor: int
        persistent_workers: bool
        use_train_augmentation: bool, whether to use data augmentation on the training set. defaults to True
        limit_total_dataset: bool, whether to limit the total dataset size for faster experiments. defaults to True
    """

    if use_train_augmentation:
        # transform = augments_transforms.get_hateful_memes_train_augmentation()
        transform = augments_transforms.get_hateful_memes_train_augmentation_albumation(seed=seed,get_advanced=False)
    else:
        transform = None

    assert 0< train_test_ratio <1

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    path = "res/data/hateful_memes_data/train.jsonl"
    data_list = generate_data_list(path)

    train_idx = int(len(data_list) * train_test_ratio)
    train_data = data_list[:train_idx]
    val_data   = data_list[train_idx:]

    if limit_total_dataset:
        train_data = train_data[:1000]
        val_data = val_data[:400]


    train_dataset = HM_Dataset(
        data=train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms=transform
    )

    val_dataset = HM_Dataset(
        data=val_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
    )

    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_dataloader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    return train_dataloader, val_dataloader


def get_easyvqa_datasets(
    batch_size: int,
    num_workers: int,
    seed:int,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    persistent_workers: bool = True,
    use_train_augmentation: bool = False  # VQA typically doesn't need augmentation
) -> typing.Tuple[DataLoader, DataLoader]:
    """
    Get the EasyVQA dataset for train and validation

    Parameters:
        batch_size: int
        num_workers: int
        pin_memory: bool
        prefetch_factor: int
        persistent_workers: bool
        use_train_augmentation: bool (not used for VQA, kept for consistency)
    """

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    train_dataset = VqaDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=True
    )

    val_dataset = VqaDataset(
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=False
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,  # Don't shuffle validation
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    return train_dataloader, val_dataloader

def get_upmc_datasets(
    train_test_ratio: float,
    batch_size: int,
    num_workers: int,
    seed:int,
    pin_memory: bool = False,
    prefetch_factor: int = 4,
    persistent_workers: bool = False,
    use_train_augmentation:bool=True,
    max_samples: typing.Optional[int] = None,
) -> typing.Tuple[DataLoader, DataLoader]:
    """
    get the UPMC food-101 dataset. per default enables data augmentation on testset

    parameters:
        train_test_ratio: float
        batch_size: int
        num_workers: int
        pin_memory: bool
        prefetch_factor: int
        persistent_workers: bool
        use_train_augmentation: bool, whether to use data augmentation on the training set. defaults to True
        max_samples: Optional[int], maximum number of samples to use from the dataset. Useful for quick experiments.
    """

    assert 0< train_test_ratio <1

    if use_train_augmentation:
        transform = augments_transforms.get_hateful_memes_train_augmentation_albumation(get_advanced=False, seed=seed)
    else: transform = None

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

    if config.machine == "remote":
        csv_path = "/mnt/ifi/iis/javier.urena/UPMC_Food-101/UPMC_Food-101/upmcfood_trainval.csv"
        img_path = "/mnt/ifi/iis/javier.urena/UPMC_Food-101/UPMC_Food-101/images"
    else:
        csv_path = "res/data/UPMC_Food-101/upmcfood_trainval.csv"
        img_path = "res/data/UPMC_Food-101/images"

    train_dataset = UPMC_Dataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=True,
        img_path=img_path,
        train_test_ratio=train_test_ratio,
        transform=transform,
        max_samples=max_samples,       # TODO
    )
    val_dataset = UPMC_Dataset(
        csv_path=csv_path,
        tokenizer=tokenizer,
        image_processor=image_processor,
        is_train=False,
        img_path=img_path,
        train_test_ratio=train_test_ratio,
    )

    print(f"num of trainsamples: {len(train_dataset)}")
    print(f"num of valsamples: {len(val_dataset)}")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        prefetch_factor=prefetch_factor,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        worker_init_fn=utils.worker_init_fn,
        generator=utils.get_seeded_generator(seed),
    )

    return  train_dataloader, val_dataloader



import os
# problem when running training on loaded models after pretraining.
# occurs because of parallelism in data loaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # You already have this


import gc
import torch; from torch.utils.data import DataLoader, Dataset

from transformers import (
    BertTokenizerFast, PreTrainedTokenizerFast, ViTImageProcessor
)

from torchvision import transforms; import torchvision

import utils
from task import Task
import datasets; from datasets import HM_Dataset, PretrainDatasetAP, PretrainDatasetMLM, PretrainDatasetMIM
from config import *
from vilbert import ViLBERT
from trainer import Trainer, PretrainingTrainer
from logger import Logger

import argparse

import analysis
import trainer

import warnings
# Suppress specific PyTorch warnings
warnings.filterwarnings("ignore", category=UserWarning, module="torch.utils.checkpoint")
warnings.filterwarnings("ignore", category=FutureWarning, module="torch.nn.modules.module")

machine = os.getenv("MACHINE_TYPE", default="home")     # remote or home
logger = Logger()

@utils.memory_cleanup
def train_and_eval_on_downstream_task(pretrained_model_path:str, use_constrastive:Optional[bool]=False):
    if pretrained_model_path==None or not os.path.exists(pretrained_model_path) :
        # use fresh vilbert
        info_str = f"Pretrained model path {pretrained_model_path} does not exist, using fresh model."
        print(info_str)
        logger.info(info_str)

        config = ViLBERTConfig()
        model = ViLBERT(config=config)

    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cp = ViLBERT.from_pretrained_checkpoint(checkpoint_path=pretrained_model_path, device=device)
        info_str = f"Loaded model from {pretrained_model_path} with config: {cp['config']}"
        print(info_str)
        logger.info(info_str)

    if not use_constrastive:
        use_constrastive=False
    # utils.freeze_all_layers(model.vit)
    # utils.freeze_all_layers(model.bert)

    path = "res/data/hateful_memes_data/train.jsonl"
    val_path = "res/data/hateful_memes_data/test.jsonl"

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    config = ViLBERTConfig()

    #TODO: also freeze co-attention layers here
    utils.params_summary(model=model)
    train_data_list = datasets.generate_data_list(path)
    # val_data_list = datasets.generate_data_list(val_path)

    # train_data_list = train_data_list[:1000]

    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:]

    # train_data = train_data[:1000]
    # val_data = val_data[:400]


    # for alignment analysis
    if machine == "remote":
        bs_alignment_analysis = 16
    else:
        bs_alignment_analysis = 48

    config.learning_rate = DOWNSTREAM_LR
    print(f"bs_alignment_analysis: {bs_alignment_analysis}, batchsize: {BATCH_SIZE_DOWNSTREAM}")


    transform_hm = transforms.Compose([

        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),

        transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),

        transforms.RandomGrayscale(p=0.1),
        # transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomErasing(),

    ])



    train_data = train_data
    train_dataset = HM_Dataset(
        train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms=transform_hm
    )
    val_dataset   = HM_Dataset(val_data, tokenizer=tokenizer, image_processor=image_processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        )

    hm_dataloader, cc_dataloader = datasets.get_alignment_dataloaders(
        batch_size=bs_alignment_analysis,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4,
        num_samples=4000
    )
    info_str = f"using contrastive: {use_constrastive}"
    print(info_str)
    logger.info(info_str)
    trainer = Trainer(
        model,
        config,
        use_contrastive_loss=use_constrastive,
        use_cosine_loss=False,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        )
    trainer.train(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=DOWNSTREAM_EPOCHS,
        hm_dataloader=hm_dataloader,
        cc_dataloader=cc_dataloader,
    )

    del model, trainer, train_dataset, val_dataset, train_loader, val_loader
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Training and evaluation on downstream task finished, cleaning up memory\n\n"+ 25*"-")


@utils.memory_cleanup
def test_on_hm():


    config = ViLBERTConfig()
    model = ViLBERT(config=config)

    utils.freeze_all_layers(model.vit)
    utils.freeze_all_layers(model.bert)

    path = "res/data/hateful_memes_data/train.jsonl"
    val_path = "res/data/hateful_memes_data/test.jsonl"

    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    config = ViLBERTConfig()

    #TODO: also freeze co-attention layers here
    utils.params_summary(model=model)
    train_data_list = datasets.generate_data_list(path)
    # val_data_list = datasets.generate_data_list(val_path)

    # train_data_list = train_data_list[:1000]

    train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
    train_data = train_data_list[:train_idx]
    val_data   = train_data_list[train_idx:]

    train_data = train_data[:1000]
    val_data = val_data[:400]


    if machine == "remote":
        bs = 48   # obout 23.3gb vrman
        bs_alignment_analysis = 32
    else:
        bs = 32
        bs_alignment_analysis = 32

    config.learning_rate = 2e-6
    print(bs)
    print(f"bs_alignment_analysis: {bs_alignment_analysis}")

    transform_hm = transforms.Compose([

        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),

        transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),

        transforms.RandomGrayscale(p=0.1),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomErasing(),

    ])

    num_workers = 4
    pin_memory= True
    prefetch_factor = 3

    train_data = train_data
    train_dataset = HM_Dataset(
        train_data,
        tokenizer=tokenizer,
        image_processor=image_processor,
        transforms=transform_hm
    )
    val_dataset   = HM_Dataset(val_data, tokenizer=tokenizer, image_processor=image_processor)

    train_loader = DataLoader(
        train_dataset,
        batch_size=bs,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=bs,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        )

    hm_dataloader, cc_dataloader = datasets.get_alignment_dataloaders(
        batch_size=bs_alignment_analysis,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4,
        num_samples=1000,       # how many samples to do the alignment eval on
    )

    trainer = Trainer(
        model,
        config,
        use_contrastive_loss=True,
        # use_cosine_loss=True,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        )
    trainer.train(
        train_dataloader=train_loader,
        test_dataloader=val_loader,
        epochs=5,
        hm_dataloader=hm_dataloader,
        cc_dataloader=cc_dataloader
    )

    del model, trainer, train_dataset, val_dataset, train_loader, val_loader
    logger.info(25*"-")

# def test_visualization():

#     config = ViLBERTConfig()
#     model = ViLBERT(config=config)

#     utils.freeze_all_layers(model.vit)
#     utils.freeze_all_layers(model.bert)

#     path = "res/data/hateful_memes_data/train.jsonl"
#     val_path = "res/data/hateful_memes_data/test.jsonl"

#     tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
#     image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
#     config = ViLBERTConfig()

#     #TODO: also freeze co-attention layers here
#     utils.params_summary(model=model)
#     train_data_list = datasets.generate_data_list(path)
#     # val_data_list = datasets.generate_data_list(val_path)

#     # train_data_list = train_data_list[:1000]

#     train_idx = int(len(train_data_list) * TRAIN_TEST_RATIO)
#     train_data = train_data_list[:train_idx]
#     val_data   = train_data_list[train_idx:]

#     train_data = train_data[:1000]
#     val_data = val_data[:400]


#     if machine == "remote":
#         bs = 48

#     config.learning_rate = 2e-6
#     transform_hm = transforms.Compose([

#         transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),

#         transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
#         transforms.RandomAffine(
#             degrees=10,
#             translate=(0.05, 0.05),
#             scale=(0.95, 1.05),
#             shear=2
#         ),
#         transforms.RandomApply([
#             transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
#         ], p=0.2),

#         transforms.RandomGrayscale(p=0.1),
#         transforms.RandomHorizontalFlip(p=0.2),
#         transforms.RandomErasing(),

#     ])

#     num_workers = 4
#     pin_memory= True
#     prefetch_factor = 3

#     train_data = train_data
#     train_dataset = CustomDataset(
#         train_data,
#         tokenizer=tokenizer,
#         image_processor=image_processor,
#         transforms=transform_hm
#     )
#     val_dataset   = CustomDataset(val_data, tokenizer=tokenizer, image_processor=image_processor)

#     train_loader = DataLoader(
#         train_dataset,
#         batch_size=bs,
#         shuffle=True,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         prefetch_factor=prefetch_factor,
#     )
#     val_loader = DataLoader(
#         val_dataset,
#         batch_size=bs,
#         shuffle=False,
#         num_workers=num_workers,
#         pin_memory=pin_memory,
#         prefetch_factor=prefetch_factor,
#         )

#     # hm_dataloader, cc_dataloader = datasets.get_alignment_dataloaders(
#     #     batch_size=bs_alignment_analysis,
#     #     num_workers=4,
#     #     pin_memory=False,
#     #     prefetch_factor=4,
#     #     num_samples=1000,       # how many samples to do the alignment eval on
#     # )

#     trainer = Trainer(
#         model,
#         config,
#         use_contrastive_loss=False,
#         # use_cosine_loss=True,
#         )
#     trainer.train(
#         train_dataloader=train_loader,
#         test_dataloader=val_loader,
#         epochs=2,
#         hm_dataloader=None,
#         cc_dataloader=None
#     )

#     analysis.get_visualisation_data(
#         dataloader=val_loader,
#         model=model,)

#     del model, trainer, train_dataset, val_dataset, train_loader, val_loader
#     logger.info(25*"-")



if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="train on hateful memes")
        p.add_argument("--path", type=str, default=None,
                    help="Path to pretrained model checkpoint (optional)")
        p.add_argument("--use-constrastive", action="store_true",)
        p.add_argument("--test", action="store_true",
                       help="only test if it is running with small subsetof training")

        arg = p.parse_args()
        is_testing = arg.test
        pretrained_model_path = arg.path
        use_constrastive = arg.use_constrastive

        if is_testing:
            test_on_hm()
        else:
            train_and_eval_on_downstream_task(
                pretrained_model_path=pretrained_model_path,
                use_constrastive=use_constrastive
            )

        # test_visualization()
    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        raise e

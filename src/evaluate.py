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
from vilbert import ViLBERT, Baseline
from trainer import HatefulMemesTrainer, PretrainingTrainer
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
def train_and_eval_on_downstream_task(pretrained_model_path:str, use_contrastive:Optional[bool]=False):


    if pretrained_model_path==None or not os.path.exists(pretrained_model_path) :
        # use fresh vilbert
        utils.set_seeds(SEED)
        info_str = f"Pretrained model path {pretrained_model_path} does not exist, using fresh model."
        print(info_str)
        logger.info(info_str)

        config = ViLBERTConfig()
        # model = ViLBERT(config=config)
        model = Baseline(config=config)

    else:

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, cp = ViLBERT.from_pretrained_checkpoint(checkpoint_path=pretrained_model_path, device=device)
        info_str = f"Loaded model from {pretrained_model_path} with config: {cp['config']}"
        print(info_str)
        logger.info(info_str)

    if not use_contrastive:
        use_contrastive=False
    # utils.freeze_all_layers(model.vit)
    # utils.freeze_all_layers(model.bert)

    config = ViLBERTConfig()

    #TODO: also freeze co-attention layers here
    utils.params_summary(model=model)


    # for alignment analysis
    if machine == "remote":
        bs_alignment_analysis = BATCH_SIZE_ANALYSIS
    else:
        bs_alignment_analysis = BATCH_SIZE_ANALYSIS

    config.learning_rate = DOWNSTREAM_LR
    print(f"bs_alignment_analysis: {bs_alignment_analysis}, batchsize: {BATCH_SIZE_DOWNSTREAM}")


    train_loader, val_loader = datasets.get_hateful_memes_datasets(
        train_test_ratio=TRAIN_TEST_RATIO,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        # use_train_augmentation=True   #TODO: for testing
        use_train_augmentation=False,
    )

    hm_dataloader, cc_dataloader, imdb_dataloader = datasets.get_alignment_dataloaders(
        batch_size=bs_alignment_analysis,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4,
        num_samples=2000
    )
    info_str = f"using contrastive: {use_contrastive}"
    print(info_str)
    logger.info(info_str)
    trainer = HatefulMemesTrainer(
        model,
        config,
        use_contrastive_loss=use_contrastive,
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

    del model, trainer, train_loader, val_loader
    with torch.no_grad():
        torch.cuda.empty_cache()
    gc.collect()
    logger.info("Training and evaluation on downstream task finished, cleaning up memory\n\n"+ 25*"-")


@utils.memory_cleanup
def test_on_hm():
    utils.set_seeds(SEED)
    config = ViLBERTConfig()
    # model = ViLBERT(config=config)
    model = Baseline(config=config)

    utils.freeze_all_layers(model.vit)
    utils.freeze_all_layers(model.bert)
    config = ViLBERTConfig()

    #TODO: also freeze co-attention layers here
    utils.params_summary(model=model)


    if machine == "remote":
        bs_alignment_analysis = 32
    else:
        bs_alignment_analysis = 32

    config.learning_rate = DOWNSTREAM_LR

    print(f"bs_alignment_analysis: {bs_alignment_analysis}")


    train_loader, val_loader = datasets.get_hateful_memes_datasets(
        train_test_ratio=TRAIN_TEST_RATIO,
        batch_size=BATCH_SIZE_DOWNSTREAM,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        prefetch_factor=PREFETCH,
        persistent_workers=PERSISTENT_WORKERS,
        use_train_augmentation=True,
        limit_total_dataset=True,
    )

    hm_dataloader, cc_dataloader, imdb_dataloader  = datasets.get_alignment_dataloaders(
        batch_size=bs_alignment_analysis,
        num_workers=4,
        pin_memory=False,
        prefetch_factor=4,
        num_samples=1000,       # how many samples to do the alignment eval on
    )

    trainer = HatefulMemesTrainer(
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

    del model, trainer, hm_dataloader, cc_dataloader, train_loader, val_loader
    logger.info(25*"-")




if __name__ == "__main__":
    try:
        p = argparse.ArgumentParser(description="train on hateful memes")
        p.add_argument("--path", type=str, default=None,
                    help="Path to pretrained model checkpoint (optional)")
        p.add_argument("--use-contrastive", action="store_true",)
        p.add_argument("--test", action="store_true",
                       help="only test if it is running with small subsetof training")

        arg = p.parse_args()
        is_testing = arg.test
        pretrained_model_path = arg.path
        use_contrastive = arg.use_contrastive

        if is_testing:
            test_on_hm()
        else:
            train_and_eval_on_downstream_task(
                pretrained_model_path=pretrained_model_path,
                use_contrastive=use_contrastive
            )

        # test_visualization()
    except Exception as e:
        logger.error(f"Error during training and evaluation: {e}")
        raise e

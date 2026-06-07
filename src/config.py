import os
import socket
from dataclasses import dataclass, field
from typing import Optional

from task import Task

SEED = 13310  #TODO INTEGRATE EVERYWHERE


MM_IMDB_NUM_GENRES = 23
EASY_VQA_NUM_CLASSES = 13
UPMC_NUM_CLASSES = 101

# --------------------------------------------------
# ViLBERT
EMBEDDING_DIM = 768
VOCAB_SIZE    = 30522
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
NUM_BI_ATTENTION_HEADS = 8
COATTN_HIDDEN_SIZE = 1024
DROPOUT_PROB        =  0.08
VIT_MODEL_NAME = "vit_base_patch16_224"
#default vals for them
DEPTH = 12          # how many co-attn layers in transformer

V_BIATTENTION_IDS = [0,1,2,3,4,5]
T_BIATTENTION_IDS   = [6,7,8,9, 10,11]

TEXT_ATTENTION_DROPOUT = 0.1
VISION_ATTENTION_DROPOUT = 0.1

CLS_FUSION_METHOD = "concat"  # available ["sum", "hardamard", "concat" ]
FUSION_METHODS = ["sum", "hardamard", "concat"]
# --------------------------------------------------
# pretraining
PRETRAIN_LEARNING_RATE = 1e-4
PRETRAIN_EPOCHS = 5 # TODO

USE_CONTRASTIVE_LOSS=False
FREEZE_UNIMODAL_ENCODERS = False
NUM_SAMPLES_CC = 500_000

OPTIMIZE_CKA = False
OPTIMIZE_CKA_LAMBDA = 0.2
# optimize mutual k-NN (differentiable surrogate)
OPTIMIZE_MUTUAL_KNN = False
OPTIMIZE_MUTUAL_KNN_LAMBDA = 0.2
MUTUAL_KNN_TEMP = 0.07
# --------------------------------------------------
# data specific
IMG_SIZE = (224, 224)
PREPROCESSED_PATH = "res/preprocessed.pkl"      # not yet used, used to store precomputed datasets (in tensor form)
TRAIN_TEST_RATIO = 0.8
# what length for text tokens; is the same as num_patches + 1: 16*16 patches + cls
TOKENIZER_MAX_LEN = 197
#all the torch dataset/dataloader stuff
NUM_WORKERS = 0
PREFETCH = None
PERSISTENT_WORKERS = False
PIN_MEMORY = False


#hateful memes specific, for the weighted loss; hardcoded as this is easier to do!
POS_COUNT_HM = 3019
NEG_COUNT_HM = 5481
# --------------------------------------------------
# for the src/evaluate.py part; finetunes on hateful memes or mmimdb
DOWNSTREAM_EPOCHS = 9
DOWNSTREAM_LR     = 3.4e-5

# --------------------------------------------------
# hardware detection: one place decides machine type, batch sizes and grad
# accumulation. Robust to unknown hostnames (no crash) and to the gpu5 case.
_GOOD_GPUS = [0, 1, 9, 6, 7, 10, 11, 12]  # gpus with 24gb vram
_GPU_PREFIX = "c703i-gpu"


def _gpu_index(hostname: str) -> Optional[int]:
    """Return the trailing GPU number for a `c703i-gpuN` host, else None."""
    if not hostname.startswith(_GPU_PREFIX):
        return None
    try:
        return int(hostname[len(_GPU_PREFIX):])
    except ValueError:
        return None


def detect_hardware():
    """
    Decide machine type + batch/accumulation sizes from MACHINE_TYPE and the
    GPU hostname. Returns a dict; never raises on an unrecognised host.
    """
    machine = os.environ.get("MACHINE_TYPE", "local")  # local (16gb) / remote (24gb)
    gpu = _gpu_index(socket.gethostname())

    # pretraining sizes follow the machine type
    if machine == "remote":
        batch_pretrain, grad_pretrain = 20, 26  # simulated batches ~512, like the og vilbert paper
    else:
        batch_pretrain, grad_pretrain = 8, 64   # simulated batches ~128

    # downstream sizes follow the specific GPU
    if gpu == 5:
        batch_down, grad_down = 4, 128          # gpu5 has less vram
    elif gpu in _GOOD_GPUS:
        batch_down, grad_down = 24, 22
    else:
        batch_down, grad_down = 8, 64

    # analysis batch: remote always 128, gpu5 64, else 128
    if machine == "remote":
        batch_analysis = 128
    elif gpu == 5:
        batch_analysis = 64
    else:
        batch_analysis = 128

    return {
        "machine": machine,
        "batch_size_pretrain": batch_pretrain,
        "gradient_accumulation": grad_pretrain,
        "batch_size_downstream": batch_down,
        "gradient_accumulation_downstream": grad_down,
        "batch_size_analysis": batch_analysis,
    }


_HW = detect_hardware()
machine                          = _HW["machine"]
BATCH_SIZE_PRETRAIN              = _HW["batch_size_pretrain"]
GRADIENT_ACCUMULATION            = _HW["gradient_accumulation"]
BATCH_SIZE_DOWNSTREAM            = _HW["batch_size_downstream"]
GRADIENT_ACCUMULATION_DOWNSTREAM = _HW["gradient_accumulation_downstream"]
BATCH_SIZE_ANALYSIS              = _HW["batch_size_analysis"]
print(f"[config] machine={machine}, host={socket.gethostname()}, "
      f"pretrain_bs={BATCH_SIZE_PRETRAIN}, downstream_bs={BATCH_SIZE_DOWNSTREAM}")

# --------------------------------------------------
# analysis.py
KNN_K = 32      #value for k in knn
NUM_SAMPLES_CLS =   2000
NUM_SAMPLES_FULL_SEQ= 200 # lower, as this is full seq; mainly used for cka
ALIGNMENT_ANALYSIS_SIZE = 1024   # samples used for the layer-wise alignment analysis

FC_HIDDEN_DIM = 512       # what hidden size in fc head


# --------------------------------------------------
# LR SCHEDULER
WARMUP_ITERATIONS = 0.1     #what fraction of total training steps is in warmup?
DECAY_ITERATIONS  = 0.9     #what fraction of total training steps is in decay?
MIN_LR_FRACTION   = 0.2    #fraction of original lr => min_lr

# --------------------------------------------------

#early stopping
USE_EARLY_STOPPING = True
ES_CONTINUE_THRESH = 0.001
ES_PATIENCE = 3
ES_MODE = "max"  # min for loss, max for acc
# --------------------------------------------------
# finetune checkpoints directory
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251010-234252_pretrained_early_fusion/"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251011-234349_pretrained_middle_fusion"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251013-010227_pretrained_late_fusion"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251014-034432_pretrained_asymmetric_fusion"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251015-081211_pretrained_optuna1"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251016-062038_pretrained_optuna2"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251010-085859_pretrained_baseline"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251028_finetune_comparison"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251025-105249_pretrained_bl_full_coattn"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251030-192145_pretrained_latefusion_cka"
FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251102-122009_pretrained_early_fusion_cka"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251111-222754_pretrained_hybrid1"
# FINETUNE_CHECKPOINTS_DIR = "res/checkpoints/20251113-080744_pretrained_hybrid2"


@dataclass
class ViLBERTConfig:
    """
    Single source of truth for model + training configuration.

    Replaces the old ViLBERTConfig/ExperimentConfig split: experiment runners
    build this directly. Dataloader knobs (num_workers, prefetch, …) and the
    machine type live here too, defaulting to the module-level globals.
    """
    # model architecture
    embedding_dim: int = EMBEDDING_DIM
    vocab_size: int = VOCAB_SIZE
    num_hidden_layers: int = NUM_HIDDEN_LAYERS
    num_attention_heads: int = NUM_ATTENTION_HEADS
    num_bi_attention_heads: int = NUM_BI_ATTENTION_HEADS
    dropout_prob: float = DROPOUT_PROB
    img_size: tuple = IMG_SIZE

    # cross-attention placement
    text_cross_attention_layers: list = field(default_factory=lambda: list(T_BIATTENTION_IDS))
    vision_cross_attention_layers: list = field(default_factory=lambda: list(V_BIATTENTION_IDS))

    # training
    learning_rate: float = PRETRAIN_LEARNING_RATE
    epochs: int = PRETRAIN_EPOCHS
    seed: int = SEED
    train_test_ratio: float = TRAIN_TEST_RATIO
    use_contrastive_loss: bool = USE_CONTRASTIVE_LOSS
    batch_size: int = BATCH_SIZE_DOWNSTREAM           # downstream / finetune batch
    pretrain_batch_size: int = BATCH_SIZE_PRETRAIN
    gradient_accumulation: int = GRADIENT_ACCUMULATION
    pretraining_tasks: list = field(
        default_factory=lambda: [Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM]
    )

    # dataloader knobs (previously bare module globals)
    num_workers: int = NUM_WORKERS
    prefetch: Optional[int] = PREFETCH
    persistent_workers: bool = PERSISTENT_WORKERS
    pin_memory: bool = PIN_MEMORY

    # analysis
    batch_size_analysis: int = BATCH_SIZE_ANALYSIS
    alignment_analysis_size: int = ALIGNMENT_ANALYSIS_SIZE

    # data / misc
    preprocessed_path: str = PREPROCESSED_PATH
    machine: str = machine

    # computed (not a constructor argument)
    depth: int = field(init=False)

    def __post_init__(self):
        assert len(self.text_cross_attention_layers) == len(self.vision_cross_attention_layers)
        if self.text_cross_attention_layers:
            assert max(self.text_cross_attention_layers) <= 12
        if self.vision_cross_attention_layers:
            assert max(self.vision_cross_attention_layers) <= 12
        assert len(self.text_cross_attention_layers) <= DEPTH
        self.depth = DEPTH + len(self.text_cross_attention_layers)

    # dict-like interface (used by experiment_tracker.save_results and serialization)
    def items(self):
        return vars(self).items()

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def __str__(self):
        return f"ViLBERTConfig({', '.join(f'{k}={v}' for k, v in self.items())})"

    def to_dict(self) -> dict:
        d = vars(self).copy()
        d["pretraining_tasks"] = [t.value for t in d["pretraining_tasks"]]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ViLBERTConfig":
        tasks = [Task(t) for t in d.get("pretraining_tasks", [])]
        return cls(
            embedding_dim=d.get("embedding_dim", EMBEDDING_DIM),
            vocab_size=d.get("vocab_size", VOCAB_SIZE),
            num_hidden_layers=d.get("num_hidden_layers", NUM_HIDDEN_LAYERS),
            num_attention_heads=d.get("num_attention_heads", NUM_ATTENTION_HEADS),
            num_bi_attention_heads=d.get("num_bi_attention_heads", NUM_BI_ATTENTION_HEADS),
            dropout_prob=d.get("dropout_prob", DROPOUT_PROB),
            img_size=d.get("img_size", IMG_SIZE),
            text_cross_attention_layers=d.get("text_cross_attention_layers", list(T_BIATTENTION_IDS)),
            vision_cross_attention_layers=d.get("vision_cross_attention_layers", list(V_BIATTENTION_IDS)),
            learning_rate=d.get("learning_rate", PRETRAIN_LEARNING_RATE),
            epochs=d.get("epochs", PRETRAIN_EPOCHS),
            seed=d.get("seed", SEED),
            train_test_ratio=d.get("train_test_ratio", TRAIN_TEST_RATIO),
            use_contrastive_loss=d.get("use_contrastive_loss", USE_CONTRASTIVE_LOSS),
            batch_size=d.get("batch_size", BATCH_SIZE_DOWNSTREAM),
            pretrain_batch_size=d.get("pretrain_batch_size", BATCH_SIZE_PRETRAIN),
            gradient_accumulation=d.get("gradient_accumulation", GRADIENT_ACCUMULATION),
            pretraining_tasks=tasks,
            num_workers=d.get("num_workers", NUM_WORKERS),
            prefetch=d.get("prefetch", PREFETCH),
            persistent_workers=d.get("persistent_workers", PERSISTENT_WORKERS),
            pin_memory=d.get("pin_memory", PIN_MEMORY),
            batch_size_analysis=d.get("batch_size_analysis", BATCH_SIZE_ANALYSIS),
            alignment_analysis_size=d.get("alignment_analysis_size", ALIGNMENT_ANALYSIS_SIZE),
            preprocessed_path=d.get("preprocessed_path", PREPROCESSED_PATH),
        )


if __name__ == "__main__":
    config = ViLBERTConfig()
    print(len(config.__dict__))

import os
from typing import Optional


from task import Task

machine = os.environ.get("MACHINE_TYPE", "local")  # local or remote: local - my gaming pc (16gb), remote - university gpu (24gb)

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
if machine == "remote":
    BATCH_SIZE_PRETRAIN = 20
    GRADIENT_ACCUMULATION = 26  # simulated batches of 512, similar to the og vilbert paper
else:
    BATCH_SIZE_PRETRAIN = 8
    GRADIENT_ACCUMULATION = 64    # simulated batches of 128

USE_CONTRASTIVE_LOSS=False
FREEZE_UNIMODAL_ENCODERS = False
NUM_SAMPLES_CC = 500_000
# --------------------------------------------------
# data specific
IMG_SIZE = (224, 224)
PREPROCESSED_PATH = "res/preprocessed.pkl"      # not yet used, used to store precomputed datasets (in tensor form)
TRAIN_TEST_RATIO = 0.8
# what length for text tokens; is the same as num_patches + 1: 16*16 patches + cls
TOKENIZER_MAX_LEN = 197
#all the torch dataset/dataloader stuff
NUM_WORKERS = 4
PREFETCH = 3
PERSISTENT_WORKERS = False
PIN_MEMORY = False
# --------------------------------------------------
# for the src/evaluate.py part; finetunes on hateful memes or mmimdb
DOWNSTREAM_EPOCHS = 9
DOWNSTREAM_LR     = 3.4e-5

if machine == "remote":
    BATCH_SIZE_DOWNSTREAM = 24
    GRADIENT_ACCUMULATION_DOWNSTREAM = 22
else:
    BATCH_SIZE_DOWNSTREAM = 8
    GRADIENT_ACCUMULATION_DOWNSTREAM = 64
# --------------------------------------------------
# analysis.py
if machine == "remote":
    BATCH_SIZE_ANALYSIS = 128
else:
    BATCH_SIZE_ANALYSIS = 128

KNN_K = 32      #value for k in knn
NUM_SAMPLES_CLS =   2000
NUM_SAMPLES_FULL_SEQ= 200 # lower, as this is full seq; mainly used for cka

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


class ViLBERTConfig:
    def __init__(
        self,
        embedding_dim=EMBEDDING_DIM,
        vocab_size=VOCAB_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dropout_prob=DROPOUT_PROB,
        learning_rate=PRETRAIN_LEARNING_RATE,
        img_size=IMG_SIZE,
        preprocessed_path=PREPROCESSED_PATH,
        train_test_ratio=TRAIN_TEST_RATIO,
        batch_size=BATCH_SIZE_PRETRAIN,
        gradient_accumulation=GRADIENT_ACCUMULATION,
        pretraining_tasks: list = [Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM],  # default tasks to pretrain on
        text_cross_attention_layers: list[int] = T_BIATTENTION_IDS,
        vision_cross_attention_layers: list[int] = V_BIATTENTION_IDS,
        seed:int = SEED,
        use_contrastive_loss: bool = USE_CONTRASTIVE_LOSS,
        num_bi_attention_heads: int = NUM_BI_ATTENTION_HEADS,
        epochs: int = PRETRAIN_EPOCHS,
    ):
        assert len(text_cross_attention_layers) == len(vision_cross_attention_layers)
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.dropout_prob = dropout_prob
        self.learning_rate = learning_rate
        self.img_size = img_size
        self.preprocessed_path = preprocessed_path
        self.train_test_ratio = train_test_ratio
        self.batch_size = batch_size
        self.gradient_accumulation = gradient_accumulation
        self.depth = DEPTH + len(text_cross_attention_layers)  # total number of layers in transformer
        self.pretraining_tasks = pretraining_tasks
        self.seed = seed
        self.text_cross_attention_layers = text_cross_attention_layers
        self.vision_cross_attention_layers = vision_cross_attention_layers
        self.use_contrastive_loss = use_contrastive_loss
        self.num_bi_attention_heads = num_bi_attention_heads
        self.epochs = epochs
        assert len(self.text_cross_attention_layers) <= DEPTH


    def items(self):
        return vars(self).items()

    def keys(self):
        return vars(self).keys()

    def values(self):
        return vars(self).values()

    def __str__(self, ):
        return f"ViLBERTConfig({', '.join([f'{k}={v}' for k, v in self.items()])})"

    def to_dict(self,):
        config_dict = self.__dict__.copy()
        config_dict["pretraining_tasks"] = [task.value for task in config_dict["pretraining_tasks"]]

        return config_dict

    @classmethod
    def from_dict(cls, config_dict):
        pretraining_tasks = config_dict.get("pretraining_tasks", [])
        pretraining_tasks = [Task(task) for task in pretraining_tasks]

        config = cls(
            embedding_dim=config_dict.get("embedding_dim", EMBEDDING_DIM),
            vocab_size=config_dict.get("vocab_size", VOCAB_SIZE),
            num_hidden_layers=config_dict.get("num_hidden_layers", NUM_HIDDEN_LAYERS),
            num_attention_heads=config_dict.get("num_attention_heads", NUM_ATTENTION_HEADS),
            dropout_prob=config_dict.get("dropout_prob", DROPOUT_PROB),
            learning_rate=config_dict.get("learning_rate", PRETRAIN_LEARNING_RATE),
            img_size=config_dict.get("img_size", IMG_SIZE),
            preprocessed_path=config_dict.get("preprocessed_path", PREPROCESSED_PATH),
            train_test_ratio=config_dict.get("train_test_ratio", TRAIN_TEST_RATIO),
            batch_size=config_dict.get("batch_size", BATCH_SIZE_PRETRAIN),
            depth=config_dict.get("depth", DEPTH),
            pretraining_tasks=pretraining_tasks,
            text_cross_attention_layers=config_dict.get("text_cross_attention_layers", T_BIATTENTION_IDS),
            vision_cross_attention_layers=config_dict.get("vision_cross_attention_layers", V_BIATTENTION_IDS),
            seed=config_dict.get("seed", SEED),
        )
        return config



if __name__ == "__main__":
    config = ViLBERTConfig()

    print(len(config.__dict__))
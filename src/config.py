import os
from typing import Optional

from task import Task

machine = os.environ.get("MACHINE_TYPE", "local")  # local or remote: local - my gaming pc (16gb), remote - university gpu (24gb)

EMBEDDING_DIM = 768
VOCAB_SIZE    = 30522
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
DROPOUT_PROB        = 0.1
LEARNING_RATE       = 3e-5

# data specific
IMG_SIZE = (224, 224)
PREPROCESSED_PATH = "res/preprocessed.pkl"      # not yet used, used to store precomputed datasets (in tensor form)
TRAIN_TEST_RATIO = 0.8

if machine == "remote":
    BATCH_SIZE = 32
    EPOCHS = 10         # TODO: not yet used
else: 
    BATCH_SIZE = 32
    EPOCHS = 10

TOKENIZER_MAX_LEN = 192


FC_HIDDEN_DIM = 1024        # what hidden size in fc head
DEPTH = 4                  # how many co-attn layers in transformer
CROSS_ATTENTION_LAYERS = [0,2]      # first and 3rd layer are coattn


VIT_MODEL_NAME = "vit_base_patch16_224"





class ViLBERTConfig: 
    def __init__(
        self, 
        embedding_dim=EMBEDDING_DIM,
        vocab_size=VOCAB_SIZE,
        num_hidden_layers=NUM_HIDDEN_LAYERS,
        num_attention_heads=NUM_ATTENTION_HEADS,
        dropout_prob=DROPOUT_PROB,
        learning_rate=LEARNING_RATE,
        img_size=IMG_SIZE,
        preprocessed_path=PREPROCESSED_PATH,
        train_test_ratio=TRAIN_TEST_RATIO,
        batch_size=BATCH_SIZE, 
        depth=DEPTH,
        pretraining_tasks: list = [Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM],  # default tasks to pretrain on
        cross_attention_layers: list[int]= CROSS_ATTENTION_LAYERS
    ):
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
        self.depth = depth
        self.pretraining_tasks = pretraining_tasks
        self.cross_attention_layers = cross_attention_layers
        
        assert depth >= len(cross_attention_layers)
        
    
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
            learning_rate=config_dict.get("learning_rate", LEARNING_RATE),
            img_size=config_dict.get("img_size", IMG_SIZE),
            preprocessed_path=config_dict.get("preprocessed_path", PREPROCESSED_PATH),
            train_test_ratio=config_dict.get("train_test_ratio", TRAIN_TEST_RATIO),
            batch_size=config_dict.get("batch_size", BATCH_SIZE),
            depth=config_dict.get("depth", DEPTH),
            pretraining_tasks=pretraining_tasks, 
            cross_attention_layers=config_dict.get("cross_attention_layers", CROSS_ATTENTION_LAYERS)
        )
        return config
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
    BATCH_SIZE = 88
    EPOCHS = 10         # TODO: not yet used
else: 
    BATCH_SIZE = 32
    EPOCHS = 10

TOKENIZER_MAX_LEN = 192


FC_HIDDEN_DIM = 1024        # what hidden size in fc head
DEPTH = 4                  # how many co-attn layers in transformer


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
        pretraining_tasks: list = [Task.ALIGNMENT_PREDICTION, Task.MASKED_LM, Task.MASKED_IM]  # default tasks to pretrain on
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
        
    
    def items(self):
        return vars(self).items()
    
    def keys(self):
        return vars(self).keys()
    
    def values(self):
        return vars(self).values()
    
    def __str__(self, ): 
        return f"ViLBERTConfig({', '.join([f'{k}={v}' for k, v in self.items()])})"
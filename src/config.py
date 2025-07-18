EMBEDDING_DIM = 768
VOCAB_SIZE    = 30522
NUM_HIDDEN_LAYERS = 12
NUM_ATTENTION_HEADS = 12
DROPOUT_PROB        = 0.1


# data specific
IMG_SIZE = (224, 224)
PREPROCESSED_PATH = "res/preprocessed.pkl"      # not yet used, used to store precomputed datasets (in tensor form)
TRAIN_TEST_RATIO = 0.8

BATCH_SIZE = 32
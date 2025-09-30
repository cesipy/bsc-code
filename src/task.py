from enum import Enum

# pretraining tasks here
class Task(Enum):
    ALIGNMENT_PREDICTION = 1
    MASKED_LM = 2
    MASKED_IM = 3       # masked image modelling
    PLACEHOLDER = -1

    #aliases
    MLM = 2
    MIM = 3


# normal tasks here
# TODO integrate everywhere
all_task_list = ["mm_imdb", "hateful_memes", "upmc_food"]


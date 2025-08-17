from enum import Enum

class Task(Enum): 
    ALIGNMENT_PREDICTION = 1
    MASKED_LM = 2
    MASKED_IM = 3       # masked image modelling
    
    #aliases 
    MLM = 2
    MIM = 3
    
    
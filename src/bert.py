import torch
from torch import nn


from config import * 


# im not quite sure if i need this. 
# this is a individual implemetation of bert, seems like overkill. 
# i only need pretrained bert, then finetune on the tasks. 

class Bert(nn.Module): 
    def __init__(
        self,
        hidden_dim: int, 
        num_hidden_layers: int, 
        num_attention_heads: int, 
        vocab_size: int, 
    ) -> None:
        """
        @parm hidden_dim: embedding dimension for tokenized image and tokenized text
        @param num_hidden_layers: numbers of hidden layers
        @param num_attention_heads: numbers of attention heads
        @vocab_size: int classification of tokens to predict
        """
        super(Bert, self).__init__()
        self.dim        = hidden_dim
        self.attentions = self.init_attentions()
        self.mlp        = FeedFoward()
        
        
    def init_attentions(num_attention_heads: int): 
        attentions = []
        for i in range(num_attention_heads): 
            attention = AttentionBlock()
            attentions.append(attention)
            
        return attentions
        
    def forward(self, x): 
        ...
        

class AttentionBlock(nn.Module): 
    def __init__(
        self,
        hidden_dim: int, 
    ): 
        self.dim = hidden_dim
        
        self.relu = nn.ReLU()
        self.attention = nn.Softmax(dim=-1)
        
        
        # for q: 
        q = nn.Linear(hidden_dim, hidden_dim)
        
        #for k: 
        v = nn.Linear(hidden_dim, hidden_dim)
        
        # for v: 
        k = nn.Linear(hidden_dim, hidden_dim)
        
    def forward(self, x): 
        # implement method for gettin q, k, v from input
        
        q, k, v = self.get_projections(x)
        
        # why the transposing
        qk_scaled = torch.matmul(q, k.transpose(-2, -1)) / (self.dim ** 0.5)
        
        attended = self.attention(qk_scaled)
        
        attended_v = torch.matmul(attended, v)
        
        return attended_v
         
        
        
    
class FeedFoward(nn.Module): 
    ...
    


def main(): 
    bert = Bert()
    
    
if __name__ == '__main__':
    main()
    
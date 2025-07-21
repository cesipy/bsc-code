import torch
from torch import nn
from einops import rearrange, repeat       
from einops.layers.torch import Rearrange  
from config import * 


class Attention_Block(nn.Module):
    # Some inspiration from the deep learning assignment 3, 
    # Credits to javier urena
  def __init__(self, dim, heads=8, dropout=0.):
    super(Attention_Block, self).__init__()

    self.dk = dim // heads # inner head dimension. Dim and number of heads must be multiple numbers between them
    self.heads = heads

    self.norm = nn.LayerNorm(dim)
    self.dropout = nn.Dropout(dropout)

    self.query = nn.Linear(in_features=dim, out_features=dim)
    self.key   = nn.Linear(in_features=dim, out_features=dim)
    self.value = nn.Linear(in_features=dim, out_features=dim)

    self.softmax = nn.Softmax(dim=-1)
    
    self.rearrange_qkv = Rearrange('b n (h d) -> b h n d', h=heads)
    

    self.to_out = nn.Sequential(
        nn.Linear(dim, dim),
        nn.Dropout(dropout)
    ) if dim != self.dk else nn.Identity()

  def forward(self, x):
    x = self.norm(x)
    
    x_proj_k = self.key(x)
    x_proj_q = self.query(x)
    x_proj_v = self.value(x)
    
    q = self.rearrange_qkv(x_proj_q)
    k = self.rearrange_qkv(x_proj_k)
    v = self.rearrange_qkv(x_proj_v)

    #attention mechanism softmax(Q, K) /sqrt(dk)
    qk_scaled = torch.matmul(q, k.transpose(-1, -2)) * self.dk ** -0.5
    attention_qk = self.softmax(qk_scaled)
    
    attention = torch.matmul(attention_qk, v)
    
    attention = rearrange(attention, 'b h n d -> b n (h d)')
    
    return self.to_out(attention)

class CrossAttention(nn.Module): 
    def __init__(self, dim, heads=8, dropout=0.): 
        super(CrossAttention, self).__init__()
        
        self.dk = dim // heads  # inner head dimension. Dim and number of heads must be multiple numbers between them
        
        self.dim = dim
        self.heads = heads
        self.dropout= dropout
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        
        self.query1 = nn.Linear(in_features=dim, out_features=dim)
        self.key1   = nn.Linear(in_features=dim, out_features=dim)
        self.value1 = nn.Linear(in_features=dim, out_features=dim)
        
        self.query2 = nn.Linear(in_features=dim, out_features=dim)
        self.key2   = nn.Linear(in_features=dim, out_features=dim)
        self.value2 = nn.Linear(in_features=dim, out_features=dim)
        
        self.softmax = nn.Softmax(dim=-1)
        self.rearrange_qkv = Rearrange('b n (h d) -> b h n d', h=heads)
        
        self.to_out_proj1 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 
        
        self.to_out_proj2 = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        ) 
        
    def forward(self, text_tensor, vision_tensor): 
        
        # store the inputs for later residuals
        text_residual = text_tensor
        vision_residual = vision_tensor
        
        text_tensor = self.norm1(text_tensor)
        vision_tensor = self.norm2(vision_tensor)
        
        text_proj_k = self.key1(text_tensor)
        text_proj_q = self.query1(text_tensor)
        text_proj_v = self.value1(text_tensor)
        
        vision_proj_k = self.key2(vision_tensor)
        vision_proj_q = self.query2(vision_tensor)
        vision_proj_v = self.value2(vision_tensor)
        
        
        text_q = self.rearrange_qkv(text_proj_q) 
        text_k = self.rearrange_qkv(text_proj_k)
        text_v = self.rearrange_qkv(text_proj_v)
        
        vision_q = self.rearrange_qkv(vision_proj_q)
        vision_k = self.rearrange_qkv(vision_proj_k)
        vision_v = self.rearrange_qkv(vision_proj_v)
        
        # attention mechanism softmax(Q, K) /sqrt(dk)
        # text query, vision key and value
        text_qk_scaled = torch.matmul(text_q, vision_k.transpose(-1, -2)) * self.dk ** -0.5
        text_attention_qk = self.softmax(text_qk_scaled)
        
        # TODO: add attention dropout
        # vision_attention_qk = self.dropout(vision_attention_qk)
        # text_attention_qk = self.dropout(text_attention_qk)
        
        attention_1 = torch.matmul(text_attention_qk, vision_v)
        attention_1 = rearrange(attention_1, 'b h n d -> b n (h d)')
        
        # vision query, text key and value
        vision_qk_scaled = torch.matmul(vision_q, text_k.transpose(-1, -2)) * self.dk ** -0.5
        vision_attention_qk = self.softmax(vision_qk_scaled)
        
        # TODO: add attention dropout
        # vision_attention_qk = self.dropout(vision_attention_qk)

        
        attention_2 = torch.matmul(vision_attention_qk, text_v)
        attention_2 = rearrange(attention_2, 'b h n d -> b n (h d)')
        
        return attention_1, attention_2, text_residual, vision_residual
        
        # text_out = self.to_out_proj1(attention_1)
        # vision_out = self.to_out_proj2(attention_2)
        
        #residuals missing?
        # return text_out, vision_out
        
class CrossAttentionOutput(nn.Module): 
    # here the residuals are handled
    def __init__(self, dim): 
        super(CrossAttentionOutput, self).__init__()
        # TODO: add layernorma like in vilbert implementation 
        # https://github.com/facebookresearch/vilbert-multi-task/blob/f22b84a9918a9aea2106e14ac1f6b32ad71492e3/vilbert/vilbert.py#L831
        self.dim = dim
        self.projection_text = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(DROPOUT_PROB)
        )
        
        self.projection_vision = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(DROPOUT_PROB)
        )
        
        self.norm_text = nn.LayerNorm(dim)
        self.norm_vision = nn.LayerNorm(dim)
        
    def forward(self, text_attn, vision_attn, text_input,  vision_input):
        # text_embedded: [bs, seq_len, embedding_dim]
        # text_input: [bs, seq_len, embedding_dim]
        # vision_embedded: [bs, num_patches, embedding_dim]
        # vision_input: [bs, num_patches, embedding_dim]
        
        current_text = self.projection_text(text_attn)
        current_vision = self.projection_vision(vision_attn)
        
        # residual conns: 
        current_text = current_text + text_input
        current_vision = current_vision + vision_input
        
        
        # TODO: normalization
        current_text = self.norm_text(current_text)
        current_vision = self.norm_vision(current_vision)
        
        return current_text, current_vision

class CrossAttentionBlock(nn.Module): 
    def __init__(self, dim, heads=8, dropout=0.): 
        super(CrossAttentionBlock, self).__init__()
        self.dim = dim
        self.heads = heads
        
        self.cross_attention = CrossAttention(dim=dim, heads=heads, dropout=dropout)
        self.output = CrossAttentionOutput(dim)
        
    def forward(self, text_tensor, vision_tensor): 
        
        text_attn, vision_attn, text_residual, vision_residual = self.cross_attention(
            text_tensor=text_tensor, 
            vision_tensor=vision_tensor
        )
        text_output, vision_output = self.output(
            text_attn=text_attn, 
            vision_attn=vision_attn, 
            text_input=text_residual, 
            vision_input=vision_residual
        )
        
        return text_output, vision_output

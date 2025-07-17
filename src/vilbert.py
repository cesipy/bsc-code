import torch
from torch import nn

from einops import rearrange, repeat        # functions for tensor dimension reordering and repeating
from einops.layers.torch import Rearrange   # same as rearrange but works as a torch layer
from transformers import (
    #BERT stuff
    BertModel, 
    BertConfig, 
    BertTokenizer, 
    BertTokenizerFast, 
    
    # ViT stuff
    ViTConfig, 
    ViTModel,
    ViTImageProcessor,
    
    # type hinting stuff
    PreTrainedTokenizerFast,
)

from PIL import Image


DIM = 768

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

class CrossAttentionBlock(nn.Module): 
    def __init__(self, dim, heads=8, dropout=0.): 
        super(CrossAttentionBlock, self).__init__()
        
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
        attention_1 = torch.matmul(text_attention_qk, vision_v)
        attention_1 = rearrange(attention_1, 'b h n d -> b n (h d)')
        
        # vision query, text key and value
        vision_qk_scaled = torch.matmul(vision_q, text_k.transpose(-1, -2)) * self.dk ** -0.5
        vision_attention_qk = self.softmax(vision_qk_scaled)
        attention_2 = torch.matmul(vision_attention_qk, text_v)
        attention_2 = rearrange(attention_2, 'b h n d -> b n (h d)')
        
        text_out = self.to_out_proj1(attention_1)
        vision_out = self.to_out_proj2(attention_2)
        
        #residuals missing?
        return text_out, vision_out
        
        


class ViLBERT(nn.Module): 
    def __init__(self, config: BertConfig): 
        super(ViLBERT, self).__init__()
        
        # loads pretrained transformers, no head for task. with transformers.BertFor.... I 
        # could download pretrained transformers for specific tasks
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        self.attention_layer = Attention_Block(dim=DIM, heads=1, dropout=0.1)
        
        self.coattention_layer = CrossAttentionBlock(dim=DIM, heads=1, dropout=0.1)
        # cross_attention_layers = []
        # for i in range(config.num_hidden_layers): 
        #     cross_attention_layers.append(CrossAttentionLayer())
            
        # self.cross_attention = nn.ModuleList(cross_attention_layers)
    
    def forward_coattention(
        self, 
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None, 
        output_attentions=False,
        output_hidden_states=False,
    ): 
        text_output = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        image_output = self.vit(
            pixel_values=image_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        text_tensor = text_output.last_hidden_state
        image_tensor = image_output.last_hidden_state
        # shape text: [bs, seq_len, embedding_dim
        # shape image: [bs, num_patches, embedding_dim]

        text_embeding, vision_embedding = self.coattention_layer(text_tensor, image_tensor)
        return text_embeding, vision_embedding

    def forward_concat(
        self, 
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None, 
        output_attentions=False,
        output_hidden_states=False,
    ): 
        text_output = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        image_output = self.vit(
            pixel_values=image_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        text_tensor = text_output.last_hidden_state
        image_tensor = image_output.last_hidden_state
        # dim 1, bc we want to add the image embeddings to the text embeddings
        # shape text: [bs, seq_len, embedding_dim
        # shape image: [bs, num_patches, embedding_dim]
        concat_embedding = torch.concat([text_tensor, image_tensor], dim=1)
    
        return self.attention_layer(concat_embedding)


    def forward_naive(
        self, 
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None, 
        output_attentions=False,
        output_hidden_states=False,
    ): 
        text_output = self.bert(
            input_ids=text_input_ids,
            attention_mask=text_attention_mask,
            token_type_ids=text_token_type_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        image_output = self.vit(
            pixel_values=image_pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        
        return text_output.last_hidden_state, image_output.last_hidden_state


def main(): 
    config = BertConfig()
    model = ViLBERT(config)
    
    tokenizer: PreTrainedTokenizerFast = BertTokenizerFast.from_pretrained("bert-base-uncased")
    
    text = "hello, what is you name?"
    text_tokens = tokenizer(text, return_tensors="pt")
    # can be simply passed to the model without further processing
    #print(type(text_tokens))        
    # transformers.tokenization_utils_base.BatchEncoding, simple dictionary

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    image = Image.open("res/image.png").convert("RGB")
    image_processed = image_processor(images=image, return_tensors="pt")

    res_linguistic, res_visual = model.forward_naive(
        text_input_ids=text_tokens["input_ids"],
        text_attention_mask=text_tokens["attention_mask"],
        text_token_type_ids=text_tokens.get("token_type_ids", None),
        image_pixel_values=image_processed["pixel_values"],
        image_attention_mask=image_processed.get("attention_mask", None),
    )
    
    concat = torch.cat((res_linguistic, res_visual), dim=1)
    print(f"lingu shape: {res_linguistic.shape}")
    print(f"visual shape: {res_visual.shape}")
    print(f"concat shape: {concat.shape}")    
    
    # console outputs
    # lingu shape: torch.Size([1, 9, 768])
    # visual shape: torch.Size([1, 197, 768])
    # concat shape: torch.Size([1, 206, 768])  


    print("-"*15)
    print("concatted model now")
    res = model.forward_concat(
        text_input_ids=text_tokens["input_ids"],
        text_attention_mask=text_tokens["attention_mask"],
        text_token_type_ids=text_tokens.get("token_type_ids", None),
        image_pixel_values=image_processed["pixel_values"],
        image_attention_mask=image_processed.get("attention_mask", None),
    )
    
    print(f"concatted shape: {res.shape}")
    
    print("-"*15)
    print("cattention model now")
    res_text, res_vision = model.forward_coattention(
        text_input_ids=text_tokens["input_ids"],
        text_attention_mask=text_tokens["attention_mask"],
        text_token_type_ids=text_tokens.get("token_type_ids", None),
        image_pixel_values=image_processed["pixel_values"],
        image_attention_mask=image_processed.get("attention_mask", None),
    )
    
    print(f"text shape: {res_text.shape}")
    print(f"vision shape: {res_vision.shape}")

    
    
    
    
if __name__ == "__main__":
    main()
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

from attention import Attention_Block, CrossAttention, CrossAttentionBlock, FeedForward_Block

import utils
from config import *

FC_HIDDEN_DIM = 1024

class ViLBERT(nn.Module): 
    def __init__(self,): 
        super(ViLBERT, self).__init__()
        
        # loads pretrained transformers, no head for task. with transformers.BertFor.... I 
        # could download pretrained transformers for specific tasks
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        utils.freeze_all_layers(self.bert)
        utils.freeze_all_layers(self.vit)
        
        self.attention_layer = Attention_Block(dim=EMBEDDING_DIM, heads=1, dropout=DROPOUT_PROB)
        
        # self.cross_attention = CrossAttentionBlock(
        #     dim=EMBEDDING_DIM,
        #     heads=1,
        # )
        
        self.cross_attention = []

        self.depth = 5
        for i in range(self.depth): 
            self.cross_attention.append(CrossAttentionBlock(dim=EMBEDDING_DIM, heads=8, dropout=DROPOUT_PROB))

        self.cross_attention = nn.ModuleList(self.cross_attention)

        # pretrain heads
        self.alignment_fc = nn.Linear(2*EMBEDDING_DIM, 2)
        self.mlm = nn.Linear(EMBEDDING_DIM, self.bert.config.vocab_size)
        # TODO: implement, skip for now. i still need an object detector doing it
        # self.mim = nn.Linear(EMBEDDING_DIM, NUM_VISUAL_CLASSES)
        
        # for hateful memes classifiction
        self.fc = nn.Sequential(
            nn.Linear(2*EMBEDDING_DIM, FC_HIDDEN_DIM),
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(FC_HIDDEN_DIM, FC_HIDDEN_DIM//2), 
            nn.ReLU(inplace=True),
            nn.Dropout(DROPOUT_PROB),
            nn.Linear(FC_HIDDEN_DIM//2, 1), 
        )

        
        
    def forward(
        self,
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        
        text_embedding, image_embedding = self.forward_coattention(
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            text_token_type_ids=text_token_type_ids,
            image_pixel_values=image_pixel_values,
            image_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        # print(f"shape of text_embedding: {text_embedding.shape}")
        # print(f"shape of image_embedding: {image_embedding.shape}")
        concatted_embedding = torch.cat([text_embedding, image_embedding], dim=1)
        # print(f"shape of concatted_embedding: {concatted_embedding.shape}")
        out = self.fc(concatted_embedding)
        return out
    
    def forward_coattention(
        self, 
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None, 
        output_attentions=False,
        output_hidden_states=False,
        extract_cls=True            # used for pretraining. should the cls token be extracted?
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

        # text_embedding, vision_embedding = self.cross_attention(text_tensor, image_tensor)
        # TODO: fix naming conflict - fix input param name
        text_embedding = text_tensor
        vision_embedding = image_tensor
        
        for i in range(len(self.cross_attention)):
            text_embedding, vision_embedding = self.cross_attention[i](
                text_tensor=text_embedding, 
                vision_tensor=vision_embedding
            )


        if not extract_cls: 
            return text_embedding, vision_embedding
            
        
        #extract the cls token. 
        # TODO: implement also gap pooling for comparison
        text_embedding = text_embedding[:, 0, :]  
        vision_embedding = vision_embedding[:, 0, :] 
        return text_embedding, vision_embedding

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

    def forward_pretrain(
        self, 
        text_input_ids,
        text_attention_mask=None,
        text_token_type_ids=None,
        image_pixel_values=None,
        image_attention_mask=None, 
        output_attentions=False,
        output_hidden_states=False,
        tasks:list[str]= None,      #TODO: make tasks an enunm
    ): 

        
        
        if "mlm" in tasks:
            # text_embedding_tensor = self.__forward_masked_text(
            #     ...
            # )
            text_seqs, vision_seqs = self.forward_coattention(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                image_pixel_values=image_pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                extract_cls=False,  # do not extract cls token for pretraining
            )
            mlm_logits = self.mlm(text_seqs)
            
            
        # TODO: later, needs an object regonition model to learn the output
        # distribution from  - minimize kl div for this task
        # if 'task_masked_image' in tasks:
        #     image_embedding_tensor = self.__forward_masked_image(
        #         ...
        #     )
        
        if "alignment_prediction" in tasks: 
            text_embedding, image_embedding = self.forward_coattention(
                text_input_ids=text_input_ids,
                text_attention_mask=text_attention_mask,
                text_token_type_ids=text_token_type_ids,
                image_pixel_values=image_pixel_values,
                image_attention_mask=image_attention_mask,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                extract_cls=True, # only cls needed. nothing else
            )
            shared_embedding = torch.cat([text_embedding, image_embedding], dim=1)
            alignment_logits = self.alignment_fc(shared_embedding)
            
            
        
        

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
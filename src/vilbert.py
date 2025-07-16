import torch
from torch import nn
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

class CrossAttentionLayer: 
    def __init__(self,): 
        ...


class ViLBERT(nn.Module): 
    def __init__(self, config: BertConfig): 
        super(ViLBERT, self).__init__()
        
        # loads pretrained transformers, no head for task. with transformers.BertFor.... I 
        # could download pretrained transformers for specific tasks
        self.bert = BertModel.from_pretrained("google-bert/bert-base-uncased")
        self.vit = ViTModel.from_pretrained("google/vit-base-patch16-224")
        
        # cross_attention_layers = []
        # for i in range(config.num_hidden_layers): 
        #     cross_attention_layers.append(CrossAttentionLayer())
            
        # self.cross_attention = nn.ModuleList(cross_attention_layers)
    
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
    #print(type(text_tokens))        # transformers.tokenization_utils_base.BatchEncoding, simple dictionary

    image_processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    image = Image.open("res/image.png").convert("RGB")
    image_processed = image_processor(images=image, return_tensors="pt")

    res_linguistic, res_visual = model(
        text_input_ids=text_tokens["input_ids"],
        text_attention_mask=text_tokens["attention_mask"],
        text_token_type_ids=text_tokens.get("token_type_ids", None),
        image_pixel_values=image_processed["pixel_values"],
        image_attention_mask=image_processed.get("attention_mask", None),
    )
    
    print(f"lingu shape: {res_linguistic.shape}")
    print(f"visual shape: {res_visual.shape}")
    
    
if __name__ == "__main__":
    main()
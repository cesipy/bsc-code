import torch
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from ckatorch import CKA


# the below modules wrap vilbert to one modality, so i can use ckatorch library.
# this library takes two models as input and calculates the batch cka for them
# however, my tests found, that this is more or less the same as using cka_batch in my own implementation.
# the only point: they efficently use pytorch hooks to save memory.

class ViLBERTTextWrapper(nn.Module):
    def __init__(
        self, vilbert_model,
        layer_selection: int = -1
    ):
        super().__init__()
        self.vilbert = vilbert_model
        self.layer_idx = layer_selection
        self.output_layer = nn.Identity()

    def forward(self, **batch_data):
        # all the data from batch from dataloader are passed
        device = next(self.vilbert.parameters()).device
        text = {}
        for k, v in batch_data["text"].items():
            if v.dim() > 2:
                text[k] = v.squeeze(1).to(device)

            else:
                text[k] = v.to(device)

        image = {}
        for k, v in batch_data["img"].items():
            if v.dim() > 4:
                image[k] = v.squeeze(1).to(device)
            else:
                image[k] = v.to(device)

        res = self.vilbert.forward_coattention(
            text_input_ids=text["input_ids"],
            text_attention_mask=text["attention_mask"],
            text_token_type_ids=text.get("token_type_ids", None),
            image_pixel_values=image["pixel_values"],
            image_attention_mask=image.get("attention_mask", None),
            extract_cls=True,
            save_intermediate_representations=True
        )

        text_embedding, vision_embedding, intermediate_reprs = res
        if self.layer_idx == -1:
            text_repr = intermediate_reprs[-1]["text_embedding"]
        else:
            text_repr = intermediate_reprs[self.layer_idx]["text_embedding"]

        # did not work with cls, as cka_batch needs shape [bs, num_tokens, dim]
        # cls_token = text_repr[:, 0, :].to(device)
        text_repr = text_repr.to(device)
        return self.output_layer(text_repr)


class ViLBERTVisionWrapper(nn.Module):

    def __init__(
        self,
        vilbert_model,
        layer_selection: int = -1
    ):
        super().__init__()
        self.vilbert = vilbert_model
        self.layer_idx = layer_selection
        self.output_layer = nn.Identity()

    def forward(self, **batch_data):
        device = next(self.vilbert.parameters()).device
        text = {}
        for k, v in batch_data["text"].items():
            if v.dim() > 2:
                text[k] = v.squeeze(1).to(device)
            else:
                text[k] = v.to(device)




        image = {}
        for k, v in batch_data["img"].items():
            if v.dim() > 4:
                image[k] = v.squeeze(1).to(device)
            else:
                image[k] = v.to(device)


        res = self.vilbert.forward_coattention(
            text_input_ids=text["input_ids"],
            text_attention_mask=text["attention_mask"],
            text_token_type_ids=text.get("token_type_ids", None),
            image_pixel_values=image["pixel_values"],
            image_attention_mask=image.get("attention_mask", None),
            extract_cls=True,
            save_intermediate_representations=True
        )


        _,_, intermediate_reprs = res


        # get the correct layer
        if self.layer_idx == -1:
            vision_repr = intermediate_reprs[-1]["vision_embedding"]
        else:
            vision_repr = intermediate_reprs[self.layer_idx]["vision_embedding"]

        # as above: did not work with cls, as cka_batch needs shape [bs, num_tokens, dim]
        # cls_token = vision_repr[:, 0, :].to(device)
        vision_repr= vision_repr.to(device)
        return self.output_layer(vision_repr)



#----------------------------------






def analyze_layer_alignment_with_ckatorch(
    model,
    dataloader,
    layer_idx: int = -1,
    epochs: int = 5,
    device: str = "cuda"
):

    #wrappers
    text_model = ViLBERTTextWrapper(model, layer_selection=layer_idx)
    vision_model = ViLBERTVisionWrapper(model, layer_selection=layer_idx)
    text_model = text_model.to(device)
    vision_model = vision_model.to(device)


    cka_calculator = CKA(
        first_model=text_model,
        second_model=vision_model,
        layers=["output_layer"],  # hook for outputlayer
        first_name=f"text_layer_{layer_idx}",
        second_name=f"vision_layer_{layer_idx}",
        device=device
    )


    with torch.no_grad():
        cka_matrix = cka_calculator(dataloader, epochs=epochs)


    # 1x1
    cka_score = cka_matrix[0, 0].item()

    print(f"Layer {layer_idx} CKA alignment score: {cka_score:.4f}")

    return cka_score

def analyze_all_layers_alignment(model, dataloader, epochs: int = 3):
    """Analyze alignment across all layers"""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.eval()

    print("Analyzing layer-wise alignment with ckatorch...")

    layer_scores = {}

    for layer_idx in range(model.depth):

        cka_score = analyze_layer_alignment_with_ckatorch(
            model=model,
            dataloader=dataloader,
            layer_idx=layer_idx,
            epochs=epochs,
            device=device
        )
        layer_scores[layer_idx] = cka_score

        # TODO: incorporate into my existing modules (analysis.py)
        is_cross_attn = layer_idx in model.config.cross_attention_layers
        print(f"Layer {layer_idx} (cross-attn: {is_cross_attn}): CKA = {cka_score:.4f}")


    return layer_scores
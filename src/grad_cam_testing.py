import torch
import cv2
import numpy as np
import pytorch_grad_cam

from pytorch_grad_cam import (
    GradCAM, FEM, HiResCAM, ScoreCAM, GradCAMPlusPlus,
    AblationCAM, XGradCAM, EigenCAM, EigenGradCAM,
    LayerCAM, FullGrad, GradCAMElementWise, KPCA_CAM, ShapleyCAM,
    FinerCAM
)
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import (
    show_cam_on_image, deprocess_image, preprocess_image
)
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget, ClassifierOutputReST


from vilbert import ViLBERT
from config import *

import datasets
import utils

def decode_batch_text(text_tokens, tokenizer, batch_size):
    """Decode tokenized text back to readable strings"""
    decoded_texts = []

    for i in range(batch_size):
        # Extract tokens for sample i
        input_ids = text_tokens['input_ids'][i]
        # Decode back to text
        decoded_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        decoded_texts.append(decoded_text)

    return decoded_texts

def convert_original_photo( image_pixels):
    img_for_display = image_pixels.permute(0, 2, 3, 1).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_for_display = (img_for_display * std + mean).clip(0, 1)
    return img_for_display


def convert_to_display(image_pixels_copy, grayscale_cam, i ):
    img_for_display = image_pixels_copy[i].permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_for_display = (img_for_display * std + mean).clip(0, 1)

    cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)
    return cam_image

#from https://jacobgil.github.io/pytorch-gradcam-book/vision_transformers.html
def reshape_transform(tensor, height=14, width=14):
    result = tensor[:, 1 :  , :].reshape(tensor.size(0),
        height, width, tensor.size(2))

    # Bring the channels to the first dimension,
    # like in CNNs.
    result = result.transpose(2, 3).transpose(1, 2)
    return result


class ViLBERTWrapper(torch.nn.Module):
    def __init__(self, model, text_inputs):
        super().__init__()
        self.model = model
        self.text_inputs = text_inputs

    def forward(self, image_pixel_values):
        text_cls, vision_cls = self.model(
            text_input_ids=self.text_inputs["input_ids"],
            text_attention_mask=self.text_inputs["attention_mask"],
            image_pixel_values=image_pixel_values,
        )

        combined = text_cls * vision_cls
        return self.model.fc(combined)

def calculate_entropy(intermediate_result):

    attn_probs = intermediate_result / (np.sum(intermediate_result) + 1e-10)

    entr = - np.sum(attn_probs * np.log(attn_probs + 1e-10))

    return entr


def analyse_batch(grayscale_cam, image_pixels_copy,text_list,  filename:str, layer_indx:int,dir="res/attn-plots", ):
    os.makedirs(name=dir, exist_ok=True)
    s = grayscale_cam.shape[0]
    entrs = np.zeros(s)

    for i in range(s):
        curr_grayscale_cam = grayscale_cam[i, :]

        entr = calculate_entropy(curr_grayscale_cam)
        entrs[i] = entr

        current_text = text_list[i]
        with open(f"{dir}/{filename}_b{i}_text.txt", "w") as f:
            f.write(current_text)

        og_photo = convert_original_photo(image_pixels_copy[i:i+1])
        cv2.imwrite(f"{dir}/{filename}_b{i}_og.jpg", cv2.cvtColor(np.uint8(og_photo[0]*255), cv2.COLOR_RGB2BGR))

        curr_cam_image = convert_to_display(image_pixels_copy.clone(), curr_grayscale_cam, i=i)
        if layer_indx == None:
            cv2.imwrite(f"{dir}/{filename}_b{i}.jpg", cv2.cvtColor(curr_cam_image, cv2.COLOR_RGB2BGR))
        else:
            cv2.imwrite(f"{dir}/{filename}_b{i}_{layer_indx}.jpg", cv2.cvtColor(curr_cam_image, cv2.COLOR_RGB2BGR))


    print(f"Avg entropy for {filename}, layer {layer_indx}: {np.mean(entrs)}")


# .---------------------------------------------------------------
utils.set_seeds(SEED)

# train_dl, test_dl = datasets.get_hateful_memes_datasets(
#     train_test_ratio=TRAIN_TEST_RATIO,
#     batch_size=50,
#     num_workers=NUM_WORKERS,
#     pin_memory=PIN_MEMORY,
#     prefetch_factor=PREFETCH,
#     persistent_workers=PERSISTENT_WORKERS,
#     use_train_augmentation=False,

# )
hm_dl, dl, imdb_dl = datasets.get_alignment_dataloaders(
    batch_size=50,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH,

)
tok = datasets.BertTokenizerFast.from_pretrained("bert-base-uncased")
train_item = next(iter(dl))
text = "gun"
# text_tokens = datasets.get_text_embedding(text=text, tokenizer=tok)
# bs = train_item["img"]["pixel_values"].shape[0]
# text_tokens = {k:v.expand(bs, -1) for k,v in text_tokens.items()}

print(train_item.keys())    # dict_keys(['img', 'label', 'text'])
print(train_item["img"]["pixel_values"].shape)   # torch.Size([2, 1, 3, 224, 224])


# already tokenized and preprocessed for vit
text_tokens = {k: v.squeeze(1) for k, v in train_item["text"].items()}
decoded_texts = decode_batch_text(text_tokens, tok, dl.batch_size)

print(text_tokens.keys())   # dict_keys(['input_ids', 'token_type_ids', 'attention_mask'])
image_pixels = train_item["img"]["pixel_values"].squeeze(1)

text_tokens_copy = {k: v.clone() for k, v in text_tokens.items()}
image_pixels_copy = image_pixels.clone()

print(f"text_token shape: {text_tokens['input_ids'].shape}, imgage_pixels shape: {image_pixels.shape}")

config = ViLBERTConfig()
model_vilbert_untrained = ViLBERT(config=config)


model_vilbert = ViLBERT.load_model(load_path="res/checkpoints/hm_finetuned_e2.pt", )


model = ViLBERTWrapper(model=model_vilbert, text_inputs=text_tokens)
model_untrained = ViLBERTWrapper(model=model_vilbert_untrained, text_inputs=text_tokens_copy)

# TODO: better fetching
# clayers are not working,as they return tuple, grad-cam expects single tensor
# target_layers = [model.model.vit.blocks[4], model.model.vit.blocks[5], model.model.vit.blocks[6]]
# target_layers_untrained = [model_untrained.model.vit.blocks[4], model_untrained.model.vit.blocks[5], model_untrained.model.vit.blocks[6]]

for i in range(12):
    target_layers = [ model.model.vit.blocks[i] ]
    target_layers_untrained = [ model_untrained.model.vit.blocks[i] ]

    #TODO: maybe use other method for it, in import there are many, also here:
    # https://github.com/jacobgil/pytorch-grad-cam/blob/781dbc0d16ffa95b6d18b96b7b829840a82d93d1/cam.py#L67

    with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as c:


        grayscale_cam = c(input_tensor=image_pixels, targets=None)
        analyse_batch(grayscale_cam=grayscale_cam,text_list=decoded_texts, image_pixels_copy=image_pixels_copy,layer_indx=i,filename= "grad_cam_attention_finetuned")


    with GradCAM(model=model_untrained, target_layers=target_layers_untrained, reshape_transform=reshape_transform) as c:

        grayscale_cam = c(input_tensor=image_pixels_copy, targets=None)
        analyse_batch(grayscale_cam=grayscale_cam, image_pixels_copy=image_pixels_copy,layer_indx=i,filename= "grad_cam_attention_untrained", text_list=decoded_texts)


# one for all layers, is automatically averaged
target_layers = [ model.model.vit.blocks[i] for i in range(12)]
target_layers_untrained = [ model_untrained.model.vit.blocks[i] for i in range(12)]

with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as c:

        grayscale_cam = c(input_tensor=image_pixels, targets=None)
        analyse_batch(grayscale_cam=grayscale_cam, image_pixels_copy=image_pixels_copy,layer_indx=None,filename= "grad_cam_attention_finetuned", text_list=decoded_texts.copy())

with GradCAM(model=model_untrained, target_layers=target_layers_untrained, reshape_transform=reshape_transform) as c:

        grayscale_cam = c(input_tensor=image_pixels_copy, targets=None)
        analyse_batch(grayscale_cam=grayscale_cam, image_pixels_copy=image_pixels_copy,layer_indx=None,filename= "grad_cam_attention_untrained", text_list=decoded_texts.copy())

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


# .---------------------------------------------------------------
utils.set_seeds(SEED)

train_dl, test_dl = datasets.get_hateful_memes_datasets(
    train_test_ratio=TRAIN_TEST_RATIO,
    batch_size=2,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    prefetch_factor=PREFETCH,
    persistent_workers=PERSISTENT_WORKERS,
    use_train_augmentation=False,

)
train_item = next(iter(train_dl))

print(train_item.keys())    # dict_keys(['img', 'label', 'text'])
print(train_item["img"]["pixel_values"].shape)   # torch.Size([2, 1, 3, 224, 224])

# already tokenized and preprocessed for vit
text_tokens = {k: v.squeeze(1) for k, v in train_item["text"].items()}
image_pixels = train_item["img"]["pixel_values"].squeeze(1)

print(f"text_token shape: {text_tokens['input_ids'].shape}, imgage_pixels shape: {image_pixels.shape}")

config = ViLBERTConfig()
model_vilbert_untrained = ViLBERT(config=config)


model_vilbert = ViLBERT.load_model(load_path="hm_finetune.pt", )




model = ViLBERTWrapper(model=model_vilbert, text_inputs=text_tokens)
model_untrained = ViLBERTWrapper(model=model_vilbert_untrained, text_inputs=text_tokens)

# TODO: better fetching
# clayers are not working,as they return tuple, grad-cam expects single tensor
target_layers = [model.model.vit.blocks[4]]
target_layers_untrained = [model_untrained.model.vit.blocks[4]]

#TODO: maybe use other method for it, in import there are many, also here:
# https://github.com/jacobgil/pytorch-grad-cam/blob/781dbc0d16ffa95b6d18b96b7b829840a82d93d1/cam.py#L67

with GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform) as c:

    grayscale_cam = c(input_tensor=image_pixels, targets=None)
    print(grayscale_cam.shape)

    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape)   # (224, 224)
    img_for_display = image_pixels[0].permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_for_display = (img_for_display * std + mean).clip(0, 1)

    cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)
    cv2.imwrite( "grad_cam_attention_trained.jpg",    cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))



with GradCAM(model=model_untrained, target_layers=target_layers, reshape_transform=reshape_transform) as c:

    grayscale_cam = c(input_tensor=image_pixels, targets=None)
    print(grayscale_cam.shape)

    grayscale_cam = grayscale_cam[0, :]
    print(grayscale_cam.shape)   # (224, 224)
    img_for_display = image_pixels[0].permute(1, 2, 0).cpu().numpy()
    mean, std = np.array([0.485, 0.456, 0.406]), np.array([0.229, 0.224, 0.225])
    img_for_display = (img_for_display * std + mean).clip(0, 1)

    cam_image = show_cam_on_image(img_for_display, grayscale_cam, use_rgb=True)
    cv2.imwrite("grad_cam_attention_untrained.jpg", cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR))






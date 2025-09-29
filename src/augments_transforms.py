import torchvision; from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import resolve_data_config

import albumentations as A
import cv2

from config import VIT_MODEL_NAME, SEED

##https://explore.albumentations.ai/
# to test out augmentations


def get_transform_unmasked():
    """ data augmentation for MIM. this is the weak augmentation for the unmasked image"""
    transforms_unmasked = torchvision.transforms.Compose([
        transforms.RandomRotation(5),
        transforms.RandomRotation(degrees=5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
        transforms.RandomResizedCrop(size=224, scale=(0.95, 1.0)),
    ])
    return transforms_unmasked

def get_transform_masked():
    """ data augmentation for MIM. this is the strong augmentation"""
    transforms_masked = torchvision.transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomResizedCrop(size=224, scale=(0.7, 1.0)),  # More aggressive cropping
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
    ])
    return transforms_masked

def get_full_vit_transform():
    config = resolve_data_config({}, model=VIT_MODEL_NAME)
    # vit_transform = create_transform(**config)        # this was used before
    vit_transform_full = transforms.Compose([
        transforms.Resize(256, interpolation=InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    return vit_transform_full

def get_minimal_vit_transform():
    """ data augmentation for ViT. As images are already resized, no need to resize, only normalize"""
    vit_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD)
    ])
    return vit_transform


def get_mm_imdb_train_augmentation(seed:int):
    """ data augmentation for MM-IMDB dataset training"""

    transform_augmentation = transforms.Compose([

        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),

        transforms.RandomResizedCrop(size=224, scale=(0.92, 1.0), ratio=(0.95, 1.05 )),
        transforms.RandomAffine(
            degrees=5,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),

        transforms.RandomPerspective(distortion_scale=0.08, p=0.25),

        transforms.RandomGrayscale(p=0.1),
        # transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomErasing(),

    ],)
    return transform_augmentation

def get_hateful_memes_train_augmentation():
    """ data augmentation for hateful memes dataset training. the same as `get_mm_imdb_train_augmentation`"""

    return get_mm_imdb_train_augmentation()



def get_hateful_memes_train_augmentation_albumation(seed:int,get_advanced=False, ):

    hm_transforms = A.Compose([
        A.ColorJitter(
            brightness=(0.7,1.3),
            contrast=(0.7,1.3),
            saturation=(0.8, 1.2),
            hue=(-0.01, 0.01),
            p=.6
            ),
        # A.RandomResizedCrop(size=(224, 224), scale=(0.92, 1.0), ratio=(0.95, 1.05), p=1.0),

        # A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.5), p=0.2),
        # A.Perspective(scale=(0.05, 0.08), p=0.25),
        A.ToGray(p=0.1),
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.5), p=1.0),
            A.MotionBlur(
                blur_limit=(3,5),
                allow_shifted=False,
                angle_range=(0,45)
            ),
            A.ImageCompression(quality_range=(65, 90)),
        ], p=0.25),
        A.OneOf([
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.2),
                hole_width_range=(0.05, 0.2),
                fill=0,
                p=1.
            ),
            A.CoarseDropout(
                num_holes_range=(1, 2),
                hole_height_range=(0.05, 0.2),
                hole_width_range=(0.05, 0.2),
                fill="random_uniform",
                p=1.
            )
        ], p=0.25),
        A.Affine(
            scale=(0.7,0.98 ),
            rotate=(-10,10),
            shear=(0.05,0.05),
            border_mode=cv2.BORDER_REPLICATE,
            p=0.5,
        ),
    ], seed=seed)

    hm_transforms_improved = A.Compose([
        # Your existing transforms (keep these)
        A.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15, p=1.0),
        A.RandomResizedCrop(size=(224, 224), scale=(0.92, 1.0), ratio=(0.95, 1.05), p=1.0),  # FIXED
        A.Affine(
            rotate=(-5, 5),
            translate_percent=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=(-2, 2),
            p=1.0
        ),
        A.ToGray(p=0.1),


        A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),  # Better contrast
        A.GridDistortion(num_steps=5, distort_limit=0.05, p=0.2),  # Mild geometric distortion
        A.GaussNoise(std_range=(0.1, 0.1)),

        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 3), sigma_limit=(0.1, 0.5), p=1.0),
            A.MotionBlur(blur_limit=3, p=1.0),
            A.ImageCompression(quality_range=(65, 90)),
        ], p=0.25),

        A.Perspective(scale=(0.05, 0.08), p=0.25),
    ], seed=seed)

    if get_advanced:
        return hm_transforms_improved

    return hm_transforms
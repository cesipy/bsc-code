import torchvision; from torchvision import transforms
from torchvision.transforms import InterpolationMode
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import resolve_data_config

from config import VIT_MODEL_NAME


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


def get_mm_imdb_train_augmentation():
    """ data augmentation for MM-IMDB dataset training"""

    transform_augmentation = transforms.Compose([

        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15),

        transforms.RandomResizedCrop(size=224, scale=(0.85, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomAffine(
            degrees=10,
            translate=(0.05, 0.05),
            scale=(0.95, 1.05),
            shear=2
        ),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
        ], p=0.2),

        transforms.RandomGrayscale(p=0.1),
        # transforms.RandomHorizontalFlip(p=0.2),
        # transforms.RandomErasing(),

    ])
    return transform_augmentation

def get_hateful_memes_train_augmentation():
    """ data augmentation for hateful memes dataset training. the same as `get_mm_imdb_train_augmentation`"""

    return get_mm_imdb_train_augmentation()



"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/19 13:10
"""

import torch
import albumentations
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REAL_TRAIN_DIR = "data/faces/train"
REAL_VAL_DIR = "data/faces/val"
FAKE_TRAIN_DIR = "data/GFaces/train"
FAKE_VAL_DIR = "data/GFaces/val"
LEARNING_RATE = 2e-4
BATCH_SIZE = 16
NUM_WORKERS = 2
IMAGE_SIZE = 256
CHANNELS_IMG = 3
L1_LAMBDA = 100
LAMBDA_GP = 10
NUM_EPOCHS = 10
LOAD_MODEL = False
SAVE_MODEL = True
CHECKPOINT_DISC = "models/disc.pth.tar"
CHECKPOINT_GEN = "models/gen.pth.tar"

transform_only_input = albumentations.Compose(
    [
        albumentations.HorizontalFlip(p=0.5),
        albumentations.ColorJitter(p=0.2),
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)

transform_only_mask = albumentations.Compose(
    [
        albumentations.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], max_pixel_value=255.0,),
        ToTensorV2(),
    ]
)
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 15:23
"""

import os
import torch
import albumentations
from torchvision import transforms
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DATASET_DIR = "data/img_align_celeba"
TRAIN_DIR = "data/celeba/train"
VAL_DIR = "data/celeba/val"
WIDTH,HEIGHT = 128,128
LEARNING_RATIO = 1e-4
BATCH_SIZE_T = 2
BATCH_SIZE_V = 4
SHUFFLE = True
NUM_WORKERS = 0
SAVE_MODELS = "models"
LOAD_MODELS = "models"
SAVE_IMAGES = "images"
NUM_EPOCHS = 1
DOWNSCALE = 4
NEW_SIZE_W,NEW_SIZE_H = WIDTH // DOWNSCALE,HEIGHT // DOWNSCALE
LOSS_RATIO = 1e-3
BEAT1 = 0.9
BEAT2 = 0.9
EPSILON = 1e-5

#对32 x 32和128 x 128的图像进行增强
Transform = albumentations.Compose([
    albumentations.HorizontalFlip(p = 0.5),
    albumentations.VerticalFlip(p = 0.5),
    albumentations.RandomBrightness(limit=0.2),
    albumentations.HueSaturationValue(hue_shift_limit=20,sat_shift_limit=30),
    albumentations.RandomContrast(limit=0.2),
    albumentations.Normalize(mean=[0.5,0.5,0.5],std=[0.5,0.5,0.5]),
    ToTensorV2()
])
#对128 x 128的图像进行裁剪到32 x 32
down_Transform = albumentations.Compose([
    albumentations.Resize(height=NEW_SIZE_W,width=NEW_SIZE_H)
])

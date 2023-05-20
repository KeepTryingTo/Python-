"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/19 13:19
"""

"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/4 14:44
"""

import os
import torch
import config
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class MapDataset(Dataset):
    def __init__(self,real_dir,fake_dir):
        super(MapDataset, self).__init__()

        self.real_dir = real_dir
        self.fake_dir = fake_dir

        self.real_imgs = os.listdir(self.real_dir)
        self.real_imgs.sort(key=lambda x: int(x.split('.')[0]))
        self.fake_imgs = os.listdir(self.fake_dir)
        self.fake_imgs.sort(key=lambda x: int(x.split('.')[0]))

    def __len__(self):
        # print(len(self.real_imgs))
        # print(len(self.fake_imgs))
        return len(self.real_imgs)

    def __getitem__(self, index):
        real_img_file = self.real_imgs[index]
        fake_img_file = self.fake_imgs[index]
        real_img_path = os.path.join(self.real_dir,real_img_file)
        fake_img_path = os.path.join(self.fake_dir,fake_img_file)
        real_image = np.array(Image.open(real_img_path))
        fake_image = np.array(Image.open(fake_img_path))
        #图像增强
        target_image = config.transform_only_mask(image = real_image)["image"]
        input_image = config.transform_only_input(image = fake_image)["image"]

        return input_image,target_image

if __name__ == '__main__':
    dataset = MapDataset(real_dir="data/faces/val",fake_dir="data/GFaces/val")
    print(dataset.__len__())

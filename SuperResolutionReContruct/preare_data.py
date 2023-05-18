"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 15:53
"""

import os
import torch
import config
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader,Dataset

class MyDataset(Dataset):
    def __init__(self,root_dir):
        super(MyDataset, self).__init__()
        self.root_dir = root_dir
        self.imgs_list = os.listdir(root_dir)

    def __len__(self):
        return len(self.imgs_list)

    def __getitem__(self, index):
        img_name = self.imgs_list[index]
        img_path = os.path.join(self.root_dir,img_name)
        image = np.array(Image.open(img_path).convert("RGB"))
        labels = config.Transform(image=image)["image"]
        features = config.Transform(image=config.down_Transform(image=image)["image"])["image"]
        return features,labels

if __name__ == '__main__':
    mydataT = MyDataset(config.TRAIN_DIR)
    print(mydataT.__len__())
    print("labels.shape: {}--feautres.shape: {}".format(np.shape(mydataT[0][0]),np.shape(mydataT[0][1])))
    mydataV = MyDataset(config.VAL_DIR)
    print(mydataV.__len__())
    print("labels.shape: {}--feautres.shape: {}".format(np.shape(mydataV[0][0]), np.shape(mydataV[0][1])))
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/19 12:04
"""

import os
import cv2
import torch
import random
import numpy as np


#添加高斯噪声
# loc：float
#     此概率分布的均值（对应着整个分布的中心centre）
# scale：float
#     此概率分布的标准差（对应于分布的宽度，scale越大越矮胖，scale越小，越瘦高）
def Add_Gauss_Noise(img_path='',img_name = '',mean=0.0,val=0.01,save_path = "data/GFaces/train"):
    """
    :param img_path:
    :param img_name:
    :param mean:
    :param val:
    :param save_path:
    :return:
    """
    image=cv2.imread(img_path)
    # image=cv2.resize(src=image,dsize=(256,256))
    image=image/255.0
    noise=np.random.normal(mean,val**0.5,image.shape)
    image=image+noise
    #保存图片
    file,etc = os.path.splitext(img_name)
    save_img_path=os.path.join(save_path,str(file)+'.png')
    noise_image=image*255.0
    cv2.imwrite(save_img_path,noise_image)


# 生成椒盐噪声prob:噪声比例
def Papper_Noise(img_path='',img_name = '', prob=0.1,save_path = "data/GFaces/train"):
    """
    :param img_path:
    :param img_name:
    :param prob:
    :param save_path:
    :return:
    """
    image = cv2.imread(img_path)
    # image = cv2.resize(src=img, dsize=(256, 256))
    noise = np.zeros(image.shape, np.uint8)
    thres = 1 - prob

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            # 产生0-1之间的浮点数（大于等于0且小于1之间的小数）
            rand = random.random()
            # 如果随机数小于给定的噪声比例
            if rand < prob:
                noise[i][j] = 0
            elif rand > thres:
                noise[i][j] = 255
            else:
                noise[i][j] = image[i][j]
    # 保存图片
    file, etc = os.path.splitext(img_name)
    save_img_path = os.path.join(save_path, str(file) + '.png')
    cv2.imwrite(save_img_path, noise)


def read_imgs(dataset_dir = "data/faces",train_val_ratio = 0.9):
    """
    :param dataset_dir:
    :return:
    """
    imgs_list = os.listdir(dataset_dir)
    #按照文件名的顺序读取
    imgs_list.sort(key=lambda x: int(x.split('.')[0]))
    n_train = int(len(imgs_list) * train_val_ratio)
    n_val = int(len(imgs_list) * (1 - train_val_ratio))
    for i in range(n_train):
        img_name = imgs_list[i]
        img_path = os.path.join(dataset_dir,img_name)
        Add_Gauss_Noise(img_path = img_path,img_name = img_name,save_path="data/GFaces/train")
        # break
    for i in range(n_train,n_train + n_val):
        img_name = imgs_list[i]
        img_path = os.path.join(dataset_dir,img_name)
        Add_Gauss_Noise(img_path = img_path,img_name = img_name,save_path="data/GFaces/val")
        # break

if __name__ == '__main__':
    read_imgs()
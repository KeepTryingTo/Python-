"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 16:03
"""

import os
import config
import random
from PIL import Image

def crop_images(train_val_ratio = 0.9):
    """
    :param train_val_ratio: 训练集和验证集数量划分
    :return:
    """
    #列出训练集和验证集的图片路径
    train_files = os.listdir(config.DATASET_DIR)
    # 得到图片对应的索引
    shuffle_id = list(range(len(train_files)))
    # 将索引随机打散
    random.shuffle(shuffle_id)
    # 获取训练集大小
    n_train = int(len(train_files) * train_val_ratio)
    n_val = int(len(train_files) - n_train)
    # 划分训练集和测试集
    train_id = shuffle_id[:n_train]
    val_id = shuffle_id[n_train:]
    #裁剪训练集图片
    for i,id in enumerate(train_id):
        file = train_files[id]
        resize_image(file,config.DATASET_DIR,config.TRAIN_DIR)
    print("train done ...")
    #采集验证集图片
    for i,id in enumerate(val_id):
        file = train_files[id]
        resize_image(file,config.DATASET_DIR,config.VAL_DIR)
    print("val done ...")

def resize_image(filename,hr_dir,resize_dir):
    """
    :param filename: 缩放的图片名称
    :param hr_dir: 图片的父级目录
    :param resize_dir: 缩放之后保存的文件路径
    :return:
    """
    #将图像裁剪到128 x 128
    image_size = 128
    image = Image.open(os.path.join(hr_dir,filename))
    half_the_width = image.size[0] / 2
    half_the_height = image.size[1] / 2
    #Image.crop(left, up, right, below)
    box = (half_the_width - 64,half_the_height - 64,
           half_the_width + 64,half_the_height + 64
    )
    image = image.crop(box)
    file,ext = os.path.splitext(filename)
    image.save(os.path.join(resize_dir,file+'-resized.png'))

if __name__ == '__main__':
    # crop_images(train_val_ratio=0.9)
    print(len(os.listdir(config.TRAIN_DIR)))
    print((len(os.listdir(config.VAL_DIR))))
    pass
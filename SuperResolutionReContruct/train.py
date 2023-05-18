"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 16:23
"""

import os
import torch
import config
import numpy as np
from tqdm import tqdm
from torchinfo import summary
import torchvision.models.vgg

import utils
from preare_data import MyDataset
from net.Generator import Generator
from net.Discriminator import Discriminator
from torch.utils.data import DataLoader,Dataset


def vgg19():
    """
    vgg19主要是用来在这里提取特征，进生成器生成的图片输入到vgg19中和将标签图片输入到vgg19中，
    将两个样本最后输出的特征之间的距离作为最后计算损失值
    :return:
    """
    vgg19 = torchvision.models.vgg.vgg19(pretrained = True,progress = True)
    # vgg19.classifier = torch.nn.Sequential()
    # vgg19.avgpool = torch.nn.Sequential()
    #去掉最后的全连接层和avg_pooling层
    vgg19 = torch.nn.Sequential(*(list(vgg19.children())[:-2])[0][:36])
    # summary(vgg19,input_size=(1,3,128,128))
    # print(vgg19)
    return vgg19

#生成器损失
def create_g_loss(d_output,g_output,labels,loss_model,loss_fn_gen):
    """
    :param d_output: 判别器输出结果
    :param g_output: 生成器生成的图片
    :param labels: 标签值
    :param loss_model: 损失模型
    :return:
    """
    gene_ce_loss = loss_fn_gen(d_output,torch.ones_like(d_output))
    # print('labels.shape: {}'.format(np.shape(labels)))
    # print('g_output.shape: {}'.format(np.shape(g_output)))
    vgg_loss = torch.mean(torch.square(loss_model(labels) - loss_model(g_output)))
    g_loss = vgg_loss + config.LOSS_RATIO * gene_ce_loss
    return g_loss

#判别器损失值
def create_d_loss(disc_real_output,disc_fake_output,loss_fn_real,loss_fn_fake):
    """
    :param disc_real_output: 标签值输入到判别器的输出结果
    :param disc_fake_output: 生成器生成的图片输入到判别器中的输出结果
    :return:
    """
    disc_real_loss = loss_fn_real(disc_real_output,torch.ones_like(disc_real_output))
    disc_fake_loss = loss_fn_fake(disc_fake_output,torch.zeros_like(disc_fake_output))
    disc_loss =  disc_fake_loss + disc_real_loss
    return disc_loss

def train_step(features,labels,loss_model,gen,disc,opt_gen,opt_disc,loss_fn_gen,loss_fn_real,loss_fn_fake):
    """
    :param features: 低分辨率图像
    :param labels: 高分辨率图像
    :param loss_model: 使用VGG19计算输入图像的特征
    :param gen:生成器
    :param disc:判别器
    :param opt_gen:生成器优化器
    :param opt_disc:判别器优化器
    :return:
    """
    fake_img = gen(features)
    real_disc = disc(labels)
    fake_disc = disc(fake_img)
    g_loss = create_g_loss(fake_disc,fake_img,labels,loss_model,loss_fn_gen)
    #注意fake_disc.detach()
    d_loss = create_d_loss(real_disc,fake_disc.detach(),loss_fn_real,loss_fn_fake)

    opt_gen.zero_grad()
    g_loss.backward()
    opt_gen.step()

    opt_disc.zero_grad()
    d_loss.backward()
    opt_disc.step()
    return g_loss,d_loss



def train():
    #加载模型
    gen = Generator().to(config.DEVICE)
    disc = Discriminator(in_channels=3,out_channels=1).to(config.DEVICE)
    #加载数据集
    trainDataset = MyDataset(root_dir=config.TRAIN_DIR)
    valDataset = MyDataset(root_dir=config.VAL_DIR)
    trainLoader = DataLoader(
        dataset=trainDataset,
        batch_size=config.BATCH_SIZE_T,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    valLoader = DataLoader(
        dataset=valDataset,
        batch_size=config.BATCH_SIZE_V,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    #定义优化器
    opt_gen = torch.optim.Adam(params=gen.parameters(),lr = config.LEARNING_RATIO,
                               betas=(config.BEAT1,config.BEAT2),eps=config.EPSILON)
    opt_disc = torch.optim.Adam(params=disc.parameters(),lr = config.LEARNING_RATIO,
                                betas=(config.BEAT1,config.BEAT2),eps=config.EPSILON)

    #定义损失函数
    loss_fn_gen = torch.nn.BCELoss()
    loss_fn_real = torch.nn.BCELoss()
    loss_fn_fake = torch.nn.BCELoss()

    loss_model = vgg19()
    loss_model = loss_model.to(config.DEVICE)

    gen_loss = []
    disc_loss = []
    for epoch in range(config.NUM_EPOCHS):
        all_g_cost,all_d_cost = 0,0
        loop = tqdm(trainLoader,leave=True)
        loop.set_description(desc="training: ")
        gen.train()
        disc.train()
        for step,data in enumerate(loop):
            imgs,labels = data
            imgs,labels = imgs.to(config.DEVICE),labels.to(config.DEVICE)

            g_loss,d_loss = train_step(imgs,labels,loss_model,gen,disc,opt_gen,opt_disc,
                                       loss_fn_gen,loss_fn_real,loss_fn_fake)

            all_g_cost += g_loss
            all_d_cost += d_loss
            loop.set_description(desc="training: ")

            if step % 50 == 0 and step > 0:
                loop.set_postfix(epoch = epoch,g_loss = g_loss.item(),d_loss = d_loss.item())
                print("\n--------------------------------------g_loss: {:.6f}--------------------------------------".format(g_loss.item()))
                print("--------------------------------------d_loss: {:.6f}--------------------------------------".format(d_loss.item()))
        gen_loss.append(all_g_cost / len(trainLoader))
        disc_loss.append(all_d_cost / len(trainLoader))

        gen.eval()
        disc.eval()
        with torch.no_grad():
            loop = tqdm(valLoader,leave=True)
            loop.set_description(desc="valing: ")
            for step,data in enumerate(loop):
                imgs,labels = data
                imgs,labels = imgs.to(config.DEVICE),labels.to(config.DEVICE)

                fake_img = gen(imgs)
                real_disc = disc(labels)
                fake_disc = disc(fake_img)
                g_loss = create_g_loss(fake_disc, fake_img, labels, loss_model, loss_fn_gen)
                # 注意fake_disc.detach()
                d_loss = create_d_loss(real_disc, fake_disc.detach(), loss_fn_real, loss_fn_fake)

                if step % 100 == 0 and step > 0:
                    utils.save_images(gen,epoch,step,imgs)
                    loop.set_postfix(epoch=epoch, g_loss=g_loss.item(), d_loss=d_loss.item())
                    print("\n--------------------------------------g_loss: {:.6f}--------------------------------------".format(g_loss.item()))
                    print("--------------------------------------d_loss: {:.6f}--------------------------------------".format(d_loss.item()))
        if epoch % 10 == 0:
            utils.save_model(gen, opt_gen, epoch)
    utils.draw(gen_loss,disc_loss)


if __name__ == '__main__':
    # vgg19()
    train()

# import tensorflow as tf
#
# vgg19 = tf.keras.applications.vgg19.VGG19(include_top = False,weights='imagenet',input_shape=(128,128,3))
# vgg19.trainable = False
# for l in vgg19.layers:
#     l.trainable = False
#
# loss_models = tf.keras.Model(inputs = vgg19.input,outputs = vgg19.get_layer("block5_conv4").output)
# loss_models.trainable = False
# loss_models.summary()
"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 15:26
"""
import os
import torch
import config
import numpy as np
import matplotlib.pyplot as plt

#保存模型
def save_model(model,optimizer,epoch):
    """
    :param model:
    :param epoch:
    :return:
    """
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, os.path.join(config.SAVE_MODELS,str(epoch)+'gen.tar'))


def load_checkpoin(checkpoint_file, model, optimizer, lr):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_group:
        param_group["lr"] = lr

def generate_and_save_images(save_dir,features,gen,epoch):
    """
    :param save_dir:
    :param gen:
    :param epoch:
    :return:
    """
    predictions = gen(features)
    pass

def draw(gen_loss,disc_loss):
    """
    :param gen_loss:
    :param disc_loss:
    :return:
    """
    plt.plot(range(1, len(gen_loss) + 1), gen_loss, label='genLoss')
    plt.plot(range(1, len(disc_loss) + 1), disc_loss, label='discLoss')
    plt.legend()
    plt.title('GEN_DISC-LOSS')
    plt.savefig('logs/figure.png')

def save_images(model,epoch,step,val_loader):
    imgs = model(val_loader).detach().cpu().numpy()
    imgs = np.squeeze(imgs)
    fig = plt.figure(figsize=(2,2))
    for i in range(4):
        plt.subplot(2,2,i + 1)
        plt.imshow(np.transpose((imgs[i] + 1) / 2,(1,2,0)))
        plt.axis("off")
    plt.savefig(os.path.join(config.SAVE_IMAGES,str(epoch)+"_"+str(step)+'.png'))
    print("================================>Saving images...")
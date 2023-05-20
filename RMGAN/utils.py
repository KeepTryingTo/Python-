"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/19 13:11
"""
import os
import cv2
import torch
import config
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.utils as vutils
from torchvision.utils import save_image

def save_some_examples(inputs,y_fakes, idx,epoch, folder = "images"):
    """
    :param y_fake:
    :param target:
    :param epoch:
    :param folder:
    :return:
    """
    y_fake, inputs = y_fakes[0].detach().cpu(),inputs[0].numpy()
    y_fake = np.squeeze(y_fake.numpy())
    # plt.imshow(np.transpose(inputs * 0.5 + 0.5,(1,2,0)))
    # plt.savefig(os.path.join(folder, f"inputs_{epoch}_{idx}.png"))
    # plt.axis("off")
    # plt.imshow(np.transpose(y_fake * 0.5 + 0.5,(1,2,0)))
    # plt.axis("off")
    # plt.savefig(os.path.join(folder, f"y_fake_{epoch}_{idx}.png"))

    inputs = np.transpose(inputs * 0.5 + 0.5,(1,2,0))
    y_fake = np.transpose(y_fake * 0.5 + 0.5,(1,2,0))
    output = np.hstack((inputs,y_fake))
    cv2.imwrite(os.path.join(folder,f"y_fake_{epoch}_{idx}.png"), output * 255)


def save_checkpoint(model, optimizer, filename="my_checkpoint.pth.tar"):
    print("\n===================================> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr):
    print("\n=========================> Loading checkpoint")
    checkpoint = torch.load(checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    # If we don't do this then it will just have learning rate of old checkpoint
    # and it will lead to many hours of debugging \:
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

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

if __name__ == '__main__':
    pass
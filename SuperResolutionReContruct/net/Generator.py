"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 13:27
"""

import torch
import numpy as np
from torchinfo import summary

#残差模块
class _IdentityBlock(torch.nn.Module):
    #data_format判断输入图像的通道C位置[w,h,c]或者[c,w,h]
    def __init__(self,in_channels,out_channels,stride = (1,1),data_format = "channels_first"):
        super(_IdentityBlock, self).__init__()
        self.bn_axis = 1 if data_format == "channels_first" else 3
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(3,3),
                            stride=stride,padding=(1,1),bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.PReLU(num_parameters=1),

            torch.nn.Conv2d(in_channels = out_channels,out_channels = out_channels,kernel_size=(3,3),
                            stride=(1,1),padding=(1,1),bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels)
        )
    def forward(self,x):
        x_resnet = self.conv(x)
        out = x + x_resnet
        return out

def phaseShift(inputs,scale,shape_1,shape_2):
    x = torch.reshape(inputs,shape_1)
    x = torch.reshape(x,[0,1,3,2,4])
    return torch.reshape(x,shape_2)

#使用PixelShuffle进行上采样
def PixelShuffle(inputs,scale = 2):
    """
    :param inputs: 进行上采样的输入
    :param scale: 上采样的倍率
    :return:
    """
    size = np.shape(inputs)
    batch_size = size[0]
    h = size[1]
    w = size[2]
    c = size[-1]
    #进行上采样之后需要进行通道数1/4
    channel_target = c // (scale * scale)
    #获得上采样因子
    channel_factor = c // channel_target
    shape_1 = [batch_size,h,w,channel_factor // scale,channel_target//scale]
    shape_2 = [batch_size,h * scale,w * scale]
    #reshape and transpose for periods shuffle for each channel
    input_split = torch.split(inputs,channel_target,dim=3)
    output = torch.cat([phaseShift(x,scale,shape_1,shape_2) for x in input_split],dim=3)
    return output

# 生成器
class Generator(torch.nn.Module):
    def __init__(self,upscale = 2,data_format = "channels_last"):
        super(Generator, self).__init__()
        self.upscale = 2
        if data_format == "channels_first":
            self._input_shape = [-1,3,32,32]
            self.bn_axis = 1
        else:
            assert data_format == "channels_last"
            self._input_shape=[-1,32,32,3]
            self.bn_axis = 3

        self.initial_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(9, 9), stride=(1, 1),
                            padding=(4,4),bias=False),
            torch.nn.PReLU(num_parameters=1)
        )

        #使用残差模块
        self.identityBlocks = [_IdentityBlock(in_channels=64,out_channels=64) for _ in range(16)]
        self.Blocks = torch.nn.Sequential(
            *self.identityBlocks
        )

        self.conv2 = torch.nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1))

        # 由于进行了一次上采样，通道数减为原来的1/4，所以输入通道数为256=>64
        self.upconv1 = torch.nn.Conv2d(in_channels=64,out_channels=256,kernel_size=(3,3),stride=(1,1),
                        padding=(1,1))
        self.prelu1 = torch.nn.PReLU(num_parameters=1)
        #由于进行了一次上采样，通道数减为原来的1/4，所以输入通道数为256=>64
        self.upconv2 = torch.nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1),
                        padding=(1, 1))
        self.prelu2 = torch.nn.PReLU(num_parameters=1)
        self.conv3 = torch.nn.Conv2d(in_channels=64, out_channels=3, kernel_size=(9, 9), stride=(1, 1),
                        padding=(4, 4), bias=False)
    def forward(self,x):
        # x = torch.reshape(x,self._input_shape)
        x = self.initial_conv(x)

        x_resnet = self.Blocks(x)
        x_resnet = self.conv2(x_resnet)
        x = x + x_resnet

        #进行第一次上采样
        x = self.upconv1(x)
        x = torch.nn.PixelShuffle(self.upscale)(x)
        x = self.prelu1(x)

        #进行第二次上采样
        x = self.upconv2(x)
        x = torch.nn.PixelShuffle(self.upscale)(x)
        x = self.prelu2(x)

        x = self.conv3(x)
        x = torch.tanh(x)

        return x


if __name__ == '__main__':
    x = torch.randn(size = (1,3,32,32))
    gen = Generator(upscale=2,data_format="channels_first")
    summary(gen,input_size=(1,3,32,32))
    print(gen(x).shape)


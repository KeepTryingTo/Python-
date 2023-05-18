"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/16 14:30
"""

import torch
from torchinfo import summary

class ConvBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels = in_channels,out_channels=out_channels,kernel_size=(3,3),
                            stride=(1,1),padding=(1,1),bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(3, 3),
                            stride=(2, 2), padding=(1, 1), bias=False),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.01)
        )
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class Discriminator(torch.nn.Module):
    def __init__(self,in_channels,out_channels,data_format = "channels_first"):
        super(Discriminator, self).__init__()
        if data_format == "channels_first":
            self._input_shape = [-1,3,128,128]
            self.bn_axis = 1
        else:
            assert data_format == "channels_last"
            self._input_shape=[-1,128,128,3]
            self.bn_axis = 3

        self.conv1 = ConvBlock(in_channels=in_channels,out_channels = 64)
        self.conv2 = ConvBlock(in_channels=64,out_channels=128)
        self.conv3 = ConvBlock(in_channels=128,out_channels=256)
        self.conv4 = ConvBlock(in_channels=256,out_channels=512)
        # self.fc1 = torch.nn.Linear(in_features=512 * 8 * 8,out_features=256)
        # self.fc2 = torch.nn.Linear(in_features=256,out_features=out_channels)
        self.conv5 = torch.nn.Conv2d(in_channels=512, out_channels=out_channels, kernel_size=(3, 3),
                            stride=(2, 2), padding=(1, 1), bias=False)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self,x):
        # x = torch.reshape(x,self._input_shape)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)

        # x = x.view(-1,512 * 8 * 8)
        # x = self.fc1(x)
        # x = self.fc2(x)
        x = self.conv5(x)
        x = self.sigmoid(x)

        return x

if __name__ == '__main__':
    x = torch.randn(size = (1,3,128,128))
    model = Discriminator(in_channels=3,out_channels=1)
    summary(model,input_size=(1,3,128,128))
    print(model(x).shape)
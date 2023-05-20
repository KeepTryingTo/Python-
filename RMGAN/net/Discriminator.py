"""
@Author : Keep_Trying_Go
@Major  : Computer Science and Technology
@Hobby  : Computer Vision
@Time   : 2023/5/4 13:29
"""

import torch
from torchinfo import summary

class CNNBlock(torch.nn.Module):
    def __init__(self,in_channels,out_channels,stride=2):
        super(CNNBlock, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=out_channels,kernel_size=(4,4),
                            stride=(stride,stride),bias=False,padding_mode='reflect'),
            torch.nn.BatchNorm2d(num_features=out_channels),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )
    def forward(self,input):
        x = self.conv(input)
        return x


class Discriminator(torch.nn.Module):
    def __init__(self,in_channels=3,features=[32,64,128,256]):
        super(Discriminator, self).__init__()
        self.initial = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channels,out_channels=features[0],kernel_size=(4,4),
                            stride=(2,2),padding=(1,1),padding_mode='reflect'),
            torch.nn.LeakyReLU(negative_slope=0.2)
        )

        layers=[]
        in_channels=features[0]
        for feature in features[1:]:
            layers.append(
                CNNBlock(in_channels,feature,stride=1 if feature == features[-1] else 2)
            )
            in_channels=feature
        self.model = torch.nn.Sequential(
            *layers,
            torch.nn.Conv2d(in_channels=in_channels,out_channels=1,kernel_size=(4,4),
                            stride=(1,1),padding=(1,1),padding_mode='reflect')
        )
    def forward(self,x):
        x = self.initial(x)
        x = self.model(x)
        return x

if __name__ == '__main__':
    dis = Discriminator(in_channels=3,features=[32,64,128,256])
    x = torch.randn(size = (1,3,96,96))
    preds = dis(x)
    summary(dis,input_size=[1,3,96,96])
    print(preds.shape)

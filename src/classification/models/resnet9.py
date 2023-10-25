import torch
import torch.nn as nn
from .resnet18 import ConvBlock,ResBlock


class Resnet9(nn.Module):
    def __init__(self,in_channels:int,nClasses:int):
        super().__init__()
        self.nClasses=nClasses
        self.conv1=ConvBlock(in_channels,64,kernel_size=7,stride=2,padding=3)
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResBlock(64,64),
        )
        self.conv3_x = nn.Sequential(
            ResBlock(64,128,stride=2),
        )
        self.conv4_x = nn.Sequential(
            ResBlock(128,256,stride=2),
        )
        self.conv5_x = nn.Sequential(
            ResBlock(256,512,stride=2),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1,1)),
            nn.Flatten(),
            nn.Linear(512,self.nClasses)
        )
    def forward(self,x):
        y1 = self.conv1(x)
        y2 = self.conv2_x(y1)
        y3 = self.conv3_x(y2)
        y4 = self.conv4_x(y3)
        y5 = self.conv5_x(y4)
        y = self.head(y5)
        return y


if __name__=='__main__':
    input = torch.randn((4,3,224,224))
    model = Resnet9(3,20)
    output = model(input)
    print('terminé')

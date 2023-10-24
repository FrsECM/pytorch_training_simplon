import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels:int,out_channels:int,kernel_size:int=3,stride:int=1,padding:int=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
    def forward(self,x):
        y = self.conv(x)
        y = self.bn(y)
        y = self.act(y)
        return y

class ResBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int=3,
                 stride:int=1,
                 padding:int=1):
        super().__init__()
        self.conv1 =ConvBlock(in_channels,out_channels,kernel_size,stride,padding)
        self.conv2 = ConvBlock(out_channels,out_channels,kernel_size,padding=padding)
        # Shortcut management
        self.shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)

    def forward(self,x):
        shortcut = self.shortcut(x)
        # First Stage
        y1 = self.conv1(x)

        # Second Stage
        y2 = self.conv2(y1)
        y = y2 + shortcut
        return y

class Resnet18(nn.Module):
    def __init__(self,in_channels:int,nClasses:int):
        super().__init__()
        self.nClasses=nClasses
        self.conv1=ConvBlock(in_channels,64,kernel_size=7,stride=2,padding=3)
        self.conv2_x = nn.Sequential(
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
            ResBlock(64,64),
            ResBlock(64,64)
        )
        self.conv3_x = nn.Sequential(
            ResBlock(64,128,stride=2),
            ResBlock(128,128),
        )
        self.conv4_x = nn.Sequential(
            ResBlock(128,256,stride=2),
            ResBlock(256,256),
        )
        self.conv5_x = nn.Sequential(
            ResBlock(256,512,stride=2),
            ResBlock(512,512),
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
    model = Resnet18(3,20)
    output = model(input)
    print('terminé')

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


class ResnetBlock(nn.Module):
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
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3)
    def forward(self,x):
        pass


if __name__=='__main__':
    input = torch.randn((4,3,224,224))
    model = ResnetBlock(3,64,stride=2)
    output = model(input)
    print('terminé')

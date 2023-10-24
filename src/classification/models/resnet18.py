import torch
import torch.nn as nn

class ResnetBlock(nn.Module):
    def __init__(self,
                 in_channels:int,
                 out_channels:int,
                 kernel_size:int=3,
                 stride:int=1,
                 padding:int=1):
        super().__init__()
        self.conv1 =nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size,padding=padding)
        self.bn2 = nn.BatchNorm2d(out_channels)
        # Shortcut management
        self.shortcut = nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=stride)

    def forward(self,x):
        shortcut = self.shortcut(x)
        # First Stage
        y1 = self.conv1(x)
        y1 = self.bn1(y1)
        y1 = self.act(y1)
        # Second Stage
        y2 = self.conv2(y1)
        y2 = self.bn2(y2)
        y2 = self.act(y2)
        y = y2 + shortcut
        return y


if __name__=='__main__':
    input = torch.randn((4,3,224,224))
    model = ResnetBlock(3,64,stride=2)
    output = model(input)
    print('terminé')

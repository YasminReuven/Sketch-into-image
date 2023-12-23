import torch
from torch import nn
import config


class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=2):
        super().__init__()
        self.convBlock = nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride,
                                                 padding=1, bias=False),
                                       nn.InstanceNorm2d(out_channels, affine=True),
                                       nn.LeakyReLU(0.2))

    def forward(self, x):
        return self.convBlock(x)


class Discriminator(nn.Module):
    def __init__(self, in_channels=3, features=None):
        super().__init__()
        if features is None:
            features = [64, 128, 256, 512]
        self.initial = nn.Sequential(nn.Conv2d(in_channels*2, features[0]
                                               , kernel_size=4, stride=2, bias=False, padding=1),
                                     nn.LeakyReLU(0.2))
        layers = []
        in_channels = features[0]
        for out_channels in features:
            layers.append(CnnBlock(in_channels, out_channels, stride=1 if out_channels != 512 else 1))
            in_channels = out_channels

        layers.append(nn.Sequential(nn.Conv2d(in_channels, out_channels=1, kernel_size=4, stride=1, padding=1)))
        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)




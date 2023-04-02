import torch
import torch.nn as nn
import torch.nn.functional as F

class Gnenertor(nn.Module):
    def __init__(self,
                 channell:int=3) -> None:
        super().__init__()

        def upsample(in_, out_, normalize=True):
            layers = [nn.ConvTranspose2d(in_, out_, 4, stride=2, padding=1)]
            if normalize: layers.append(nn.BatchNorm2d(out_, 0.8))
            layers.append(nn.ReLU())
            return layers
        def downsample(in_, out_, normalize=True):
            layers = [nn.Conv2d(in_, out_, 4, stride=2, padding=1)]
            if normalize: layers.append(nn.BatchNorm2d(out_, 0.8))
            layers.append(nn.LeakyReLU(0.2))
            return layers
        
        self.model = nn.Sequential(
            *downsample(channell, 64),
            *downsample(64, 64),
            *downsample(64, 128),
            *downsample(128, 256),
            *downsample(256, 512),

            nn.Conv2d(512, 4000, 1),

            *upsample(4000, 512),
            *upsample(512, 256),
            *upsample(256, 128),
            *upsample(128, 64),

            nn.Conv2d(54, channell, 3, 1, 1),
            nn.Tanh())

    def forward(self, x):
        return self.model(x)
    

class Discriminator(nn.Module):
    def __init__(self, 
                 channel:int=3) -> None:
        super().__init__()

        def block(in_, out_, stride, normalize=True):
            layers =  [nn.Conv2d(in_, out_, 3, stride=stride, padding=1)]
            if normalize: layers.append(nn.InstanceNorm2d(out_))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        self.model = nn.Sequential(
            *block(channel, 64, 2, normalize=False),
            *block(64, 128, 2),
            *block(128, 256, 2),
            *block(256, 512, 1),

            nn.Conv2d(512, 1, 3, 1, 1)

        )
    def forward(self, x):
        return self.model(x)
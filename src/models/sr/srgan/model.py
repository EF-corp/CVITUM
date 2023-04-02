import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import vgg19

import numpy as np
import math

class ExtractFeature(nn.Module):
    def __init__(self) -> None:
        super().__init__()

        vgg19_model = vgg19(pretrained=True)

        self.F_E = nn.Sequential(*list(vgg19_model.features.children())[:18])


    def forward(self, x):
        return self.F_E(x)
    


class ResBlock(nn.Module):
    def __init__(self,
                 in_:int) -> None:
        super().__init__()

        def part_block(in_:int, prelu:bool = True) -> list:
            layer = [nn.Conv2d(in_, in_, kernel_size=3, stride=1, padding=1)]
            if prelu: 
                layer.append(nn.PReLU())
            layer.append(nn.BatchNorm2d(in_, 0.8))
            return layer
        
        self.block = nn.Sequential(
            *part_block(in_=in_),
            *part_block(in_=in_, prelu=False)
        )
    

    def forward(self, x):
        return x + self.block(x)
    


class GeneratorResNet(nn.Module):
    def __init__(self,
                 in_:int=3,
                 out_:int=3,
                 n_resblock:int=16) -> None:
        super().__init__()

        def block():
            return [
                nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.PixelShuffle(upscale_factor=2),
                nn.PReLU()
            ]
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_, 64, kernel_size=9, stride=1, padding=4),
            nn.PReLU()
        )

        self.res_blocks = nn.Sequential(
            *[ResBlock(64) for _ in range(n_resblock)]
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=4),
            nn.BatchNorm2d(64, 0.8)
        )

        self.upsampling = nn.Sequential(
            *block(),
            *block()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, out_, kernel_size=9, stride=1, padding=4),
            nn.Tanh()
        )

    
    def forward(self, x):

        out1 = self.conv1(x)

        out = self.res_blocks(out1)

        out2 = self.conv2(out)

        out = torch.add(out1, out2)

        out = self.upsampling(out)

        return self.conv3(out)
    

class Discriminator(nn.Module):
    def __init__(self,
                 inp) -> None:
        super().__init__()

        def block(in_filters:int, 
                  out_filters:int, 
                  first:bool=False) -> list:
            layer = [nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1)]

            if not first: 
                layer.append(nn.BatchNorm2d(out_filters))

            layers += [
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(out_filters),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return layer
        
        self.input_shape = inp
        in_ch, in_h, in_w = self.input_shape

        path_h, path_w = in_h // 2**4, in_w // 2**4

        self.out_shape = (1, path_h, path_w)

        self.model_D = nn.Sequential(
        
            *block(in_filters=in_ch, out_filters=64, first=True),
            *block(in_filters=64, out_filters=128),
            *block(in_filters=128, out_filters=256),
            *block(in_filters=256, out_filters=512),
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1)

        )


    def forward(self, x):
        return self.model_D(x)

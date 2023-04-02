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

        self.F_E = nn.Sequential(*list(vgg19_model.features.children())[:35])


    def forward(self, x):
        return self.F_E(x)
    


class DenseResBlock(nn.Module):
    def __init__(self,
                 filters,
                 res_scale:float=0.2) -> None:
        super().__init__()

        def block(in_:int,
                  non_linear:bool=True):
            layer = [nn.Conv2d(in_, filters, kernel_size=3, stride=1, padding=1, bias=True)]

            if non_linear:
                layer.append(nn.LeakyReLU())

            return nn.Sequential(*layer)
        
        self.res_scale = res_scale
        self.blocks = []

        for i in range(1, 5):
            self.blocks.append(block(in_= i * filters))

        self.blocks.append(block(in_= 5 * filters, non_linear=False))

    def forward(self, x):
        inp = x

        for block in self.blocks:
            out = block(inp)
            inp = torch.cat([inp, out], 1)

        result = out.mul(self.res_scale) + x

        return result



class ResInResDenseBlock(nn.Module):
    def __init__(self,
                 filters,
                 res_scale:float=0.2) -> None:
        super().__init__()

        self.res_scale = res_scale

        self.blocks = nn.Sequential(
            *[DenseResBlock(filters=filters) for _ in range(3)]
        )
        
    def forward(self,  x):

        return self.blocks(x).mul(self.res_scale) + x
    


class GeneratorRRDB(nn.Module):
    def __init__(self,
                 channels:int=3,
                 filters:int=64,
                 num_block:int=16,
                 num_upsample:int=2) -> None:
        super().__init__()

        def block():
            return [
                nn.Conv2d(filters, filters*4, kernel_size=3, stride=1, padding=1),
                #nn.BatchNorm2d(256),
                nn.LeakyReLU(),
                nn.PixelShuffle(upscale_factor=2),
                #nn.PReLU()
            ]
        
        self.conv1 = nn.Conv2d(channels, filters, kernel_size=3, stride=1, padding=1)

        self.res_blocks = nn.Sequential(
            *[ResInResDenseBlock(filters=filters) for _ in range(num_block)]
        )
        
        self.conv2 = nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1)

        ups = []
        for _ in range(num_upsample):
            ups += block()

        self.upsampling = nn.Sequential(*ups)
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(filters, filters, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(),
            nn.Conv2d(filters, channels, kernel_size=3, stride=1, padding=1)
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

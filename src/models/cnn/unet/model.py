import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

class CNA(nn.Module):
    def __init__(self,
                 in_:int,
                 out_:int,
                 stride:int=1) -> None:
        super().__init__()

        self.cna = nn.Sequential(
            nn.Conv2d(in_, out_, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_),
            nn.GELU()

        )

    def forward(self, x):
        return self.cna(x)
    


class UnetBlock(nn.Module):
    def __init__(self,
                 in_:int,
                 out_:int,
                 inner:int,
                 inner_block=False) -> None:
        super().__init__()

        self.inner_block = inner_block

        self.conv1 = CNA(in_=in_, out_=inner, stride=2)

        self.conv2, self.conv3 = [CNA(in_=inner, out_=inner)] * 2

        self.conv4 = nn.Conv2d(in_+inner, out_, 3, padding=1)


    def forward(self, x):

        h, w = x.shape[2:]

        out = self.conv1(x)

        out = self.conv2(out)

        if self.inner_block is not None:
            out = self.inner_block(out)

        out = self.conv3(out)

        out = F.upsample(out, size=(h, w), mode="bilinear")

        out = torch.cat((x, out), axis=1)

        return self.conv4(out) 



class Unet(nn.Module):
    def __init__(self,
                 in_:int=1,
                 out_:int=1,
                 nc:int=32,
                 num_downs:int=32) -> None:
        super().__init__()

        self.conv1 = CNA(in_=in_, out_=nc)
        self.conv2, self.conv3 = [CNA(in_=nc, out_=nc)] * 2

        unet = None
        for i in range(num_downs-3):
            unet = UnetBlock(8*nc, 8*nc, 8*nc, unet)
        unet = UnetBlock(4*nc, 8*nc, 4*nc, unet)
        unet = UnetBlock(2*nc, 4*nc, 2*nc, unet)
        
        self.unet_block = UnetBlock(1*nc, 2*nc, 1*nc, unet)

        
        self.conv4 = nn.Conv2d(nc, out_, 3, padding=1)

    
    def forward(self, x):

        out = self.conv1(x)
        out = self.conv2(x)
        out = self.unet_block(x)
        out = self.conv3(x)
        out = self.conv4(x)

        return out



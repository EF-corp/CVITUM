import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import (Tuple, 
                    Union)

from torchvision.models import vgg19

import numpy as np
import math

from src.models.utils.anime_utils import (gram, 
                                          rgb_to_yuv, 
                                          initialize_weights,
                                          spectral_norm_)




# =======CONVs=======


class DownConv(nn.Module):

    def __init__(self, 
                 channels, 
                 bias=False):
        super().__init__()

        self.conv1 = SeparableConv2D(channels, 
                                     channels, 
                                     stride=2, 
                                     bias=bias)
        
        self.conv2 = SeparableConv2D(channels, 
                                     channels, 
                                     stride=1, 
                                     bias=bias)

    def forward(self, x):

        out1 = self.conv1(x)
        out2 = F.interpolate(x, scale_factor=0.5, mode='bilinear')
        out2 = self.conv2(out2)

        return out1 + out2


class UpConv(nn.Module):
    def __init__(self, 
                 channels, 
                 bias=False):
        
        super().__init__()

        self.conv = SeparableConv2D(channels, 
                                    channels, 
                                    stride=1, 
                                    bias=bias)

    def forward(self, x):

        out = F.interpolate(x, scale_factor=2.0, mode='bilinear')
        out = self.conv(out)

        return out


class SeparableConv2D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 stride=1, 
                 bias=False):
        
        super().__init__()

        self.depthwise = nn.Conv2d(in_channels, 
                                   in_channels, 
                                   kernel_size=3, 
                                   stride=stride, 
                                   padding=1, 
                                   groups=in_channels, 
                                   bias=bias)
        
        self.pointwise = nn.Conv2d(in_channels, 
                                   out_channels, 
                                   kernel_size=1, 
                                   stride=1, 
                                   bias=bias)

        self.ins_norm1 = nn.InstanceNorm2d(in_channels)
        self.activation1 = nn.LeakyReLU(0.2, True)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation2 = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):

        out = self.depthwise(x)
        out = self.ins_norm1(out)
        out = self.activation1(out)

        out = self.pointwise(out)
        out = self.ins_norm2(out)

        return self.activation2(out)


class ConvBlock(nn.Module):
    def __init__(self, 
                 channels, 
                 out_channels, 
                 kernel_size=3, 
                 stride=1, 
                 padding=1, 
                 bias=False):
        super().__init__()

        self.conv = nn.Conv2d(channels, 
                              out_channels, 
                              kernel_size=kernel_size, 
                              stride=stride, 
                              padding=padding, 
                              bias=bias)
        self.ins_norm = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):

        out = self.conv(x)
        out = self.ins_norm(out)
        out = self.activation(out)

        return out


class InvertedResBlock(nn.Module):
    def __init__(self, 
                 channels=256, 
                 out_channels=256, 
                 expand_ratio=2, 
                 bias=False):
        super().__init__()

        bottleneck_dim = round(expand_ratio * channels)

        self.conv_block = ConvBlock(channels, bottleneck_dim, kernel_size=1, stride=1, padding=0, bias=bias)

        self.depthwise_conv = nn.Conv2d(bottleneck_dim, 
                                        bottleneck_dim, 
                                        kernel_size=3, 
                                        groups=bottleneck_dim, 
                                        stride=1, 
                                        padding=1, 
                                        bias=bias)
        
        self.conv = nn.Conv2d(bottleneck_dim, 
                              out_channels, 
                              kernel_size=1, 
                              stride=1, 
                              bias=bias)

        self.ins_norm1 = nn.InstanceNorm2d(out_channels)
        self.ins_norm2 = nn.InstanceNorm2d(out_channels)
        self.activation = nn.LeakyReLU(0.2, True)

        initialize_weights(self)

    def forward(self, x):

        out = self.conv_block(x)
        out = self.depthwise_conv(out)
        out = self.ins_norm1(out)
        out = self.activation(out)
        out = self.conv(out)
        out = self.ins_norm2(out)

        return out + x
    


# =====VGG19=====


class Vgg19(nn.Module):
    def __init__(self,
                 device:Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
        super().__init__()

        self.vgg19 = self.get_vgg19().eval()

        vgg_mean = torch.tensor([0.485, 0.456, 0.406]).float()
        vgg_std = torch.tensor([0.229, 0.224, 0.225]).float()

        vgg_mean.to(device=device)
        vgg_std.to(device=device)

        self.mean = vgg_mean.view(-1, 1 ,1)
        self.std = vgg_std.view(-1, 1, 1)

    def forward(self, x):

        return self.vgg19(self.normalize_vgg(x))


    @staticmethod
    def get_vgg19(last_layer='conv4_4'):

        vgg = vgg19(pretrained=True).features
        model_list = []

        i = 0
        j = 1

        for layer in vgg.children():
            if isinstance(layer, nn.MaxPool2d):
                i = 0
                j += 1

            elif isinstance(layer, nn.Conv2d):
                i += 1

            name = f'conv{j}_{i}'

            if name == last_layer:
                model_list.append(layer)
                break

            model_list.append(layer)


        model = nn.Sequential(*model_list)
        return model


    def normalize_vgg(self, image):

        image = (image + 1.0) / 2.0
        return (image - self.mean) / self.std



# =====LOSSES=====


class ColorLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.l1 = nn.L1Loss()
        self.huber = nn.SmoothL1Loss()

    def forward(self, 
                image, 
                image_g):

        image = rgb_to_yuv(image)
        image_g = rgb_to_yuv(image_g)


        return (self.l1(image[:, :, :, 0], image_g[:, :, :, 0]) +
                self.huber(image[:, :, :, 1], image_g[:, :, :, 1]) +
                self.huber(image[:, :, :, 2], image_g[:, :, :, 2]))


class AnimeGanLoss:
    def __init__(self, 
                 wadvg,
                 wadvd,
                 wcon,
                 wgra,
                 wcol,
                 gan_loss,
                 device:Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
        

        self.content_loss = nn.L1Loss().to(device=device)
        self.gram_loss = nn.L1Loss().to(device=device)
        self.color_loss = ColorLoss().to(device=device)
        self.wadvg = wadvg
        self.wadvd = wadvd
        self.wcon = wcon
        self.wgra = wgra
        self.wcol = wcol
        self.vgg19 = Vgg19().to(device=device).eval()
        self.adv_type = gan_loss
        self.bce_loss = nn.BCELoss()


    def compute_loss_G(self, fake_img, img, fake_logit, anime_gray):

        fake_feat = self.vgg19(fake_img)
        anime_feat = self.vgg19(anime_gray)
        img_feat = self.vgg19(img).detach()

        return [
            self.wadvg * self.adv_loss_g(fake_logit),
            self.wcon * self.content_loss(img_feat, fake_feat),
            self.wgra * self.gram_loss(gram(anime_feat), gram(fake_feat)),
            self.wcol * self.color_loss(img, fake_img),
        ]


    def compute_loss_D(self, fake_img_d, real_anime_d, real_anime_gray_d, real_anime_smooth_gray_d):
        return self.wadvd * (
            self.adv_loss_d_real(real_anime_d) +
            self.adv_loss_d_fake(fake_img_d) +
            self.adv_loss_d_fake(real_anime_gray_d) +
            0.2 * self.adv_loss_d_fake(real_anime_smooth_gray_d)
        )


    def content_loss_vgg(self, image, recontruction):
        feat = self.vgg19(image)
        re_feat = self.vgg19(recontruction)

        return self.content_loss(feat, re_feat)

    def adv_loss_d_real(self, pred):

        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 - pred))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.ones_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')

    def adv_loss_d_fake(self, pred):

        if self.adv_type == 'hinge':
            return torch.mean(F.relu(1.0 + pred))

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


    def adv_loss_g(self, pred):

        if self.adv_type == 'hinge':
            return -torch.mean(pred)

        elif self.adv_type == 'lsgan':
            return torch.mean(torch.square(pred - 1.0))

        elif self.adv_type == 'normal':
            return self.bce_loss(pred, torch.zeros_like(pred))

        raise ValueError(f'Do not support loss type {self.adv_type}')


class LossSummary:
    def __init__(self):

        self.reset()

    def reset(self):

        self.loss_g_adv = []
        self.loss_content = []
        self.loss_gram = []
        self.loss_color = []
        self.loss_d_adv = []

    def update_loss_G(self, adv, gram, color, content):

        self.loss_g_adv.append(adv.cpu().detach().numpy())
        self.loss_gram.append(gram.cpu().detach().numpy())
        self.loss_color.append(color.cpu().detach().numpy())
        self.loss_content.append(content.cpu().detach().numpy())

    def update_loss_D(self, loss):

        self.loss_d_adv.append(loss.cpu().detach().numpy())

    def avg_loss_G(self):

        return (
            self._avg(self.loss_g_adv),
            self._avg(self.loss_gram),
            self._avg(self.loss_color),
            self._avg(self.loss_content),
        )

    def avg_loss_D(self):

        return self._avg(self.loss_d_adv)


    @staticmethod
    def _avg(losses):

        return sum(losses) / len(losses)
    



# =====ANIMEGan=====

class Generator(nn.Module):
    def __init__(self,
                 dataset_name:str=""):
        super().__init__()

        self.name = f'generator_{dataset_name}'
        bias = False

        self.encode_blocks = nn.Sequential(

            ConvBlock(3, 64, bias=bias),
            ConvBlock(64, 128, bias=bias),
            DownConv(128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            SeparableConv2D(128, 256, bias=bias),
            DownConv(256, bias=bias),
            ConvBlock(256, 256, bias=bias),
        )

        self.res_blocks = nn.Sequential(

            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
            InvertedResBlock(256, 256, bias=bias),
        )

        self.decode_blocks = nn.Sequential(

            ConvBlock(256, 128, bias=bias),
            UpConv(128, bias=bias),
            SeparableConv2D(128, 128, bias=bias),
            ConvBlock(128, 128, bias=bias),
            UpConv(128, bias=bias),
            ConvBlock(128, 64, bias=bias),
            ConvBlock(64, 64, bias=bias),
            nn.Conv2d(64, 3, kernel_size=1, stride=1, padding=0, bias=bias),
            nn.Tanh(),
        )

        initialize_weights(self)

    def forward(self, x):

        out = self.encode_blocks(x)
        out = self.res_blocks(out)
        img = self.decode_blocks(out)

        return img


class Discriminator(nn.Module):
    def __init__(self, 
                 use_sn:bool,
                 d_layers:int,
                 dataset_name:str=""):
        
        super().__init__()

        self.name = f'discriminator_{dataset_name}'
        self.bias = False
        channels = 32

        layers = [
            nn.Conv2d(3, channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=self.bias),
            nn.LeakyReLU(0.2, True)
        ]

        for i in range(d_layers):
            layers += [
                nn.Conv2d(channels, channels * 2, 
                          kernel_size=3, 
                          stride=2, 
                          padding=1, 
                          bias=self.bias),
                nn.LeakyReLU(0.2, True),
                nn.Conv2d(channels * 2, 
                          channels * 4, 
                          kernel_size=3, 
                          stride=1, 
                          padding=1, 
                          bias=self.bias),
                nn.InstanceNorm2d(channels * 4),
                nn.LeakyReLU(0.2, True),
            ]
            channels *= 4

        layers += [
            nn.Conv2d(channels, 
                      channels, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=self.bias),

            nn.InstanceNorm2d(channels),

            nn.LeakyReLU(0.2, True),

            nn.Conv2d(channels, 1, 
                      kernel_size=3, 
                      stride=1, 
                      padding=1, 
                      bias=self.bias),
        ]


        if use_sn:
            for i in range(len(layers)):
                if isinstance(layers[i], nn.Conv2d):
                    layers[i] = spectral_norm_(layers[i])

        self.discriminate = nn.Sequential(*layers)

        initialize_weights(self)

    def forward(self, 
                img):
        
        return self.discriminate(img)
    

if __name__ == "__main__":
    pass

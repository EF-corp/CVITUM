import torch
import torch.nn as nn
from typing import Union


kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()

def gram(inp):

    b, c, w, h = inp.size()

    x = inp.view(b * c, w * h)

    G = torch.mm(x, x.T)

    return G.div(b * c * w * h)

def rgb_to_yuv(image,
               device:Union[str, torch.device] = "cuda" if torch.cuda.is_available() else "cpu"):
    
    image = (image + 1.0) / 2.0

    yuv_img = torch.tensordot(
        image,
        kernel.to(device=device),
        dims=([image.ndim - 3], [0]))

    return yuv_img


def initialize_weights(model:nn.Module):
     
     for m in model.modules():

        try:
            if isinstance(m, nn.Conv2d):

                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):

                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):

                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):

                m.weight.data.fill_(1)
                m.bias.data.zero_()

        except Exception as e:
            print(e)


def spectral_norm_(x):
    return torch.nn.utils.spectral_norm(x)
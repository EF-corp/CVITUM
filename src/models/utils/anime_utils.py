import torch
import torch.nn as nn
from typing import Union
import urllib
import gc

from src.models.generative.anime_gan.dataset import AnimeDataset
from src.models.utils.general import DownloadProgressBar
from PIL import Image
import cv2

import os

kernel = torch.tensor([
    [0.299, -0.14714119, 0.61497538],
    [0.587, -0.28886916, -0.51496512],
    [0.114, 0.43601035, -0.10001026]
]).float()


def proc_img(image, 
             mode:str = "norm",
             dtype:Union[torch.Tensor, str]=None):
    
    if mode == "norm":
        return image / 127.5 - 1.0
    
    if mode == "denorm":

        img = image*127.5 + 127.5

        if dtype is not None:
            if isinstance(img, torch.Tensor):
                img = img.type(dtype=dtype)

            else:
                img = img.astype(dtype)

        return img

    

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

def get_dataloader(self, 
                   root:str,
                   batch_size:int=4,
                   n_workers:int=1):

    return torch.utils.data.DataLoader(
        AnimeDataset(root=root),
        batch_size=batch_size,
        num_workers=n_workers,
        shuffle=True
    )


class To_Anime:

    def __init__(self, 
                 model:nn.Module=None) -> None:
        

        assert model != None, "Need given AnimeGan model"

        self.model = model

    
    def image2anime(self,
                    image_path:str=None):
        
        image = Image.open(image_path)
        ...

    def video2anime(self,
                    image_video:str=None):
        
        ...
        

        
def save_checkpoint(model:torch.nn.Module, 
                    optimizer, 
                    epoch:int, 
                    checkpoint_dir:str,
                    **kwargs):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
    }
    path = os.path.join(checkpoint_dir, f'{model.name}{kwargs.get("postfix")}.pth')
    torch.save(checkpoint, path)


ASSET_HOST = "https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0"

def download_from_url(weight):

    filename = f'generator_{weight.lower()}.pth'
    os.makedirs('.cache', exist_ok=True)
    url = f'{ASSET_HOST}/{filename}'
    save_path = f'.cache/{filename}'

    if os.path.isfile(save_path):
        return save_path

    desc = f'Downloading {url} to {save_path}'
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urllib.request.urlretrieve(url, save_path, reporthook=t.update_to)

    return save_path

def load_w(model, checkpoint_dir, **kwargs):

    path = os.path.join(checkpoint_dir, f'{model.name}{kwargs.get("postfix")}.pth')
    return load_(model, path)


def load_(model, weigh):

    if weigh.lower() in {'hayao','shinkai'}:
        weigh = download_from_url(weigh)

    checkpoint = torch.load(weigh,  map_location='cuda:0') if torch.cuda.is_available() else torch.load(weigh,  map_location='cpu')

    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    epoch = checkpoint['epoch']
    del checkpoint
    torch.cuda.empty_cache()
    gc.collect()

    


def set_lr(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

if __name__ == "__main__":

    anim = To_Anime()

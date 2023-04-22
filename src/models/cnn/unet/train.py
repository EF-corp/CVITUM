import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

import numpy as np
from src.models.utils.unet_utils import n_parameters
from .dataset import UnetDataset
from .model import (Unet,
                    UnetBlock,
                    CNA)

from tqdm import tqdm
from PIL import Image
import os
import math
import cv2

class UNETTrainer:

    def __init__(self,
                 path2img: str,
                 path2mask: str,
                 n_epochs: int = 200,
                 batch_size: int = 4,
                 lr: float = 1e-4,
                 betas: tuple = (0.9, 0.999),
                 n_workers: int = 4,
                 nc: int = 32,
                 out: int = 1,
                 num_downs: int = 5,
                 gamma: float = 0.9,
                 h: int = 256,
                 w: int = 256,
                 chanell: int = 3,
                 sample_interval: int = 5,
                 save_every: int = 5,
                 pretrain: bool = False,
                 path2pretrain = None,
                 ) -> None:
        
        def get_dataloader():
            return torch.utils.data.DataLoader(
                UnetDataset(img_dir=path2img,
                            mask_dir=path2mask,
                            transform=self.transform),
                batch_size=self.batch_size,
                shuffle=True,
                n_workers=self.n_workers
            )
        
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.n_workers = n_workers
        self.out = out
        self.nc = nc
        self.chanell = chanell
        self.sample_interval = sample_interval
        self.save_every = save_every
        self.num_downs = num_downs
        self.gamma = gamma
        self.h_w = (h, w)
        
        self.transform = transforms.Compose([]) #  доделать

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.unet_model = Unet(in_=self.chanell, nc=self.nc, out_=self.out, num_downs=self.num_downs)
        self.loss = nn.BCEWithLogitsLoss()
        
        self.unet_model = self.unet_model.to(self.device)
        self.loss = self.loss.to(self.device)
        
        if pretrain:
            self.unet_model.load_state_dict(torch.load(path2pretrain))
        
        self.optimizer = torch.optim.Adam(self.unet_model.parameters(), lr=self.lr, betas=self.betas)
        self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.gamma)

        self.model_parameters = n_parameters(self.unet_model)

        self.train_loader = get_dataloader()


    def train(self):

        path = os.path.join(os.path.realpath("./"), "/saves")
        os.makedirs(path, exist_ok=True)

        for epoch in range(self.n_epochs):

            loss_val = 0
            acc_val = 0

            for sample in (pbar := tqdm(self.train_loader)):

                img, mask = sample
                img = img.to(self.device)
                mask = mask.to(self.device)
                self.optimizer.zero_grad()
                
                pred = self.unet_model(img)
                loss = self.loss(pred, mask)

                loss.backward()
                loss_item = loss.item()
                loss_val += loss_item

                self.optimizer.step()
            
            self.scheduler.step()
            pbar.set_description(f'loss: {loss_item:.5f}\tlr: {self.scheduler.get_last_lr}')
            #print(f'{loss_val/len(self.train_loader)}\t lr: {self.scheduler.get_last_lr()}')

            if epoch % self.save_every == 0:
                torch.save(self.unet_model.state_dict(), os.path.join(path, f"unet_{epoch}.pth"))

    
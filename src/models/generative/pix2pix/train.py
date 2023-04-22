import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image
import torchvision.transforms as transforms

from typing import List
import time
import numpy as np

from .dataset import Pix2PixDataset
from .model import (Discriminator, 
                    GeneratorUNet, 
                    UNetDown, 
                    UNetUp)

from tqdm import tqdm
from PIL import Image
import os
import math

class Pix2PixTrainer:
    
    def __init__(self,
                 n_epochs: int = 200,
                 batch_size: int = 4,
                 lr: float = 2e-4,
                 betas: tuple = (0.5, 0.999),
                 n_workers: int = 6,
                 #decay:int = 100,
                 hr_h: int = 256,
                 hr_w: int = 256,
                 chanell: int = 3,
                 sample_interval: int = 15,
                 checkpoint_interval: int = 15,
                 path2data: str = "./",
                 dataset_mode: List[str] = ["train"],
                 save_every: int = 5,
                 lambda_pixel : int = 100,
                 pretrain : bool = False,
                 path2pretrain:dict=None,) -> None:

        def get_dataloader() -> List:
            return [
                torch.utils.data.DataLoader(
                    Pix2PixDataset(root=self.root,
                                   transforms_=self.transform),
                    batch_size=self.batch_size,
                    shuffle=True,
                    num_workers=self.n_workers
                ),

                torch.utils.data.DataLoader(
                    Pix2PixDataset(root=self.root,
                                   transforms_=self.transform,
                                   mode=["val"]),
                    batch_size=10,
                    shuffle=True,
                    num_workers=1
                )
            ]

        def weghts_init(model):
            classname = model.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(model.bias.data, 0.0)

        os.makedirs("./images/", exist_ok=True)
        os.makedirs("./saved_models/", exist_ok=True)


        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.n_workers = n_workers

        self.hr_shape = (hr_h, hr_w)
        self.patch = (1, hr_h // 2 ** 4, hr_w // 2 ** 4)

        self.checkpoint_interval = checkpoint_interval

        self.chanell = chanell
        self.root = path2data
        self.lambda_pixel = lambda_pixel
        self.sample_interval = sample_interval
        self.save_every = save_every
        self.losses_dict = {"Generator": [], "Discriminator": []}

        self.transform = [
            transforms.Resize(self.hr_shape, Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
        
        self.generator = GeneratorUNet()
        self.discriminator = Discriminator()

        self.gan_loss = torch.nn.MSELoss()
        self.pixel_loss = torch.nn.L1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.gan_loss.to(self.device)
        self.pixel_loss.to(self.device)

        if dataset_mode != ["use"]:
            self.train_dataloader, self.val_dataloader = get_dataloader()

        if pretrain:
            self.generator.load_state_dict(torch.load(path2pretrain["generator"]))
            self.discriminator.load_state_dict(torch.load(path2pretrain["discriminator"]))

        else:
            self.generator.apply(weghts_init)
            self.discriminator.apply(weghts_init)


        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)



    def save_sample(self, n:int):

        imgs = next(iter(self.val_dataloader))
        real_A = Variable(imgs["B"].type(self.Tensor))
        real_B = Variable(imgs["A"].type(self.Tensor))

        fake_B = self.generator(real_A)
        img_sample = torch.cat((real_A.data, fake_B.data, real_B.data), -2)

        save_image(img_sample, f"images/{n}.png" , nrow=5, normalize=True)

    def using(self, image_path:str):

        name = os.path.basename(image_path).split(".")[0]

        img = Image.open(image_path)
        img = self.transform(img)

        img_t = Variable(img.type(self.Tensor))

        fake = self.generator(img_t)

        save_image(fake, f"images/{name}.png" , nrow=5, normalize=True)

    def train(self):


        for epoch in range(self.n_epochs):
            for i, batch in enumerate((pbar := tqdm(self.train_dataloader))):

                start_time = time.time()

                real_A = Variable(batch["B"].type(self.Tensor))
                real_B = Variable(batch["A"].type(self.Tensor))


                valid = Variable(self.Tensor(np.ones((real_A.size(0), *self.patch))), requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((real_A.size(0), *self.patch))), requires_grad=False)


                self.optimG.zero_grad()


                fake_B = self.generator(real_A)
                pred_fake = self.discriminator(fake_B, real_A)
                loss_GAN = self.gan_loss(pred_fake, valid)

                loss_pixel = self.pixel_loss(fake_B, real_B)


                loss_G = loss_GAN + self.lambda_pixel * loss_pixel

                loss_G.backward()

                self.optimG.step()


                self.optimD.zero_grad()


                pred_real = self.discriminator(real_B, real_A)
                loss_real = self.gan_loss(pred_real, valid)


                pred_fake = self.discriminator(fake_B.detach(), real_A)
                loss_fake = self.gan_loss(pred_fake, fake)


                loss_D = 0.5 * (loss_real + loss_fake)

                loss_D.backward()
                self.optimD.step()


                batches_done = epoch * len(self.train_dataloader) + i
                batches_left = self.n_epochs * len(self.train_dataloader) - batches_done
                prev_time = time.time()

                if batches_done % self.sample_interval == 0:
                    self.losses_dict["Generator"].append(loss_G.item())
                    self.losses_dict["Discriminator"].append(loss_D.item())
                    self.save_sample(n=batches_done)

                pbar.set_description(
                    "[Epoch {}/{}]\t[Batch {}/{}]\t[D loss: {}]\t[G loss: {}, adv: {}, pixel: {}]\t[time: {}]".format(
                        epoch, self.n_epochs, i, len(self.dataloader), loss_D.item(), loss_G.item(),
                        loss_GAN.item(), loss_pixel.item(), prev_time-start_time))
                
            if  epoch % self.checkpoint_interval == 0:
                
                torch.save(self.generator.state_dict(), f"saved_models/generator_{epoch}.pth" )
                torch.save(self.discriminator.state_dict(), f"saved_models/discriminator_{epoch}.pth")

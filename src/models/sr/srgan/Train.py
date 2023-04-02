import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

import numpy as np

from .Dataset import SRDataset
from .model import (ExtractFeature, 
                   Discriminator,
                   GeneratorResNet,
                   ResBlock)

from tqdm import tqdm
from PIL import Image
import os
import math


class SRTrainer:
    
    def __init__(self,
                 n_epochs:int=200,
                 batch_size:int=4,
                 lr:float=2e-4,
                 betas:tuple=(0.5, 0.999),
                 n_workers:int=4,
                 latent:int=100,
                 hr_h:int=256,
                 hr_w:int=256,
                 chanell:int=3,
                 sample_interval:int=5,
                 path2data:str="./",
                 save_every:int=5,
                 pretrain:bool=False,
                 path2pretrain=None,
                 mode_train:bool=False #for debug
                 ) -> None:
        
        def get_dataloader():
            return torch.utils.data.DataLoader(
                SRDataset(self.root,
                          hr_shpe=self.hr_shape),
                batch_size=self.batch_size, 
                shuffle=True,
                n_workers=self.n_workers
            )
        
        def weghts_init(model):
            classname = model.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(model.bias.data, 0.0)
            
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.n_workers = n_workers
        self.latent = latent
        self.hr_shape = (hr_h, hr_w)
        self.chanell = chanell
        self.root = path2data
        self.sample_interval = sample_interval
        self.save_every = save_every
        self.losses_dict = {"Generator": [], "Discriminator": []}

        self.generator = GeneratorResNet()
        self.feature_extractor = ExtractFeature()
        self.discriminator = Discriminator(inp=(self.chanell, *self.hr_shape))

        self.feature_extractor.eval()

        self.gan_loss = nn.MSELoss()
        self.content_loss = nn.L1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.gan_loss.to(self.device)
        self.content_loss.to(self.device)

        if mode_train:
            self.generator.apply(weghts_init)
            self.discriminator.apply(weghts_init)
        
        if pretrain:
            self.generator.load_state_dict(torch.load(path2pretrain))
            self.discriminator.load_state_dict(torch.load(path2pretrain))

        self.dataloader = get_dataloader()

        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)



    def save_sample(self, imgs_lr, gen_hr, n):
        imgs_lr = nn.functional.interpolate(imgs_lr, scale_factor=4)
        gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        img_grid = torch.cat((imgs_lr, gen_hr), -1)
        save_image(img_grid, f"images/{n}.png", normalize=False)


    def train(self):
        
        path = "./checkpoint/"
        os.makedirs(path, exist_ok=True)
        os.makedirs("./images", exist_ok=True)
        for epoch in range(self.n_epochs):
            for i, imgs in enumerate((pbar := tqdm(self.train_dataloader))):

                imgs_lr = Variable(imgs["low"].type(self.Tensor))
                imgs_hr = Variable(imgs["high"].type(self.Tensor))

                valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))), requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))), requires_grad=False)


                self.optimG.zero_grad()


                gen_hr = self.generator(imgs_lr)


                loss_GAN = self.gan_loss(self.discriminator(gen_hr), valid)


                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr)
                loss_content = self.criterion_content(gen_features, real_features.detach())
                
                loss_G = loss_content + 1e-3 * loss_GAN

                loss_G.backward()
                self.optimG.step()


                self.optimD.zero_grad()


                loss_real = self.gan_loss(self.discriminator(imgs_hr), valid)
                loss_fake = self.gan_loss(self.discriminator(gen_hr.detach()), fake)

                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimD.step()

                #print(
                #    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G adv: {}, pixel: {}]".format(
                #     epoch, self.n_epochs, i, len(self.train_dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                #)

                # Generate sample at sample interval
                batches_done = epoch * len(self.train_dataloader) + i

                if batches_done % self.sample_interval == 0:
                    self.losses_dict["Generator"].append(loss_G.item())
                    self.losses_dict["Discriminator"].append(loss_D.item())
                    self.save_sample(imgs_lr=imgs_lr, gen_hr=gen_hr, n=batches_done)

                # update pbar
                pbar.set_description("[Epoch {}/{}]\t[Batch {}/{}]\t[D loss: {}]\t[G loss: {}]".format(epoch, self.n_epochs, i, len(self.dataloader), loss_D.item(), loss_G.item()))

            if epoch % self.save_every == 0:
                torch.save(self.generator.state_dict(), os.path.join(path, f"generator_{epoch}.pth"))
                torch.save(self.discriminator.state_dict(), os.path.join(path,f"discriminator_{epoch}.pth"))


if __name__ == "__main__":
    Trainer = SRTrainer()
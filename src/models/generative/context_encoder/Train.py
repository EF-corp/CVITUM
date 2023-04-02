import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image
import torchvision.transforms as transforms

import numpy as np

from .dataset import ContextEncodingDataset
from .model import Discriminator, Generator

from tqdm import tqdm
from PIL import Image
import os
import math



class ContextEncoderTrainer:
    
    def __init__(self,
                 n_epochs:int=200,
                 batch_size:int=8,
                 lr:float=2e-4,
                 betas:tuple=(0.5, 0.999),
                 n_workers:int=4,
                 latent:int=100,
                 img_size:int=128,
                 mask_size:int=64,
                 chanell:int=3,
                 sample_interval:int=5,
                 path2data:str="./") -> None:

        def weghts_init(model):
            classname = model.__class__.__name__
            if classname.find("Conv") != -1:
                torch.nn.init.normal_(model.weight.data, 0.0, 0.02)
            elif classname.find("BatchNorm2d") != -1:
                torch.nn.init.normal_(model.weight.data, 1.0, 0.02)
                torch.nn.init.constant_(model.bias.data, 0.0)


        def get_dataloader(mode="train"):
            return torch.utils.data.DataLoader(
                ContextEncodingDataset(self.root,
                                       Transforms=[
                                            transforms.Resize((self.img_size, self.img_size), Image.BICUBIC),
                                            transforms.ToTensor(),
                                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                       ],
                                       mode=mode),
                batch_size=self.batch_size, 
                shuffle=True,
                n_workers=self.n_workers
            )


        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.lr = lr
        self.betas = betas
        self.n_workers = n_workers
        self.latent = latent
        self.img_size = img_size
        self.mask_size = mask_size
        self.chanell =  chanell
        self.root = path2data
        self.sample_interval = sample_interval

        self.patchW, self.patchH = [self.mask_size // 2**3]*2
        self.patch = (1, self.patchW, self.patchH)

        self.a_loss = nn.MSELoss()
        self.p_loss = nn.L1Loss()

        self.train_dataloader = get_dataloader()
        self.test_dataloader = get_dataloader(mode="valid")

        self.generator =  Generator(chanell=self.chanell)
        self.discriminator = Discriminator(chanell=self.chanell)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.a_loss.to(self.device)
        self.p_loss.to(self.device)

        self.generator.apply(weghts_init)
        self.discriminator.apply(weghts_init)

        self.optimG = torch.optim.Adam(self.generator.parameters(), lr=self.lr, betas=self.betas)
        self.optimD = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr, betas=self.betas)



    def save_sample(self, n):

        samples, masked_samples, i = next(iter(self.test_dataloader))
        samples = Variable(samples.type(self.Tensor))
        masked_samples = Variable(masked_samples.type(self.Tensor))

        i = i[0].item()

        gen_mask = self.generator(masked_samples)

        filled_samples = masked_samples.clone()
        filled_samples[:, :, i:i+self.mask_size, i:i+self.mask_size] = gen_mask

        sample = torch.cat((masked_samples.data, filled_samples.data, samples.data), -2)
        save_image(sample, f"./results/gen_image{n}.png", nrow=6, normalize=True)



    def train(self):

        os.makedirs("./results/")

        for epoch in range(self.n_epochs):
            for i, (imgs, masked_imgs, masked_parts) in enumerate((pbar := tqdm(self.train_dataloader))):

                valid = Variable(self.Tensor(imgs.shape[0], *self.patch).fill_(1.0), requires_grad=False)
                fake = Variable(self.Tensor(imgs.shape[0], *self.patch).fill_(0.0), requires_grad=False)

                imgs = Variable(imgs.type(self.Tensor))
                masked_imgs = Variable(masked_imgs.type(self.Tensor))
                masked_parts = Variable(masked_parts.type(self.Tensor))


                self.optimG.zero_grad()


                gen_parts = self.generator(masked_imgs)


                g_adv = self.a_loss(self.discriminator(gen_parts), valid)
                g_pixel = self.p_loss(gen_parts, masked_parts)

                g_loss = 0.001 * g_adv + 0.999 * g_pixel

                g_loss.backward()
                self.optimG.step()


                self.optimD.zero_grad()


                real_loss = self.a_loss(self.discriminator(masked_parts), valid)
                fake_loss = self.a_loss(self.discriminator(gen_parts.detach()), fake)
                d_loss = 0.5 * (real_loss + fake_loss)

                d_loss.backward()
                self.optimD.step()

                #print(
                #    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G adv: {}, pixel: {}]".format(
                #     epoch, self.n_epochs, i, len(self.train_dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                #)

                # Generate sample at sample interval
                batches_done = epoch * len(self.train_dataloader) + i
                if batches_done % self.sample_interval == 0:
                    self.save_sample(batches_done)
                # update pbar
                pbar.set_description("[Epoch {}/{}]\t[Batch {}/{}]\t[D loss: {}]\t[G adv: {}, pixel: {}]".format(epoch, self.n_epochs, i, len(self.train_dataloader), d_loss.item(), g_adv.item(), g_pixel.item()))



        
        
        

if __name__ == "__main__":
    ...


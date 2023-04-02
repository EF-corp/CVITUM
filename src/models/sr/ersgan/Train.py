import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torchvision.utils import save_image, make_grid
import torchvision.transforms as transforms

import numpy as np

from .dataset import ERSDataset
from .model import (ExtractFeature,
                   Discriminator,
                   GeneratorRRDB,
                   ResInResDenseBlock)

from tqdm import tqdm
from PIL import Image
import os
import math


class ERSTrainer:

    def __init__(self,
                 n_epochs: int = 200,
                 batch_size: int = 4,
                 lr: float = 2e-4,
                 betas: tuple = (0.5, 0.999),
                 n_workers: int = 4,
                 latent: int = 100,
                 hr_h: int = 256,
                 hr_w: int = 256,
                 chanell: int = 3,
                 res_block: int = 23,
                 filters: int = 64,
                 warmup_batch: int = 500,
                 lambda_adv: float = 5e-3,
                 lambda_pixel: float = 1e-2,
                 sample_interval: int = 5,
                 path2data: str = "./",
                 save_every: int = 5,
                 pretrain: bool = False,
                 path2pretrain=None,
                 mode_train: bool = False  # for debug
                 ) -> None:

        def get_dataloader():
            return torch.utils.data.DataLoader(
                ERSDataset(self.root,
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
        self.warmup = warmup_batch
        self.lambda_adv = lambda_adv
        self.lambda_pixel = lambda_pixel
        self.sample_interval = sample_interval
        self.save_every = save_every
        self.res_block = res_block
        self.mean = np.array([0.485, 0.456, 0.406])
        self.std = np.array([0.229, 0.224, 0.225])
        self.filters = filters
        self.losses_dict = {"Generator": [], "Discriminator": []}

        self.generator = GeneratorRRDB(self.chanell, filters=self.filters, num_block=self.res_block)
        self.feature_extractor = ExtractFeature()
        self.discriminator = Discriminator(inp=(self.chanell, *self.hr_shape))

        self.feature_extractor.eval()

        self.gan_loss = nn.MSELoss()
        self.content_loss = nn.L1Loss()
        self.pixel_loss = nn.L1Loss()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

        self.generator.to(self.device)
        self.discriminator.to(self.device)
        self.gan_loss.to(self.device)
        self.content_loss.to(self.device)
        self.pixel_loss.to(self.device)

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
        # gen_hr = make_grid(gen_hr, nrow=1, normalize=True)
        # imgs_lr = make_grid(imgs_lr, nrow=1, normalize=True)
        img_grid = self.denormalize(torch.cat((imgs_lr, gen_hr), -1))
        save_image(img_grid, f"images/{n}.png", nrow=1, normalize=False)

    def denormalize(self, tensor):
        for i in range(3):
            tensor[:, i].mul_(self.std[i]).add_(self.mean[i])

    def train(self):

        path = "./checkpoint/"
        os.makedirs(path, exist_ok=True)
        os.makedirs("./images", exist_ok=True)

        for epoch in range(self.n_epochs):
            for i, imgs in enumerate((pbar := tqdm(self.train_dataloader))):

                imgs_lr = Variable(imgs["low"].type(self.Tensor))
                imgs_hr = Variable(imgs["high"].type(self.Tensor))

                valid = Variable(self.Tensor(np.ones((imgs_lr.size(0), *self.discriminator.output_shape))),
                                 requires_grad=False)
                fake = Variable(self.Tensor(np.zeros((imgs_lr.size(0), *self.discriminator.output_shape))),
                                requires_grad=False)

                self.optimG.zero_grad()

                gen_hr = self.generator(imgs_lr)

                loss_pixel = self.pixel_loss(gen_hr, imgs_hr)

                batches_done = epoch * len(self.train_dataloader) + i

                if batches_done < self.warmup:
                    loss_pixel.backward()
                    self.optimG.step()
                    pbar.set_description(
                        "[Epoch {}/{}]\t[Batch {}/{}]\t[G loss  pixel: {}]".format(epoch, self.n_epochs, i,
                                                                                   len(self.dataloader),
                                                                                   loss_pixel.item()))
                    continue

                #
                pred_real = self.discriminator(imgs_hr).detach()
                pred_fake = self.discriminator(gen_hr)

                loss_GAN = self.gan_loss(pred_fake - pred_real.mean(0, keepdim=True), valid)

                gen_features = self.feature_extractor(gen_hr)
                real_features = self.feature_extractor(imgs_hr).detach()
                loss_content = self.content_loss(gen_features, real_features.detach())

                loss_G = loss_content + self.lambda_adv * loss_GAN + self.lambda_pixel * loss_pixel

                loss_G.backward()
                self.optimG.step()

                self.optimD.zero_grad()

                loss_real = self.gan_loss(
                    self.discriminator(imgs_hr) - self.discriminator(gen_hr.detach()).mean(0, keepdim=True), valid)
                loss_fake = self.gan_loss(
                    self.discriminator(gen_hr.detach()) - self.discriminator(imgs_hr).mean(0, keepdim=True), fake)

                loss_D = (loss_real + loss_fake) / 2

                loss_D.backward()
                self.optimD.step()

                # print(
                #    "[Epoch {}/{}] [Batch {}/{}] [D loss: {}] [G adv: {}, pixel: {}]".format(
                #     epoch, self.n_epochs, i, len(self.train_dataloader), d_loss.item(), g_adv.item(), g_pixel.item())
                # )

                # Generate sample at sample interval

                if batches_done % self.sample_interval == 0:
                    self.losses_dict["Generator"].append(loss_G.item())
                    self.losses_dict["Discriminator"].append(loss_D.item())
                    self.save_sample(imgs_lr=imgs_lr, gen_hr=gen_hr, n=batches_done)

                # update pbar
                pbar.set_description(
                    "[Epoch {}/{}]\t[Batch {}/{}]\t[D loss: {}]\t[G loss: {}, content: {}, adv: {}, pixel: {}]".format(
                        epoch, self.n_epochs, i, len(self.dataloader), loss_D.item(), loss_G.item(),
                        loss_content.item(), loss_GAN.item(), loss_pixel.item()))

            if epoch % self.save_every == 0:
                torch.save(self.generator.state_dict(), os.path.join(path, f"generator_{epoch}.pth"))
                torch.save(self.discriminator.state_dict(), os.path.join(path, f"discriminator_{epoch}.pth"))


if __name__ == "__main__":
    Trainer = ERSTrainer()

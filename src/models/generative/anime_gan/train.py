import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import *
import os
import cv2
import numpy as np

from tqdm import tqdm
from multiprocessing import cpu_count

from .model import (Generator, 
                    Discriminator, 
                    AnimeGanLoss, 
                    LossSummary)

from src.models.utils.anime_utils import proc_img, load_w, set_lr, save_checkpoint

class AnimeGanTrainer:

    def __init__(self,
                 data_root:str,
                 dataset_name:str,
                 batch_size:int=6,
                 epochs:int=100,
                 init_epochs:int=5,
                 pretrain_path:Tuple[str]=(None, "GD"),
                 n_workers:int=cpu_count(),
                 save_every:int=5,
                 wadvg:float=10.0,
                 wadvd:float=10.0,
                 wcon:float=1.5,
                 wgra:float=3.0,
                 wcol:float=30.0,
                 d_layers:int=3,
                 betas:Tuple[float]=(0.5, 0.999),
                 lrs:Tuple[float]=(2e-4, 4e-4),
                 mode_gan_loss:str="lsgan",
                 **kwargs) -> None:
        

        def check():

            if not os.path.exists(self.dataroot):
                raise FileNotFoundError(f'Dataset not found {self.dataroot}')

            assert self.mode_gan_loss in ['lsgan', 'hinge', 'bce'], f'{self.mode_gan_loss} is not supported'



        self.dataroot = data_root
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.epochs = epochs
        self.save_every = save_every
        self.d_layers = d_layers
        self.betas = betas
        self.n_workers = n_workers
        self.init_epochs = init_epochs

        self.mode_gan_loss = mode_gan_loss
        self.wadvg = wadvg
        self.wadvd = wadvd
        self.wcon = wcon
        self.wgra = wgra
        self.wcol = wcol

        self.gaussian_mean = torch.tensor(0.0)
        self.gaussian_std = torch.tensor(0.1)

        check()

        self.lrG, self.lrD = lrs
        self.loss_tracker = LossSummary()
        self.loss_fn = AnimeGanLoss(wadvg=self.wadvg,
                                    wadvd=self.wadvd,
                                    wcon=self.wcon,
                                    wgra=self.wgra,
                                    wcol=self.wcol,
                                    gan_loss=self.mode_gan_loss)


        self.generator = Generator(dataset_name=self.dataset_name)
        self.discriminator = Discriminator(d_layers=self.d_layers, dataset_name=self.dataset_name)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if pretrain_path[0] is not None:
            try:
                if pretrain_path[-1] == "GD" or pretrain_path[-1] == "DG":

                    load_w(self.generator, pretrain_path[0])
                    load_w(self.discriminator, pretrain_path[0])

                elif pretrain_path[-1] == "G":
                    load_w(self.generator, pretrain_path[0])

                elif pretrain_path[-1] == "D":

                    load_w(self.discriminator, pretrain_path[0])

            except Exception as ex:
                print(ex)

        self.discriminator.to(device=self.device)
        self.generator.to(device=self.device)

        self.optimizer_g = torch.optim.Adam(self.generator.parameters(), lr=self.lrG, betas=self.betas)
        
        self.optimizer_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lrD, betas=self.betas)

        self.save_sample_dir = "./sample_images/"
        self.save_models_dir = "./saved_models/"


        os.makedirs(self.save_sample_dir, exist_ok=True)
        os.makedirs(self.save_models_dir, exist_ok=True)


    def gaussian_noise(self):

        return torch.normal(self.gaussian_mean, self.gaussian_std)
    
    def save_sample(self, 
                    name:str, 
                    max_img:int=2):

        self.generator.eval()

        max_iter = (max_img // self.batch_size) + 1
        fake_imgs = []

        for i, (img, *_) in enumerate(self.loader):
            with torch.no_grad():

                fake_img = self.generator(img.cuda())
                fake_img = fake_img.detach().cpu().numpy()

                fake_img  = fake_img.transpose(0, 2, 3, 1)
                fake_imgs.append(proc_img(fake_img, dtype=np.int16, mode="denorm"))

            if i + 1 == max_iter:
                break

        fake_imgs = np.concatenate(fake_imgs, axis=0)

        for i, img in enumerate(fake_imgs):

            cv2.imwrite(os.path.join(self.save_sample_dir, f'{name}_{i}.jpg'), img[..., ::-1])


    def train(self):


        for e in range(self.epochs):

            print(f"Epoch {e}/{self.epochs}")
            bar = tqdm(self.loader)
            self.generator.train()

            init_losses = []

            if e < self.init_epochs:

                set_lr(self.optimizer_g, self.init_lr)
                for img, *_ in bar:
                    img = img.cuda()
                    
                    self.optimizer_g.zero_grad()

                    fake_img = self.generator(img)
                    loss = self.loss_fn.content_loss_vgg(img, fake_img)
                    loss.backward()
                    self.optimizer_g.step()

                    init_losses.append(loss.cpu().detach().numpy())
                    avg_content_loss = sum(init_losses) / len(init_losses)
                    bar.set_description(f'[Init Training G] content loss: {avg_content_loss:2f}')

                set_lr(self.optimizer_g, self.lrG)
                save_checkpoint(self.generator, self.optimizer_g, e, checkpoint_dir=self.save_models_dir, posfix='_init')
                self.save_sample(name='init')
                continue

            self.loss_tracker.reset()
            for img, anime, anime_gray, anime_smt_gray in bar:

                img = img.to(self.device)
                anime = anime.to(self.device)
                anime_gray = anime_gray.to(self.device)
                anime_smt_gray = anime_smt_gray.to(self.device)


                self.optimizer_d.zero_grad()
                fake_img = self.generator(img).detach()


                if bool(np.random.randint(0, 1)):
                    fake_img += self.gaussian_noise()
                    anime += self.gaussian_noise()
                    anime_gray += self.gaussian_noise()
                    anime_smt_gray += self.gaussian_noise()

                fake_d = self.discriminator(fake_img)
                real_anime_d = self.discriminator(anime)
                real_anime_gray_d = self.discriminator(anime_gray)
                real_anime_smt_gray_d = self.discriminator(anime_smt_gray)

                loss_d = self.loss_fn.compute_loss_D(
                    fake_d, real_anime_d, real_anime_gray_d, real_anime_smt_gray_d)

                loss_d.backward()
                self.optimizer_d.step()

                self.loss_tracker.update_loss_D(loss_d)

                self.optimizer_g.zero_grad()

                fake_img = self.generator(img)
                fake_d = self.discriminator(fake_img)

                adv_loss, con_loss, gra_loss, col_loss = self.loss_fn.compute_loss_G(fake_img, 
                                                                                     img, 
                                                                                     fake_d, 
                                                                                     anime_gray)

                loss_g = adv_loss + con_loss + gra_loss + col_loss

                loss_g.backward()
                self.optimizer_g.step()

                self.loss_tracker.update_loss_G(adv_loss, 
                                                gra_loss, 
                                                col_loss, 
                                                con_loss)

                avg_adv, avg_gram, avg_color, avg_content = self.loss_tracker.avg_loss_G()
                avg_adv_d = self.loss_tracker.avg_loss_D()
                bar.set_description(f'loss G: adv {avg_adv:2f} con {avg_content:2f} gram {avg_gram:2f} color {avg_color:2f} / loss D: {avg_adv_d:2f}')

            if e %self.save_every == 0:
                save_checkpoint(self.generator, self.optimizer_g, e, checkpoint_dir=self.save_models_dir, posfix='')
                save_checkpoint(self.discriminator, self.optimizer_d, e, checkpoint_dir=self.save_models_dir, posfix='')
                self.save_sample(name=f"end_epoch{e}")


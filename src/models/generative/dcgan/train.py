
import torch
import torch.nn as nn
from DCGan import *
import os
from torch.autograd import Variable
import cv2

import torchvision.transforms as transforms
from torchvision.utils import *
from torchvision import datasets
discriminator = DCDiscriminator()
generator = DCGenerator()
generator.apply(normal_weights)
discriminator.apply(normal_weights)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

generator.to(device=device)
discriminator.to(device=device)
batch_size = 200
img_size = 32
lr = 2e-4
b1 = 0.5
b2 = 0.999
sample_interval = 1
Epochs = 100
latent_dim = 100

class  DCGanTrainer:
    def __init__(self, **kwargs) -> None:
        self.img_size = kwargs.get("img_size")
        self.dataloader = kwargs.get("dataloader")
        self.discriminator = kwargs.get("discriminator")
        self.generator = kwargs.get("generator")
        ngpu = kwargs.get("ngpu")
        self.lr = kwargs.get("lr")
        self.betas = kwargs.get("betas") #tuple: (0.5, 0.999)
        self.batch_size = kwargs.get("batch_size")
        self.latent_dim = kwargs.get("latent_dim")
        self.epochs = kwargs.get("epochs")
        self.optim = ...

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.discriminator.to(self.device)
        self.generator.to(self.device)

        def parallel(model):
            model = nn.DataParallel(model, list(range(ngpu)))

        if kwargs.get("parallel") is True:
            parallel(self.discriminator)
            parallel(self.generator)

        self.generator.apply(normal_weights)
        self.discriminator.apply(normal_weights)

    def train():
        ...








def load_test_dataset(path:str="./data/mnist"):
    os.makedirs(path, exist_ok=True)
    return torch.utils.data.DataLoader(
                                            datasets.MNIST(
                                                path,
                                                train=True,
                                                download=True,
                                                transform=transforms.Compose(
                                                    [transforms.Resize(img_size), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]
                                                ),
                                            ),
                                            batch_size=batch_size,
                                            shuffle=True,
                                        )
    

class VideoGenerator():
    def __init__(self, size: tuple = (256, 256), result_path:str="./video", format_out:str="MP4V", n_second:int=None):
        self.Images = list()
        self.format = format_out
        self.n_second = n_second
        self.path = result_path
    def add_image(self, image:np.array):
        self.Images.append(image)
        
    def generate(self):
        if self.n_second is None:
            fps = 1
        else:
            fps = len(self.Images)//self.n_second
        name = "gan_traine/"
        path = os.path.join(self.path, name)
        os.makedirs(path, exist_ok=True)
        vid = cv2.VideoCapture(path)
        #vid_name = os.path.basename(video)
        #total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        #fps = vid.get(cv2.CAP_PROP_FPS)
        #w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        #h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*self.format)
        out = cv2.VideoWriter(f'{path}GAN.mp4',codec , fps, self.size)
        for img in self.Images:
            out.write(img)
        out.release()


loss = nn.BCELoss()
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(b1, b2))
def train(dataloader, n_epochs):
    for epoch in range(n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

        
            valid = Variable(Tensor(imgs.shape[0], 1).fill_(1.0), requires_grad=False)
            fake = Variable(Tensor(imgs.shape[0], 1).fill_(0.0), requires_grad=False)

            
            real_imgs = Variable(imgs.type(Tensor))
            optimizer_G.zero_grad()

          
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], latent_dim))))

           
            gen_imgs = generator(z)

            g_loss = loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

           
            optimizer_D.zero_grad()

            real_loss = loss(discriminator(real_imgs), valid)
            fake_loss = loss(discriminator(gen_imgs.detach()), fake)
            d_loss = (real_loss + fake_loss) / 2

            d_loss.backward()
            optimizer_D.step()

            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )

            batches_done = epoch * len(dataloader) + i
            if batches_done % sample_interval == 0:

                save_image(gen_imgs.data[:25], f"images/{batches_done}.png", nrow=5, normalize=True)

if __name__ == "__main__":
    os.makedirs("./images", exist_ok=True)
    #video = VideoGenerator(size=(img_size, img_size), result_path="./data/res_video")
    dataloader = load_test_dataset()
    train(dataloader=dataloader, n_epochs=Epochs)


"""
not bad dataset: https://www.kaggle.com/datasets/vikramtiwari/pix2pix-dataset
"""
import torch
import torchvision.transforms as transforms
from PIL import Image

import glob
import os
import numpy as np
from typing import List, Any


class Pix2PixDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 root : str, 
                 transforms_ = None, 
                 mode : List[str] = ["train", "val"]):
        self.transform = transforms.Compose(transforms_)

        self.files = []

        for name in mode:
            self.files.extend(sorted(glob.glob(os.path.join(root, name) + "/*.*")))


    def __getitem__(self, index):

        img = Image.open(self.files[index % len(self.files)])
        w, h = img.size
        img_A = img.crop((0, 0, w / 2, h))
        img_B = img.crop((w / 2, 0, w, h))

        if np.random.random() < 0.5:
            img_A = Image.fromarray(np.array(img_A)[:, ::-1, :], "RGB")
            img_B = Image.fromarray(np.array(img_B)[:, ::-1, :], "RGB")

        img_A = self.transform(img_A)
        img_B = self.transform(img_B)

        return {"A": img_A, "B": img_B}

    def __len__(self):
        return len(self.files)
    

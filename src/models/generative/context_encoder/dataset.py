from glob import glob
import numpy as np
import torch
import os
import random
from PIL import Image
import torchvision.transforms as transforms


class ContextEncodingDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path:str, 
                 Transforms=None, 
                 img_size:int=128,
                 mask_size:int=64,
                 mode:str="train",
                 per_train:float=0.8):
        super().__init__()
        self.transform = transforms.Compose(Transforms)
        self.img_size = img_size
        self.mask_size = mask_size
        self.mode = mode
        self.images = list(range(20))#sorted(glob(f"{path}/*.jpg")) # ??????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????????
        
        self.images = self.images[-int(len(self.images)*per_train):] if self.mode == "train" else self.images[:-int(len(self.images)*per_train)]

    def random_masked(self, img):
        y1, x1 = np.random.randint(0, self.img_size - self.mask_size, 2)
        y2, x2 = y1 + self.mask_size, x1 + self.mask_size
        masked = img[:, y1:y2, x1:x2]

        masked_img = img.copy()
        masked_img[:, y1:y2, x1:x2] = 1
        return masked_img, masked
    
    def fixed_masked(self, img):
        ...

    def __getitem__(self, item):
        img = Image.open(self.images[item % len(self.images)])
        img = self.transform(img)

        masked_img, aux = self.random_masked(img=img)
        return img, masked_img, aux
    def __len__(self):
        return len(self.images)










if __name__ == "__main__":
    test = ContextEncodingDataset(path="./test data",
                                mode="valid")
    print(test.images)









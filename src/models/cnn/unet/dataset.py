import torch
from PIL import Image
import cv2

import numpy as np

import os
import glob



class UNETdataset(torch.utils.data.Dataset):

    def __init__(self,
                 img_dir:str,
                 mask_dir:str,
                 transform=None) -> None:
        super().__init__()

        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform 

        self.filenames_img = sorted(os.listdir(self.img_dir))


    def __len__(self):
        return len(self.filenames_img)
    
    def __getitem__(self, idx):
        img_name = self.filenames_img[idx % len(self.filenames_img)]
        img = cv2.imread(os.path.join(self.img_dir, img_name),  cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(os.path.join(self.mask_dir, img_name),  cv2.IMREAD_COLOR)[:, :, 0:1]
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)

        img = img.astype(np.float32)
        img /= 255.0

        mask = mask.astype(np.float32)
        mask /= 255.0

        if self.transform is not None:
            tran = self.transform(mask=mask, image=img)
            img, mask =  tran["mask"], tran["image"]
            mask = np.transpose(mask, (2,0,1))

        return img , mask
            
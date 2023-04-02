import torch
import numpy as np
import glob
import os
import torchvision.transforms as transforms

from PIL import Image


class SRDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 path:str, 
                 hr_shape) -> None:

        super().__init__()

        hr_h, hr_w = hr_shape
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        self.low_transform = transforms.Compose([
            transforms.Resize((hr_h // 4, hr_w // 4), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.high_transform = transforms.Compose([
            transforms.Resize((hr_h, hr_w), Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        self.files = sorted(glob.glob(os.path.join(path, "/*.*")))

    def __getitem__(self, indx):

        img = Image.open(self.files[indx % len(self.files)])
        low_img = self.low_transform(img)
        high_img = self.high_transform(img)

        return {"low": low_img, "high": high_img}
    
    def __len__(self):
        return len(self.files)
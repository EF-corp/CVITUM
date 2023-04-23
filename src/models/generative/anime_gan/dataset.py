import torch
from PIL import Image
import cv2

from tqdm import tqdm

import numpy as np

import os
import glob

from src.models.utils.anime_utils import proc_img

class AnimeDataset(torch.utils.data.Dataset):

    def __init__(self,               
                 root:str,
                 dataset_name:str,
                 transform=None):
        """
        structure folder:

        root:
            --train_images
            .
            . (images)
            .
            --style
            .
            . (images)
            .
            --smooth
            .
            . (images)
            .

        
        """
        super().__init__()

        def check(path:str) -> None:

            if not os.path.exists(path):
                raise FileNotFoundError(f" Folder {path} is not found")

        def _data_mean(data_path:str):
            check(path=data_path)
            img_files = os.listdir(path=data_path)
            total = np.zeros(3)

            for file in tqdm(img_files):
                path = os.path.join(data_path, file)
                img = cv2.imread(path)

                total += img.mean(axis=(0,1))

            mean = total/len(img_files)

            return np.mean(mean) - mean[...,::-1]

        self.root = root 
        self.dataset_name = dataset_name
        self.transform = transform

        self.style_path = os.path.join(self.root, "style")
        self.anim_path = os.path.join(self.root, dataset_name)

        check(self.root)
        check(self.anim_path)

        self.mean = _data_mean(self.style_path)

        self.images_paths = {}

        self.dummy = torch.zeros(3, 256, 256)
        self.photo, self.style, self.smooth = "train_img", "style", "smooth"

        for _ in [self.photo, self.style, self.smooth]:

            folder = os.path.join(self.root, _)
            files = os.listdir(folder)

            self.images_paths[_] = [os.path.join(folder, fi) for fi in files]


    

    def __len__(self):

        return len(self.images_paths[self.photo])
    

    def __getitem__(self, idx):

        image = self.load_photo(idx)
        anm_idx = idx
        if anm_idx > self.images_paths[self.style] - 1:
            anm_idx -= self.images_paths[self.style] * (idx // self.images_paths[self.style])

        anime, anime_gray = self.load_anime(anm_idx)
        smooth_gray = self.load_anime_smooth(anm_idx)

        return image, anime, anime_gray, smooth_gray

    def _transform(self, img, addmean=True):
        if self.transform is not None:
            img =  self.transform(image=img)['image']

        img = img.astype(np.float32)
        if addmean:
            img += self.mean
    
        return proc_img(img, mode="norm")

    def load_photo(self, index):

        fpath = self.images_paths[self.photo][index]
        image = cv2.imread(fpath)[:,:,::-1]
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)

    def load_anime(self, index):

        fpath = self.images_paths[self.style][index]
        image = cv2.imread(fpath)[:,:,::-1]

        image_gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
        image_gray = np.stack([image_gray, image_gray, image_gray], axis=-1)
        image_gray = self._transform(image_gray, addmean=False)
        image_gray = image_gray.transpose(2, 0, 1)

        image = self._transform(image, addmean=True)
        image = image.transpose(2, 0, 1)

        return torch.tensor(image), torch.tensor(image_gray)

    def load_anime_smooth(self, index):

        fpath = self.images_paths[self.smooth][index]
        image = cv2.imread(fpath, cv2.IMREAD_GRAYSCALE)
        image = np.stack([image, image, image], axis=-1)
        image = self._transform(image, addmean=False)
        image = image.transpose(2, 0, 1)
        return torch.tensor(image)
    
    



if __name__ == "__main__":

    dataset_test = AnimeDataset(...)


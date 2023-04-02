import torch

import os
import numpy as np
import cv2

class myDataset(torch.utils.data.Dataset): 
    def __init__(self, *args):
        super().__init__()
        
        paths = list(args)
        #print(paths)
        self.all_dict_path = dict(zip(paths, [sorted(os.listdir(path)) for path in paths]))
        #print(self.all_dict_path)
        self.classes_dir = dict(zip(list(range(len(paths))), [path for path in paths]))
        #print(self.classes_dir)

    
    def __len__(self):

        return sum([len(i)  for i in self.all_dict_path.values()])
    

    def __getitem__(self, idx):

        class_img, n = 0, idx
        for i in self.all_dict_path.values():
            if n < len(i):
                break
            else:
                n -= len(i)
                class_img+=1
        
        img_path = os.path.join(self.classes_dir[class_img], self.all_dict_path[self.classes_dir[class_img]][n])
        #print(img_path)
        img = cv2.imread(img_path,  cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32)
        img /= 255.0
        img1 = img.transpose((2,0,1))
        t_img = torch.from_numpy(img1)
        t_class = torch.Tensor([class_img])
        return {"img": t_img, "class": t_class}

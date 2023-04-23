import os
import glob
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import os
from PIL import Image
import cv2
import numpy as np

from tqdm import tqdm

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_nline_project(root: str = None,
                      need_format: List[str] = ["py",]) -> int:

    files = []
    prog_files = []
    nline = 0

    for p in Path(root).rglob('*'):
        files.append(os.path.join(str(p.parent), p.name))
    
    for path in files:
        if path.split(".")[-1] in need_format:
            prog_files.append(path)

    for path2prog in prog_files:

        with open(path2prog, "r") as prog:
            n = len(prog.read().split("\n")) + 1
            nline += n


    return f"The project takes {nline} lines."





def generate(pipe,
             prompt:str="",
             save_path:str=None):
  if not torch.cuda.is_available():
    raise OSError("you need have cuda gpu")

  with torch.autocast(device):
    image = pipe(prompt, guidance_scale=8.5).images[0]

  if save_path != None:
    image.save(save_path)
    return os.path.realpath(save_path)

  else:
    img = Image.new("RGB", size=image.size)
    img.paste(image)

    plt.figure(figsize=(10,10))
    plt.imshow(img)
    plt.axis("off")

    return "Done!"

def load_stable_diffusion(auth_token:str=None):
   try:
        os.startfile(r'./transformer_lib_load.sh')
        from diffusers import StableDiffusionPipeline

        assert device == "cuda", "for start you need gpu with cuda"

        model_id = "CompVis/stable-diffusion-v1-4"
        pipe = StableDiffusionPipeline.from_pretrained(model_id, revision="fp16", torch_dtype=torch.float16, use_auth_token=auth_token)
        pipe.to(device)
        
        return pipe
   except Exception as ex:
      print(ex)


def n_param(model):
    return sum([p.numel() for p in model.parameters() if p.requires_grad])

class ColorizatorOpenCV:
    def __init__(self,
                 path2prototext:str,
                 path2model:str,
                 path2points:str,
                 save_path:str) -> None:
        
        self.prototext = path2prototext
        self.model = path2model
        self.points = path2points 
        self.save_path = save_path 

        self.net = cv2.dnn.readFromCaffe(self.prototext, self.model) 
        self.pts = np.load(self.points)

        self.class8 = self.net.getLayerId("class8_ab")
        self.conv8 = self.net.getLayerId("class8_313_rh")

        self.pts = self.pts.transpose().reshape(2, 313, 1, 1)

        self.net.getLayer(self.class8).blobs = [self.pts.astype("np.float32")]
        self.net.getLayer(self.conv8).blobs = [np.full([1, 313, 2.606], dtype="float32")]


    def colorize_video(self,
                       path2video:str,
                       width:int=500,
                       output_format:str="MP4V"):
        
        def color_frame(frame):

            img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

            sc = img.astype(np.float32)
            sc /= 255.0

            lab = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
            rs = cv2.resize(lab, (224, 224))
            L = cv2.split(rs)[0]
            L -= 50

            self.net.setInput(cv2.dnn.blobFromImage(L))
            ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
            ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

            L = cv2.split(lab)[0]
            color = np.concatenate((L[:, :, :, np.newaxis], ab), axis=2)

            color = cv2.cvtColor(color, cv2.COLOR_LAB2RGB)
            color = np.clip(color, 0, 1)
            color = (255 * color).astype("uint8")
            color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

            return color
        

        vid = cv2.VideoCapture(path2video)
        #vid_name = os.path.basename(video)
        #total = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = vid.get(cv2.CAP_PROP_FPS)
        w = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = cv2.VideoWriter_fourcc(*output_format)
        out = cv2.VideoWriter(f'color_{os.path.basename(path2video).split(".")[0]}.mp4',codec , fps, (w, h))
        while True:
            s, frame = vid.read()
            if s:
                color_image = color_frame(frame=frame)
                out.write(color_image)
            else: 
                break
        vid.release()
        out.release()

    def colorize_img(self,
                     path2img:str,
                     width:int=500):
        path = self.save_path+"/Color_img"
        os.makedirs(path, exist_ok=True)
        img = cv2.imread(path2img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        sc = img.astype(np.float32)
        sc /= 255.0

        lab = cv2.cvtColor(sc, cv2.COLOR_RGB2LAB)
        rs = cv2.resize(lab, (224, 224))
        L = cv2.split(rs)[0]
        L -= 50

        self.net.setInput(cv2.dnn.blobFromImage(L))
        ab = self.net.forward()[0, :, :, :].transpose((1, 2, 0))
        ab = cv2.resize(ab, (img.shape[1], img.shape[0]))

        L = cv2.split(lab)[0]
        color = np.concatenate((L[:, :, :, np.newaxis], ab), axis=2)

        color = cv2.cvtColor(color, cv2.COLOR_LAB2RGB)
        color = np.clip(color, 0, 1)
        color = (255 * color).astype("uint8")
        color = cv2.cvtColor(color, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(path, f"colorized_{os.path.basename(path2img).split('.')[0]}"), color)

        return color

class DownloadProgressBar(tqdm):
    
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

if __name__ == "__main__":
    print(get_nline_project(root="D:\\works\\CIVITUM\\",
                            need_format=["py","sh"]))
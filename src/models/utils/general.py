import os
import glob
from typing import List
from pathlib import Path

import matplotlib.pyplot as plt
import os
from PIL import Image

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


if __name__ == "__main__":
    print(get_nline_project(root="D:\\works\\CIVITUM\\",
                            need_format=["py","sh"]))
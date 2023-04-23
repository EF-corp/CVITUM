import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch

def visualize_augment(path_to_img:str,
                      path_to_mask:str=None, 
                      model=None,
                      device:str="cpu",
                      transform=None):
    
    
    img = np.array(Image.open(path_to_img))
    img = img.astype(np.float32)
    img /= 255.0
    #print(img)
    if path_to_img is not None and model is None:
        mask = np.array(Image.open(path_to_mask))[:, :, 0:1]
        mask = mask.astype(np.float32)
        mask /= 255.0

    elif model is not None and path_to_img is None and transform is not None:
        img_t = transform(img=img)
        img_t = img_t["img"].unsqueze(0)
        pred = model(img_t.to(device))

        mask = torch.nn.functional.sigmoid(pred.detach()).cpu().numpy()[0].transpose(1,2,0)


    else:
        raise ValueError("не может быть model и mask")
    print(img.shape, mask.shape)
    figure, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 24))
    ax[0].imshow(img)
    ax[1].imshow(mask, interpolation="nearest")
    ax[0].set_axis_off()
    ax[1].set_axis_off()
    
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    visualize_augment("D:\\works\\cv_for_noobs\\src\\datasets\\celeba\\img_align_celeba\\000001.jpg")
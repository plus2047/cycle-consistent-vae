import torch
from collections import defaultdict
import numpy as np
from PIL import Image
from PIL import ImageOps

import sys
sys.path.append("PyCasia")
from pycasia.CASIA import CASIA

def resize_and_padding(img, size):
    x, y = size
    ix, iy = img.size
    k = min(x / ix, y / iy)
    tx, ty = int(ix * k), int(iy * k)
    img = img.resize((tx, ty))
    
    px, py = (x - tx) // 2, (y - ty) // 2
    print(px, py)
    return ImageOps.expand(img, (px, py, px, py), fill=255)

def process(dataset="competition-gnt", output_filename="train.pt"):
    casia = CASIA()

    origin_image_list, label_list = [], []
    for image, label in casia.load_dataset(dataset):
        origin_image_list.append(image)
        label_list.append(label)

    def resize_and_padding(img, size):
        x, y = size
        ix, iy = img.size
        k = min(x / ix, y / iy)
        tx, ty = int(ix * k), int(iy * k)
        img = img.resize((tx, ty))
        
        px1, py1 = (x - tx) // 2, (y - ty) // 2
        px2, py2 = x - tx - px1, y - ty - py1
        return ImageOps.expand(img, (px1, py1, px2, py2), fill=255)


    image_list = [resize_and_padding(i, (128, 128)) for i in origin_image_list]
    image_list = [torch.tensor(np.asarray(im, dtype=np.uint8), dtype=torch.uint8) for im in image_list]

    word2index = {}
    for label in label_list:
        if label not in word2index:
            word2index[label] = len(word2index)

    index_list = [word2index[w] for w in label_list]

    data_tensor = torch.stack(image_list)
    index_tensor = torch.tensor(index_list)

    torch.save((data_tensor, index_tensor), output_filename)


if __name__ == "__main__":
    process()

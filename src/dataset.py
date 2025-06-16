import os
import pandas as pd
from helpers import get_path, get_device

import torch
from torch.utils.data import Dataset
from torchvision import transforms as T

from PIL import Image

device = get_device()


class ObjectDataset(Dataset):
    def __init__(self, images_path, boxes_path):
        self.images_path = get_path(images_path) / "image_2"
        self.object_path = get_path(boxes_path) / "training/label_2"

        self.images = os.listdir(self.images_path)
        self.objects = os.listdir(self.object_path)

    def __getitem__(self, index):
        # read image and transform
        image = self.images[index]
        image_path = self.images_path / image
        image_res = Image.open(image_path).convert("RGB")
        image_res = T.ToTensor()(image_res)

        # read boxes and tranforms as tl, bl, tr, br
        image_name = image.split(".")[0]
        box = image_name + ".txt"
        box_path = self.object_path / box
        box_res = pd.read_csv(box_path, header=None)
        box_res = box_res.to_numpy()

        box_cords = []
        categories = []
        gt_labes = []

        """Turn cordinates to format x, y, w, h"""
        for item in box_res:
            pre_box = item[0].split(" ")
            if "DontCare" not in pre_box:
                categories.append(pre_box[0])
                box_cords.append(torch.tensor([float(x) - 1 for x in pre_box[4:8]]))
                gt_labes.append(1)
            gt_labes.append(-1)

        box_cords = torch.stack(box_cords)

        return {
            "image": image_res,
            "cords": box_cords,
            "labels": categories,
            "gt_labels": gt_labes,
        }

    def __len__(self):
        return len(self.objects)

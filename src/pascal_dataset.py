import glob
import os
import random

import torch
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataset import Dataset
import xml.etree.ElementTree as ET


class VOCDataset(Dataset):
    def __init__(self, split, im_dir, ann_dir):
        self.split = split
        self.im_dir = im_dir
        self.ann_dir = ann_dir
        classes = [
            "person",
            "bird",
            "cat",
            "cow",
            "dog",
            "horse",
            "sheep",
            "aeroplane",
            "bicycle",
            "boat",
            "bus",
            "car",
            "motorbike",
            "train",
            "bottle",
            "chair",
            "diningtable",
            "pottedplant",
            "sofa",
            "tvmonitor",
        ]
        classes = sorted(classes)
        classes = ["background"] + classes
        self.label2idx = {classes[idx]: idx for idx in range(len(classes))}
        self.idx2label = {idx: classes[idx] for idx in range(len(classes))}
        self.images_info = self.load_images_and_anns(im_dir, ann_dir, self.label2idx)

    def load_images_and_anns(self, im_dir, ann_dir, label2idx):
        im_infos = []
        for ann_file in tqdm(glob.glob(os.path.join(ann_dir, "*.xml"))):
            im_info = {}
            im_info["img_id"] = os.path.basename(ann_file).split(".xml")[0]
            im_info["filename"] = os.path.join(
                im_dir, "{}.jpg".format(im_info["img_id"])
            )
            ann_info = ET.parse(ann_file)
            root = ann_info.getroot()
            size = root.find("size")
            width = int(size.find("width").text)
            height = int(size.find("height").text)
            im_info["width"] = width
            im_info["height"] = height
            detections = []

            for obj in ann_info.findall("object"):
                det = {}

                label = label2idx[obj.find("name").text]
                bbox_info = obj.find("bndbox")
                bbox = [
                    int(float(bbox_info.find("xmin").text)) - 1,
                    int(float(bbox_info.find("ymin").text)) - 1,
                    int(float(bbox_info.find("xmax").text)) - 1,
                    int(float(bbox_info.find("ymax").text)) - 1,
                ]
                det["label"] = label
                det["bbox"] = bbox
                detections.append(det)
            im_info["detections"] = detections
            im_infos.append(im_info)
        return im_infos

    def __len__(self):
        return len(self.images_info)

    def __getitem__(self, index):
        im_info = self.images_info[index]
        im = Image.open(im_info["filename"])
        to_flip = False
        if self.split == "train" and random.random() < 0.5:
            to_flip = True
            im = im.transpose(Image.FLIP_LEFT_RIGHT)
        im_tensor = torchvision.transforms.ToTensor()(im)
        box_cords = torch.as_tensor(
            [detection["bbox"] for detection in im_info["detections"]]
        )
        categories = torch.as_tensor(
            [detection["label"] for detection in im_info["detections"]]
        )
        if to_flip:
            for idx, box in enumerate(box_cords):
                x1, y1, x2, y2 = box
                w = x2 - x1
                im_w = im_tensor.shape[-1]
                x1 = im_w - x1 - w
                x2 = x1 + w
                box_cords[idx] = torch.as_tensor([x1, y1, x2, y2])
        # return im_tensor, targets, im_info["filename"]
        return {
            "image": im_tensor,
            "cords": box_cords,
            "labels": self.idx2label,
            "gt_labels": categories,
        }


# batch = VOCDataset(
#     "train", "VOCdevkit/VOC2012/JPEGImages", "VOCdevkit/VOC2012/Annotations"
# )[0]
# from torch.utils.data import DataLoader

# traning_data = DataLoader(
#     VOCDataset(
#         "train", "VOCdevkit/VOC20012/JPEGImages", "VOCdevkit/VOC20012/Annotations"
#     ),
#     batch_size=1,
#     shuffle=True,
# )
# image, gt_boxes, labels, gt_labels = (
#     batch["image"],
#     batch["cords"],
#     batch["labels"],
#     batch["gt_labels"],
# )

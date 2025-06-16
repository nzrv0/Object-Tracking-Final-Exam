from pathlib import Path

# import matplotlib.pyplot as plt
import numpy as np

import torch
from torchsummary import summary


def get_path(subpath: str) -> Path:
    path = Path("./")
    path = path / subpath
    return path


# def show_image(image):
#     plt.imshow(image)
#     plt.show()


def get_device():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return device


def summary_model(model_):
    summary(model_)


def iou_calc(boxes1, boxes2):
    r"""
    Interseciton over union
    :param boxes1 are ground truth boxes (Tensor shape (N x 4))
    :param boxes2 are anchors that we are generated (Tensor shape (M x 4))
    :return (Tensor shape of NxM)
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    # Get top left x1,y1 coordinate
    x_left = torch.max(boxes1[:, None, 0], boxes2[:, 0])
    y_top = torch.max(boxes1[:, None, 1], boxes2[:, 1])

    # Get bottom right x2,y2 coordinate
    x_right = torch.min(boxes1[:, None, 2], boxes2[:, 2])
    y_bottom = torch.min(boxes1[:, None, 3], boxes2[:, 3])

    intersection_area = (x_right - x_left).clamp(min=0) * (y_bottom - y_top).clamp(
        min=0
    )
    union = area1[:, None] + area2 - intersection_area
    iou = intersection_area / union
    return iou


def sample_pos_neg(lables, positive, total):
    """
    Random sampling positive and negative labels
    """
    pos = torch.where(lables >= 1)[0]
    neg = torch.where(lables == 0)[0]

    # selecting num of pos and neg if there're not enough pos we increase size of negative
    num_pos = min(pos.numel(), positive)
    num_neg = total - num_pos
    num_neg = min(neg.numel(), num_neg)

    # random sampling
    num_rand_pos_idx = torch.randperm(pos.numel(), device=pos.device)[:num_pos]
    num_rand_neg_idx = torch.randperm(neg.numel(), device=neg.device)[:num_neg]

    pos_idx = pos[num_rand_pos_idx]
    neg_idx = neg[num_rand_neg_idx]

    # with this zeros like matrix we can spescify Trues and return boolen like torch tensor
    mask_pos = torch.zeros_like(lables, dtype=torch.bool)
    mask_neg = torch.zeros_like(lables, dtype=torch.bool)
    mask_pos[pos_idx] = True
    mask_neg[neg_idx] = True

    return mask_neg, mask_pos

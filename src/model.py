from rpn import RegionProposalNetwork
from roi import ROI
from feature_extractor import FeatureExtractor
from anchor_boxes import generate_anchor_maps, boxes_to_original
from helpers import get_device

import torch
from torch import nn

device = get_device()


class FasterRcnn(nn.Module):
    def __init__(self):
        super().__init__()
        self.feature_extractor = FeatureExtractor()
        self.rpn = RegionProposalNetwork(512)
        self.roi = ROI(21)

    def normalize(self, image, gt_boxes=None):
        min_size = 600
        max_size = 1000

        # normalize image by mean and standart deviation
        mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32, device=device)
        std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32, device=device)

        image = (image - mean[:, None, None]) / std[:, None, None]

        # resize image
        h, w = image.shape[-2:]
        im_shape = torch.tensor(image.shape[-2:])
        img_min_size = torch.min(im_shape).to(dtype=torch.float32)
        img_max_size = torch.max(im_shape).to(dtype=torch.float32)
        scale = torch.min(
            float(min_size) / img_min_size, float(max_size) / img_max_size
        )
        image = torch.nn.functional.interpolate(
            image,
            size=None,
            scale_factor=scale.item(),
            mode="bilinear",
            recompute_scale_factor=True,
            align_corners=False,
        )
        if gt_boxes is not None:
            ratios = [
                torch.tensor(s, dtype=torch.float32, device=device)
                / torch.tensor(s_orig, dtype=torch.float32, device=device)
                for s, s_orig in zip(image.shape[-2:], (h, w))
            ]
            ratio_height, ratio_width = ratios
            # print(gt_boxes, gt_boxes.unbind(1))
            # gt_boxes = gt_boxes.unsqueeze(0)
            xmin, ymin, xmax, ymax = gt_boxes.unbind(1)
            xmin = xmin * ratio_width
            xmax = xmax * ratio_width
            ymin = ymin * ratio_height
            ymax = ymax * ratio_height
            gt_boxes = torch.stack((xmin, ymin, xmax, ymax), dim=1)
        return image, gt_boxes

    def forward(
        self,
        image,
        gt_labels=None,
        gt_boxes=None,
    ):
        # old_shape = image.shape[-2:]
        # if self.training:
        #     image, gt_boxes = self.normalize(image, gt_boxes)
        # else:
        #     image, _ = self.normalize(image, None)
        features = self.feature_extractor(image)
        anchors = generate_anchor_maps(image, features)

        rpn = self.rpn(features, anchors, image.shape, gt_boxes)

        roi = self.roi(
            features, rpn["proposals"], image.shape[-2:], gt_boxes, gt_labels
        )
        # if not self.training:
        #     roi["boxes"] = boxes_to_original(roi["boxes"], image.shape[-2:], old_shape)

        return rpn, roi

from anchor_boxes import generate_pred_boxes, clamp_boxes, boxes_to_targets
from helpers import iou_calc, sample_pos_neg

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.ops import nms
from torch.nn.functional import smooth_l1_loss, binary_cross_entropy_with_logits


class RegionProposalNetwork(nn.Module):
    def __init__(self, feature_map_channels):
        super().__init__()
        num_anchors = 9
        self.conv1 = nn.Conv2d(
            in_channels=feature_map_channels,
            out_channels=feature_map_channels,
            kernel_size=(3, 3),
            stride=1,
            padding="same",
        )
        self.classes = nn.Conv2d(
            in_channels=feature_map_channels,
            out_channels=num_anchors,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
        )
        self.boxes = nn.Conv2d(
            in_channels=feature_map_channels,
            out_channels=num_anchors * 4,
            kernel_size=(1, 1),
            stride=1,
            padding="same",
        )

        # initialize weights
        self.conv1.weight.data.normal_(mean=0.0, std=0.01)
        self.conv1.bias.data.zero_()
        self.classes.weight.data.normal_(mean=0.0, std=0.01)
        self.classes.bias.data.zero_()
        self.boxes.weight.data.normal_(mean=0.0, std=0.01)
        self.boxes.bias.data.zero_()

    def filter_proposals(self, preds, scores, im_shape):
        """
        Filtering based on the paper describtion
        """

        # Pre nms filtering
        scores = scores.reshape(-1)
        scores = torch.sigmoid(scores)

        # selecting only top 10_000 scores, original paper applies 6000
        sc = min(len(scores), 6000)
        topk_values, topk_idx = torch.topk(scores, sc)
        scores = scores[topk_idx]
        preds = preds[topk_idx]

        # clipping boxes to image boundries
        preds = clamp_boxes(preds, im_shape)

        # remove anything less than 16 pixels
        ws, hs = preds[:, 2] - preds[:, 0], preds[:, 3] - preds[:, 1]
        keep = (ws >= 16) & (hs >= 16)
        keep = torch.where(keep)[0]
        preds = preds[keep]
        scores = scores[keep]

        # applying nms
        keep_mask = torch.zeros_like(scores)
        keep_indxs = nms(preds, scores, 0.7)
        keep_mask[keep_indxs] = True
        keep_indxs = torch.where(keep_mask)[0]

        # sorting by scores
        post_nms_indxs = keep_indxs[scores[keep_indxs].sort(descending=True)[1]]

        # in the original paper we use 300
        preds = preds[post_nms_indxs[:300]]
        scores = scores[post_nms_indxs[:300]]

        return preds, scores

    def assign_targets_to_anchors(self, anchors, gt_boxes):
        """
        Assigning labels for foreground (1), background (0), ignored anchors (-1)
        :param anchors (Tensor shape (num of anchors x 4))
        :param gt_boxes (Tensor shape (num of ground truth boxes (anchors) x 4))
        """
        iou_matrix = iou_calc(gt_boxes, anchors)

        # selecting top overlap values for each anchor
        best_match_iou, best_match_iou_idx = iou_matrix.max(dim=0)

        best_match_gt_idx_pre_thresholding = best_match_iou_idx.clone()

        below_thershold = best_match_iou < 0.3
        between_thershold = (best_match_iou < 0.7) & (best_match_iou >= 0.3)

        best_match_iou_idx[below_thershold] = -1
        best_match_iou_idx[between_thershold] = -2

        # getting highest iou value amongst the all anchors
        best_anchor_iou, _ = iou_matrix.max(dim=1)

        # getting indexses of the best matching ious
        highest_iou_idxs = torch.where(iou_matrix == best_anchor_iou[:, None])[1]

        best_match_iou_idx[highest_iou_idxs] = best_match_gt_idx_pre_thresholding[
            highest_iou_idxs
        ]

        # only matching values for foreground ignore background (-1) and ignored anchors (-2)
        matched_gt_boxes = gt_boxes[best_match_iou_idx.clamp(min=0)]

        # setting all foreground as 1
        labels = (best_match_iou_idx >= 0).to(dtype=torch.float32)

        # setting all bacgkround as 0
        bg_anchors = best_match_iou_idx == -1
        labels[bg_anchors] = 0.0

        # finally all ignored anchors set to -1
        ignored_anchors = best_match_iou_idx == -2
        labels[ignored_anchors] = -1.0

        return labels, matched_gt_boxes

    def forward(self, feature_map, anchors, image_shape, gt_boxes=None):
        """
        :param feature_map (Tensor shape (Number of anchors x 4))
        :param anchor_map (Tensor shape (9 x 4))
        :param gt_boxes (Tensor shape (number of object x 4))
        :param image_shape (Tensor shape (im_resized_height x im_resized_width))
        """

        # rpn layer
        y = F.relu(self.conv1(feature_map))
        objectness_scores_map = self.classes(y)
        box_pred = self.boxes(y)

        objectness_scores = objectness_scores_map.view(-1).to(dtype=torch.float64)
        box_pred = box_pred.reshape(-1, 4)

        # transform as 1 class

        # transforming anchors based on predictions
        box_pred_transform = box_pred.detach().reshape(-1, 4)
        preds = generate_pred_boxes(
            box_pred_transform.detach().reshape(-1, 1, 4), anchors
        )
        preds = preds.reshape(-1, 4)
        
        preds, scores = self.filter_proposals(
            preds, objectness_scores.detach(), image_shape
        )

        rpn_output = {"proposals": preds, "scores": scores}

        if not self.training:
            return rpn_output
        else:
            # assigning labels to foregorund, background, and ignored anchors
            labels, matched_gt_boxes = self.assign_targets_to_anchors(anchors, gt_boxes)

            # transform matched anchors foregorund anchors from format x1, y1, x2, y2, to x, y, w, h
            regression_targets = boxes_to_targets(matched_gt_boxes, anchors)

            # random sampling as positive and negative
            mask_neg_idx, mask_pos_idx = sample_pos_neg(labels, 128, 256)

            # selecting all of the neg and pos instances
            sampled_idxs = torch.where(mask_neg_idx | mask_pos_idx)[0]

            # calculating localization loss, and binary cross entropy loss
            localization_loss = smooth_l1_loss(
                box_pred_transform[mask_pos_idx],
                regression_targets[mask_pos_idx],
                beta=1 / 9,
                reduction="sum",
            ) / (sampled_idxs.numel())

            cls_loss = binary_cross_entropy_with_logits(
                objectness_scores[sampled_idxs].flatten(),
                labels[sampled_idxs].flatten(),
            )

            rpn_output["reg_loss"] = localization_loss
            rpn_output["cls_loss"] = cls_loss

            return rpn_output

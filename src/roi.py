from helpers import iou_calc, get_device, sample_pos_neg
from anchor_boxes import boxes_to_targets, generate_pred_boxes, clamp_boxes

import torch
import torch.nn as nn
from torch.nn.functional import cross_entropy, smooth_l1_loss, softmax
from torchvision.ops import roi_pool, nms


device = get_device()


class ROI(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # including bacgkround
        # num_classes = num_classes + 1

        # roi layers
        self.fully_connected_layer = nn.Sequential(
            nn.Linear(in_features=512 * 7 * 7, out_features=1024),
            nn.ReLU(),
            nn.Linear(in_features=1024, out_features=1024),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(in_features=1024, out_features=num_classes)
        self.regressor = nn.Linear(in_features=1024, out_features=num_classes * 4)

        # initialize weights
        self.classifier.weight.data.normal_(mean=0.0, std=0.01)
        self.classifier.bias.data.zero_()
        self.regressor.weight.data.normal_(mean=0.0, std=0.001)
        self.regressor.bias.data.zero_()

    def assign_target_to_proposal(self, proposals, gt_boxes=None, gt_labels=None):
        iou = iou_calc(gt_boxes, proposals)

        best_matched_iou, best_match_iou_idx = iou.max(dim=0)

        bg_proposal = (best_matched_iou < 0.7) & (best_matched_iou >= 0.3)
        ignored_proposal = best_matched_iou < 0.3

        best_matched_iou[bg_proposal] = -1
        best_matched_iou[ignored_proposal] = -2

        # we are getting every ground truth boxes for proposals even background
        matched_gt_boxes = gt_boxes[best_match_iou_idx.clamp(min=0)]

        labels = gt_labels[best_match_iou_idx.clamp(min=0)].to(device=device)

        labels[bg_proposal] = 0
        labels[ignored_proposal] = -1

        return labels, matched_gt_boxes

    def filter_predictions(self, pred_boxes, pred_labels, pred_scores):
        """
        We are gonna apply in order:
        1. Filter low score boxes
        2. Remove small sized boxes
        3. nms for each class
        4. keep only top values
        """
        # remove low scoring boxes
        keep = torch.where(pred_scores > 0.05)[0]
        pred_boxes, pred_scores, pred_labels = (
            pred_boxes[keep],
            pred_scores[keep],
            pred_labels[keep],
        )

        # Remove small boxes
        min_size = 16
        ws, hs = (
            pred_boxes[:, 2] - pred_boxes[:, 0],
            pred_boxes[:, 3] - pred_boxes[:, 1],
        )
        keep = (ws >= min_size) & (hs >= min_size)
        keep = torch.where(keep)[0]
        pred_boxes, pred_scores, pred_labels = (
            pred_boxes[keep],
            pred_scores[keep],
            pred_labels[keep],
        )

        # Class wise nms
        keep_mask = torch.zeros_like(pred_scores, dtype=torch.bool)
        for class_id in torch.unique(pred_labels):
            curr_indices = torch.where(pred_labels == class_id)[0]
            curr_keep_indices = nms(
                pred_boxes[curr_indices], pred_scores[curr_indices], 0.3
            )
            keep_mask[curr_indices[curr_keep_indices]] = True

        keep_indices = torch.where(keep_mask)[0]
        post_nms_keep_indices = keep_indices[
            pred_scores[keep_indices].sort(descending=True)[1]
        ]
        keep = post_nms_keep_indices[:100]
        pred_boxes, pred_scores, pred_labels = (
            pred_boxes[keep],
            pred_scores[keep],
            pred_labels[keep],
        )
        return pred_boxes, pred_labels, pred_scores

    def forward(self, features, proposals, image_shape, gt_boxes, gt_labels):
        proposals = proposals.to(dtype=torch.float32)
        if self.training and gt_boxes is not None and gt_labels is not None:
            # concatign proposals with grouth truth boxes
            proposals = torch.cat([proposals, gt_boxes], dim=0)

            labels, matched_boxes = self.assign_target_to_proposal(
                proposals, gt_boxes, gt_labels
            )
            pos_idx, neg_idx = sample_pos_neg(labels, 32, 128)

            sampled_idxs = torch.where(pos_idx | neg_idx)[0]

            # keeping only sampled proposals
            proposals = proposals[sampled_idxs]
            labels = labels[sampled_idxs]
            matched_boxes = matched_boxes[sampled_idxs]
            regression_targets = boxes_to_targets(matched_boxes, proposals)

        possible_sacles = []

        # we need to set scale factro because of the downscale form image to feature map
        # for vgg16 it would be 1/16 (0.0625)
        for s1, s2 in zip(features.shape[-2:], image_shape):
            approx_scale = float(s1) / float(s2)
            scale = 2 ** float(torch.tensor(approx_scale).log2().round())
            possible_sacles.append(scale)

        proposal_roi_pool_feats = roi_pool(
            features, [proposals], output_size=7, spatial_scale=possible_sacles[0]
        ).flatten(start_dim=1)

        fc = self.fully_connected_layer(proposal_roi_pool_feats)

        cls_scores = self.classifier(fc)
        box_transform_pred = self.regressor(fc)

        # calculating loss
        num_boxes, num_classes = cls_scores.shape
        box_transform_pred = box_transform_pred.reshape(num_boxes, num_classes, 4)

        output = {}
        if self.training and gt_boxes is not None and gt_labels is not None:
            classification_loss = cross_entropy(cls_scores, labels)

            # Compute loss only for foreground
            fg_propsal_idxs = torch.where(labels > 0)[0]
            fg_cls_labels = labels[fg_propsal_idxs]

            localizaiton_loss = (
                smooth_l1_loss(
                    box_transform_pred[fg_propsal_idxs, fg_cls_labels],
                    regression_targets[fg_propsal_idxs],
                    beta=1 / 9,
                    reduction="sum",
                )
                / labels.numel()
            )

            output["classificaiton_loss"] = classification_loss
            output["localizaiton_loss"] = localizaiton_loss
        if self.training:
            return output
        else:
            pred_boxes = generate_pred_boxes(box_transform_pred, proposals)
            pred_scores = softmax(cls_scores, dim=-1)

            # creating labels for each prediction
            pred_labels = torch.arange(num_classes, device=device)
            pred_labels = pred_labels.view(1, -1).expand_as(pred_scores)

            # clamp boxes to boundry
            pred_boxes = clamp_boxes(pred_boxes, image_shape)

            # remove predictions with the background label
            pred_boxes = pred_boxes[:, 1:]
            pred_scores = pred_scores[:, 1:]
            pred_labels = pred_labels[:, 1:]

            pred_boxes = pred_boxes.reshape(-1, 4)
            pred_scores = pred_scores.reshape(-1)
            pred_labels = pred_labels.reshape(-1)

            pred_boxes, pred_labels, pred_scores = self.filter_predictions(
                pred_boxes, pred_labels, pred_scores
            )
            output["boxes"] = pred_boxes
            output["scores"] = pred_scores
            output["labels"] = pred_labels
            return output

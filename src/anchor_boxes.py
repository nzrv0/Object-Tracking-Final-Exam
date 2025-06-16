import numpy as np
import torch


def anchor_size():
    areas = np.array([[64, 64], [64 * 2, 64 * 2], [64 * 4, 64 * 4]])
    ratio = np.array([0.5, 1.0, 2.0])

    return (areas[:, None] * ratio[:, None]).reshape(-1, 2)


def generate_anchor_maps(image, feature_map):
    """
    Generate anchors based on image grid centers
    :param image: (Tensor shape (im_height x im_width))
    :param feature_map: (Tensor shape (512 x f_heihgt x f_width))
    :return anchors: (Numpy shape (number_anchors, 4)), valid_anchors (Numpy shape (number_anchors x 4))
    """

    # generating 3 anchros based on 3 size
    anchors = anchor_size()

    # creating tl, bl, tr, br
    anc_len = anchors.shape[0]
    anc_temp = np.empty((anc_len, 4))
    anc_temp[:, 0:2] = -0.5 * anchors
    anc_temp[:, 2:4] = +0.5 * anchors
    """
        [[ -32.  -32.   32.   32.]
        [ -64.  -64.   64.   64.]
        [-128. -128.  128.  128.]
        [ -64.  -64.   64.   64.]
        [-128. -128.  128.  128.]
        [-256. -256.  256.  256.]
        [-128. -128.  128.  128.]
        [-256. -256.  256.  256.]
        [-512. -512.  512.  512.]]

    """
    f_width = feature_map.shape[-1]
    f_height = feature_map.shape[-2]

    im_width = image.shape[-1]
    im_height = image.shape[-2]

    # calculate shift in that how original image differs from feature map for creating grid
    w_stride = im_width / f_width
    h_stride = im_height / f_height
    shiftX = np.arange(0, f_width) * w_stride
    shiftY = np.arange(0, f_height) * h_stride

    xv, yv = np.meshgrid(shiftX, shiftY)
    shifts = np.vstack((xv.ravel(), yv.ravel(), xv.ravel(), yv.ravel())).T

    # sum up n dimensions with anch_temp
    anchors = shifts.reshape((shifts.shape[0], 1, 4)) + anc_temp
    anchors = anchors.reshape(-1, 4)
    anchors = torch.tensor(anchors)
    return anchors


def clamp_boxes(boxes, image_shape):
    """
    Clamp anchors or proposals to image boundires
    """
    x1, y1, x2, y2 = boxes[..., 0], boxes[..., 1], boxes[..., 2], boxes[..., 3]
    h, w = image_shape[-2:]

    x1, x2 = x1.clamp(min=0, max=w), x2.clamp(min=0, max=w)
    y1, y2 = y1.clamp(min=0, max=h), y2.clamp(min=0, max=h)

    return torch.cat(
        (x1[..., None], y1[..., None], x2[..., None], y2[..., None]), dim=-1
    )


def generate_pred_boxes(box_pred, anchors_or_props):
    """
    Generating predicted boxes via transforming box_pred for all anchors or proposals
    :param box_pred (Tensor shape (num_anchors_or_proposals, num_classes, 4))
    :param anchors (Tensor shape (num_anchors, 4))
    :return pred_box (Tensor shape (num_anchors_or_proposals, num_classes, 4))
    """

    # reshape to format num of anchors or proposals, classes, 4 corners
    box_pred = box_pred.reshape(box_pred.size(0), -1, 4)

    # Get cx, cy, w, h from x1, y1, x2, y2
    w = anchors_or_props[:, 2] - anchors_or_props[:, 0]
    h = anchors_or_props[:, 3] - anchors_or_props[:, 1]
    center_x = anchors_or_props[:, 0] + 0.5 * w
    center_y = anchors_or_props[:, 1] + 0.5 * h

    dx = box_pred[..., 0]
    dy = box_pred[..., 1]
    dw = box_pred[..., 2]
    dh = box_pred[..., 3]

    pred_center_x = dx * w[:, None] + center_x[:, None]
    pred_center_y = dy * h[:, None] + center_y[:, None]
    pred_w = torch.exp(dw) * w[:, None]
    pred_h = torch.exp(dh) * h[:, None]

    # converting again the same format x1, y1, x2, y2
    pred_box_x1 = pred_center_x - 0.5 * pred_w
    pred_box_y1 = pred_center_y - 0.5 * pred_h
    pred_box_x2 = pred_center_x + 0.5 * pred_w
    pred_box_y2 = pred_center_y + 0.5 * pred_h

    # stacking vertically like in the format x1, y1, x2, y2
    pred_boxes = torch.stack(
        (pred_box_x1, pred_box_y1, pred_box_x2, pred_box_y2), dim=2
    )
    return pred_boxes


def boxes_to_targets(gt_boxes, anchors):
    """
    Transform anchors or proposals in the format of x1, y1, x2, y2 to transformed form tx, ty, tw, th
    :param gt_boxes (Tensor shape (N x 4))
    :param anchors (Tensor shape (N x 4))
    :return (Tensor shape (N x 4))
    """

    # Get cx, cy, w, h from x1, y1, x2, y2 for anchors
    w = anchors[:, 2] - anchors[:, 0]
    h = anchors[:, 3] - anchors[:, 1]
    center_x = anchors[:, 0] + 0.5 * w
    center_y = anchors[:, 1] + 0.5 * h

    # Get center_x,center_y,w,h from x1,y1,x2,y2 for gt_boxes
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_center_x = gt_boxes[:, 0] + 0.5 * w
    gt_center_y = gt_boxes[:, 1] + 0.5 * h

    # transform as top left, bottom left, top rigth, bottom rigth
    target_dx = (gt_center_x - center_x) / w
    target_dy = (gt_center_y - center_y) / h
    target_dw = torch.log(gt_w / w)
    target_dh = torch.log(gt_h / h)

    # stack as tl, bl, tr, br in row wise
    regression_targets = torch.stack(
        (target_dx, target_dy, target_dw, target_dh), dim=1
    )

    return regression_targets


def boxes_to_original(boxes, new_size, original_size):
    """
    Turning images and boxes from scaled form to original form
    """
    ratios = [
        torch.tensor(s_orig, dtype=torch.float32, device=boxes.device)
        / torch.tensor(s, dtype=torch.float32, device=boxes.device)
        for s, s_orig in zip(new_size, original_size)
    ]
    ratio_height, ratio_width = ratios
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    xmin = xmin * ratio_width
    xmax = xmax * ratio_width
    ymin = ymin * ratio_height
    ymax = ymax * ratio_height
    return torch.stack((xmin, ymin, xmax, ymax), dim=1)

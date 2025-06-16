from model import FasterRcnn
from dataset import ObjectDataset
from helpers import get_path, get_device

import torch
from torch.utils.data import DataLoader
from torch.optim import SGD
from torch.optim.lr_scheduler import MultiStepLR
from tqdm import tqdm
import numpy as np

device = get_device()


def load_dataset():
    image_path = "training"
    box_path = "data_object_label_2"
    # shuffle true in the colab
    traning_data = DataLoader(ObjectDataset(image_path, box_path), batch_size=1)
    return traning_data


def load_model():
    model = FasterRcnn().to(device)
    model.train()
    optim = SGD(params=model.parameters(), lr=0.001, weight_decay=5e-4, momentum=0.9)
    scheduler = MultiStepLR(optim, [12, 16], gamma=0.1)
    return model, optim, scheduler


"""EDIT THIS BEFORE UPLOADING TO COLAB"""
if __name__ == "__main__":
    dataset = load_dataset()
    rpn_cls_losses = []
    rpn_reg_losses = []
    roi_cls_losses = []
    roi_reg_losses = []

    model, optim, scheduler = load_model()

    epochs = 20
    for epoch in range(epochs):
        # optim.zero_grad()

        for batch in tqdm(dataset):
            image, gt_boxes, labels, gt_labels = (
                batch["image"],
                batch["cords"],
                batch["labels"],
                batch["gt_labels"],
            )

            gt_labels = torch.tensor(gt_labels)

            gt_boxes = gt_boxes.squeeze(0)
            rpn, roi = model(image, gt_labels, gt_boxes)

            rpn_reg_loss, rpn_cls_loss = rpn["reg_loss"], rpn["cls_loss"]
            roi_reg_loss, roi_cls_loss = (
                roi["localizaiton_loss"],
                roi["classificaiton_loss"],
            )
            rpn_cls_losses.append(rpn_cls_loss.item())
            rpn_reg_losses.append(rpn_reg_loss.item())
            roi_cls_losses.append(roi_cls_loss.item())
            roi_reg_losses.append(roi_reg_loss.item())

            total_loss = rpn_reg_loss + rpn_cls_loss + roi_reg_loss + roi_cls_loss

            total_loss.backward()

            optim.step()
            optim.zero_grad()

        torch.save(model.state_dict(), f"model{epoch}")

        # optim.step()
        # optim.zero_grad()

        loss_output = ""
        loss_output += "RPN Classification Loss : {:.4f}".format(
            np.mean(rpn_cls_losses)
        )
        loss_output += " | RPN Localization Loss : {:.4f}".format(
            np.mean(rpn_reg_losses)
        )
        loss_output += " | FRCNN Classification Loss : {:.4f}".format(
            np.mean(roi_cls_losses)
        )
        loss_output += " | FRCNN Localization Loss : {:.4f}".format(
            np.mean(roi_reg_losses)
        )
        print(loss_output)

        scheduler.step()

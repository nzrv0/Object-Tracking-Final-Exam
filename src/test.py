from dataset import ObjectDataset
from pascal_dataset import VOCDataset
from model import FasterRcnn
from helpers import get_path

import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torchvision import transforms as T


image_path = "training"
box_path = "data_object_label_2"

dataset = VOCDataset(
    "test", "VOCdevkit/VOC2012/JPEGImages", "VOCdevkit/VOC2012/Annotations"
)
# dataset = ObjectDataset(image_path, box_path)

model = FasterRcnn()

model_path = get_path("models")

model.load_state_dict(
    torch.load(
        model_path / "model19.pth",
        weights_only=True,
        map_location=torch.device("cpu"),
    )
)
model.eval()

data = dataset[0]

image, gt_boxes, labels, gt_labels = (
    data["image"],
    data["cords"],
    data["labels"],
    data["gt_labels"],
)

gt_labels = torch.tensor(gt_labels)

gt_boxes = gt_boxes.squeeze(0)

# torch.autograd.set_detect_anomaly(True)


def visualise(image=None):
    labels = [
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
        "background",
    ]
    labels = sorted(labels)

    image_path = get_path("training") / "image_2"
    images = os.listdir(image_path)
    image = Image.open(image_path / images[-3]).convert("RGB")

    image = T.ToTensor()(image)

    image = image[None, :]

    rpn, roi = model(image)
    max_el = torch.where(roi["scores"] * 100 >= 10)

    roi_labels = roi["labels"].detach().cpu().numpy()
    labels = np.array(labels)
    labels = labels[roi_labels]
    print(roi["boxes"][max_el].detach().cpu())
    # roi_ll = np.zeros(roi_labels.shape, dtype=object)

    # for key in labels:
    #     roi_ll[roi_labels == key] = labels[key]

    # roi_ll = roi_ll[max_el]
    # roi_ll = [roi_ll] if isinstance(roi_ll, str) else roi_ll

    font = get_path("fonts") / "Roboto_SemiCondensed-Medium.ttf"
    print(font.resolve())
    drawn_boxes = draw_bounding_boxes(
        image.squeeze().detach().cpu(),
        roi["boxes"][max_el].detach().cpu(),
        labels=labels[max_el],
        colors="red",
        width=1,
        font=font,
        font_size=24,
    )
    img_with_boxes = F.to_pil_image(drawn_boxes)

    plt.imshow(img_with_boxes)
    plt.show()
    return np.array(img_with_boxes)


def video_test():
    import cv2 as cv

    test_video = get_path("videos") / "test.mp4"
    cap = cv.VideoCapture(test_video)

    cv.namedWindow("Video", cv.WINDOW_NORMAL)

    cv.resizeWindow("Video", 1200, 800)

    while cap.isOpened():
        ret, frame = cap.read()
        frame = F.to_tensor(frame)

        frame = visualise(frame)

        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        cv.imshow("Video", frame)
        if cv.waitKey(1) == ord("q"):
            break

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    visualise(image)
    # video_test()

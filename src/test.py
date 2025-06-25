from dataset import ObjectDataset
from pascal_dataset import VOCDataset

from model import FasterRcnn
from helpers import get_path
import matplotlib.pyplot as plt

from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
import numpy as np

import torch


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
        model_path / "model4.pth",
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


def visualise(image):
    image = image[None, :]
    rpn, roi = model(image, gt_labels, gt_boxes)
    # max_el = torch.sort(roi["scores"] * 100, dim=0, descending=True)[1][:40]
    roi_labels = roi["labels"].detach().cpu().numpy()
    roi_ll = np.zeros(roi_labels.shape, dtype=object)
    # for key in labels:
    #     roi_ll[roi_labels == key] = labels[key]
    # print(roi_ll[max_el])
    from helpers import iou_calc

    print(gt_boxes)

    ious = iou_calc(gt_boxes.unsqueeze(0), roi["boxes"])

    max_elements = []
    for item in ious:
        max_el = torch.sort(item * 100, dim=0, descending=True)[1][:1]
        max_elements.append(max_el.item())

    drawn_boxes = draw_bounding_boxes(
        image.squeeze().detach().cpu(),
        roi["boxes"][max_elements].detach().cpu(),
        # labels=roi_ll[max_el],
        colors="red",
        width=1,
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

    cv.resizeWindow("Video", 1200, 800)  # width x height

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

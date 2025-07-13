from model import FasterRcnn
from helpers import get_path

import streamlit as st
from PIL import Image
import numpy as np

import torch
from torchvision.utils import draw_bounding_boxes
import torchvision.transforms.functional as F
from torchvision import transforms as T

from huggingface_hub import hf_hub_download


def load_model():
    repo_id = "nzrv0/object-tracking-model"
    filename = "model19.pth"

    model = FasterRcnn()

    model_path = hf_hub_download(repo_id=repo_id, filename=filename)

    model.load_state_dict(
        torch.load(
            model_path,
            weights_only=True,
            map_location=torch.device("cpu"),
        )
    )
    model.eval()
    return model


def show_boxes(image_path):
    image = Image.open(image_path).convert("RGB")
    image = T.ToTensor()(image)

    image = image[None, :]

    model = load_model()
    rpn, roi = model(image)

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

    max_el = torch.where(roi["scores"] * 100 >= 10)

    roi_labels = roi["labels"].detach().cpu().numpy()
    labels = np.array(labels)
    labels = labels[roi_labels][max_el]
    labels = [labels] if isinstance(labels, str) else labels

    import os

    font = os.path.abspath("app/static/Roboto_SemiCondensed-Medium.ttf")

    rpn_head = draw_bounding_boxes(
        image.squeeze().detach().cpu(),
        rpn["proposals"].detach().cpu(),
        colors="blue",
        width=1,
    )

    roi_head = draw_bounding_boxes(
        image.squeeze().detach().cpu(),
        roi["boxes"][[max_el]].detach().cpu(),
        labels=labels,
        colors="red",
        width=2,
        font=font,
        font_size=24,
    )
    rpn_head = F.to_pil_image(rpn_head)
    roi_head = F.to_pil_image(roi_head)

    return rpn_head, roi_head


def run_web_app():
    st.title("Detecting Objects With FasterRCNN")
    col1, col2, col3 = st.columns([1, 4, 1])

    st.markdown("---")

    left, center, right = st.columns([1, 20, 1])

    with center:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, use_container_width=False, width=1600)
            rpn, roi = show_boxes(uploaded_file)

    with col1:
        if st.button("RPN Output") and uploaded_file:
            with center:
                st.image(rpn, use_container_width=False, width=1600)
    with col2:
        if st.button("FasterRCNN Output") and uploaded_file:
            with center:
                st.image(roi, use_container_width=False, width=1600)


run_web_app()

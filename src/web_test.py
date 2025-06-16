from model import FasterRcnn
from pathlib import Path

import streamlit as st
from PIL import Image

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

    max_el = torch.sort(roi["scores"] * 100, dim=0, descending=False)[1][:5]

    rpn_head = draw_bounding_boxes(
        image.squeeze().detach().cpu(),
        rpn["proposals"].detach().cpu(),
        colors="blue",
        width=1,
    )
    roi_head = draw_bounding_boxes(
        image.squeeze().detach().cpu(),
        roi["boxes"].detach().cpu(),
        colors="red",
        width=1,
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

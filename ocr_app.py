import io

import pandas as pd
from PIL import Image
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import hashlib
import pypdfium2

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
import subprocess
import re

MAX_WIDTH = 1000


def replace_katex_invalid(string):
    return re.sub(r'\\tag\{.*?\}', '', string)

@st.cache_resource()
def load_model_cached():
    return load_model()


@st.cache_resource()
def load_processor_cached():
    return load_processor()


@st.cache_data()
def infer_image(pil_image, bbox):
    input_img = pil_image.crop(bbox)
    model_output = batch_inference([input_img], model, processor)
    return model_output[0]


def open_pdf(pdf_file):
    stream = io.BytesIO(pdf_file.getvalue())
    return pypdfium2.PdfDocument(stream)


@st.cache_data()
def get_page_image(pdf_file, page_num, dpi=96):
    doc = open_pdf(pdf_file)
    renderer = doc.render(
        pypdfium2.PdfBitmap.to_pil,
        page_indices=[page_num - 1],
        scale=dpi / 72,
    )
    png = list(renderer)[0]
    png_image = png.convert("RGB")
    return png_image


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


@st.cache_data()
def get_canvas_hash(pil_image):
    return hashlib.md5(pil_image.tobytes()).hexdigest()


@st.cache_data()
def get_image_size(pil_image):
    if pil_image is None:
        return 800, 600
    height, width = pil_image.height, pil_image.width
    if width > MAX_WIDTH:
        scale = MAX_WIDTH / width
        height = int(height * scale)
        width = MAX_WIDTH
    return height, width


st.set_page_config(layout="wide")
col1, col2 = st.columns([.7, .3])

model = load_model_cached()
processor = load_processor_cached()

pdf_file = st.sidebar.file_uploader("PDF file:", type=["pdf"])
if pdf_file is None:
    st.stop()

page_count = page_count(pdf_file)
page_number = st.sidebar.number_input(f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count)

pil_image = get_page_image(pdf_file, page_number)
canvas_hash = get_canvas_hash(pil_image) if pil_image else "canvas"

with col1:
    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.1)",  # Fixed fill color with some opacity
        stroke_width=1,
        stroke_color="#FFAA00",
        background_color="#FFF",
        background_image=pil_image,
        update_streamlit=True,
        height=get_image_size(pil_image)[0],
        width=get_image_size(pil_image)[1],
        drawing_mode="rect",
        point_display_radius=0,
        key=canvas_hash,
    )

if canvas_result.json_data is not None:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    if objects.shape[0] > 0:
        boxes = objects[objects["type"] == "rect"][["left", "top", "width", "height"]]
        boxes["right"] = boxes["left"] + boxes["width"]
        boxes["bottom"] = boxes["top"] + boxes["height"]
        bbox_list = boxes[["left", "top", "right", "bottom"]].values.tolist()
        with col2:
            inferences = [infer_image(pil_image, bbox) for bbox in bbox_list]
            for idx, inference in enumerate(inferences):
                st.markdown(f"### {idx + 1}")
                katex_markdown = replace_katex_invalid(inference)
                st.markdown(katex_markdown)
                st.code(inference)
                st.divider()
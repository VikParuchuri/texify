import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

import io

import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas
import hashlib
import pypdfium2

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from texify.output import replace_katex_invalid
from PIL import Image

MAX_WIDTH = 800
MAX_HEIGHT = 1000


@st.cache_resource()
def load_model_cached():
    return load_model()


@st.cache_resource()
def load_processor_cached():
    return load_processor()


@st.cache_data()
def infer_image(pil_image, bbox, temperature):
    input_img = pil_image.crop(bbox)
    model_output = batch_inference([input_img], model, processor, temperature=temperature)
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
def get_uploaded_image(in_file):
    return Image.open(in_file).convert("RGB")


def resize_image(pil_image):
    if pil_image is None:
        return
    pil_image.thumbnail((MAX_WIDTH, MAX_HEIGHT), Image.Resampling.LANCZOS)


@st.cache_data()
def page_count(pdf_file):
    doc = open_pdf(pdf_file)
    return len(doc)


def get_canvas_hash(pil_image):
    return hashlib.md5(pil_image.tobytes()).hexdigest()


@st.cache_data()
def get_image_size(pil_image):
    if pil_image is None:
        return MAX_HEIGHT, MAX_WIDTH
    height, width = pil_image.height, pil_image.width
    return height, width


st.set_page_config(layout="wide")

top_message = """### Texify

After the model loads, upload an image or a pdf, then draw a box around the equation or text you want to OCR by clicking and dragging. Texify will convert it to Markdown with LaTeX math on the right.

If you have already cropped your image, select "OCR image" in the sidebar instead.
"""

st.markdown(top_message)
col1, col2 = st.columns([.7, .3])

model = load_model_cached()
processor = load_processor_cached()

in_file = st.sidebar.file_uploader("PDF file or image:", type=["pdf", "png", "jpg", "jpeg", "gif", "webp"])
if in_file is None:
    st.stop()

filetype = in_file.type
whole_image = False
if "pdf" in filetype:
    page_count = page_count(in_file)
    page_number = st.sidebar.number_input(f"Page number out of {page_count}:", min_value=1, value=1, max_value=page_count)

    pil_image = get_page_image(in_file, page_number)
else:
    pil_image = get_uploaded_image(in_file)
    whole_image = st.sidebar.button("OCR image")

# Resize to max bounds
resize_image(pil_image)

temperature = st.sidebar.slider("Generation temperature:", min_value=0.0, max_value=1.0, value=0.0, step=0.05)

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

if canvas_result.json_data is not None or whole_image:
    objects = pd.json_normalize(canvas_result.json_data["objects"])  # need to convert obj to str because PyArrow
    bbox_list = None
    if objects.shape[0] > 0:
        boxes = objects[objects["type"] == "rect"][["left", "top", "width", "height"]]
        boxes["right"] = boxes["left"] + boxes["width"]
        boxes["bottom"] = boxes["top"] + boxes["height"]
        bbox_list = boxes[["left", "top", "right", "bottom"]].values.tolist()
    if whole_image:
        bbox_list = [(0, 0, pil_image.width, pil_image.height)]

    if bbox_list:
        with col2:
            inferences = [infer_image(pil_image, bbox, temperature) for bbox in bbox_list]
            for idx, inference in enumerate(reversed(inferences)):
                st.markdown(f"### {len(inferences) - idx}")
                katex_markdown = replace_katex_invalid(inference)
                st.markdown(katex_markdown)
                st.code(inference)
                st.divider()

with col2:
    tips = """
    ### Usage tips
    - Don't make your boxes too small or too large.  See the examples and the video in the [README](https://github.com/vikParuchuri/texify) for more info.
    - Texify is sensitive to how you draw the box around the text you want to OCR. If you get bad results, try selecting a slightly different box, or splitting the box into multiple.
    - You can try changing the temperature value on the left if you don't get good results.  This controls how "creative" the model is.
    - Sometimes KaTeX won't be able to render an equation (red error text), but it will still be valid LaTeX.  You can copy the LaTeX and render it elsewhere.
    """
    st.markdown(tips)
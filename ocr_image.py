import argparse

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from PIL import Image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OCR an image of a LaTeX equation.")
    parser.add_argument("image", type=str, help="Path to image to OCR.")
    args = parser.parse_args()

    image_path = args.image
    model = load_model()
    processor = load_processor()

    image = Image.open(image_path)
    text = batch_inference([image], model, processor)

    print(text[0])


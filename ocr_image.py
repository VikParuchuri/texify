import argparse
import os.path

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from PIL import Image

from texify.output import replace_katex_invalid
from texify.settings import settings
from texify.util import is_valid_image
import json


def inference_single_image(image_path, json_path, model, processor, katex_compatible=False):
    image = Image.open(image_path)
    text = batch_inference([image], model, processor)
    if katex_compatible:
        text = [replace_katex_invalid(t) for t in text]
    write_data = [{"image_path": image_path, "text": text[0]}]
    with open(json_path, "w+") as f:
        json_repr = json.dumps(write_data, indent=4)
        f.write(json_repr)


def inference_image_dir(image_dir, json_path, model, processor, max=None, katex_compatible=False):
    image_paths = [os.path.join(image_dir, image_name) for image_name in os.listdir(image_dir)]
    image_paths = [ip for ip in image_paths if is_valid_image(ip)]
    if max:
        image_paths = image_paths[:max]

    write_data = []
    for i in range(0, len(image_paths), settings.BATCH_SIZE):
        batch = image_paths[i:i+settings.BATCH_SIZE]
        images = [Image.open(image_path) for image_path in batch]
        text = batch_inference(images, model, processor)
        for image_path, t in zip(batch, text):
            if katex_compatible:
                t = replace_katex_invalid(t)
            write_data.append({"image_path": image_path, "text": t})

    with open(json_path, "w+") as f:
        json_repr = json.dumps(write_data, indent=4)
        f.write(json_repr)


def main():
    parser = argparse.ArgumentParser(description="OCR an image of a LaTeX equation.")
    parser.add_argument("image", type=str, help="Path to image or folder of images to OCR.")
    parser.add_argument("--max", type=int, help="Maximum number of images to OCR if a folder is passes.", default=None)
    parser.add_argument("--json_path", type=str, help="Path to JSON file to save results to.", default=os.path.join(settings.DATA_DIR, "results.json"))
    parser.add_argument("--katex_compatible", action="store_true", help="Make output KaTeX compatible.", default=False)
    args = parser.parse_args()

    image_path = args.image
    model = load_model()
    processor = load_processor()

    json_path = os.path.abspath(args.json_path)
    os.makedirs(os.path.dirname(json_path), exist_ok=True)

    if os.path.isfile(image_path):
        inference_single_image(image_path, json_path, model, processor, args.katex_compatible)
    else:
        inference_image_dir(image_path, json_path, model, processor, args.max, args.katex_compatible)

    print(f"Wrote results to {json_path}")


if __name__ == "__main__":
    main()




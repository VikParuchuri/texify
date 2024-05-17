import argparse
import os.path
import random
import time
from functools import partial

import evaluate
from tabulate import tabulate
from tqdm import tqdm

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from PIL import Image
from texify.settings import settings
import json
import base64
import io
from rapidfuzz.distance import Levenshtein


def normalize_text(text):
    # Replace fences
    text = text.replace("$", "")
    text = text.replace("\[", "")
    text = text.replace("\]", "")
    text = text.replace("\(", "")
    text = text.replace("\)", "")
    text = text.strip()
    return text


def score_text(predictions, references):
    bleu = evaluate.load("bleu")
    bleu_results = bleu.compute(predictions=predictions, references=references)

    meteor = evaluate.load('meteor')
    meteor_results = meteor.compute(predictions=predictions, references=references)

    lev_dist = []
    for p, r in zip(predictions, references):
        lev_dist.append(Levenshtein.normalized_distance(p, r))

    return {
        'bleu': bleu_results["bleu"],
        'meteor': meteor_results['meteor'],
        'edit': sum(lev_dist) / len(lev_dist)
    }


def image_to_pil(image):
    decoded = base64.b64decode(image)
    return Image.open(io.BytesIO(decoded))


def load_images(source_data):
    images = [sd["image"] for sd in source_data]
    images = [image_to_pil(image) for image in images]
    return images


def inference_texify(source_data, model, processor):
    images = load_images(source_data)

    write_data = []
    for i in tqdm(range(0, len(images), settings.BATCH_SIZE), desc="Texify inference"):
        batch = images[i:i+settings.BATCH_SIZE]
        text = batch_inference(batch, model, processor)
        for j, t in enumerate(text):
            eq_idx = i + j
            write_data.append({"text": t, "equation": source_data[eq_idx]["equation"]})

    return write_data


def inference_pix2tex(source_data):
    from pix2tex.cli import LatexOCR
    model = LatexOCR()

    images = load_images(source_data)
    write_data = []
    for i in tqdm(range(len(images)), desc="Pix2tex inference"):
        try:
            text = model(images[i])
        except ValueError:
            # Happens when resize fails
            text = ""
        write_data.append({"text": text, "equation": source_data[i]["equation"]})

    return write_data


def image_to_bmp(image):
    img_out = io.BytesIO()
    image.save(img_out, format="BMP")
    return img_out


def inference_nougat(source_data, batch_size=1):
    import torch
    from nougat.postprocessing import markdown_compatible
    from nougat.utils.checkpoint import get_checkpoint
    from nougat.utils.dataset import ImageDataset
    from nougat.utils.device import move_to_device
    from nougat import NougatModel

    # Load images, then convert to bmp format for nougat
    images = load_images(source_data)
    images = [image_to_bmp(image) for image in images]
    predictions = []

    ckpt = get_checkpoint(None, model_tag="0.1.0-small")
    model = NougatModel.from_pretrained(ckpt)
    if settings.TORCH_DEVICE_MODEL != "cpu":
        move_to_device(model, bf16=settings.CUDA, cuda=settings.CUDA)
    model.eval()

    dataset = ImageDataset(
        images,
        partial(model.encoder.prepare_input, random_padding=False),
    )

    # Batch sizes higher than 1 explode memory usage on CPU/MPS
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        shuffle=False,
    )

    for idx, sample in tqdm(enumerate(dataloader), desc="Nougat inference", total=len(dataloader)):
        model.config.max_length = settings.MAX_TOKENS
        model_output = model.inference(image_tensors=sample, early_stopping=False)
        output = [markdown_compatible(o) for o in model_output["predictions"]]
        predictions.extend(output)
    return predictions


def main():
    parser = argparse.ArgumentParser(description="Benchmark the performance of texify.")
    parser.add_argument("--data_path", type=str, help="Path to JSON file with source images/equations", default=os.path.join(settings.DATA_DIR, "bench_data.json"))
    parser.add_argument("--result_path", type=str, help="Path to JSON file to save results to.", default=os.path.join(settings.DATA_DIR, "bench_results.json"))
    parser.add_argument("--max", type=int, help="Maximum number of images to benchmark.", default=None)
    parser.add_argument("--pix2tex", action="store_true", help="Run pix2tex scoring", default=False)
    parser.add_argument("--nougat", action="store_true", help="Run nougat scoring", default=False)
    args = parser.parse_args()

    source_path = os.path.abspath(args.data_path)
    result_path = os.path.abspath(args.result_path)
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    model = load_model()
    processor = load_processor()

    with open(source_path, "r") as f:
        source_data = json.load(f)

    if args.max:
        random.seed(1)
        source_data = random.sample(source_data, args.max)

    start = time.time()
    predictions = inference_texify(source_data, model, processor)
    times = {"texify": time.time() - start}
    text = [normalize_text(p["text"]) for p in predictions]
    references = [normalize_text(p["equation"]) for p in predictions]

    scores = score_text(text, references)

    write_data = {
        "texify": {
            "scores": scores,
            "text": [{"prediction": p, "reference": r} for p, r in zip(text, references)]
        }
    }

    if args.pix2tex:
        start = time.time()
        predictions = inference_pix2tex(source_data)
        times["pix2tex"] = time.time() - start

        p_text = [normalize_text(p["text"]) for p in predictions]

        p_scores = score_text(p_text, references)

        write_data["pix2tex"] = {
            "scores": p_scores,
            "text": [{"prediction": p, "reference": r} for p, r in zip(p_text, references)]
        }

    if args.nougat:
        start = time.time()
        predictions = inference_nougat(source_data)
        times["nougat"] = time.time() - start
        n_text = [normalize_text(p) for p in predictions]

        n_scores = score_text(n_text, references)

        write_data["nougat"] = {
            "scores": n_scores,
            "text": [{"prediction": p, "reference": r} for p, r in zip(n_text, references)]
        }

    score_table = []
    score_headers = ["bleu", "meteor", "edit"]
    score_dirs = ["⬆", "⬆", "⬇", "⬇"]

    for method in write_data.keys():
        score_table.append([method, *[write_data[method]["scores"][h] for h in score_headers], times[method]])

    score_headers.append("time taken (s)")
    score_headers = [f"{h} {d}" for h, d in zip(score_headers, score_dirs)]
    print()
    print(tabulate(score_table, headers=["Method", *score_headers]))
    print()
    print("Higher is better for BLEU and METEOR, lower is better for edit distance and time taken.")
    print("Note that pix2tex is unbatched (I couldn't find a batch inference method in the docs), so time taken is higher than it should be.")

    with open(result_path, "w") as f:
        json.dump(write_data, f, indent=4)


if __name__ == "__main__":
    main()
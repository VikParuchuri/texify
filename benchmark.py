import argparse
import os.path

import evaluate

from texify.inference import batch_inference
from texify.model.model import load_model
from texify.model.processor import load_processor
from PIL import Image
from texify.settings import settings
import json
import base64
import io
from rapidfuzz.distance import Levenshtein


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


def inference_benchmark_images(source_data, model, processor):
    images = [sd["image"] for sd in source_data]
    images = [image_to_pil(image) for image in images]

    write_data = []
    for i in range(0, len(images), settings.BATCH_SIZE):
        batch = images[i:i+settings.BATCH_SIZE]
        text = batch_inference(batch, model, processor)
        for j, t in enumerate(text):
            eq_idx = i + j
            write_data.append({"text": t, "equation": source_data[eq_idx]["equation"]})

    return write_data


def main():
    parser = argparse.ArgumentParser(description="Benchmark the performance of texify.")
    parser.add_argument("--data_path", type=str, help="Path to JSON file with source images/equations", default=os.path.join(settings.DATA_DIR, "bench_data.json"))
    parser.add_argument("--result_path", type=str, help="Path to JSON file to save results to.", default=os.path.join(settings.DATA_DIR, "bench_results.json"))
    parser.add_argument("--max", type=int, help="Maximum number of images to benchmark.", default=None)
    args = parser.parse_args()

    source_path = os.path.abspath(args.data_path)
    result_path = os.path.abspath(args.result_path)
    model = load_model()
    processor = load_processor()

    with open(source_path, "r") as f:
        source_data = json.load(f)

    if args.max:
        source_data = source_data[:args.max]

    predictions = inference_benchmark_images(source_data, model, processor)
    text = [p["text"] for p in predictions]
    references = [p["equation"] for p in predictions]

    scores = score_text(text, references)

    print(scores)

    write_data = {
        "scores": scores,
        "text": [{"prediction": p, "reference": r} for p, r in zip(text, references)]
    }
    with open(result_path, "w") as f:
        json.dump(write_data, f, indent=4)


if __name__ == "__main__":
    main()




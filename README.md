# Texify

Texify converts equations and surrounding text into Markdown with LaTex math that can be rendered by MathJax or KaTex ($$ and $ are delimiters).  It will work with images or pdfs, and can run on CPU, GPU, or MPS.

There isn't a clear comparison to Texify, but the closest are Latex-OCR and Nougat:
- Compared to [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR), Texify is more accurate on a wider range of documents.
- Compared to [Nougat](https://github.com/facebookresearch/nougat), Texify is optimized for equations and small page regions.  It is more accurate and hallucinates less for this use case.

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

# Installation

This has been tested on Mac and Linux (Ubuntu and Debian).  You'll need python 3.10+ and [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

- `git clone https://github.com/VikParuchuri/texify.git`
- `cd texify`
- `poetry install --without dev` # This skips the dev dependencies
- Set your `TORCH_DEVICE` according to your system (see below).

The first time you run it, model weights will be automatically downloaded.

# Usage

First, some configuration:

- Set your torch device in the `local.env` file.  For example, `TORCH_DEVICE=cuda` or `TORCH_DEVICE=mps`.  `cpu` is the default.
- Inspect the other settings in `texify/settings.py`.  You can override any settings in the `local.env` file, or by setting environment variables.

## App for interactive conversion

I've included a streamlit app that can be used to interactively select and convert equations from images or PDF files.  To run it, do this:

```
streamlit run ocr_app.py
```

The app will allow you to select the specific equations you want to convert on each page, then render the results with Katex and enable easy copying.

## Convert an image or directory of images

Run `ocr_image.py`, like this:

```
python ocr_image.py /path/to/folder_or_file --max 8 --json_path results.json
```

- `--max` is how many images in the folder to convert at most.  Omit this to convert all images in the folder.
- `--json_path` is an optional path to a json file where the results will be saved.  If you omit this, the results will be saved to `data/results.json`.

# Limitations

OCR is complicated, and texify is not perfect.  Here are some known limitations:

- It will OCR equations and surrounding text, but is not good for general purpose OCR.  Think sections of a page instead of a whole page.
- Texify was mostly trained with 96 DPI images, and only at a max 420x420 resolution.  Very wide or very tall images may not work well.
- It's English-only for now.
- The output format will be markdown with embedded latex for equations (close to Github flavored markdown).  It will not be pure latex.

# Benchmarks

Benchmarking OCR quality is hard - you ideally need a parallel corpus that models haven't been trained on.  I've sampled some images from across a range of sources (web, arxiv, im2latex) to create a representative benchmark set.  Both Latex-OCR and Texify were trained on parts of this data, so it isn't a perfect benchmark.  However, it should simulate real-world usage.

## Running your own benchmarks

You can benchmark the performance of texify on your machine.  

- Clone the repo if you haven't already (see above for manual installation instructions)
- Install dev dependencies with `poetry install`
  - If you want to use pix2tex, run `pip install pix2tex`
  - If you want to use nougat, run `pip install nougat-ocr`
- Download the benchmark data [here]() and put it in the `data` folder.
- Run `benchmark.py` like this:

```
python benchmark.py --max 100 --pix2tex --data_path data/bench_data.json --result_path data/bench_results.json
```

This will benchmark marker against Latex-OCR.  It will do batch inference with texify, but not with Latex-OCR, since I couldn't find an option for batching.

- `--max` is how many benchmark images to convert at most.
- `--data_path` is the path to the benchmark data.  If you omit this, it will use the default path.
- `--result_path` is the path to the benchmark results.  If you omit this, it will use the default path.
- `--pix2tex` specifies whether to run pix2tex (Latex-OCR) or not.  If you omit this, it will only run texify.

# Training

Texify was trained on latex images and paired equations from across the web.  It includes the [im2latex](https://github.com/guillaumegenthial/im2latex) dataset.  Training happened on 4x A6000 GPUs for 3 days.

# Commercial usage

This model is trained on top of the openly licensed [Donut](https://huggingface.co/naver-clova-ix/donut-base) model, and thus can be used for commercial purposes.

# Thanks

This work would not have been possible without lots of amazing open source work.  I particularly want to acknowledge Lukas Blecher, whose work on Nougat and Latex-OCR was key for this project.  I learned a lot from his code, and used parts of it for texify.

- [im2latex](https://github.com/guillaumegenthial/im2latex) - one of the datasets used for training
- [Donut](https://huggingface.co/naver-clova-ix/donut-base) from Naver, the base model for texify
- [Nougat](https://github.com/facebookresearch/nougat) - I used the tokenized from Nougat
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR) - The original open source Latex OCR project
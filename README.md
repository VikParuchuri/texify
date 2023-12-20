# Texify

Texify converts equations and surrounding text into Markdown with Latex math blocks ($$ and $ are delimiters).  It will work with images or pdfs, and can run on CPU, GPU, or MPS.

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

# Installation

This has been tested on Mac and Linux (Ubuntu and Debian).  You'll need python 3.10+ and [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

- `git clone https://github.com/VikParuchuri/texify.git`
- `cd texify`
- `poetry install`
- Set your `TORCH_DEVICE` according to your system (see below).

The first time you run it, model weights will be automatically downloaded.

# Usage

First, some configuration:

- Set your torch device in the `local.env` file.  For example, `TORCH_DEVICE=cuda` or `TORCH_DEVICE=mps`.  `cpu` is the default.
- Inspect the other settings in `texify/settings.py`.  You can override any settings in the `local.env` file, or by setting environment variables.

## Convert an image or directory of images

Run `ocr_image.py`, like this:

```
python ocr_image.py /path/to/folder_or_file --max 8 --json_path results.json
```

- `--max` is how many images in the folder to convert at most.  Omit this to convert all images in the folder.
- `--json_path` is an optional path to a json file where the results will be saved.  If you omit this, the results will be saved to `data/results.json`.

## App for converting PDF regions

I've included a streamlit app that can be used to select equations from PDF files.  To run it, do this:

```
streamlit run ocr_app.py
```

The app will allow you to select the specific equations you want to convert on each page, then render the results with Katex and enable easy copying.

# Limitations

OCR is complicated, and texify is not perfect.  Here are some known limitations:

- It will OCR equations and surrounding text, but is not good for general purpose OCR.  Think sections of a page instead of a whole page.
- Texify was mostly trained with 96 DPI images, and only at a max 420x420 resolution.  Make sure you don't feed it images that are too large.
- It's English-only for now.
- The output format will be markdown with embedded latex for equations (close to Github flavored markdown).  It will not be pure latex.

# Training

Texify was trained on latex images and paired equations from across the web.  It includes the [im2latex](https://github.com/guillaumegenthial/im2latex) dataset.  Training happened on 4x A6000 GPUs for 3 days.

# Commercial usage

This model is trained on top of the openly licensed [Donut](https://huggingface.co/naver-clova-ix/donut-base) model, and thus can be used for commercial purposes.

# Thanks

This work would not have been possible without lots of amazing open source work.  I particularly want to acknowledge Lukas Blecher, whose work on Nougat and Latex-OCR was key for this project.

- [im2latex](https://github.com/guillaumegenthial/im2latex)
- [Donut](https://huggingface.co/naver-clova-ix/donut-base) from Naver
- [Nougat](https://github.com/facebookresearch/nougat)
- [Latex-OCR](https://github.com/lukas-blecher/LaTeX-OCR)
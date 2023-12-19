# Texify

Texify converts equations and surrounding text into markdown with embedded Latex blocks that can be parsed by MathJax.  You can use a web app to select your equations from pdf files, or run it on a folder of images.

# Community

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

# Limitations

OCR is complicated, and texify is not perfect.  Here are some known limitations:

- It will OCR equations and surrounding text, but is not good for general purpose OCR.  Think sections of a page instead of a whole page.
- It is English-only for now.
- The output format will be markdown with embedded mathjax for equations.  It will not be pure latex.

# Installation

This has been tested on Mac and Linux (Ubuntu and Debian).  You'll need python 3.10+ and [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

- `git clone https://github.com/VikParuchuri/texify.git`
- `cd texify`
- `poetry install`

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

This app will allow you to select the specific equations you want to convert to Latex on each page, and copy out the results.

# Commercial usage

This model is trained on top of the openly licensed [Donut](https://huggingface.co/naver-clova-ix/donut-base) model, and thus can be used for commercial purposes.

# Thanks

This work would not have been possible without lots of amazing open source work.  I particularly want to acknowledge Nougat and Latex-OCR by Lukas Blecher, which were the inspiration for this project.
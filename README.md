# Texify

Texify converts equations and surrounding text into markdown with LaTeX math that can be rendered by MathJax ($$ and $ are delimiters).  It will work with images or pdfs, and can run on CPU, GPU, or MPS.

https://github.com/VikParuchuri/texify/assets/913340/39b1f139-872f-4ae8-9f31-39e396953bd9

> **Example**
> 
> ![image](data/examples/0.png)
> 
> **Detected Text** The potential $V_{i}$ of cell $\mathcal{C}_{j}$ centred at position $\mathbf{r}_{i}$ is related to the surface charge densities $\sigma_{j}$ of cells $\mathcal{E}_{j}$ $j\in[1,N]$ through the superposition principle as:
> 
> $$V_{i}\,=\,\sum_{j=0}^{N}\,\frac{\sigma_{j}}{4\pi\varepsilon_{0}}\,\int_{\mathcal{E}_{j}}\frac{1}{\left|\mathbf{r}_{i}-\mathbf{r}^{\prime}\right|}\,\mathrm{d}^{2}\mathbf{r}^{\prime}\,=\,\sum_{j=0}^{N}\,Q_{ij}\,\sigma_{j},$$
> 
> where the integral over the surface of cell $\mathcal{C}_{j}$ only depends on $ \mathcal{C}_{j} $ shape and on the relative position of the target point $\mathbf{r}_{i}$ with respect to $\mathcal{C}_{j}$ location, as $\sigma_{j}$ is assumed constant over the whole surface of cell $\mathcal{C}_{j}$.

The closest open source comparisons to texify are pix2tex and nougat, although they're designed for different purposes:

- Compared to [pix2tex](https://github.com/lukas-blecher/LaTeX-OCR), texify can detect text and inline equations. Pix2tex is designed for block LaTeX equations, and hallucinates more on text.
- Compared to [nougat](https://github.com/facebookresearch/nougat), texify is optimized for equations and small page regions.  Nougat is designed to OCR entire pages, and hallucinates more on small images.

I created texify to render equations in [marker](https://github.com/VikParuchuri/marker), but realized it could also be valuable on its own.

See more details in the [benchmarks](#benchmarks) section.

## Community

[Discord](https://discord.gg//KuZwXNGnfH) is where we discuss future development.

# Installation

This has been tested on Mac and Linux (Ubuntu and Debian).  You'll need python 3.10+ and [poetry](https://python-poetry.org/docs/#installing-with-the-official-installer).

- `git clone https://github.com/VikParuchuri/texify.git`
- `cd texify`
- `poetry install --without dev` # This skips the dev dependencies

Model weights will automatically download the first time you run it.

# Usage

First, some configuration:

- Inspect the settings in `texify/settings.py`.  You can override any settings in a `local.env` file, or by setting environment variables.
- Your torch device will be automatically detected, but you can override this.  For example, `TORCH_DEVICE=cuda` or `TORCH_DEVICE=mps`.

## App for interactive conversion

I've included a streamlit app that lets you interactively select and convert equations from images or PDF files.  To run it, do this:

```
streamlit run ocr_app.py
```

The app will allow you to select the specific equations you want to convert on each page, then render the results with KaTeX and enable easy copying.

## Convert an image or directory of images

Run `ocr_image.py`, like this:

```
python ocr_image.py /path/to/folder_or_file --max 8 --json_path results.json
```

- `--max` is how many images in the folder to convert at most.  Omit this to convert all images in the folder.
- `--json_path` is an optional path to a json file where the results will be saved.  If you omit this, the results will be saved to `data/results.json`.

# Limitations

OCR is complicated, and texify is not perfect.  Here are some known limitations:

- Texify will OCR equations and surrounding text, but is not good for general purpose OCR.  Think sections of a page instead of a whole page.
- Texify was mostly trained with 96 DPI images, and only at a max 420x420 resolution.  Very wide or very tall images may not work well.
- It works best with English, although it should support other languages with similar character sets.
- The output format will be markdown with embedded LaTeX for equations (close to Github flavored markdown).  It will not be pure LaTeX.

# Benchmarks

Benchmarking OCR quality is hard - you ideally need a parallel corpus that models haven't been trained on.  I've sampled some images from across a range of sources (web, arxiv, im2latex) to create a representative benchmark set.

Of these, here is what is known about the training data:

- Nougat was trained on arxiv.
- Pix2tex was trained on im2latex and web images.
- Texify was trained on im2latex and web images.

## Running your own benchmarks

You can benchmark the performance of texify on your machine.  

- Clone the repo if you haven't already (see above for manual installation instructions)
- Install dev dependencies with `poetry install`
  - If you want to use pix2tex, run `pip install pix2tex`
  - If you want to use nougat, run `pip install nougat-ocr`
- Download the benchmark data [here]() and put it in the `data` folder.
- Run `benchmark.py` like this:

```
python benchmark.py --max 100 --pix2tex --nougat --data_path data/bench_data.json --result_path data/bench_results.json
```

This will benchmark marker against Latex-OCR.  It will do batch inference with texify, but not with Latex-OCR, since I couldn't find an option for batching.

- `--max` is how many benchmark images to convert at most.
- `--data_path` is the path to the benchmark data.  If you omit this, it will use the default path.
- `--result_path` is the path to the benchmark results.  If you omit this, it will use the default path.
- `--pix2tex` specifies whether to run pix2tex (Latex-OCR) or not.
- `--nougat` specifies whether to run nougat or not.

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
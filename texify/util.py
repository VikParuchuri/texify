import os
from PIL import Image


def is_valid_image(file_path):
    if not os.path.isfile(file_path):
        return False

    filename = os.path.basename(file_path)
    if filename.startswith("."):
        return False

    try:
        with Image.open(file_path) as img:
            img.verify()
        return True
    except Exception:
        return False
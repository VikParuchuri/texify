import re
from ftfy import fix_text


def replace_katex_invalid(string):
    return re.sub(r'\\tag\{.*?\}', '', string)


def postprocess(text):
    text = fix_text(text)
    return text
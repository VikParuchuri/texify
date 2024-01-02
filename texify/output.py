import re
from ftfy import fix_text


def remove_labels(text):
    pattern = r'\\label\{[^}]*\}'
    text = re.sub(pattern, '', text)

    ref_pattern = r'\\ref\{[^}]*\}'
    text = re.sub(ref_pattern, '', text)

    pageref_pattern = r'\\pageref\{[^}]*\}'
    text = re.sub(pageref_pattern, '', text)
    return text


def remove_inner_dollars(text):
    def replace_dollar(match):
        # Replace single $ with nothing, keep $$ intact
        math_block = match.group(1)
        return '$$' + math_block.replace('$', '') + '$$'

    # Regex to find $$...$$ blocks, including new lines
    pattern = r'\$\$(.*?)\$\$'

    return re.sub(pattern, replace_dollar, text, flags=re.DOTALL)


def replace_katex_invalid(string):
    # KaTeX cannot render all LaTeX, so we need to replace some things
    string = re.sub(r'\\tag\{.*?\}', '', string)
    string = re.sub(r'\\(?:Bigg?|bigg?)\{(.*?)\}', r'\1', string)
    string = re.sub(r'\\quad\\mbox\{(.*?)\}', r'\1', string)
    string = re.sub(r'\\mbox\{(.*?)\}', r'\1', string)
    string = remove_inner_dollars(string)
    return string


def postprocess(text):
    text = fix_text(text)
    # Remove latex labels and references
    text = remove_labels(text)
    return text
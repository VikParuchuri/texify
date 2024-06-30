import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1" # For some reason, transformers decided to use .isin for a simple op, which is not supported on MPS

from texify.settings import settings
from texify.output import postprocess


def batch_inference(images, model, processor, temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS):
    images = [image.convert("RGB") for image in images]
    encodings = processor(images=images, return_tensors="pt", add_special_tokens=False)
    pixel_values = encodings["pixel_values"].to(model.dtype)
    pixel_values = pixel_values.to(model.device)

    additional_kwargs = {}
    if temperature > 0:
        additional_kwargs["temperature"] = temperature
        additional_kwargs["do_sample"] = True
        additional_kwargs["top_p"] = 0.95

    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=max_tokens,
        decoder_start_token_id=processor.tokenizer.bos_token_id,
        **additional_kwargs,
    )

    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = [postprocess(text) for text in generated_text]
    return generated_text
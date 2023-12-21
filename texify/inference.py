from texify.settings import settings
from texify.output import postprocess


def batch_inference(images, model, processor):
    images = [image.convert("RGB") for image in images]
    encodings = processor(images=images, return_tensors="pt", add_special_tokens=False)
    pixel_values = encodings["pixel_values"].to(settings.MODEL_DTYPE)
    pixel_values = pixel_values.to(settings.TORCH_DEVICE_MODEL)

    generated_ids = model.generate(
        pixel_values=pixel_values,
        max_new_tokens=settings.MAX_TOKENS,
        decoder_start_token_id=processor.tokenizer.bos_token_id
    )

    generated_text = processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    generated_text = [postprocess(text) for text in generated_text]
    return generated_text
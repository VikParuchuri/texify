from transformers import DonutSwinConfig, VisionEncoderDecoderConfig


def get_config(model_checkpoint):
    config = VisionEncoderDecoderConfig.from_pretrained(model_checkpoint)
    encoder_config = vars(config.encoder)
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder
    return config


class VariableDonutSwinConfig(DonutSwinConfig):
    pass
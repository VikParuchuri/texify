from transformers import VisionEncoderDecoderConfig, PretrainedConfig, MBartConfig


def get_config(model_checkpoint):
    config = TexifyConfig.from_pretrained(model_checkpoint)
    encoder_config = config.encoder
    encoder = VariableDonutSwinConfig(**encoder_config)
    config.encoder = encoder

    decoder_config = config.decoder
    decoder = MBartConfig(**decoder_config)
    config.decoder = decoder
    return config


class TexifyConfig(VisionEncoderDecoderConfig):
    model_type = "vision-encoder-decoder"
    is_composition = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        encoder_config = kwargs.pop("encoder")
        decoder_config = kwargs.pop("decoder")

        self.encoder = encoder_config
        self.decoder = decoder_config
        self.is_encoder_decoder = True


class VariableDonutSwinConfig(PretrainedConfig):
    model_type = "donut-swin"

    attribute_map = {
        "num_attention_heads": "num_heads",
        "num_hidden_layers": "num_layers",
    }

    def __init__(
        self,
        image_size=224,
        patch_size=4,
        num_channels=3,
        embed_dim=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4.0,
        qkv_bias=True,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        drop_path_rate=0.1,
        hidden_act="gelu",
        use_absolute_embeddings=False,
        use_2d_embeddings=False,
        initializer_range=0.02,
        layer_norm_eps=1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.patch_size = patch_size
        self.num_channels = num_channels
        self.embed_dim = embed_dim
        self.depths = depths
        self.num_layers = len(depths)
        self.num_heads = num_heads
        self.window_size = window_size
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.drop_path_rate = drop_path_rate
        self.hidden_act = hidden_act
        self.use_absolute_embeddings = use_absolute_embeddings
        self.use_2d_embeddings = use_2d_embeddings
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        # we set the hidden_size attribute in order to make Swin work with VisionEncoderDecoderModel
        # this indicates the channel dimension after the last stage of the model
        self.hidden_size = int(embed_dim * 2 ** (len(depths) - 1))
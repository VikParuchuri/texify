from typing import Dict, List

from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: str = None
    MAX_TOKENS: int = 384
    MAX_IMAGE_SIZE: Dict = {"height": 420, "width": 420}
    MODEL_CHECKPOINT: str = "vikp/texify"
    BATCH_SIZE: int = 16 # Should use ~5GB of RAM
    DATA_DIR: str = "data"

    @computed_field
    @property
    def TORCH_DEVICE_MODEL(self) -> str:
        if self.TORCH_DEVICE is not None:
            return self.TORCH_DEVICE

        if torch.cuda.is_available():
            return "cuda"

        if torch.backends.mps.is_available():
            return "mps"

        return "cpu"

    @computed_field
    @property
    def CUDA(self) -> bool:
        return "cuda" in self.TORCH_DEVICE_MODEL

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.float32 if self.TORCH_DEVICE_MODEL == "cpu" else torch.float16


    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
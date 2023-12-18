from dotenv import find_dotenv
from pydantic import computed_field
from pydantic_settings import BaseSettings
import torch


class Settings(BaseSettings):
    # General
    TORCH_DEVICE: str = "cpu"
    MAX_TOKENS: int = 512
    MODEL_CHECKPOINT: str = "vikp/texify"

    @computed_field
    @property
    def CUDA(self) -> bool:
        return "cuda" in self.TORCH_DEVICE

    @computed_field
    @property
    def MODEL_DTYPE(self) -> torch.dtype:
        return torch.bfloat16 if self.CUDA else torch.float32

    class Config:
        env_file = find_dotenv("local.env")
        extra = "ignore"


settings = Settings()
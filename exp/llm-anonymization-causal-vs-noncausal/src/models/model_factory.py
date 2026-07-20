from src.configs import ModelConfig

from .model import BaseModel
from .open_ai import OpenAIGPT
from .hf_model import HFModel
from .ner_baseline import NERModel
from .ollama import OllamaModel
from .together import TogetherModel
from .anthropic import AnthropicModel


def get_model(config: ModelConfig) -> BaseModel:
    if config.provider == "openai" or config.provider == "azure":
        return OpenAIGPT(config)
    elif config.provider == "hf":
        return HFModel(config)
    elif config.provider == "ner":
        return NERModel(config)
    elif config.provider == "ollama":
        return OllamaModel(config)
    elif config.provider == "together":
        return TogetherModel(config)
    elif config.provider == "anthropic":
        return AnthropicModel(config)
    elif config.provider == "loc":
        raise NotImplementedError

    else:
        raise NotImplementedError

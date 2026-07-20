from .azure_anonymizer import AzureAnonymizer
from .llm_anonymizers import LLMFullAnonymizer, LLMBaselineAnonymizer
from .anonymizer import Anonymizer

from src.configs import AnonymizationConfig
from src.models.model_factory import get_model

def get_anonymizer(cfg: AnonymizationConfig) -> Anonymizer:
    
    if cfg.anonymizer.anon_type == "azure":
        return AzureAnonymizer(cfg.anonymizer)
    elif cfg.anonymizer.anon_type == "llm":
        model = get_model(cfg.anon_model)
        return LLMFullAnonymizer(cfg.anonymizer, model)
    elif cfg.anonymizer.anon_type == "llm_base":
        model = get_model(cfg.anon_model)
        return LLMBaselineAnonymizer(cfg.anonymizer, model)
    else:
        raise ValueError(f"Unknown anonymizer type {cfg.anonymizer.anon_type}")
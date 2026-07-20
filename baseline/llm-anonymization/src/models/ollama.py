from typing import Dict, List, Iterator, Tuple
import os

from ollama import generate as ol_generate
import ollama
from src.configs import ModelConfig
from src.prompts import Prompt
from .model import BaseModel


class OllamaModel(BaseModel):
    model_to_port = {}

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        # Run shell script to pull the model

        curr_models = ollama.list()

        names = [model["name"] for model in curr_models["models"]]

        if config.name not in names:
            ollama.pull(config.name)

    def predict(self, input: Prompt, **kwargs):
        text = input.get_prompt().rstrip()

        if input.system_prompt is None:
            input.system_prompt = "You are an expert investigator and detective with years of experience in online profiling and text analysis."

        response = ol_generate(
            text,
            model=self.config.name,
            system=input.system_prompt,
        )

        return response

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        for input in inputs:
            text = input.get_prompt().rstrip()

            if input.system_prompt is None:
                input.system_prompt = "You are an expert investigator and detective with years of experience in online profiling and text analysis."

            response = ol_generate(
                prompt=text,
                model=self.config.name,
                system=input.system_prompt,
                stream=False,
            )

            yield input, response["response"]

    def predict_string(self, input: str, **kwargs) -> str:
        return self.predict(Prompt(intermediate=input), **kwargs)

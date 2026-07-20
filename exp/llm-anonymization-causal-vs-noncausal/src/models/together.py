from typing import List, Dict, Tuple, Iterator

import os
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from together import Together

from src.configs import ModelConfig
from src.prompts import Prompt, Conversation

from .model import BaseModel


class TogetherModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

        if "temperature" not in self.config.args.keys():
            self.config.args["temperature"] = 0.0
        if "max_tokens" not in self.config.args.keys():
            self.config.args["max_tokens"] = 2000

        self.client = Together(api_key=os.environ.get("TOGETHER_API_KEY"))

    def _predict_call(self, input: List[Dict[str, str]]) -> str:

        # self.config.name = "mistralai/Mixtral-8x22B-Instruct-v0.1"

        response = self.client.chat.completions.create(
            model=self.config.name,
            messages=input,
            temperature=self.config.args["temperature"],
            max_tokens=self.config.args["max_tokens"],
        )
        print(response.choices[0].message.content)

        return response.choices[0].message.content

    def predict(self, input: Prompt, **kwargs) -> str:
        messages: List[Dict[str, str]] = []

        if input.system_prompt is not None:
            messages.append(
                {
                    "role": "system",
                    "content": input.system_prompt,
                }
            )
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are an expert investigator and detective with years of experience in online profiling and text analysis.",
                }
            ]

        messages += [
            {"role": "user", "content": self.apply_model_template(input.get_prompt())}
        ]

        if "llama" in self.config.name:
            # Remove system message for Llama models
            messages = messages[1:]

        guess = self._predict_call(messages)

        return guess

    def predict_string(self, input: str, **kwargs) -> str:
        input_list = [
            {
                "role": "system",
                "content": "You are an helpful assistant.",
            },
            {"role": "user", "content": input},
        ]

        guess = self._predict_call(input_list)

        return guess

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 4
        base_timeout = kwargs["timeout"] if "timeout" in kwargs else 600

        self.predict(inputs[0])

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

            # q = self.predict(inputs[38])

            while len(ids_to_do) > 0 and retry_ctr <= len(inputs):
                # executor.map will apply the function to every item in the iterable (prompts), returning a generator that yields the results
                results = executor.map(
                    lambda id: (id, inputs[id], self.predict(inputs[id])),
                    ids_to_do,
                    timeout=timeout,
                )
                try:
                    for res in tqdm(
                        results,
                        total=len(ids_to_do),
                        desc="Profiles",
                        position=1,
                        leave=False,
                    ):
                        id, orig, answer = res
                        yield (orig, answer)
                        # answered_prompts.append()
                        ids_to_do.remove(id)
                except TimeoutError:
                    print(f"Timeout: {len(ids_to_do)} prompts remaining")
                except Exception as e:
                    print(f"Exception: {e}")
                    time.sleep(10)
                    continue

                if len(ids_to_do) == 0:
                    break

                time.sleep(2 * retry_ctr)
                timeout *= 2
                timeout = min(120, timeout)
                retry_ctr += 1

        # return answered_prompts

from typing import List, Dict, Tuple, Iterator

import openai
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from openai.error import RateLimitError

from azure.ai.textanalytics import TextAnalyticsClient, DocumentError
from azure.core.credentials import AzureKeyCredential
import credentials

from src.configs import ModelConfig
from src.prompts import Prompt, Conversation

from .model import BaseModel


def recognize_entities(comments: List[str], threshhold: float):
    target_entities = [
        "Person",
        "PersonType",
        "Location",
        "Organization",
        "Event",
        "Address",
        "PhoneNumber",
        "Email",
        "URL",
        "IP",
        "DateTime",
        ("Quantity", ["Age", "Currency", "Number"]),
    ]

    text_analytics_client = TextAnalyticsClient(
        endpoint=credentials.azure_language_endpoint,
        credential=AzureKeyCredential(credentials.azure_language_key),
    )
    all_entities = []

    for i in range(0, len(comments), 5):
        end = min(i + 5, len(comments))
        comment_batch = comments[i:end]

        anonymized_results = text_analytics_client.recognize_entities(comment_batch)

        for comment, res in zip(comment_batch, anonymized_results):
            text = comment

            if isinstance(res, DocumentError):
                if res.id == "2":
                    new_entities = []
                    for k in range(0, len(text), 5000):
                        end = min(k + 5000, len(text))
                        res = text_analytics_client.recognize_entities([text[k:end]])[0]
                        new_entities.extend(res.entities)
                    res.entities = new_entities

                else:
                    assert False, f"Error {res.id}: {res.error.message}"

            entities = [
                (entity.text, entity.category, entity.subcategory)
                for entity in res.entities
                if entity.confidence_score > threshhold
            ]

            all_entities.extend(entities)

    return all_entities


class NERModel(BaseModel):
    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self.config = config

    def _get_attribute_answer(
        self, attribute: str, entities: List[Tuple[str, str, str]]
    ):
        relevant_entities = []

        if attribute == "age":
            relevant_entities = [
                entity
                for entity in entities
                if entity[1] == "Quantity" and entity[2] == "Age"
            ]
        elif attribute == "location":
            ent_list = ["Location", "Address"]
            relevant_entities = [entity for entity in entities if entity[1] in ent_list]
        elif attribute == "gender":
            relevant_entities = [
                entity for entity in entities if entity[1] == "PersonType"
            ]
        # elif attribute == "occupation":
        #     relevant_entities = [
        #         entity for entity in entities if entity[1] == "Organization"
        #     ]
        elif attribute == "income":
            relevant_entities = [
                entity
                for entity in entities
                if entity[1] == "Quantity" and entity[2] == "Currency"
            ]

        if len(relevant_entities) == 0:
            return None, ""

        # Put entities into dict with count (based on their number of occurrences in the text)
        entity_dict = {}
        for entity in relevant_entities:
            if entity[0] in entity_dict:
                entity_dict[entity[0]] += 1
            else:
                entity_dict[entity[0]] = 1

        # Sort entities by their occurrence in the text
        relevant_entities = sorted(
            entity_dict.items(), key=lambda x: x[1], reverse=True
        )

        # Return the first three entities
        num_return = min(3, len(relevant_entities))
        return relevant_entities, ";".join(
            [entity[0] for entity in relevant_entities[:num_return]]
        )

    def predict(self, input: Prompt, **kwargs) -> str:
        original_text = [c.text for c in input.original_point.comments]

        actual_messages = original_text
        target_attributes = [input.gt] if isinstance(input.gt, str) else input.gt

        # Use azure language understanding to extract the attributes from the intermediate text
        entities = recognize_entities(actual_messages, 0.4)

        # Generate answer from the extracted attributes

        answer = ""

        for attribute in target_attributes:
            top_entities, entities_str = self._get_attribute_answer(attribute, entities)

            answer += f"Type: {attribute}\n"
            answer += f"Inference: {top_entities}\n"
            answer += f"Guess: {entities_str}\n"
            answer += "\n"

        return answer

    def predict_string(self, input: str, **kwargs) -> str:
        # TODO

        input_prompt = Prompt(
            system_prompt="",
            header="",
            intermediate=input,
            footer="",
            target=[],
            original_point=None,
            gt=[],
            answer="",
            shots=[],
            id="",
        )

        return self.predict(input_prompt)

    def predict_multi(
        self, inputs: List[Prompt], **kwargs
    ) -> Iterator[Tuple[Prompt, str]]:
        max_workers = kwargs["max_workers"] if "max_workers" in kwargs else 4
        base_timeout = kwargs["timeout"] if "timeout" in kwargs else 120

        for input in inputs:
            yield (input, self.predict(input))
        return

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            ids_to_do = list(range(len(inputs)))
            retry_ctr = 0
            timeout = base_timeout

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
                except RateLimitError as r:
                    print(f"Rate_limit {r}")
                    time.sleep(30)
                    continue
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

    def continue_conversation(self, input: Conversation, **kwargs) -> str:
        input_list: List[Dict[str, str]] = [
            {
                "role": "system",
                "content": input.system_prompt,
            }
        ]

        for message in input.prompts:
            assert message.role is not None
            input_list.append(
                {
                    "role": message.role,
                    "content": message.get_prompt(),  # Simply returns the intermediate text
                }
            )

        guess = None
        while guess is None:
            try:
                guess = self._predict_call(input_list)
            except RateLimitError as r:
                time.sleep(30)
                continue

        return guess

import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class LLMModel(ABC):

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system: str = "",
    ) -> str:
        pass


class DeepSeekLLM(LLMModel):

    def __init__(self):
        from huggingface_hub import (
            InferenceClient,
        )

        self._client = InferenceClient(
            api_key=os.getenv(
                "HUGGINGFACE_API_KEY"
            )
        )
        self._model = (
            "deepseek-ai/DeepSeek-V3-0324"
        )

    def generate(
        self,
        prompt: str,
        system: str = "",
    ) -> str:
        messages = []
        if system:
            messages.append({
                "role": "system",
                "content": system,
            })
        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = (
            self._client.chat_completion(
                model=self._model,
                messages=messages,
                max_tokens=2048,
            )
        )
        text = response.choices[0].message.content
        import re
        text = re.sub(
            r"<think>.*?</think>\s*",
            "", text, flags=re.DOTALL,
        )
        return text.strip()


class OpenAILLM(LLMModel):

    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self._model = "gpt-4o-mini"

    def generate(
        self,
        prompt: str,
        system: str = "",
    ) -> str:
        messages = []
        if system:
            messages.append({
                "role": "system",
                "content": system,
            })
        messages.append({
            "role": "user",
            "content": prompt,
        })

        response = (
            self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=2048,
            )
        )
        return response.choices[0].message.content


class GeminiLLM(LLMModel):

    def __init__(self):
        import google.generativeai as genai

        genai.configure(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self._model = genai.GenerativeModel(
            "gemini-2.5-flash"
        )

    def generate(
        self,
        prompt: str,
        system: str = "",
    ) -> str:
        full_prompt = prompt
        if system:
            full_prompt = (
                f"{system}\n\n{prompt}"
            )

        response = self._model.generate_content(
            full_prompt
        )
        return response.text


LLM_MODELS = {
    "deepseek": DeepSeekLLM,
    "openai": OpenAILLM,
    "gemini": GeminiLLM,
}


def get_llm(
    name: str = "deepseek",
) -> LLMModel:
    model_class = LLM_MODELS.get(name)
    if not model_class:
        raise ValueError(
            f"Unknown LLM: {name}. "
            f"Options: "
            f"{list(LLM_MODELS.keys())}"
        )
    return model_class()

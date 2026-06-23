import os
from abc import ABC, abstractmethod

from dotenv import load_dotenv

load_dotenv()


class EmbeddingModel(ABC):

    @abstractmethod
    def embed(
        self, texts: list[str]
    ) -> list[list[float]]:
        pass

    @property
    @abstractmethod
    def dimension(self) -> int:
        pass


class GeminiEmbedding(EmbeddingModel):

    def __init__(self):
        import google.generativeai as genai

        genai.configure(
            api_key=os.getenv("GOOGLE_API_KEY")
        )
        self._genai = genai
        self._model = "models/text-embedding-004"

    def embed(
        self, texts: list[str]
    ) -> list[list[float]]:
        results = []
        for text in texts:
            response = (
                self._genai.embed_content(
                    model=self._model,
                    content=text,
                )
            )
            results.append(
                response["embedding"]
            )
        return results

    @property
    def dimension(self) -> int:
        return 768


class OpenAIEmbedding(EmbeddingModel):

    def __init__(self):
        from openai import OpenAI

        self._client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY")
        )
        self._model = "text-embedding-3-small"

    def embed(
        self, texts: list[str]
    ) -> list[list[float]]:
        response = self._client.embeddings.create(
            model=self._model,
            input=texts,
        )
        return [
            item.embedding
            for item in response.data
        ]

    @property
    def dimension(self) -> int:
        return 1536


class SentenceTransformerEmbedding(EmbeddingModel):

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        from sentence_transformers import (
            SentenceTransformer,
        )

        self._model = SentenceTransformer(
            model_name
        )
        self._dimension = (
            self._model
            .get_embedding_dimension()
        )

    def embed(
        self, texts: list[str]
    ) -> list[list[float]]:
        embeddings = self._model.encode(
            texts, show_progress_bar=False
        )
        return embeddings.tolist()

    @property
    def dimension(self) -> int:
        return self._dimension


class DeepSeekEmbedding(EmbeddingModel):

    def __init__(
        self,
        model_name: str = (
            "BAAI/bge-small-en-v1.5"
        ),
    ):
        from huggingface_hub import (
            InferenceClient,
        )

        self._client = InferenceClient(
            api_key=os.getenv(
                "HUGGINGFACE_API_KEY"
            )
        )
        self._model = model_name

    def embed(
        self, texts: list[str]
    ) -> list[list[float]]:
        results = []
        for text in texts:
            response = (
                self._client.feature_extraction(
                    text, model=self._model
                )
            )
            if isinstance(response[0], list):
                # Mean pooling over token embeddings
                import numpy as np
                arr = np.array(response)
                pooled = arr.mean(axis=1)[0]
                results.append(pooled.tolist())
            else:
                results.append(list(response))
        return results

    @property
    def dimension(self) -> int:
        return 384


EMBEDDING_MODELS = {
    "gemini": GeminiEmbedding,
    "openai": OpenAIEmbedding,
    "sentence-transformer": (
        SentenceTransformerEmbedding
    ),
    "deepseek": DeepSeekEmbedding,
}


def get_embedding_model(
    name: str = "sentence-transformer",
) -> EmbeddingModel:
    model_class = EMBEDDING_MODELS.get(name)
    if not model_class:
        raise ValueError(
            f"Unknown embedding model: {name}. "
            f"Options: "
            f"{list(EMBEDDING_MODELS.keys())}"
        )
    return model_class()

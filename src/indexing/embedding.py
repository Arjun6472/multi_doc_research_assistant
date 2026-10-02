from __future__ import annotations

import hashlib
import logging
import math
from abc import ABC, abstractmethod

from ..models import Settings
from .sparse import tokenize

logger = logging.getLogger(__name__)


class Embedder(ABC):
    @property
    @abstractmethod
    def dim(self) -> int:
        raise NotImplementedError

    @abstractmethod
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class HashingEmbedder(Embedder):
    def __init__(self, dim: int = 256) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be greater than zero: {dim}")
        self._dim = dim

    @property
    def dim(self) -> int:
        return self._dim

    def embed_one(self, text: str) -> list[float]:
        vec = [0.0] * self._dim
        for token in tokenize(text):
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:8], "big") % self._dim
            sign = 1.0 if digest[8] & 1 else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(value * value for value in vec))
        if norm == 0.0:
            return vec
        return [value / norm for value in vec]

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        return [self.embed_one(text) for text in texts]


class SentenceTransformerEmbedder(Embedder):
    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info("loading sentence-transformers: %s", self._model_name)
            self._model = SentenceTransformer(self._model_name)
        return self._model

    @property
    def dim(self) -> int:
        model = self._ensure_model()
        return int(model.get_sentence_embedding_dimension())

    def embed_text(self, texts: list[str]) -> list[list[float]]:
        model = self._ensure_model()
        vectors = model.encode(
            texts,
            normalize_embedding=True,
            convert_to_numpy=True,
        )
        return [list(map(float, row)) for row in vectors.tolist()]


def get_embedder(settings: Settings, *, fake: bool = False) -> Embedder:
    if fake:
        return HashingEmbedder()
    return SentenceTransformerEmbedder(settings.embedding_model)

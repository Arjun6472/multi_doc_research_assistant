from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from ..models import Settings, RetrievelResult
from .sparse import tokenize

logger = logging.getLogger(__name__)

def _rescored(candidate: RetrievelResult, score: float, rank: int) -> RetrievelResult:
    return candidate.model_copy(update={"score": score, "rank": rank})

class Reranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: list[RetrievelResult],
        top_k: int,
    ) -> list[RetrievelResult]:
        ...

class LexicalOverlapReranker(Reranker):
    def rerank(
            self,
            query: str,
            candidates: list[RetrievelResult],
            top_k: int,
    ) -> list[RetrievelResult]:
        if top_k <= 0 or not candidates:
            return []
        
        query_tokens = set(tokenize(query))
        scored: list[tuple[float, int, RetrievelResult]] = []
        for incoming_index, candidate in enumerate(candidates):
            overlap = float(len(query_tokens & set(tokenize(candidate.text))))
            scored.append((overlap, incoming_index, candidate))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _rescored(candidate, score, rank=final_rank)
            for final_rank, (score, _, candidate) in enumerate(scored[:top_k], start=1)
        ]


class IdentifyReranker(Reranker):
    def rerank(
            self,
            query: str,
            candidates: list[RetrievelResult],
            top_k: int
    ) -> list[RetrievelResult]:
        if top_k <= 0:
            return []
        return[
            candidate.model_copy(update={"rank": final_rank})
            for final_rank, candidate in enumerate(candidates[:top_k], start=1)
        ]

class CrossEncoderReranker(Reranker):

    def __init__(self, model_name: str) -> None:
        self._model_name = model_name
        self._model: object | None = None

    def _ensure_model(self) -> object:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info("Loading cross-encoder reranker: %s", self._model_name)
            self._model = CrossEncoder(self._model_name)
        return self._model

    def rerank(
            self,
            query: str,
            candidates: list[RetrievelResult],
            top_k: int,
    ) -> list[RetrievelResult]:
        if top_k <= 0 or not candidates:
            return []
        model = self._ensure_model()
        pairs = [[query, candidate.text] for candidate in candidates]
        raw = model.predict(pairs)
        scores = [float(value) for value in raw]
        scored = list(zip(scores, range(len(candidates)), candidates, strict=True))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            _rescored(candidate, score, rank=final_rank)
            for final_rank, (score, _, candidate) in enumerate(scored[:top_k], start=1)
        ]

def get_reranker(settings: Settings, *, fake: bool=False) -> Reranker:
    if fake or settings.rerank_kind == "lexical":
        return LexicalOverlapReranker()
    if settings.rerank_kind == "none":
        return IdentifyReranker()
    return CrossEncoderReranker(settings.reranker_model)
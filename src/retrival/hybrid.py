from __future__ import annotations
import logging
from typing import Any
from ..models import Settings
from ..indexing.embedding import Embedder
from ..indexing.sparse import BM25Index
from ..indexing.vector_store import VectorStore
from .dense import DenseRetriever
from .fusion import reciprocal_rank_fusion
from ..models import RetrievelResult
from .rerank import Reranker
from .sparse import SparseRetrival

logger = logging.getLogger(__name__)

RERANK_POOL_MULTIPLIER = 4

def coerce_list(val: object) -> list[str]:
    if isinstance(val, list):
        return [str(item) for item in val]
    return []

def coerce_dict(val: object) -> dict[str, Any]:
    if isinstance(val, dict):
        return {str(k): v for k, v in val.items()}
    return {}


class HybridRetriever:
    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        bm25: BM25Index,
        reranker: Reranker,
        settings: Settings,
    ) -> None:
        self._settings = settings
        self._store = store
        self._reranker = reranker
        self._dense = DenseRetriever(embedder, store)
        self._sparse = SparseRetrival(bm25)

    def retrieve(self, query: str, *, k: int | None = None) -> list[RetrievelResult]:
        final_k = self._settings.final_top_k if k is None else k
        if final_k <= 0 or not query.strip():
            return []

        dense_hits = self._dense.retrieve(query, self._settings.dense_top_k)
        sparse_hits = self._sparse.retrieve(query, self._settings.sparse_top_k)

        dense_ids = [chunk_id for chunk_id, _ in dense_hits]
        sparse_ids = [chunk_id for chunk_id, _ in sparse_hits]

        dense_set = set(dense_ids)
        sparse_set = set(sparse_ids)

        fused = reciprocal_rank_fusion([dense_ids, sparse_ids], k=self._settings.rrf_k)
        if not fused:
            return []

        pool_size = max(final_k, final_k * RERANK_POOL_MULTIPLIER)
        fused_pool = fused[:pool_size]
        pool_ids = [chunk_id for chunk_id, _ in fused_pool]

        payloads = self._store.get_payloads(pool_ids)
        results: list[RetrievelResult] = []
        for provisional_rank, (chunk_id, fused_score) in enumerate(fused_pool, start=1):
            payload = payloads.get(chunk_id)
            if payload is None:
                logger.warning("no payload for fused chunk_id=%s; dropping from results", chunk_id)
                continue
            sources: list[str] = []
            if chunk_id in dense_set:
                sources.append("dense")
            if chunk_id in sparse_set:
                sources.append("sparse")
            results.append(
                RetrievelResult(
                    chunk_id=chunk_id,
                    score=fused_score,
                    rank=provisional_rank,
                    text=str(payload.get("text", "")),
                    rel_path=str(payload.get("rel_path", "")),
                    heading_path=coerce_list(payload.get("heading_path")),
                    metadata=coerce_dict(payload.get("metadata")),
                    sources=sources,
                )
            )

        if self._settings.rerank_kind != "none":
            return self._reranker.rerank(query, results, final_k)

        return [
            result.model_copy(update={"rank": final_rank})
            for final_rank, result in enumerate(results[:final_k], start=1)
        ]
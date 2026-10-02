from __future__ import annotations
import logging
from ..indexing.embedding import Embedder
from ..indexing.vector_store import VectorStore

logger = logging.getLogger(__name__)

class DenseRetriever:
    def __init__(self, embedder: Embedder, store: VectorStore) -> None:
        self._embedder = embedder
        self._store = store
    def retrieve(self, query:str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        vector = self._embedder.embed_text([query])[0]
        results = self._store.search(vector, k)
        logger.debug("dense retrieve: query=%r k=%d -> %d hits", query, k, len(results))
        return results
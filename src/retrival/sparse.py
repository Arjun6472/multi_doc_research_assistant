from __future__ import annotations
import logging
from ..indexing.sparse import BM25Index, tokenize
from ..models import Settings

logger = logging.getLogger(__name__)

class SparseRetrival:

    def __init__(self, bm25: BM25Index) -> None:
        self._bm25 = bm25

    @classmethod
    def from_storage(cls, settings: Settings) -> SparseRetrival:
        bm25 = BM25Index.load(settings.index_dir)
        logger.info(
            "loaded BM25 index from %s with %d chunks",
            settings.index_dir,
            len(bm25.chunk_ids),
        )
        return cls(bm25)

    def retrieve(self, query: str, k: int) -> list[tuple[str, float]]:
        if k <= 0:
            return []
        results = self._bm25.query(query, k)
        logger.debug("sparse retrieve: query=%r k=%d -> %d hits", query, k, len(results))
        return results


SparseRetriever = SparseRetrival
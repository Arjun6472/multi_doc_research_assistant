from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

from ..models import Chunk

logger = logging.getLogger(__name__)
TOKEN_SPLIT_RE = re.compile(r"[^a-z0-9]+")

BM25_FILENAME = "bm25.json"
SCHEMA_VERSION = 1
TOKENIZER_ID = "lower_alnum_v1"

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75


def tokenize(text: str) -> list[str]:
    return [token for token in TOKEN_SPLIT_RE.split(text.lower()) if token]


class BM25Index:
    def __init__(self, *, K1: float = DEFAULT_K1, b: float = DEFAULT_B) -> None:
        self.chunk_ids: list[str] = []
        self.corpus_tokens: list[list[str]] = []
        self.params: dict[str, float] = {"k1": K1, "b": b}
        self._bm25: Any | None = None

    def build(self, chunks: list[Chunk]) -> None:
        from rank_bm25 import BM25Okapi

        self.chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.corpus_tokens = [tokenize(chunk.text) for chunk in chunks]

        if self.corpus_tokens:
            self._bm25 = BM25Okapi(
                self.corpus_tokens,
                k1=self.params["k1"],
                b=self.params["b"],
            )
        else:
            self._bm25 = None
        logger.info("built BM25 index over %d chunks", len(self.chunk_ids))

    def save(self, directory: Path) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / BM25_FILENAME
        payload = {
            "schema_version": SCHEMA_VERSION,
            "tokenizer": TOKENIZER_ID,
            "params": self.params,
            "chunk_ids": self.chunk_ids,
            "corpus_tokens": self.corpus_tokens,
        }

        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True),
            encoding="utf-8",
        )

        return path

    @classmethod
    def load(cls, directory: Path) -> BM25Index:
        path = directory / BM25_FILENAME
        payload = json.loads(path.read_text(encoding="utf-8"))
        version = payload.get("schema_version")
        if version != SCHEMA_VERSION:
            raise ValueError(f"unsupported BM25 schema_version {version!r} (expected {SCHEMA_VERSION})")

        params = payload.get("params", {})
        index = cls(K1=float(params.get("k1", DEFAULT_K1)), b=float(params.get("b", DEFAULT_B)))
        index.chunk_ids = list(payload.get("chunk_ids", []))
        index.corpus_tokens = [list(tokens) for tokens in payload.get("corpus_tokens", [])]
        if index.corpus_tokens:
            from rank_bm25 import BM25Okapi

            index._bm25 = BM25Okapi(
                index.corpus_tokens,
                k1=index.params["k1"],
                b=index.params["b"],
            )
        else:
            index._bm25 = None

        return index

    def query(self, text: str, k: int) -> list[tuple[str, float]]:
        if self._bm25 is None or not self.chunk_ids:
            return []

        scores = self._bm25.get_scores(tokenize(text))
        ranked = sorted(enumerate(scores), key=lambda pair: (-float(pair[1]), pair[0]))

        top = ranked[: max(0, k)]

        return [(self.chunk_ids[index], float(score)) for index, score in top]
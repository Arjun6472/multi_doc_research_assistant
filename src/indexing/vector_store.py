from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Protocol

from ..models import Chunk, Settings

logger = logging.getLogger(__name__)

POINT_ID_NAMESPACE = uuid.UUID("6f9619ff-8b86-d011-b42d-00cf4fc964ff")


def point_id_for(chunk_id: str) -> uuid.UUID:
    return uuid.uuid5(POINT_ID_NAMESPACE, chunk_id)


def ponit_id_for(chunk_id: str) -> uuid.UUID:
    return point_id_for(chunk_id)


def _payload(chunk: Chunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "rel_path": chunk.rel_path,
        "ordinal": chunk.ordinal,
        "heading_path": chunk.heading_path,
        "text": chunk.text,
        "metadata": chunk.metadata,
    }


class VectorStore(Protocol):
    def ensure_collection(self, dim: int) -> None: ...

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def count(self) -> int: ...

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]: ...

    def get_payload(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]: ...


class QdrantVectorStore:
    def __init__(
        self,
        *,
        collection: str,
        url: str | None = None,
        location: str | None = None,
        path: str | None = None,
        api_key: str | None = None,
    ) -> None:
        modes = {"url": url, "location": location, "path": path}
        provided = [name for name, value in modes.items() if value is not None]
        if len(provided) != 1:
            raise ValueError("provide exactly one of 'url', 'location', or 'path'" f" (got: {provided or 'none'})")

        try:
            from qdrant_client import QdrantClient
        except ImportError as exc:  # pragma: no cover - dependency installation guidance
            raise RuntimeError("qdrant-client is required for QdrantVectorStore. Install it with 'pip install qdrant-client'.") from exc

        self.collection = collection
        if location is not None:
            self._client = QdrantClient(location=location)
        elif path is not None:
            self._client = QdrantClient(path=path)
        else:
            self._client = QdrantClient(url=url, api_key=api_key)

    @classmethod
    def in_memory(cls, collection: str) -> QdrantVectorStore:
        return cls(collection=collection, location=":memory:")

    @classmethod
    def from_settings(cls, settings: Settings) -> QdrantVectorStore:
        if settings.qdrant_path:
            Path(settings.qdrant_path).mkdir(parents=True, exist_ok=True)
            return cls(collection=settings.qdrant_collection, path=settings.qdrant_path)
        return cls(
            collection=settings.qdrant_collection,
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
        )

    def ensure_collection(self, dim: int) -> None:
        from qdrant_client import models as qm

        if self._client.collection_exists(self.collection):
            try:
                collection = self._client.get_collection(self.collection)
                config = getattr(collection, "config", None)
                params = getattr(config, "params", None)
                vectors = getattr(params, "vectors", None)
                existing_dim = getattr(vectors, "size", None)
            except Exception:  # pragma: no cover - defensive for client/version differences
                existing_dim = None

            if existing_dim is not None and int(existing_dim) == dim:
                logger.info("reusing existing collection %s (dim=%d)", self.collection, dim)
                return

            if existing_dim is not None:
                logger.warning(
                    "collection %s already exists with dim=%s; requested dim=%d; recreating the storage",
                    self.collection,
                    existing_dim,
                    dim,
                )
            else:
                logger.info("recreating collection %s (clean rebuild, dim=%d)", self.collection, dim)
            self._client.delete_collection(self.collection)

        self._client.create_collection(
            collection_name=self.collection,
            vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE),
        )

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError(f"chunks/vectors length mismatch: {len(chunks)} != {len(vectors)}")
        if not chunks:
            return

        from qdrant_client import models as qm

        points = [
            qm.PointStruct(
                id=point_id_for(chunk.chunk_id),
                vector=vector,
                payload=_payload(chunk),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]

        self._client.upsert(collection_name=self.collection, points=points)

    def count(self) -> int:
        response = self._client.count(collection_name=self.collection, exact=True)
        return int(response.count)

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        response = self._client.query_points(
            collection_name=self.collection,
            query=vector,
            limit=k,
            with_payload=True,
        )
        results: list[tuple[str, float]] = []
        for hit in response.points:
            chunk_id = (hit.payload or {}).get("chunk_id")
            if chunk_id is None:
                logger.warning("search hit %s has no chunk_id payload: skipping", hit.id)
                continue
            results.append((str(chunk_id), float(hit.score)))
        return results

    def get_payload(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        if not chunk_ids:
            return {}

        unique_ids = list(dict.fromkeys(chunk_ids))
        point_ids = [point_id_for(chunk_id) for chunk_id in unique_ids]

        records = self._client.retrieve(
            collection_name=self.collection,
            ids=point_ids,
            with_payload=True,
            with_vectors=False,
        )
        payloads: dict[str, dict[str, object]] = {}
        for record in records:
            payload = dict(record.payload or {})
            chunk_id = payload.get("chunk_id")
            if chunk_id is None:
                logger.warning("payload for point %s has no chunkid; skipping", record.id)
                continue
            payloads[str(chunk_id)] = payload
        return payloads

    def get_payloads(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        return self.get_payload(chunk_ids)


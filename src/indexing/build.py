from __future__ import annotations
import logging
from typing import Any 
from ..models import Settings, get_settings
from .embedding import Embedder, get_embedder
from .meta import META_FILENAME, build_meta, write_meta
from .sparse import BM25_FILENAME, BM25Index
from .vector_store import QdrantVectorStore, VectorStore

from ..ingestion import chunk_corpus, load_corpus

logger = logging.getLogger(__name__)

def build_index(
        settings: Settings,
        *,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
) -> dict[str, Any]:
    settings.index_dir.mkdir(parents=True, exist_ok=True)

    documents = load_corpus(settings.corpus_dir)
    chunks = chunk_corpus(documents, settings)
    logger.info("loaded %d documents -> documents -> %d chunks", len(documents), len(chunks))

    meta_path = settings.index_dir / META_FILENAME
    bm25_path = settings.index_dir / BM25_FILENAME

    embedder = embedder or get_embedder(settings)

    if not chunks:
        logger.warning(
                "no chunks produced from chunks_dir %s; writing empty meta + bm25 index",
                settings.corpus_dir

        )

        empty_bm25 = BM25Index()
        empty_bm25.build(chunks)
        empty_bm25.save(settings.index_dir)
        meta = build_meta(
            settings=settings,
            chunks=chunks,
            n_documents=len(documents),
            embedding_model = settings.embedding_model,
            embedding_dim=0,
        )
        write_meta(meta_path, meta)
        return {
            "n_documents": len(documents),
            "n_chunks": 0,
            "embedding_dim": 0,
            "vector_count": 0,
            "meta_path": str(meta_path),
            "bm25_path": str(bm25_path)
        }

    store = store or QdrantVectorStore.from_settings(settings)
    vectors = embedder.embed_text([chunk.text for chunk in chunks])
    dim = embedder.dim
    if len(vectors) != len(chunks):
        raise ValueError(f"embedder returned {len(vectors)} vectors for {len(chunks)} chunks")

    store.ensure_collection(dim)
    store.upsert(chunks, vectors)
    logger.info("upserted %d vectors (dim=%d) into the dense store", len(vectors), dim)

    bm25 = BM25Index()
    bm25.build(chunks)
    bm25.save(settings.index_dir)
    logger.info("saved BM25 index to %s", bm25_path)

    meta = build_meta(
        settings=settings,
        chunks=chunks,
        n_documents=len(documents),
        embedding_model=settings.embedding_model,
        embedding_dim=dim,

    )

    write_meta(meta_path, meta)
    logger.info("wrote provenece to %s", meta_path)

    return {
        "n_documents": len(documents),
        "n_chunks": len(chunks),
        "embedding_dim": dim,
        "vector_count": store.count(),
        "meta_path": str(meta_path),
        "bm25_path": str(bm25_path),

    }


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    summary = build_index(get_settings())
    logger.info("info build complete: %s", summary)


if __name__ == "__main__":
    main()


from __future__ import annotations

from pathlib import Path

from ..models import Chunk, Document, Settings
from .chunkers import chunker_for
from .loaders import load_path


def load_corpus(corpus_dir: Path) -> list[Document]:
    return list(load_path(corpus_dir))


def chunk_corpus(documents: list[Document], settings: Settings) -> list[Chunk]:
    chunker = chunker_for(
        settings.chunking_strategy,
        chunk_size_tokens=settings.chunk_size_tokens,
        overlap_tokens=settings.chunk_overlap_tokens,
        semantic_min=settings.semantic_chunk_min_tokens,
        semantic_max=settings.semantic_chunk_max_tokens,
    )
    return [chunk for document in documents for chunk in chunker.chunk(document)]
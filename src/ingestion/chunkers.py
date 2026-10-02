from __future__ import annotations

import hashlib
import re

from abc import ABC, abstractmethod
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass

from ..models import Chunk, ChunkingStrategy, Document

try:
    import tiktoken

    _ENCODER = tiktoken.get_encoding("cl100k_base")
except Exception:
    _ENCODER = None


def count_tokens(text: str) -> int:
    if _ENCODER is not None:
        return len(_ENCODER.encode(text))
    return max(1, len(text) // 4)


def stable_chunk_id(doc_source: str, position: int, text: str) -> str:
    value = f"{doc_source}|{position}|{text}".encode()
    return f"chunk{hashlib.sha256(value).hexdigest()[:16]}"


@dataclass(frozen=True, slots=True)
class TextSlice:
    text: str
    char_start: int
    char_end: int


class Chunker(ABC):
    strategy: ChunkingStrategy

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        raise NotImplementedError


class FixedTokenChunker(Chunker):
    strategy = ChunkingStrategy.FIXED

    def __init__(self, chunk_size_tokens: int = 512, overlap_tokens: int = 64):
        if chunk_size_tokens <= 0 or not 0 <= overlap_tokens < chunk_size_tokens:
            raise ValueError("overlap_tokens must be >= 0 and < chunk_size_tokens")
        self._size = chunk_size_tokens
        self._overlap = overlap_tokens

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []
        width = max(1, self._size * 4)
        overlap = self._overlap * 4
        result: list[TextSlice] = []
        start = 0
        while start < len(doc.text):
            end = min(len(doc.text), start + width)
            result.append(TextSlice(doc.text[start:end], start, end))
            if end == len(doc.text):
                break
            start = end - overlap
        return slices_to_chunks(doc, result, self.strategy)


class RecursiveCharacterChunker(Chunker):
    strategy = ChunkingStrategy.RECURSIVE
    DEFAULT_SEPARATORS = ("\n\n", "\n", ". ", " ")

    def __init__(
        self,
        *,
        chunk_size_tokens: int = 512,
        overlap_tokens: int = 64,
        separators: Iterable[str] = DEFAULT_SEPARATORS,
    ):
        if chunk_size_tokens <= 0 or not 0 <= overlap_tokens < chunk_size_tokens:
            raise ValueError("overlap_tokens must be >= 0 and < chunk_size_tokens")
        self._size = chunk_size_tokens
        self._overlap = overlap_tokens
        self._separators = tuple(separators)

    def chunk(self, doc: Document) -> list[Chunk]:
        if not doc.text:
            return []
        pieces = self._split(doc.text, 0, self._separators)
        pieces = self._add_overlap(pieces, doc.text)
        return slices_to_chunks(doc, pieces, self.strategy)

    def _split(
        self, text: str, offset: int, separators: tuple[str, ...]
    ) -> list[TextSlice]:
        if count_tokens(text) <= self._size:
            return [TextSlice(text, offset, offset + len(text))]
        if not separators:
            width = max(1, self._size * 4)
            return [
                TextSlice(text[i : i + width], offset + i, offset + min(i + width, len(text)))
                for i in range(0, len(text), width)
            ]
        separator = separators[0]
        result: list[TextSlice] = []
        cursor = 0
        for part in text.split(separator):
            if part:
                result.extend(self._split(part, offset + cursor, separators[1:]))
            cursor += len(part) + len(separator)
        return result

    def _add_overlap(self, pieces: list[TextSlice], full_text: str) -> list[TextSlice]:
        if not pieces or self._overlap == 0:
            return pieces
        overlap = self._overlap * 4
        result = [pieces[0]]
        for piece in pieces[1:]:
            start = max(0, piece.char_start - overlap)
            result.append(TextSlice(full_text[start : piece.char_end], start, piece.char_end))
        return result


SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


class SentenceChunker(Chunker):
    strategy = ChunkingStrategy.SEMANTIC

    def __init__(
        self,
        *,
        embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
        min_tokens: int = 200,
        max_tokens: int = 900,
        similarity_threshold: float = 0.75,
    ):
        if min_tokens >= max_tokens:
            raise ValueError("min_tokens must be < max_tokens")
        self._embed_fn = embed_fn
        self._min_tokens = min_tokens
        self._max_tokens = max_tokens
        self._threshold = similarity_threshold

    def chunk(self, doc: Document) -> list[Chunk]:
        base = RecursiveCharacterChunker(chunk_size_tokens=self._max_tokens, overlap_tokens=0)
        return [item.model_copy(update={"strategy": self.strategy}) for item in base.chunk(doc)]

    async def chunk_async(self, doc: Document) -> list[Chunk]:
        if self._embed_fn is None:
            return self.chunk(doc)
        sentences = split_sentences(doc.text)
        if not sentences:
            return []
        embeddings = await self._embed_fn([text for text, _, _ in sentences])
        pieces = merged_by_similarity(
            sentences,
            embeddings,
            min_tokens=self._min_tokens,
            max_tokens=self._max_tokens,
            threshold=self._threshold,
        )
        return slices_to_chunks(doc, pieces, self.strategy)


def split_sentences(text: str) -> list[tuple[str, int, int]]:
    result: list[tuple[str, int, int]] = []
    cursor = 0
    for piece in SENTENCE_RE.split(text):
        if piece.strip():
            start = text.find(piece, cursor)
            result.append((piece, start, start + len(piece)))
            cursor = start + len(piece)
        else:
            cursor += len(piece)
    return result


def _cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(left * right for left, right in zip(a, b, strict=False))
    norm_a = sum(value * value for value in a) ** 0.5 or 1.0
    norm_b = sum(value * value for value in b) ** 0.5 or 1.0
    return dot / (norm_a * norm_b)


def merged_by_similarity(
    sentences: list[tuple[str, int, int]],
    embeddings: list[list[float]],
    *,
    min_tokens: int,
    max_tokens: int,
    threshold: float,
) -> list[TextSlice]:
    if not sentences or not embeddings:
        return []
    result: list[TextSlice] = []
    current_text, current_start, current_end = sentences[0]
    current_embedding = embeddings[0]
    for (text, start, end), embedding in zip(sentences[1:], embeddings[1:], strict=False):
        candidate = f"{current_text} {text}"
        merge = count_tokens(candidate) <= max_tokens and (
            _cosine(current_embedding, embedding) >= threshold
            or count_tokens(current_text) < min_tokens
        )
        if merge:
            current_text = candidate
            current_end = end
            current_embedding = [
                (left + right) / 2
                for left, right in zip(current_embedding, embedding, strict=False)
            ]
        else:
            result.append(TextSlice(current_text, current_start, current_end))
            current_text, current_start, current_end = text, start, end
            current_embedding = embedding
    result.append(TextSlice(current_text, current_start, current_end))
    return result


def slices_to_chunks(
    doc: Document, pieces: list[TextSlice], strategy: ChunkingStrategy
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for position, piece in enumerate(pieces):
        text = piece.text.strip()
        if not text:
            continue
        chunks.append(
            Chunk(
                chunk_id=stable_chunk_id(doc.source, position, text),
                title=doc.title,
                text=text,
                strategy=strategy,
                position=position,
                char_start=piece.char_start,
                char_end=piece.char_end,
                metadata=dict(doc.metadata),
                doc_id=doc.source,
                rel_path=str(doc.metadata.get("path", doc.source)),
                ordinal=position,
                heading_path=[],
            )
        )
    return chunks


def chunker_for(
    strategy: ChunkingStrategy,
    *,
    chunk_size_tokens: int = 512,
    overlap_tokens: int = 64,
    embed_fn: Callable[[list[str]], Awaitable[list[list[float]]]] | None = None,
    semantic_min: int = 200,
    semantic_max: int = 900,
) -> Chunker:
    if strategy is ChunkingStrategy.FIXED:
        return FixedTokenChunker(chunk_size_tokens, overlap_tokens)
    if strategy is ChunkingStrategy.RECURSIVE:
        return RecursiveCharacterChunker(
            chunk_size_tokens=chunk_size_tokens, overlap_tokens=overlap_tokens
        )
    if strategy is ChunkingStrategy.SEMANTIC:
        return SentenceChunker(
        embed_fn=embed_fn,
        min_tokens=semantic_min,
        max_tokens=semantic_max,
    )


__all__ = [
    "Chunker",
    "FixedTokenChunker",
    "RecursiveCharacterChunker",
    "SentenceChunker",
    "chunker_for",
]
from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock

from src.assistant import ResearchAssistant
from src.generation import FabricatingFakeLLMClient, FakeLLMClient
from src.indexing.build import build_index
from src.indexing.embedding import HashingEmbedder
from src.indexing.sparse import BM25Index
from src.indexing.vector_store import QdrantVectorStore
from src.models import Answer, Citation, RetrievalResult, Settings
from src.retrival.hybrid import HybridRetriever
from src.retrival.rerank import LexicalOverlapReranker
from src.verification import verify_answer


class MemoryVectorStore:
    def __init__(self) -> None:
        self.points: dict[str, tuple[list[float], dict[str, object]]] = {}

    def ensure_collection(self, dim: int) -> None:
        self.points.clear()

    def upsert(self, chunks, vectors) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.points[chunk.chunk_id] = (
                vector,
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "rel_path": chunk.rel_path,
                    "heading_path": chunk.heading_path,
                    "metadata": chunk.metadata,
                },
            )

    def count(self) -> int:
        return len(self.points)

    def search(self, vector: list[float], k: int) -> list[tuple[str, float]]:
        scored = [
            (chunk_id, sum(left * right for left, right in zip(vector, stored, strict=True)))
            for chunk_id, (stored, _) in self.points.items()
        ]
        return sorted(scored, key=lambda row: -row[1])[:k]

    def get_payloads(self, chunk_ids: list[str]) -> dict[str, dict[str, object]]:
        return {
            chunk_id: self.points[chunk_id][1]
            for chunk_id in chunk_ids
            if chunk_id in self.points
        }


class StubRetriever:
    def __init__(self, contexts: list[RetrievalResult]) -> None:
        self.contexts = contexts

    def retrieve(self, question: str) -> list[RetrievalResult]:
        return self.contexts


class CitationVerificationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.context = RetrievalResult(
            chunk_id="chunk-1",
            score=0.8,
            rank=1,
            text="Reciprocal Rank Fusion combines ranked search lists.",
            rel_path="notes/rrf.md",
        )
        self.settings = Settings()

    def test_normalized_quote_is_accepted(self) -> None:
        answer = Answer(
            text="RRF combines lists.",
            citations=[
                Citation(
                    chunk_id="chunk-1",
                    rel_path="notes/rrf.md",
                    supporting_quote="RECIPROCAL   Rank Fusion combines",
                )
            ],
        )
        report = verify_answer(answer, [self.context])
        self.assertTrue(report.is_grounded)
        self.assertEqual(report.citation_accuracy, 1.0)

    def test_missing_chunk_is_rejected(self) -> None:
        answer = Answer(
            text="Unsupported.",
            citations=[Citation(chunk_id="missing", supporting_quote="invented")],
        )
        self.assertFalse(verify_answer(answer, [self.context]).is_grounded)

    def test_fabricated_fake_answer_is_withheld(self) -> None:
        assistant = ResearchAssistant(
            retriever=StubRetriever([self.context]),
            llm=FabricatingFakeLLMClient(),
            settings=self.settings,
        )
        answer = assistant.ask("What does RRF do?")
        self.assertTrue(answer.is_idk)
        self.assertEqual(answer.citations, [])

    def test_empty_retrieval_abstains(self) -> None:
        assistant = ResearchAssistant(
            retriever=StubRetriever([]),
            llm=FakeLLMClient(),
            settings=self.settings,
        )
        self.assertTrue(assistant.ask("Unknown question").is_idk)

    def test_sample_corpus_runs_through_index_and_answer(self) -> None:
        project_root = Path(__file__).resolve().parents[1]
        with TemporaryDirectory() as directory:
            settings = Settings(
                corpus_dir=project_root / "documents",
                index_dir=Path(directory) / "index",
                chunk_size_tokens=64,
                chunk_overlap_tokens=8,
                dense_top_k=5,
                sparse_top_k=5,
                final_top_k=3,
                rerank_kind="lexical",
            )
            store = MemoryVectorStore()
            embedder = HashingEmbedder()
            summary = build_index(settings, embedder=embedder, store=store)
            retriever = HybridRetriever(
                embedder=embedder,
                store=store,
                bm25=BM25Index.load(settings.index_dir),
                reranker=LexicalOverlapReranker(),
                settings=settings,
            )
            answer = ResearchAssistant(
                retriever=retriever,
                llm=FakeLLMClient(),
                settings=settings,
            ).ask("How does Reciprocal Rank Fusion combine ranked results?")

        self.assertGreater(summary["n_chunks"], 0)
        self.assertFalse(answer.is_idk)
        self.assertEqual(answer.citation_accuracy, 1.0)
        self.assertTrue(answer.citations[0].supporting_quote)

    def test_qdrant_store_reuses_existing_persistent_collection(self) -> None:
        store = object.__new__(QdrantVectorStore)
        store.collection = "research_assistant"
        client = MagicMock()
        client.collection_exists.return_value = True
        client.get_collection.return_value = MagicMock(
            config=MagicMock(
                params=MagicMock(vectors=MagicMock(size=768))
            )
        )
        store._client = client

        store.ensure_collection(768)

        client.delete_collection.assert_not_called()
        client.create_collection.assert_not_called()


if __name__ == "__main__":
    unittest.main()
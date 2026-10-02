from __future__ import annotations

import argparse
import json
import logging

from .assistant import ResearchAssistant
from .generation import get_llm_client
from .indexing.build import build_index
from .indexing.embedding import get_embedder
from .indexing.sparse import BM25Index
from .indexing.vector_store import QdrantVectorStore
from .models import get_settings
from .retrival.hybrid import HybridRetriever
from .retrival.rerank import get_reranker


def _build_assistant(settings, *, offline: bool = False) -> ResearchAssistant:
    embedder = get_embedder(settings, fake=offline)
    store = QdrantVectorStore.from_settings(settings)
    bm25 = BM25Index.load(settings.index_dir)
    retriever = HybridRetriever(
        embedder=embedder,
        store=store,
        bm25=bm25,
        reranker=get_reranker(settings, fake=offline),
        settings=settings,
    )
    return ResearchAssistant(
        retriever=retriever,
        llm=get_llm_client(settings, fake=offline),
        settings=settings,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="research-assistant")
    commands = parser.add_subparsers(dest="command", required=True)
    index_parser = commands.add_parser("index", help="ingest documents and build both indexes")
    index_parser.add_argument(
        "--offline",
        action="store_true",
        help="use hashing embeddings and local lexical components; no model API required",
    )
    ask_parser = commands.add_parser("ask", help="ask a question over the indexed documents")
    ask_parser.add_argument("question", nargs="+", help="question to answer")
    ask_parser.add_argument(
        "--offline",
        action="store_true",
        help="use hashing embeddings and fake generation; no model API required",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    settings = get_settings()

    if args.command == "index":
        store = QdrantVectorStore.from_settings(settings)
        summary = build_index(
            settings,
            embedder=get_embedder(settings, fake=args.offline),
            store=store,
        )
        print(json.dumps(summary, indent=2, default=str))
        return 0

    answer = _build_assistant(settings, offline=args.offline).ask(" ".join(args.question))
    print(answer.text)
    for citation in answer.citations:
        print(f"[{citation.chunk_id}] {citation.rel_path}: {citation.supporting_quote}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
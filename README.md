# Multi-Document Research Assistant

A lightweight research assistant for indexing a corpus of Markdown/text documents, retrieving relevant context with hybrid search, and answering questions grounded in source citations.

## Features

- Document ingestion and chunking from local corpora
- Dense vector retrieval with Qdrant-backed storage
- Sparse lexical retrieval with BM25
- Hybrid ranking via Reciprocal Rank Fusion
- Optional reranking for better final ordering
- Citation-grounded answer verification
- Offline mode for local hashing embeddings and fake LLM flow

## Tech Stack

- Python 3.11+
- Pydantic + Pydantic Settings
- Qdrant for dense vector storage
- Rank-BM25 for sparse retrieval
- Local corpus indexing pipeline

## Project Layout

- `src/assistant.py` — answer orchestration and verification
- `src/cli.py` — command-line interface
- `src/indexing/` — ingestion, embeddings, BM25, and vector store logic
- `src/retrival/` — dense, sparse, hybrid, and reranking retrieval
- `src/generation/` — LLM integration and fake clients
- `tests/` — regression and behavior tests

## Quick Start

### 1) Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Index a corpus

```bash
python -m src.cli index --offline
```

This builds the local index from the documents in `documents/` and stores the vector index in `.rag-qdrant/`.

### 4) Ask a question

```bash
python -m src.cli ask "What is the main idea behind RAG?" --offline
```

## Environment Variables

For online model use, create a `.env` file based on `.env.example`:

```bash
cp .env.example .env
```

Supported keys include:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `LLM_MODEL`
- `EMBEDDING_MODEL`
- `QDRANT_URL`
- `QDRANT_API_KEY`

## Local Storage

This project stores:

- BM25 index in `.rag-index/`
- Qdrant data in `.rag-qdrant/`

These folders are git-ignored and should be treated as generated artifacts.

## Example Workflow

```bash
python -m src.cli index
python -m src.cli ask "How does reciprocal rank fusion combine multiple ranked lists?"
```

## Testing

```bash
python -m unittest discover -s tests -q
```

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Roadmap

- improve packaging and release config
- add CI checks for Python versions
- add more evaluation datasets
- expand ingestion formats and chunking strategies
- add a web UI or API wrapper

## Contributing

Contributions are welcome. Please open an issue or pull request with a clear description of the change and a brief test summary.

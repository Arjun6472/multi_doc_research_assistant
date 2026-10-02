from __future__ import annotations
import datetime as dt 
import hashlib
import importlib.metadata as importlib_metadata
import json
import logging
import platform
import subprocess
from typing import Any 
from pathlib import Path
from ..models import Settings 
from ..models import Chunk

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1
META_FILENAME = "meta.json"

_TRACKED_LIBS = (
    "qdrant-client",
    "rank-bm25",
    "sentence-transformers",
    "numpy",
    "pydantic",
)

def corpus_sha256(chunks: list[Chunk]) -> str:
    hasher = hashlib.sha256()

    for chunk_id, text in sorted((c.chunk_id, c.text) for c in chunks):
        hasher.update(f"{chunk_id}\x00{len(text)}\x00{text}\x01".encode())
    return hasher.hexdigest()

def _git_sha()-> str | None:
    repo_root = Path(__file__).resolve().parents[3]
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,

        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    sha = result.stdout.strip()
    return sha or None

def _lib_version() -> dict[str, str | None]:
    versions: dict[str, str | None ] =  {"python": platform.python_version()}
    for name in _TRACKED_LIBS:
        try:
            versions[name] = importlib_metadata.version(name)
        except importlib_metadata.PackageNotFoundError:
            versions[name] = None
    return versions

def build_meta(
        *,
        settings: Settings,
        chunks: list[Chunk],
        n_documents: int,
        embedding_model: str,
        embedding_dim: int,

) -> dict[str, Any]:

    return {
        "schema_version": SCHEMA_VERSION,
        "created_at": dt.datetime.now(dt.UTC).isoformat(),
        "git_sha": _git_sha(),
        "corpus_sha": corpus_sha256(chunks),
        "counts": {
            "n_documents": n_documents,
            "n_chunks": len(chunks),
        },
        "embedding": {
            "model": embedding_model,
            "dim": embedding_dim,
        },
        "chunking": {
            "strategy": settings.chunking_strategy,
            "chunk_size": settings.chunk_size_tokens,
            "chunk_overlap": settings.chunk_overlap_tokens,

        },
        "vector_store": {
            "collection": settings.qdrant_collection,
            "distance": "cosine",
        },
        "library_version": _lib_version(),

    }

def write_meta(path: Path, meta: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8"
    )
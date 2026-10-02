from __future__ import annotations

from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal
from pathlib import Path 
from uuid import uuid4
from pydantic import BaseModel, ConfigDict, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ChunkingStrategy(str, Enum):

    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class DocumentFormat(str, Enum):

    PDF = "pdf"
    DOCX = "docx"
    HTML = "html"
    TXT = "txt"
    MARKDOWN = "markdown"

class Document(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    source: str = Field(min_length=1)
    format: DocumentFormat
    title: str = ""
    text: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

class Chunk(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    title: str = ""
    text: str = Field(min_length=1)
    strategy: ChunkingStrategy
    position: int = Field(ge=0)
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)
    doc_id: str = Field(min_length=1)
    rel_path: str = Field(min_length=1)
    ordinal: int = Field(ge=0)
    heading_path: list[str] = Field(default_factory=list)

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding = "utf-8",
        extra = "ignore",
    )

    anthropic_api_key: str | None = None
    openai_api_key: str | None = None
    llm_model: str = "claude-sonnet-4-6"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    reranker_model: str = "BAAI/bge-reranker-base"

    corpus_dir: Path = Path("./documents")
    chunking_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE
    chunk_size_tokens: int = Field(default=512, ge=64, le=2048)
    chunk_overlap_tokens: int = Field(default=64, ge=0, le=2048)
    semantic_chunk_min_tokens: int = Field(default=200, ge=64)
    semantic_chunk_max_tokens: int = Field(default=900, ge=64)
    dedup_cosine_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    dense_top_k: int = Field(default=20, ge=1, le=200)
    sparse_top_k: int = Field(default=20, ge=1, le=200)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    final_top_k: int = Field(default=5, ge=1, le=50)
    rerank_kind: Literal["cross-encoder", "lexical", "none"] = "cross-encoder"
    idk_retrival_threshold: float = Field(default=0.35, ge=0.0, le=1.0)
    judge_weight: float = Field(default=0.5, ge=0.0, le=1.0)
    index_dir: Path = Path("./.rag-index")
    eval_set_path: Path = Path("./eval/golden_qa.jsonl")
    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8100, ge=1, le=65535)
    request_timeout: float = Field(default=60.0, gt=0)
    max_retries: int = Field(default=2, ge=0)

    qdrant_url: str = "https://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_path: str | None = "./.rag-qdrant"
    qdrant_collection: str = "research_assistant"

    #@model_validator(mode="after")
    #def _chunk_chunk(self) -> Settings:
     #   if self.chunk_overlap_tokens -> self.chunk_size_tokens:
      #      raise ValueError("Chunk overlap < chunk size")
       # if self.semantic_chunk_min_tokens > self.semantic_chunk_max_tokens:
        #    raise ValueError("MAx > min")
        #return self


def get_settings() -> Settings:
    return Settings()
def load_settings(**override: Any) -> Settings:
    return Settings(**override)


class RetrievelResult(BaseModel):
    model_config = ConfigDict(frozen=True)
    chunk_id: str = Field(description="stable chunk id (matches Chunk.chunk_id).")
    score: float = Field(description="Final ranking score: RRF fused score, or the reranker score when""reranking ran, hiher is better: not comparable across configs.")
    rank: int = Field(ge=1, description="Final 1-based rank in the returned list (1 = best). Assigned once, last.",)
    text: str = Field(description="the chunk body, fetchedfrom the vector-store payload")
    rel_path: str = Field(description="source path relative to the corpus root(provenance).")
    heading_path: list[str] = Field(default_factory=list,
                                    description="markdown heading breadcrumbs for the chunk, e.g.['intro', 'setup'].",)
    metadata: dict[str, Any] = Field(default_factory=dict,
                                     description="chunk metadata copied from the stored payload.",)
    sources: list[str] = Field(default_factory=list,
                               description="Which retrievers surfaced this chunk: subset of {'dense', 'sparse'},""in a deterministic order (dense before sparse)",)


RetrievalResult = RetrievelResult


class DenseHit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    score: float = Field(ge=-1.0, le=1.0)
    rank: int = Field(ge=1)

class SparseHit(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    chunk_id: str
    score: float
    rank: int = Field(ge=1)

class FusedHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: str
    rrf_score: float = Field(ge=0.0)
    dense_rank: int | None = None
    sparse_rank: int | None = None

class RankedHit(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk: Chunk
    score: float
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rrf_score: float = 0.0

class Citation(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str = Field(description="stable chunk id (matches Chunk.chunk_id)")
    rel_path: str = ""
    supporting_quote: str = ""
    heading_path: list[str] = Field(default_factory=list)
    source: str = ""
    title: str = ""
    quote: str = ""


citation = Citation

class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str
    request_id: str = Field(default_factory=lambda: str(uuid4()))
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))

    question: str = ""
    citations: list[Citation] = Field(default_factory=list, description="Citations supporting the answer, if any.")
    used_chunks: list[RetrievalResult] = Field(default_factory=list)
    retrieval_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    citation_accuracy: float = Field(ge=0.0, le=1.0, default=0.0)
    composition_confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    is_idk: bool = Field(default=False)
    model: str = ""
    latency: float = Field(ge=0.0, default=0.0)

class EvalCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(description="The text of the evaluation case, e.g., a question or prompt.")

    case_id: str
    question: str
    expected_answer: str
    must_cite_sources: list[str] = Field(default_factory=list)
    notes: str = ""

class IngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n_documents: int = 0
    n_chunks_added: int = 0
    n_chunks_duplicated: int = 0
    n_failures: int = 0
    strategy: ChunkingStrategy
    metadata: dict[str, Any] = Field(default_factory=dict)
    
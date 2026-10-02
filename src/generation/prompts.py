from __future__ import annotations
from ..models import RetrievalResult

_SYSTEM_PREAMBLE = (
    "You are a helpful assistant that answers questions based on the provided context. "
    "Use only the supplied context. Follow these rules exactly:\n"
    "1. If the context does not answer the question, say 'I don't know.' Do not guess.\n"
    "2. Cite every factual claim using a citation's chunk_id from the supplied context.\n"
    "3. Never invent or guess a chunk_id.\n"
    "4. Keep supported answers concise and accurate."
)

def _format_context(index: int, ctx: RetrievalResult) -> str:
    heading = ">".join(ctx.heading_path) if ctx.heading_path else ""
    header = f"Context {index + 1} (chunk_id: {ctx.chunk_id}, score: {ctx.score:.4f}, heading: {heading}):"
    if heading:
        header += f"\n{heading}\n"
    return f"{header}\n{ctx.text}\n"

def build_grounding_prompt(question: str, contexts: list[RetrievalResult]) -> str:
    if contexts:
        blocks = "\n\n".join(_format_context(i, ctx) for i, ctx in enumerate(contexts))
    else:
        blocks = "(none provided)"
    return (
        f"{_SYSTEM_PREAMBLE}"
        f"\n\nQuestion: {question}\n\n"
        f"Context passages:\n{blocks}\n\n"
        "Please provide a concise and accurate answer to the question, citing the relevant context passages using"
    )

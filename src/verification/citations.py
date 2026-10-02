from __future__ import annotations

import unicodedata

from pydantic import BaseModel, ConfigDict, Field

from ..models import Answer, RetrievalResult


class CitationCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_id: str
    grounded: bool
    reason: str


class VerificationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    is_grounded: bool
    citation_accuracy: float = Field(ge=0.0, le=1.0)
    reason: str = ""
    checks: list[CitationCheck] = Field(default_factory=list)


def normalize_quote(text: str) -> str:
    normalized = unicodedata.normalize("NFC", text).casefold()
    return " ".join(normalized.split())


def verify_answer(
    answer: Answer,
    contexts: list[RetrievalResult],
) -> VerificationReport:
    contexts_by_id = {context.chunk_id: context for context in contexts}
    checks: list[CitationCheck] = []

    for citation in answer.citations:
        context = contexts_by_id.get(citation.chunk_id)
        quote = normalize_quote(citation.supporting_quote)

        if context is None:
            grounded, reason = False, "cited chunk was not retrieved"
        elif not quote:
            grounded, reason = False, "citation has no supporting quote"
        elif citation.rel_path and citation.rel_path != context.rel_path:
            grounded, reason = False, "citation source path does not match the retrieved chunk"
        elif quote not in normalize_quote(context.text):
            grounded, reason = False, "supporting quote was not found in the cited chunk"
        else:
            grounded, reason = True, "quote matches the retrieved chunk"

        checks.append(
            CitationCheck(
                chunk_id=citation.chunk_id,
                grounded=grounded,
                reason=reason,
            )
        )

    if not checks:
        return VerificationReport(
            is_grounded=answer.is_idk,
            citation_accuracy=0.0,
            reason="abstention" if answer.is_idk else "answer has no citations",
        )

    grounded_count = sum(check.grounded for check in checks)
    citation_accuracy = grounded_count / len(checks)
    all_grounded = grounded_count == len(checks)
    return VerificationReport(
        is_grounded=all_grounded,
        citation_accuracy=citation_accuracy,
        reason="all citations verified" if all_grounded else "one or more citations failed verification",
        checks=checks,
    )
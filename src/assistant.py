from __future__ import annotations

import logging
from time import perf_counter

from .generation import LLMClient, generate_answer
from .models import Answer, Settings
from .retrival.hybrid import HybridRetriever
from .verification import verify_answer

logger = logging.getLogger(__name__)


class ResearchAssistant:
    def __init__(
        self,
        *,
        retriever: HybridRetriever,
        llm: LLMClient,
        settings: Settings,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._settings = settings

    def ask(self, question: str) -> Answer:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty")

        start = perf_counter()
        contexts = self._retriever.retrieve(question)
        if not contexts:
            return Answer(
                text="I don't know. No relevant passages were retrieved.",
                question=question,
                is_idk=True,
                model=self._settings.llm_model,
                latency=(perf_counter() - start) * 1000,
            )

        draft = generate_answer(
            question,
            contexts,
            llm=self._llm,
            settings=self._settings,
        )
        report = verify_answer(draft, contexts)
        latency = (perf_counter() - start) * 1000

        if not report.is_grounded:
            logger.warning(
                "answer rejected by citation verification: question=%r reason=%s",
                question,
                report.reason,
            )
            return draft.model_copy(
                update={
                    "text": "I don't know. I could not verify the answer against the retrieved sources.",
                    "question": question,
                    "citations": [],
                    "used_chunks": contexts,
                    "citation_accuracy": report.citation_accuracy,
                    "is_idk": True,
                    "latency": latency,
                }
            )

        return draft.model_copy(
            update={
                "question": question,
                "used_chunks": contexts,
                "citation_accuracy": report.citation_accuracy,
                "latency": latency,
            }
        )
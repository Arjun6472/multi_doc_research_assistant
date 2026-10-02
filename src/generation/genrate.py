from __future__ import annotations
import logging
from ..models import Answer, RetrievalResult, Settings
from .llm import LLMClient, AnthropicLLMClient, fakeLLMClient

logger = logging.getLogger(__name__)

def generate_answer(
        query: str,
        contexts: list[RetrievalResult],
        *,
        llm: LLMClient,
        settings: Settings,
) -> Answer:
        logger.debug(
                "Generating answer: model=%s, contexts=%d",
                settings.llm_model,
                len(contexts),
        )
        return llm.generate_answer(query=query, contexts=contexts)


genrate_answer = generate_answer


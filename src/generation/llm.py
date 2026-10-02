from __future__ import annotations

import logging
from typing import Any, Protocol, runtime_checkable

from ..models import Answer, Citation, RetrievalResult, Settings
from .prompts import build_grounding_prompt

logger = logging.getLogger(__name__)

_MAX_TOKENS = 4096
_EFFORT = "high"
_FAKE_QUOTE_CHARS = 60


@runtime_checkable
class LLMClient(Protocol):
    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        ...


class AnthropicLLMClient:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._client: Any | None = None

    def _ensure_client(self) -> Any:
        if self._client is None:
            import anthropic

            self._client = anthropic.Anthropic(api_key=self.settings.anthropic_api_key)
        return self._client

    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        if not contexts:
            return Answer(
                text="I don't know.",
                question=query,
                model=self.settings.llm_model,
                is_idk=True,
            )

        prompt = build_grounding_prompt(query, contexts)
        response = self._ensure_client().messages.parse(
            model=self.settings.llm_model,
            max_tokens=_MAX_TOKENS,
            output_config={"effort": _EFFORT},
            thinking={"type": "adaptive"},
            output_format=Answer,
            messages=[{"role": "user", "content": prompt}],
        )
        parsed = response.parsed_output
        if parsed is None:
            logger.warning("LLM returned no parsed answer")
            return Answer(
                text="I don't know.",
                question=query,
                model=self.settings.llm_model,
                is_idk=True,
            )
        return parsed.model_copy(update={"question": query, "model": self.settings.llm_model})


class FakeLLMClient:
    def generate_answer(self, query: str, contexts: list[RetrievalResult]) -> Answer:
        if not contexts:
            return Answer(text="I don't know.", question=query, is_idk=True)

        context = contexts[0]
        quote = self._quote(context)
        return Answer(
            text=f"According to {context.chunk_id}: {quote}",
            question=query,
            citations=[
                Citation(
                    chunk_id=context.chunk_id,
                    rel_path=context.rel_path,
                    supporting_quote=quote,
                    heading_path=context.heading_path,
                )
            ],
        )

    def _quote(self, context: RetrievalResult) -> str:
        return context.text[:_FAKE_QUOTE_CHARS]


class FabricatingFakeLLMClient(FakeLLMClient):
    _FABRICATED_QUOTE = "zzqxv totally fabricated unsupported claim wzzqxv"

    def _quote(self, context: RetrievalResult) -> str:
        return self._FABRICATED_QUOTE


def get_llm_client(settings: Settings, *, fake: bool = False) -> LLMClient:
    if fake:
        return FakeLLMClient()
    return AnthropicLLMClient(settings)


fakeLLMClient = FakeLLMClient
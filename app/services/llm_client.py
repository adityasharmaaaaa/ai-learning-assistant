"""
LLM client abstraction.

We depend on a small `LLMClient` protocol rather than importing ChatGroq
directly throughout the codebase. This buys us two things:
  1. Testability: FakeLLMClient lets the LangGraph nodes and services be unit
     tested deterministically, with no network calls and no API key.
  2. Swap-ability: moving to another LangChain chat model (OpenAI, Anthropic,
     a self-hosted vLLM endpoint) is a one-class change.
"""
from __future__ import annotations

import logging
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def complete(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion for a system+user prompt pair."""
        ...


class GroqLLMClient:
    """Thin wrapper around langchain_groq.ChatGroq."""

    def __init__(self, api_key: str, model: str, temperature: float, timeout_s: int) -> None:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langchain_groq import ChatGroq

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. Add it to your .env file — see .env.example."
            )

        self._SystemMessage = SystemMessage
        self._HumanMessage = HumanMessage
        self._chat = ChatGroq(
            api_key=api_key,
            model=model,
            temperature=temperature,
            timeout=timeout_s,
            max_retries=2,  # transport-level retries (network/5xx), distinct from our
            # application-level generate->validate->repair retries in structured_output_graph.
        )

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        messages = [self._SystemMessage(content=system_prompt), self._HumanMessage(content=user_prompt)]
        result = self._chat.invoke(messages)
        return result.content


class FakeLLMClient:
    """
    Deterministic in-memory LLM used by tests and local dry-runs (no network,
    no API key). Pass `responses` as a queue of strings to return in order,
    or a single `fixed_response` to always return the same thing.
    """

    def __init__(self, responses: list[str] | None = None, fixed_response: str | None = None) -> None:
        self._responses = list(responses or [])
        self._fixed = fixed_response
        self.calls: list[tuple[str, str]] = []

    def queue(self, response: str) -> None:
        """Append a response to be returned on the next call(s), in FIFO order."""
        self._responses.append(response)

    def complete(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self._responses:
            return self._responses.pop(0)
        if self._fixed is not None:
            return self._fixed
        raise RuntimeError("FakeLLMClient has no more queued responses.")

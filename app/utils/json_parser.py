"""
Robust JSON extraction from raw LLM text.

LLMs frequently wrap JSON in markdown fences, add a preamble ("Here is the
roadmap:"), or add trailing commentary. This module defensively extracts the
first well-formed JSON object/array from arbitrary text instead of assuming
`json.loads(raw_text)` will just work.
"""
from __future__ import annotations

import json
import re

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


class JSONExtractionError(ValueError):
    pass


def extract_json(raw_text: str) -> dict:
    """
    Attempt, in order:
      1. Direct json.loads on the stripped text.
      2. The content of the first ```json fenced block.
      3. The substring between the first '{' and the matching last '}'.
    Raises JSONExtractionError if none succeed.
    """
    text = raw_text.strip()

    for candidate in _candidates(text):
        try:
            return json.loads(candidate)
        except (json.JSONDecodeError, TypeError):
            continue

    raise JSONExtractionError(f"Could not extract valid JSON from LLM output: {raw_text[:200]!r}")


def _candidates(text: str):
    yield text

    fence_match = _FENCE_RE.search(text)
    if fence_match:
        yield fence_match.group(1)

    first_brace = text.find("{")
    last_brace = text.rfind("}")
    if first_brace != -1 and last_brace != -1 and last_brace > first_brace:
        yield text[first_brace : last_brace + 1]

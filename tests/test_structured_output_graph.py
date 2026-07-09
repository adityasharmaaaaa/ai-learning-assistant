import json

import pytest
from pydantic import BaseModel

from app.services.graphs.structured_output_graph import (
    StructuredGenerationError,
    run_structured_generation,
)
from app.services.llm_client import FakeLLMClient


class SimpleSchema(BaseModel):
    title: str
    count: int


VALID_JSON = json.dumps({"title": "hello", "count": 3})
INVALID_JSON = "not json at all"
INVALID_SCHEMA_JSON = json.dumps({"title": "hello"})  # missing required "count"


def test_succeeds_on_first_try():
    llm = FakeLLMClient(responses=[VALID_JSON])
    result = run_structured_generation(llm, SimpleSchema, "sys", "user", max_retries=3)
    assert result.title == "hello"
    assert result.count == 3
    assert len(llm.calls) == 1


def test_repairs_after_malformed_json():
    llm = FakeLLMClient(responses=[INVALID_JSON, VALID_JSON])
    result = run_structured_generation(llm, SimpleSchema, "sys", "user", max_retries=3)
    assert result.count == 3
    assert len(llm.calls) == 2
    # second call should include the repair instruction referencing the previous bad output
    assert "Validation error" in llm.calls[1][1]


def test_repairs_after_schema_validation_failure():
    llm = FakeLLMClient(responses=[INVALID_SCHEMA_JSON, VALID_JSON])
    result = run_structured_generation(llm, SimpleSchema, "sys", "user", max_retries=3)
    assert result.count == 3
    assert len(llm.calls) == 2


def test_gives_up_after_max_retries():
    llm = FakeLLMClient(responses=[INVALID_JSON, INVALID_JSON, INVALID_JSON])
    with pytest.raises(StructuredGenerationError):
        run_structured_generation(llm, SimpleSchema, "sys", "user", max_retries=3)
    assert len(llm.calls) == 3

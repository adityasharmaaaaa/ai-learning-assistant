"""
Generic "generate -> validate -> repair" LangGraph.

This is the core piece of AI-reliability infrastructure shared by the
roadmap, project, and chat services. Rather than hoping the LLM returns
valid JSON on the first try, we model generation explicitly as a small
state machine:

    ┌─────────┐     ┌──────────┐     ┌────────────────────┐
    │ call_llm│ --> │ validate │ --> │ success? -> END     │
    └─────────┘     └──────────┘     │ else, retries left? │
         ^                           │   -> loop to call_llm│
         └───────────────────────────┘   with repair prompt │
                                     │ else -> END (failed)  │
                                     └────────────────────────┘

Using LangGraph (rather than a plain while-loop) makes the retry/repair
policy an explicit, inspectable graph -- each node transition is logged and
traceable, and the same subgraph is reused unmodified by every generation
task in the app (roadmap, project, chat answer), which is what "structured
LLM outputs + robust parsing/validation + handling malformed responses"
means in production, not just a try/except around one call.
"""
from __future__ import annotations

import logging
from typing import Generic, TypedDict, TypeVar

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, ValidationError

from app.services.llm_client import LLMClient
from app.utils.json_parser import JSONExtractionError, extract_json

logger = logging.getLogger(__name__)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

REPAIR_INSTRUCTION_TEMPLATE = """\
Your previous response could not be parsed/validated. Fix it and return ONLY
raw JSON (no markdown fences, no commentary) that satisfies the required
schema.

Validation error:
{error}

Previous response:
{previous_output}

Return the corrected JSON now.
"""


class StructuredOutputState(TypedDict):
    system_prompt: str
    current_user_prompt: str
    raw_output: str | None
    validated: dict | None
    error: str | None
    attempt: int
    max_retries: int


def _make_call_llm_node(llm_client: LLMClient):
    def call_llm(state: StructuredOutputState) -> StructuredOutputState:
        state["attempt"] += 1
        logger.info("structured_output: LLM call attempt %d/%d", state["attempt"], state["max_retries"])
        raw = llm_client.complete(state["system_prompt"], state["current_user_prompt"])
        state["raw_output"] = raw
        return state

    return call_llm


def _make_validate_node(schema: type[SchemaT]):
    def validate(state: StructuredOutputState) -> StructuredOutputState:
        raw = state["raw_output"] or ""
        try:
            parsed = extract_json(raw)
            model_instance = schema.model_validate(parsed)
            state["validated"] = model_instance.model_dump()
            state["error"] = None
            logger.info("structured_output: validation succeeded on attempt %d", state["attempt"])
        except (JSONExtractionError, ValidationError, TypeError) as exc:
            state["error"] = str(exc)
            logger.warning("structured_output: validation failed on attempt %d: %s", state["attempt"], exc)
        return state

    return validate


def _make_prepare_repair_node():
    def prepare_repair(state: StructuredOutputState) -> StructuredOutputState:
        state["current_user_prompt"] = REPAIR_INSTRUCTION_TEMPLATE.format(
            error=state["error"], previous_output=state["raw_output"]
        )
        return state

    return prepare_repair


def _route_after_validate(state: StructuredOutputState) -> str:
    if state["validated"] is not None:
        return "success"
    if state["attempt"] >= state["max_retries"]:
        return "give_up"
    return "repair"


def build_structured_output_graph(llm_client: LLMClient, schema: type[SchemaT]):
    graph = StateGraph(StructuredOutputState)
    graph.add_node("call_llm", _make_call_llm_node(llm_client))
    graph.add_node("validate", _make_validate_node(schema))
    graph.add_node("prepare_repair", _make_prepare_repair_node())

    graph.set_entry_point("call_llm")
    graph.add_edge("call_llm", "validate")
    graph.add_conditional_edges(
        "validate",
        _route_after_validate,
        {"success": END, "give_up": END, "repair": "prepare_repair"},
    )
    graph.add_edge("prepare_repair", "call_llm")

    return graph.compile()


class StructuredGenerationError(Exception):
    def __init__(self, message: str, last_error: str | None, last_raw_output: str | None) -> None:
        super().__init__(message)
        self.last_error = last_error
        self.last_raw_output = last_raw_output


def run_structured_generation(
    llm_client: LLMClient,
    schema: type[SchemaT],
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> SchemaT:
    """Run the generate->validate->repair graph to completion and return a validated model."""
    compiled = build_structured_output_graph(llm_client, schema)
    initial_state: StructuredOutputState = {
        "system_prompt": system_prompt,
        "current_user_prompt": user_prompt,
        "raw_output": None,
        "validated": None,
        "error": None,
        "attempt": 0,
        "max_retries": max_retries,
    }
    final_state = compiled.invoke(initial_state)

    if final_state["validated"] is None:
        raise StructuredGenerationError(
            f"LLM failed to produce valid {schema.__name__} after {final_state['attempt']} attempts.",
            last_error=final_state["error"],
            last_raw_output=final_state["raw_output"],
        )
    return schema.model_validate(final_state["validated"])

"""
Renders a Pydantic model's JSON schema into a compact description that gets
embedded directly in the LLM prompt. This keeps the prompt and the actual
validation schema in sync automatically -- if a field is added/renamed in
the Pydantic model, the prompt updates itself instead of drifting out of
sync with hand-written prompt text.
"""
from __future__ import annotations

import json

from pydantic import BaseModel


def render_schema_for_prompt(model: type[BaseModel]) -> str:
    schema = model.model_json_schema()
    # Trim to the fields that matter for prompting; full $defs / titles are noise.
    compact = {
        "type": "object",
        "properties": schema.get("properties", {}),
        "required": schema.get("required", []),
    }
    return json.dumps(compact, indent=2)

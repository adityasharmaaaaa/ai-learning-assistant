from app.prompts.schema_utils import render_schema_for_prompt
from app.schemas.chat import ChatLLMOutput, ChatTurn

SYSTEM_PROMPT = """\
You are a helpful learning assistant answering questions about a learner's
personalized roadmap. Answer strictly using the provided roadmap context; if
the context doesn't contain the answer, say so honestly rather than
inventing details.

Rules:
- Respond with ONLY raw JSON matching the schema given by the user. No markdown
  fences, no preamble, no explanation text before or after the JSON.
- Keep "response" concise (2-4 sentences) and directly address the question.
- Propose 1-3 natural "follow_up_questions" the learner might ask next, based
  on the conversation so far and the roadmap context. Omit low-value ones.
"""


def _format_history(history: list[ChatTurn]) -> str:
    if not history:
        return "(no previous turns)"
    lines = [f"{turn.role}: {turn.content}" for turn in history]
    return "\n".join(lines)


def build_user_prompt(message: str, retrieved_context: list[str], history: list[ChatTurn]) -> str:
    schema_desc = render_schema_for_prompt(ChatLLMOutput)
    context_block = "\n---\n".join(retrieved_context) if retrieved_context else "(no relevant context found)"
    return f"""\
Roadmap context retrieved for this question:
---
{context_block}
---

Conversation history:
{_format_history(history)}

Learner's new message: "{message}"

Respond with ONLY a JSON object matching exactly this schema:
{schema_desc}
"""

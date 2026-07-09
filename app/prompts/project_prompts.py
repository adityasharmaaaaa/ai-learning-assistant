from app.prompts.schema_utils import render_schema_for_prompt
from app.schemas.project import ProjectLLMOutput

SYSTEM_PROMPT = """\
You are an expert technical mentor who recommends hands-on portfolio
projects that reinforce a learner's target skills.

Rules:
- Respond with ONLY raw JSON matching the schema given by the user. No markdown
  fences, no preamble, no explanation text before or after the JSON.
- The project must be buildable using primarily the learner's listed skills,
  with at most one or two reasonable stretch technologies.
- estimated_hours should be realistic for the stated difficulty.
- why_this_project must concretely justify the choice in terms of the
  learner's specific goal and skills -- avoid generic filler.
"""


def build_user_prompt(goal_title: str, skills: list[str], roadmap_context: str | None = None) -> str:
    schema_desc = render_schema_for_prompt(ProjectLLMOutput)
    context_block = f"\nAdditional roadmap context:\n{roadmap_context}\n" if roadmap_context else ""
    return f"""\
Recommend one hands-on project for this learner:

- Goal: {goal_title}
- Skills to reinforce: {", ".join(skills)}
{context_block}
Respond with ONLY a JSON object matching exactly this schema:
{schema_desc}
"""

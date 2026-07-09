from app.prompts.schema_utils import render_schema_for_prompt
from app.schemas.roadmap import RoadmapLLMOutput, RoadmapRequest

SYSTEM_PROMPT = """\
You are an expert technical curriculum designer who creates personalized,
realistic learning roadmaps for software engineering careers.

Rules:
- Respond with ONLY raw JSON matching the schema given by the user. No markdown
  fences, no preamble, no explanation text before or after the JSON.
- The roadmap must be realistic for the learner's stated experience level and
  weekly time budget (do not exceed roughly weekly_hours * 12 total hours).
- Every task must have at least one concrete, actionable subtask.
- Skills already known by the learner should be reinforced, not re-taught from
  scratch, but can still appear if genuinely relevant to the goal.
- Order tasks in a sensible learning progression (foundational -> advanced).
"""


def build_user_prompt(req: RoadmapRequest) -> str:
    schema_desc = render_schema_for_prompt(RoadmapLLMOutput)
    return f"""\
Generate a personalized learning roadmap for this learner:

- Goal: {req.goal_title}
- Experience level: {req.experience}
- Known skills: {", ".join(req.known_skills) or "none"}
- Preferred learning style: {req.learning_style}
- Available time: {req.weekly_hours} hours/week

Respond with ONLY a JSON object matching exactly this schema:
{schema_desc}
"""

import json

VALID_ROADMAP_JSON = json.dumps(
    {
        "estimated_hours": 120,
        "skills": ["Python", "FastAPI", "PostgreSQL", "Docker"],
        "tasks": [
            {
                "title": "Learn FastAPI",
                "estimated_hours": 12,
                "subtasks": [{"title": "Routing"}, {"title": "Dependency Injection"}],
            },
            {
                "title": "Learn PostgreSQL",
                "estimated_hours": 15,
                "subtasks": [{"title": "Schema design"}, {"title": "Indexes"}],
            },
        ],
    }
)

VALID_PROJECT_JSON = json.dumps(
    {
        "title": "Task Management API",
        "difficulty": "Intermediate",
        "estimated_hours": 20,
        "tech_stack": ["FastAPI", "PostgreSQL", "Docker"],
        "features": ["JWT Authentication", "CRUD APIs", "Pagination"],
        "why_this_project": "Helps practice REST API development end to end.",
    }
)

VALID_CHAT_JSON = json.dumps(
    {
        "response": "Yes, Docker can be learned before PostgreSQL, though databases first helps.",
        "follow_up_questions": [
            "Would you like a suggested learning order?",
            "Do you want a project idea combining both?",
        ],
    }
)

VALID_ROADMAP_REQUEST = {
    "goal_title": "Backend Developer",
    "experience": "Less than 1 year",
    "known_skills": ["Python", "SQL"],
    "learning_style": "Project Based",
    "weekly_hours": 15,
}

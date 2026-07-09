from tests.sample_payloads import (
    VALID_PROJECT_JSON,
    VALID_ROADMAP_JSON,
    VALID_ROADMAP_REQUEST,
)


def test_create_project_from_roadmap_id(app_client):
    client, fake_llm = app_client
    fake_llm.queue(VALID_ROADMAP_JSON)
    roadmap = client.post("/roadmap", json=VALID_ROADMAP_REQUEST).json()

    fake_llm.queue(VALID_PROJECT_JSON)
    resp = client.post("/project", json={"roadmap_id": roadmap["roadmap_id"]})

    assert resp.status_code == 201
    body = resp.json()
    assert body["title"] == "Task Management API"
    assert body["difficulty"] == "Intermediate"


def test_create_project_ad_hoc_goal_and_skills(app_client):
    client, fake_llm = app_client
    fake_llm.queue(VALID_PROJECT_JSON)

    resp = client.post(
        "/project", json={"goal_title": "Backend Developer", "skills": ["Python", "FastAPI", "SQL"]}
    )

    assert resp.status_code == 201
    assert resp.json()["title"] == "Task Management API"


def test_create_project_unknown_roadmap_id_returns_404(app_client):
    client, _ = app_client
    resp = client.post("/project", json={"roadmap_id": "does-not-exist"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "roadmap_not_found"


def test_create_project_neither_mode_returns_422(app_client):
    client, _ = app_client
    resp = client.post("/project", json={})
    assert resp.status_code == 422

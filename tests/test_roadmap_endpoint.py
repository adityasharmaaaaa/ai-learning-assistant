from tests.sample_payloads import VALID_ROADMAP_JSON, VALID_ROADMAP_REQUEST


def test_create_roadmap_success(app_client):
    client, fake_llm = app_client
    fake_llm.queue(VALID_ROADMAP_JSON)

    resp = client.post("/roadmap", json=VALID_ROADMAP_REQUEST)

    assert resp.status_code == 201
    body = resp.json()
    assert body["estimated_hours"] == 120
    assert "FastAPI" in body["skills"]
    assert len(body["tasks"]) == 2
    assert body["tasks"][0]["subtasks"][0]["title"] == "Routing"
    assert "roadmap_id" in body and body["roadmap_id"]


def test_create_roadmap_missing_field_returns_422(app_client):
    client, _ = app_client
    payload = dict(VALID_ROADMAP_REQUEST)
    del payload["weekly_hours"]

    resp = client.post("/roadmap", json=payload)

    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "request_validation_error"


def test_create_roadmap_invalid_experience_enum_returns_422(app_client):
    client, _ = app_client
    payload = dict(VALID_ROADMAP_REQUEST)
    payload["experience"] = "a decade, roughly"

    resp = client.post("/roadmap", json=payload)

    assert resp.status_code == 422


def test_create_roadmap_recovers_from_malformed_llm_output(app_client):
    client, fake_llm = app_client
    fake_llm.queue("here is your roadmap: not actually json")
    fake_llm.queue(VALID_ROADMAP_JSON)

    resp = client.post("/roadmap", json=VALID_ROADMAP_REQUEST)

    assert resp.status_code == 201
    assert len(fake_llm.calls) == 2


def test_create_roadmap_llm_exhausts_retries_returns_502(app_client):
    client, fake_llm = app_client
    fake_llm.queue("garbage")
    fake_llm.queue("still garbage")
    fake_llm.queue("nope")

    resp = client.post("/roadmap", json=VALID_ROADMAP_REQUEST)

    assert resp.status_code == 502
    assert resp.json()["error"]["code"] == "llm_generation_failed"

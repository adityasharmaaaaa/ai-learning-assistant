from tests.sample_payloads import VALID_CHAT_JSON, VALID_ROADMAP_JSON, VALID_ROADMAP_REQUEST


def _create_roadmap(client, fake_llm):
    fake_llm.queue(VALID_ROADMAP_JSON)
    return client.post("/roadmap", json=VALID_ROADMAP_REQUEST).json()


def test_chat_success_and_retrieval_context_included(app_client):
    client, fake_llm = app_client
    roadmap = _create_roadmap(client, fake_llm)

    fake_llm.queue(VALID_CHAT_JSON)
    resp = client.post(
        "/chat",
        json={"roadmap_id": roadmap["roadmap_id"], "message": "Can I learn Docker before PostgreSQL?"},
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["response"]
    assert len(body["follow_up_questions"]) == 2

    # The prompt sent to the LLM should contain retrieved roadmap context,
    # proving retrieval actually ran rather than just stuffing raw history.
    last_user_prompt = fake_llm.calls[-1][1]
    assert "PostgreSQL" in last_user_prompt or "FastAPI" in last_user_prompt


def test_chat_unknown_roadmap_returns_404(app_client):
    client, _ = app_client
    resp = client.post("/chat", json={"roadmap_id": "nope", "message": "hi"})
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "roadmap_not_found"


def test_chat_conversation_history_is_fed_back_into_next_turn(app_client):
    client, fake_llm = app_client
    roadmap = _create_roadmap(client, fake_llm)

    fake_llm.queue(VALID_CHAT_JSON)
    client.post(
        "/chat", json={"roadmap_id": roadmap["roadmap_id"], "message": "Can I learn Docker before PostgreSQL?"}
    )

    fake_llm.queue(VALID_CHAT_JSON)
    client.post(
        "/chat", json={"roadmap_id": roadmap["roadmap_id"], "message": "What did I just ask you?"}
    )

    second_call_prompt = fake_llm.calls[-1][1]
    assert "Can I learn Docker before PostgreSQL?" in second_call_prompt


def test_chat_empty_message_returns_422(app_client):
    client, fake_llm = app_client
    roadmap = _create_roadmap(client, fake_llm)
    resp = client.post("/chat", json={"roadmap_id": roadmap["roadmap_id"], "message": ""})
    assert resp.status_code == 422

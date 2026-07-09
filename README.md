# AI Learning Assistant

A FastAPI backend that generates personalized learning roadmaps, recommends
projects, and answers roadmap questions through a Retrieval-Augmented
Generation (RAG) chat assistant -- built with **FastAPI**, **Pydantic**,
**LangChain**, **LangGraph**, and **Groq**.

Built for the Get Set Skilled AI Engineering Intern assignment.

---

## Table of contents

- [Architecture overview](#architecture-overview)
- [Why LangChain *and* LangGraph](#why-langchain-and-langgraph)
- [RAG design: chunking, embeddings, vector store, retrieval](#rag-design-chunking-embeddings-vector-store-retrieval)
- [Structured output, validation & repair](#structured-output-validation--repair)
- [Product feature: conversation history](#product-feature-conversation-history)
- [Setup instructions](#setup-instructions)
- [API reference](#api-reference)
- [Testing](#testing)
- [Assumptions made](#assumptions-made)
- [AI tools/frameworks used](#ai-toolsframeworks-used)
- [Prompt design decisions](#prompt-design-decisions)
- [Known limitations / what I'd do next](#known-limitations--what-id-do-next)
- [Time spent](#time-spent)

---

## Architecture overview

```
                          ┌─────────────────────────┐
                          │        FastAPI           │
                          │  /roadmap /project /chat │
                          └────────────┬─────────────┘
                                       │
                 ┌─────────────────────┼─────────────────────┐
                 ▼                     ▼                     ▼
        roadmap_service        project_service          chat_service
                 │                     │                     │
                 │                     │           ┌─────────┴─────────┐
                 │                     │           │     chat_graph     │  (LangGraph)
                 │                     │           │ retrieve -> generate -> persist │
                 │                     │           └─────────┬─────────┘
                 └──────────┬──────────┴─────────────────────┘
                            ▼
              structured_output_graph (LangGraph, shared)
              call_llm -> validate -> [repair -> call_llm]* -> done
                            │
                            ▼
                    GroqLLMClient (langchain-groq)
                            │
                            ▼
                        Groq API

  RAG pipeline (used by chat_service):
  RoadmapResponse -> chunk_roadmap() -> Embedder -> FAISS VectorStore (per roadmap_id, persisted to disk)
```

**Layers**

| Layer | Responsibility |
|---|---|
| `app/api/routes/` | HTTP contracts only -- parses request, calls a service, returns response. No business logic. |
| `app/schemas/` | Pydantic request/response/LLM-output models. The single source of truth for validation. |
| `app/services/` | Business logic: roadmap/project/chat generation, orchestrated via LangGraph. |
| `app/services/graphs/` | LangGraph state machines (`structured_output_graph`, `chat_graph`). |
| `app/services/rag/` | Chunking, embeddings, FAISS vector store, retriever. |
| `app/prompts/` | System/user prompt templates, one module per capability. |
| `app/storage/` | SQLAlchemy models + repositories (roadmaps, chat history). |
| `app/core/` | Cross-cutting concerns: logging, exceptions, request middleware. |
| `app/config.py` | All environment configuration in one typed, validated place. |

Dependency injection (`app/dependencies.py`) wires singletons (LLM client,
embedder, vector store -- expensive to construct, safe to share) against
per-request objects (DB session, repositories). This is also the seam tests
use to swap in fakes -- see [Testing](#testing).

---

## Why LangChain *and* LangGraph

- **LangChain** (`langchain-groq`, `langchain-huggingface`,
  `langchain-community`) is used as a thin, swappable abstraction over the
  LLM/embedding/vector-store providers -- `GroqLLMClient` wraps `ChatGroq`;
  `HuggingFaceEmbedder` wraps `HuggingFaceEmbeddings`; `VectorStore` wraps
  `FAISS`. Swapping Groq for another provider, or FAISS for pgvector, is a
  class-level change, not a rewrite.
- **LangGraph** models the *reliability* concerns explicitly as inspectable
  state machines rather than ad-hoc try/except blocks:
  - `structured_output_graph`: a **generate -> validate -> repair** loop
    shared by roadmap generation, project recommendation, and chat answer
    generation. One well-tested piece of retry/repair logic, reused three
    times, instead of three slightly-different try/except blocks.
  - `chat_graph`: the RAG pipeline itself (**retrieve -> generate ->
    persist**) as an explicit graph, so each stage is independently
    testable and the flow is traceable (LangGraph logs each node
    transition).

## RAG design: chunking, embeddings, vector store, retrieval

The `/chat` endpoint is required to be a *real* RAG system, not
in-memory context stuffing. Design decisions, and why:

**Chunking (`app/services/rag/chunking.py`)** -- A roadmap is structured
JSON, not prose, so it is **not** run through a generic character/token
splitter (which would risk severing a task from its own subtasks or its
hour estimate). Instead each roadmap is chunked into semantically coherent
units:
- one **summary chunk** (goal, total hours, full skill list) -- always
  included in retrieval results regardless of similarity score, so the
  model never loses global context even when a query only fuzzily matches
  one task,
- one **task chunk** per task (title + hours + its subtasks) -- so a
  question like *"how long will Docker take?"* retrieves the whole
  relevant unit, not a fragment of it.

**Embeddings (`app/services/rag/embeddings.py`)** -- Default provider is a
local `sentence-transformers/all-MiniLM-L6-v2` model via
`langchain-huggingface`, chosen because it runs on CPU with no extra API
key or per-call cost and is a reasonable choice at this scale. It's behind
an `Embedder` interface, so swapping to `text-embedding-3-small` or another
hosted embedding API is a one-class change (set `EMBEDDING_PROVIDER`
accordingly and add the provider class).

**Vector store (`app/services/rag/vector_store.py`)** -- One **FAISS**
index per `roadmap_id`, persisted to disk under `VECTOR_INDEX_DIR/` and
cached in-process so repeated `/chat` calls for the same roadmap don't
re-embed or reload from disk. FAISS is appropriate here because each
roadmap's knowledge base is small (tens of chunks) -- exact in-memory
search is both fast and simple. At real production scale (many
roadmaps, multi-tenant filtering, shared infra) this would move to a
managed/persistent vector DB such as **pgvector** or **Pinecone**; the
`VectorStore` class is the seam where that swap happens without touching
`retriever.py` or `chat_service.py`.

**Retrieval strategy (`app/services/rag/retriever.py`)** -- Top-k
similarity search (`RETRIEVAL_TOP_K`, default 4) over task-level chunks,
with the summary chunk always force-included. The roadmap is indexed
**eagerly right after generation** (`roadmap_service.py`), so the first
`/chat` call for a roadmap doesn't pay embedding cold-start latency.

## Structured output, validation & repair

Every LLM call in this service is asked to return raw JSON matching a
Pydantic schema (schema description is generated *from* the Pydantic model
via `app/prompts/schema_utils.py`, so prompt and validator can't drift out
of sync). The response then goes through `structured_output_graph`:

1. **call_llm** -- send system + user prompt, get raw text back.
2. **validate** -- robustly extract JSON from the raw text
   (`app/utils/json_parser.py` handles markdown fences, preambles, and
   trailing commentary -- not just a bare `json.loads`), then validate
   against the Pydantic schema.
3. On success -> done. On failure -> **prepare_repair** builds a follow-up
   prompt that includes the exact validation error and the model's
   previous (bad) output, then loops back to `call_llm`.
4. After `MAX_LLM_RETRIES` failed attempts, the graph gives up and the
   service raises `LLMGenerationError` -> HTTP `502` with a structured
   error body (never a raw stack trace).

This is unit-tested directly against a `FakeLLMClient` with canned
malformed/valid responses (`tests/test_structured_output_graph.py`) --
no network or API key required to verify the retry logic.

## Product feature: conversation history

`/chat` is multi-turn: each turn is persisted (`chat_messages` table,
keyed by `roadmap_id`) and the last `MAX_HISTORY_TURNS` turns are fed back
into the prompt on the next call (`app/prompts/chat_prompts.py`), so
follow-up questions like *"what did I just ask?"* or *"and the one after
that?"* work correctly. This was chosen over the other example features
because it most directly improves the RAG chat experience the assignment
centers on, and it composes naturally with the LangGraph chat pipeline
(`persist_history` is just another node).

As a secondary, low-cost addition: the per-roadmap FAISS index is cached
in-process after first build, so repeated chat turns against the same
roadmap don't re-embed the roadmap on every call.

---

## Setup instructions

### Prerequisites
- Python 3.11+
- A [Groq API key](https://console.groq.com/keys) (free tier is fine)

### 1. Clone and install

```bash
git clone https://github.com/adityasharmaaaaa/ai-learning-assistant.git
cd ai-learning-assistant
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
# then edit .env and set GROQ_API_KEY=...
```

### 3. Run

```bash
uvicorn app.main:app --reload
```

The API is now at `http://127.0.0.1:8000`. Interactive docs:
`http://127.0.0.1:8000/docs`.

### 4. Or run with Docker

```bash
docker compose up --build
```

### 5. Try it

```bash
curl -X POST http://127.0.0.1:8000/roadmap \
  -H "Content-Type: application/json" \
  -d '{
        "goal_title": "Backend Developer",
        "experience": "Less than 1 year",
        "known_skills": ["Python", "SQL"],
        "learning_style": "Project Based",
        "weekly_hours": 15
      }'
```

Take the returned `roadmap_id` and use it against `/project` or `/chat`.

---

## API reference

### `POST /roadmap`
Generates and persists a personalized roadmap. See request/response shape
in `app/schemas/roadmap.py`; matches the assignment spec (`goal_title`,
`experience`, `known_skills`, `learning_style`, `weekly_hours` in;
`roadmap_id`, `estimated_hours`, `skills`, `tasks[].subtasks[]` out).

### `POST /project`
Recommends one project. Accepts **either** `{"roadmap_id": "..."}` **or**
`{"goal_title": "...", "skills": [...]}` for ad-hoc use without a roadmap.

### `POST /chat`
RAG chat over a previously generated roadmap. `{"roadmap_id": "...",
"message": "..."}` in; `{"response": "...", "follow_up_questions": [...]}`
out. Requires the roadmap to already exist (404 otherwise).

### `GET /health`
Liveness check.

All error responses share one envelope:
```json
{"error": {"code": "roadmap_not_found", "message": "...", "details": {}}}
```

---

## Testing

```bash
pytest -v
```

21 tests, all offline (no Groq API key or network access required):
- `test_json_parser.py` -- robust JSON extraction from messy LLM text.
- `test_structured_output_graph.py` -- the generate/validate/repair
  LangGraph, including the "gives up after N attempts" path.
- `test_roadmap_endpoint.py`, `test_project_endpoint.py`,
  `test_chat_endpoint.py` -- full FastAPI integration tests via
  `TestClient`, with `FakeLLMClient` and `HashEmbedder` injected through
  `app.dependency_overrides` (see `tests/conftest.py`). This proves the
  retrieval pipeline, conversation history, and error handling
  (404/422/502) all work end to end without hitting Groq or downloading
  embedding model weights.

---

## Assumptions made

- **`goal_title` is stored on the roadmap** and returned as an additive
  field beyond the spec's example response, since `/project` and `/chat`
  need it for context reuse (e.g. an ad-hoc `/project` fallback, prompt
  context). Extra JSON fields are additive/backward compatible.
- **Persistence is SQLite**, not pure in-memory -- durable across restarts,
  zero external setup, and the ORM layer means switching `DATABASE_URL` to
  Postgres in production is a one-line change.
- **`experience` and `learning_style` are constrained to a fixed set of
  values** (matching the assignment's examples) rather than freeform
  strings, since the roadmap-quality prompt benefits from a controlled
  vocabulary. Easy to loosen to freeform `str` if the grader expects that.
- **Authentication is out of scope** per the assignment's notes ("optional,
  bonus") -- not implemented.
- **UI is out of scope** per the assignment's notes -- this is a pure
  backend service, exercised via `/docs`, `curl`, or the screen recording.

## AI tools/frameworks used

- **Claude** (Anthropic) was used as a pair-programming assistant to
  scaffold the project structure, write the LangGraph state machines, the
  RAG pipeline, the test suite, and this README, with the design decisions
  above driven by the assignment's explicit RAG/reliability requirements.
  All generated code was reviewed and run (21/21 tests passing; manual
  `/health` and `/docs` smoke tests) before inclusion.
- **LangChain** / **LangGraph** / **langchain-groq** /
  **langchain-huggingface** / **langchain-community** (FAISS integration)
  as described above.


## Prompt design decisions

- **Schema-in-prompt, generated from Pydantic** (`schema_utils.py`): the
  JSON schema shown to the model is derived from the same Pydantic class
  used to validate its output, so the prompt can't silently drift out of
  sync with the validator as the schema evolves.
- **"ONLY raw JSON, no markdown fences, no commentary"** is stated
  explicitly and repeated across all three system prompts, because Groq's
  open-weight models (Llama/GPT-OSS family) reliably wrap JSON in
  ` ```json ` fences or add a preamble unless told not to -- hence also
  the defensive multi-strategy extraction in `json_parser.py` rather than
  relying on prompt compliance alone.
- **Repair prompts include the exact validation error and the previous bad
  output** (not just "try again"), giving the model concrete, actionable
  feedback -- this is what makes 1-2 retries reliably succeed instead of
  repeating the same mistake.
- **Roadmap prompt bounds total hours** (`~weekly_hours * 12`) to keep
  estimates realistic relative to the learner's stated time budget, rather
  than letting the model invent an arbitrary-length curriculum.
- **Chat prompt explicitly instructs "answer strictly using the provided
  roadmap context... say so honestly rather than inventing details"** --
  standard RAG grounding instruction to reduce hallucination beyond the
  retrieved chunks.
- **Follow-up questions are explicitly scoped to 1-3, "omit low-value
  ones"** -- earlier drafts without this produced follow-ups even when
  none were natural; forcing a floor of zero avoids padding.

## Known limitations / what I'd do next

- FAISS indexes are per-process; a multi-instance production deployment
  would need a shared vector store (pgvector/Pinecone) instead of local
  disk persistence -- the `VectorStore` interface is designed for that
  swap.
- No caching of identical roadmap/project LLM calls (e.g. same request
  body twice) -- would add next given more time, keyed by a hash of the
  request payload.
- No streaming responses -- Groq/LangChain support token streaming; the
  synchronous `/chat` response was chosen for simplicity and because
  `follow_up_questions` requires the full structured object anyway.
- `langchain-community`'s FAISS integration is flagged as being sunset
  upstream in favor of a standalone package; it's isolated behind
  `VectorStore` here specifically so that migration is a contained change.

## Time spent

7-8 hours

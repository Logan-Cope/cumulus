# ☁️ Cumulus

**A culturally- and personally-responsive cloud-architecture mentor bot.**
Cumulus runs a guided conversation with a learner and answers strictly from a real
cloud-architecture curriculum — citing the exact lesson behind every claim, and saying
*"I don't have enough information from the curriculum"* instead of guessing.

> Built to demonstrate a production-grade, **trustworthy** RAG + agent architecture:
> grounded retrieval, source citations, corrective RAG, and a deterministic conversation
> spine. The headline property — *it only answers from the curriculum, and every answer
> is sourced* — is the same property any high-stakes advising assistant needs.

---

## Why this exists

Most "chat with your docs" demos happily hallucinate. Cumulus is the opposite by design:
the high-stakes content comes from retrieved, cited curriculum material, the conversation
flow is an explicit state machine (not a free-form chatbot), and a corrective-RAG loop
re-checks weak retrievals before the model ever answers. It's a reference implementation
of *trustworthy* applied AI.

## Architecture at a glance

```
Learner (web / API)
        │
        ▼
Conversation engine  ──(MCP)──►  Knowledge bridge  ──►  pgvector curriculum index
(LangGraph state machine)                                 (Postgres)
        │
        ├─► Postgres: conversation memory + audit
        └─► Persona config (swappable: language, tone, voice)
```

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the principles and the *why* behind
each choice.

**Core ideas:**

- **Guided interview, deterministic spine** — LangGraph nodes are phases; the model supplies the language, the graph supplies the structure.
- **Grounded + cited only** — retrieve, answer from context, cite the module/lesson, refuse when context is insufficient.
- **Corrective RAG** — grade retrieval; re-query or refuse if it's weak (capped).
- **Persona = config, not code** — add mentors without touching the engine.
- **One database** — pgvector keeps app data, memory, audit, and vectors in a single Postgres.

## Stack

| Layer | Choice |
|---|---|
| Orchestration | LangGraph |
| Models | Claude (Sonnet / Opus / Haiku, routed by task); OpenAI optional |
| Embeddings | Voyage (or OpenAI `text-embedding-3-small`) |
| Vector store | pgvector (graduates to a dedicated vector DB only at scale) |
| Knowledge bridge | MCP server |
| API | FastAPI |
| Data store / memory / audit | PostgreSQL |

## Project layout

```
cumulus/
├── app/
│   ├── graph.py            # LangGraph conversation engine (the spine)
│   ├── retrieval.py        # pgvector similarity search (the RAG core)
│   ├── mcp_server.py       # MCP tools the agent calls (the knowledge bridge)
│   ├── personas/           # persona configs (the swappable skin)
│   ├── api.py              # FastAPI front door
│   └── config.py
├── ingest/                 # offline: Notion markdown → chunk → embed → pgvector
├── db/schema.sql           # pgvector tables
├── data/curriculum/        # raw curriculum markdown (gitignored — private)
├── docs/ARCHITECTURE.md
└── tests/
```

## Getting started

```bash
# 1. Install (uv recommended)
uv sync

# 2. Configure
cp .env.example .env   # add API keys + DATABASE_URL

# 3. Create the schema (Postgres + pgvector required)
psql "$DATABASE_URL" -f db/schema.sql

# 4. Ingest curriculum (after dropping markdown into data/curriculum/)
python -m ingest.chunk && python -m ingest.embed

# 5. Run
uvicorn app.api:app --reload
```

## MCP server (knowledge bridge)

Retrieval is exposed as an [MCP](https://modelcontextprotocol.io) tool so the
knowledge layer is reusable and front-end-agnostic — any MCP client (the Cumulus
graph, Claude Desktop, another agent) can query the curriculum the same way.

```bash
# Run the server (stdio transport, the MCP default)
uv run python -m app.mcp_server
```

It exposes one tool:

- **`search_curriculum(query: str, module: str | None = None)`** — searches the
  cloud-engineering curriculum and returns the most relevant chunks, each WITH
  its `module`, `lesson`, `source_url`, and similarity `score`, so every answer
  can cite the exact lesson. Pass `module` to restrict the search to one module.

The LangGraph engine retrieves *through* this tool (`retrieve_via_mcp`), with a
direct-retrieval fallback so a transport hiccup never breaks the answer loop.

## License

[MIT](LICENSE) © 2026 Logan Cope

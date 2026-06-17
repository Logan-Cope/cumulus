# Cumulus — Build Plan (Milestone 1)

**Goal of Milestone 1:** ask Cumulus a question and get back an answer that comes
*only* from the curriculum, with a citation to the exact lesson. Everything else
(personas, voice, memory, web UI) comes after this works.

If you only do one thing: get the retrieve → answer → cite loop working end to end
on a small slice of the curriculum, then expand.

## Order of work

1. **Get the curriculum into `data/curriculum/`** (local only; gitignored).
   - Easiest: in Notion, open *Sophomore Year - AWS Cloud Engineer Bootcamp* →
     `•••` menu → Export → **Markdown & CSV** → unzip → drop the `.md` files into
     `data/curriculum/`. Do the same for the Project pages.
   - Result: a handful of markdown files, one per page.

2. **Get a Postgres + pgvector database** (Supabase — see "Database setup" below).
   Enable the `vector` extension, copy the connection string into `.env` as
   `DATABASE_URL`, then apply the schema:
   ```bash
   psql "$DATABASE_URL" -f db/schema.sql
   ```

3. **Implement ingestion** (`ingest/chunk.py`, `ingest/embed.py`) — see "Chunking".
   Run it once to fill the `chunks` table.

4. **Implement retrieval** (`app/retrieval.py`) — embed the query, pgvector
   nearest-neighbour search, return chunks *with* `module`, `lesson`, `source`.

5. **Minimal answer loop** — start as a CLI before any graph/web:
   embed question → retrieve top-k → put chunks in the prompt → Claude answers
   from context, cites the lesson, or says it doesn't have enough info.

6. **Smoke test:** ask "What is Terraform?" and "How do I create an IAM user?"
   Confirm the answer is grounded and cites the right lesson. Then wrap the loop
   in the LangGraph graph (`app/graph.py`) and add the corrective-RAG grade step.

## Database setup (Supabase)

Supabase is managed PostgreSQL in the cloud. We use its free tier for development;
the same project can carry into production (paid tiers scale it). The provider is
interchangeable — RDS, Supabase, and Neon are all just Postgres behind a connection
string, so switching later is a one-line `DATABASE_URL` change with no code edits.

Browser steps (one time):
1. Sign up at supabase.com and create a new project. Pick a region near you, set a
   database password (save it), free tier.
2. Wait ~2 min for it to provision.
3. Enable pgvector: Database → Extensions → search "vector" → enable. (Or run
   `create extension if not exists vector;` in the SQL Editor.)
4. Copy the connection string: Connection → **Session pooler**, Type **URI**.
   (Session pooler is IPv4-friendly and works cleanly with Python/psycopg; avoid
   Direct (IPv6-only) and Transaction pooler (needs prepared statements disabled).)
   Fill in your password where it says `[YOUR-PASSWORD]`.

Then locally:
5. Put it in `.env` as `DATABASE_URL=...` (never commit it — `.env` is gitignored).
6. Apply the schema: `psql "$DATABASE_URL" -f db/schema.sql`.

The app never needs special database "access" — it just uses the connection string.

## Chunking (decided, with the why)

The curriculum is markdown organized as Week/Day → topic sections (`#`/`###`
headings) → short content + code blocks. So:

**Strategy: structure-aware chunking.** Split on headings first so each chunk is one
coherent topic; only size-split a section if it's long.

- **Chunk at the topic-section level, not the whole-day level.** A whole day mixes
  several topics, which dilutes the embedding and adds noise to retrieval. A single
  section ("What is Terraform?", "CLI Commands") is one coherent idea — that's the
  right unit.
- **Carry metadata on every chunk:** `module` = the Week/Day (or Project), `lesson`
  = the section heading, `source_url` = the Notion page. This is what lets the bot
  cite "Week 5, Day 1 — What is Terraform?".
- **Size cap:** target ~800 characters per chunk with ~120 characters of overlap,
  but keep short sections whole (don't pad them). Only split a section when it
  exceeds the cap. (~800 chars ≈ ~200 tokens — a good retrieval granularity.)
- **Respect code fences.** Don't split in the middle of a ``` ``` block, and don't
  treat `#` inside code (e.g. a Python comment) as a heading. `MarkdownHeaderTextSplitter`
  handles this correctly — use it, then a `RecursiveCharacterTextSplitter` for the
  size cap.

**The tradeoff in one line:** chunks too big = diluted/noisy retrieval and wasted
prompt tokens; chunks too small = retrieved fragments that lack the context to
answer. One coherent section is the sweet spot, and your curriculum already has
those boundaries built in — lean on them.

Implementation sketch (already stubbed in `ingest/chunk.py`):

```python
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

sections = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "module"), ("##", "section"), ("###", "lesson")],
    strip_headers=False,
).split_text(markdown_text)

chunks = RecursiveCharacterTextSplitter(
    chunk_size=800, chunk_overlap=120, add_start_index=True,
).split_documents(sections)   # only splits sections that exceed the cap
```

## Orienting Claude Code

1. `cd ~/myos/Projects/cumulus && claude`
2. It auto-loads `CLAUDE.md` and `CLAUDE.local.md`.
3. First message: *"Read docs/ARCHITECTURE.md and docs/BUILD-PLAN.md, then let's do
   Milestone 1 step by step. Start with ingestion."*
4. Once `data/curriculum/` is populated, point it there.

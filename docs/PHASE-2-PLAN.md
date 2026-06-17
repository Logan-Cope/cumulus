# Cumulus Phase 2 — Path to Mid Tier

The architecture plan to take Cumulus from MVP (Milestone 1) to the Mid tier: the
production-solid, trustworthy, measurable version. Written so the pieces read as one
connected system.

## Goal & definition of done

**Goal:** every answer is grounded, precisely cited, and *measured*. The system
retrieves well, knows when to refuse, exposes its knowledge through a standard tool
layer (MCP), remembers a conversation, and every change is judged by a number.

**Mid tier is "done" when:**
- Retrieval **hit-rate@6 ≥ 0.90** and **MRR ≥ 0.80** on the eval set.
- **Citation precision ≥ 0.90** (no tangential-lesson leakage).
- **100% correct refusals** on off-curriculum questions.
- Knowledge access goes through the **MCP server**, not a direct function call.
- A conversation has **memory** (follow-ups work) and runs are **traced** (LangSmith).
- Tests + eval run via **pre-commit/CI** so quality can't silently regress.

## Target architecture (the Mid-tier system)

```
Learner question
   │
[ LangGraph state machine ]  ← conversation memory (Postgres checkpointer)
   │
 retrieve (via MCP tool)
   │
   ├─ hybrid search: dense (pgvector) + keyword (BM25)  → ~20 candidates
   └─ rerank (Cohere/Voyage) → top 4–5 most relevant
   │
 grade (corrective RAG): confident → answer | weak → rewrite+retry | none → refuse
   │
 answer (Claude Sonnet): answer ONLY from context, cite (Module > Lesson)
   │
 response + sources   →  logged/traced (LangSmith)
```

Everything above is judged continuously by the **eval harness**, and the **MCP
server** is the seam that makes the knowledge layer reusable and swappable.

## The build, component by component

1. **Eval harness** *(in progress)* — the measuring stick. Loads `tests/eval_set.yaml`,
   scores hit-rate@k + MRR, checks citations/refusals. Everything else is judged
   against it, so it's first. Deterministic core; LLM-as-judge optional later.
2. **Chunker upgrades** — merge heading-only sections into their next section (kills
   the ~8% tiny chunks); prepend each chunk's `module > lesson` breadcrumb before
   embedding; re-ingest. Measure the gain.
3. **Threshold split** — separate the refusal floor (answer at all?) from a higher
   context floor (which chunks enter the prompt and get cited). Fixes citation leakage.
4. **Reranker** — retrieve ~20 candidates, rerank (Cohere/Voyage), keep top 4–5.
   Highest-ROI precision lever; ~$0.001–0.002/query; sits behind the retrieval interface.
5. **Hybrid search** — add keyword/BM25 alongside dense vectors so exact terms
   (service names, commands, acronyms) aren't missed; merge + rerank.
6. **MCP server** *(the Pláticas deliverable)* — wrap retrieval (and future structured
   tools) as MCP tools the agent calls. Turns "how would you design the MCP server?"
   into working code; makes the knowledge layer reusable across app + embed.
7. **Consolidate on the graph** — make the LangGraph graph the single source of truth;
   CLI becomes a thin wrapper. Fully exercise the corrective-RAG loop.
8. **Memory** — swap the in-memory checkpointer for LangGraph **PostgresSaver** so
   conversations persist (follow-ups work) and you get an audit trail. Same Supabase DB.
9. **Cost hygiene** — Anthropic **prompt caching** on the static system prompt (90% off
   repeated input); route cheap steps (query rewrite) to **Haiku**.
10. **Quality gates** — **LangSmith** tracing to see what each run retrieved/said;
    **pre-commit** running fast tests before each commit; optional GitHub Actions to run
    the eval on each PR.
11. **(Edge of Mid) Minimal API** — a small **FastAPI** `/chat` endpoint as the clean
    front door for a later web UI or embed.

## Build order

| # | Step | Depends on | Why this order |
|---|---|---|---|
| 1 | Eval harness | — | Can't measure improvements without it |
| 2 | Chunker fixes + re-ingest | 1 | Measure the gain |
| 3 | Threshold split | 1 | Free citation fix, measure |
| 4 | Reranker | 1 | Biggest precision win, measure |
| 5 | Hybrid search | 4 | Pairs with rerank |
| 6 | MCP server | 4–5 | Wrap the finished retrieval |
| 7 | Consolidate on graph | 6 | Single path over MCP |
| 8 | Memory (PostgresSaver) | 7 | Add to the graph |
| 9 | Caching + routing | 7 | Cost tuning |
| 10 | LangSmith + pre-commit/CI | 1 | Lock in quality |
| 11 | FastAPI endpoint | 7 | Front door |

**Rule:** after every step, run the eval and watch the numbers move.

## How a question flows through the finished system

1. A learner asks a question. **LangGraph** picks it up, loading **conversation
   memory** for that thread from Postgres.
2. The engine calls the **MCP server's** retrieval tool — agnostic to how knowledge is stored.
3. Retrieval runs **hybrid search** over **pgvector**, pulls ~20 candidates, and a
   **reranker** sharpens to the 4–5 most relevant chunks, each with its source.
4. The **grade** step checks confidence: strong → answer; weak → rewrite + retry once;
   nothing → refuse. Only chunks above the **context floor** enter the prompt.
5. **Claude Sonnet** answers strictly from those chunks and cites `Module > Lesson`.
6. Response + sources return, the turn saves to memory, the run is **traced (LangSmith)**.
7. The **eval harness** + **pre-commit/CI** guarantee none of this silently regresses.

Elevator version: *memory → MCP → hybrid retrieve → rerank → grade → grounded cited
answer → traced + measured.*

## How this maps to Transfer Pláticas

- **MCP server** = their named core component and deliverable.
- **Hybrid + rerank + grounding** = their hallucination/validation requirement.
- **Eval harness** = their "how would you test for hallucinations / validate" answer, in code.
- **Memory + audit** = their conversation-memory + logging/audit requirement.
- **Persona = config** = their "dozens of personas without rebuilding" requirement.
- Pláticas-only extras on top of this spine: structured-data tools (College
  Scorecard/ASSIST), bilingual, voice, red-flag alerts, multi-tenant.

## Acceptance checklist

- [ ] Eval: hit-rate@6 ≥ 0.90, MRR ≥ 0.80, citation precision ≥ 0.90, refusals 100%.
- [ ] Retrieval is hybrid + reranked.
- [ ] Knowledge served through the MCP server.
- [ ] Single graph path; corrective-RAG loop exercised.
- [ ] Durable conversation memory (PostgresSaver).
- [ ] Prompt caching + Haiku routing on.
- [ ] LangSmith tracing + pre-commit/CI enforced.
- [ ] FastAPI `/chat` endpoint live.

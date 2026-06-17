# Cumulus

A culturally- and personally-responsive cloud-architecture mentor bot: a grounded
RAG + agent system that answers strictly from a real cloud curriculum and cites the
exact lesson behind every claim. See `docs/ARCHITECTURE.md` for the design and the
reasoning behind each choice.

## Conventions

- Python 3.12, `uv` for dependencies.
- Keep retrieval behind `app/retrieval.py` so the vector store stays swappable.
- Personas are config in `app/personas/`, never hard-coded into the engine.
- Ground every answer and cite sources; prefer "I don't have enough information
  from the curriculum" over guessing.

## Commits

- Concise, present tense, no em dashes.
- Small, logical commits over one large one; push after meaningful units of work.

## Testing

Write tests as you build, not after. Two layers:
- Unit tests (pytest) for plumbing: chunking produces the expected structure,
  retrieval returns results, config loads.
- An eval set: a small list of question -> expected source/lesson pairs
  (`tests/eval_set.yaml`), to measure retrieval quality and catch hallucination
  regressions. Re-run it after any change to chunking, retrieval, or prompts.

Run the eval with `uv run python -m tests.eval` (prints hit-rate@k, MRR, citation
precision, refusals). Its pytest wrapper is marked `@pytest.mark.eval` because it
makes API + DB calls; the pre-commit hook runs `pytest -m "not eval"` so only the
fast offline tests gate commits. Run the tests before committing.

## Retrieval thresholds

Two separate cosine floors in `app/config.py`: `RELEVANCE_FLOOR` (refuse below it)
and a higher `CONTEXT_FLOOR` (only chunks above it enter the cited context). Each
chunk is embedded with a `module > lesson` breadcrumb prepended so the vector
carries its place in the curriculum, not just the local prose.

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

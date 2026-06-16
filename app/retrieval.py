"""Retrieval = the RAG core. pgvector similarity search behind one function.

This is intentionally tiny: the whole point of pgvector is that retrieval is
plain SQL against the database we already run. Swapping to Qdrant later only
means rewriting THIS file — nothing upstream changes.
"""
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    content: str
    module: str | None
    lesson: str | None
    source_url: str | None
    score: float


def embed_query(text: str) -> list[float]:
    """Embed a single query string. TODO: wire to Voyage/OpenAI per config."""
    raise NotImplementedError("Wire up the embedding model (see ingest/embed.py).")


def search_curriculum(query: str, module: str | None = None, k: int = 6) -> list[RetrievedChunk]:
    """Hybrid-ready dense retrieval over curriculum chunks.

    TODO:
      1. q = embed_query(query)
      2. SELECT content, module, lesson, source_url,
                1 - (embedding <=> %(q)s) AS score
         FROM chunks
         [WHERE module = %(module)s]
         ORDER BY embedding <=> %(q)s
         LIMIT %(k)s
      3. (Phase 2) add a BM25/keyword arm and merge for hybrid search.
      4. Return results WITH source_url so the agent can cite every claim.
    """
    raise NotImplementedError

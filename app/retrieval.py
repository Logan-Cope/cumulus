"""Retrieval = the RAG core. pgvector similarity search behind one function.

This is intentionally tiny: the whole point of pgvector is that retrieval is
plain SQL against the database we already run. Swapping to Qdrant later only
means rewriting THIS file — nothing upstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import psycopg
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg import register_vector

from app.config import DATABASE_URL, EMBEDDING_MODEL, RERANK_CANDIDATES, RETRIEVAL_K
from app.rerank import rerank, rerank_enabled


@dataclass
class RetrievedChunk:
    content: str
    module: str | None
    lesson: str | None
    source_url: str | None
    score: float


@lru_cache(maxsize=1)
def _embedder() -> OpenAIEmbeddings:
    # Same model used at ingestion (ingest/embed.py) so query and document
    # vectors live in the same space.
    return OpenAIEmbeddings(model=EMBEDDING_MODEL)


def embed_query(text: str) -> list[float]:
    """Embed a single query string with the configured embedding model."""
    return _embedder().embed_query(text)


def search_curriculum(
    query: str, module: str | None = None, k: int = RETRIEVAL_K
) -> list[RetrievedChunk]:
    """Nearest-neighbour retrieval over curriculum chunks, optionally reranked.

    Returns chunks WITH module/lesson/source_url so the agent can cite every
    claim. `score` is cosine similarity in [0, 1] (1 - cosine distance) and is
    preserved even after reranking, so the cosine-calibrated floors still apply.

    When reranking is enabled we pull a wider candidate pool first, then let the
    reranker pick the top k; otherwise we return the top k directly.
    """
    q = embed_query(query)
    limit = max(k, RERANK_CANDIDATES) if rerank_enabled() else k

    sql = """
        SELECT content, module, lesson, source_url,
               1 - (embedding <=> %(q)s) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(limit)s
    """.format(where="WHERE module = %(module)s" if module else "")

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # numpy array so pgvector's adapter sends a `vector`, not a float[].
            cur.execute(
                sql,
                {"q": np.asarray(q, dtype=np.float32), "module": module, "limit": limit},
            )
            rows = cur.fetchall()

    candidates = [
        RetrievedChunk(content=c, module=m, lesson=les, source_url=s, score=float(score))
        for (c, m, les, s, score) in rows
    ]
    return rerank(query, candidates, top_n=k) if rerank_enabled() else candidates

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

from app.config import DATABASE_URL, EMBEDDING_MODEL, RETRIEVAL_K


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
    """Dense nearest-neighbour retrieval over curriculum chunks.

    Returns chunks WITH module/lesson/source_url so the agent can cite every
    claim. `score` is cosine similarity in [0, 1] (1 - cosine distance).
    """
    q = embed_query(query)

    sql = """
        SELECT content, module, lesson, source_url,
               1 - (embedding <=> %(q)s) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(k)s
    """.format(where="WHERE module = %(module)s" if module else "")

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        with conn.cursor() as cur:
            # numpy array so pgvector's adapter sends a `vector`, not a float[].
            cur.execute(sql, {"q": np.asarray(q, dtype=np.float32), "module": module, "k": k})
            rows = cur.fetchall()

    return [
        RetrievedChunk(content=c, module=m, lesson=l, source_url=s, score=float(score))
        for (c, m, l, s, score) in rows
    ]

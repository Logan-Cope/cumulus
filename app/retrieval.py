"""Retrieval = the RAG core, behind one function.

Hybrid search: dense (pgvector cosine) + keyword (Postgres full-text), fused with
Reciprocal Rank Fusion, then optionally reranked. It's still plain SQL against the
database we already run, so swapping to a dedicated vector DB later only means
rewriting THIS file - nothing upstream changes.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import psycopg
from langchain_openai import OpenAIEmbeddings
from pgvector.psycopg import register_vector

from app.config import (
    DATABASE_URL,
    EMBEDDING_MODEL,
    HYBRID_SEARCH,
    RERANK_CANDIDATES,
    RETRIEVAL_K,
    RRF_K,
)
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
    """Hybrid retrieval over curriculum chunks, optionally reranked.

    Returns chunks WITH module/lesson/source_url so the agent can cite every
    claim. `score` is always cosine similarity in [0, 1] (1 - cosine distance),
    preserved through fusion and reranking, so the cosine-calibrated floors
    downstream stay valid regardless of the final ordering.

    Flow: dense + keyword arms each pull a candidate pool, RRF fuses them, then
    (if configured) the reranker picks the top k; otherwise we take the fused top k.
    """
    q = np.asarray(embed_query(query), dtype=np.float32)
    pool = max(k, RERANK_CANDIDATES) if (HYBRID_SEARCH or rerank_enabled()) else k

    with psycopg.connect(DATABASE_URL) as conn:
        register_vector(conn)
        dense = _dense_search(conn, q, module, pool)
        if HYBRID_SEARCH:
            keyword = _keyword_search(conn, query, q, module, pool)
            candidates = _rrf_merge([dense, keyword], RRF_K)
        else:
            candidates = dense

    if rerank_enabled():
        return rerank(query, candidates[:pool], top_n=k)
    return candidates[:k]


def _rows_to_chunks(rows) -> list[RetrievedChunk]:
    return [
        RetrievedChunk(content=c, module=m, lesson=les, source_url=s, score=float(score))
        for (c, m, les, s, score) in rows
    ]


def _dense_search(conn, q_vec, module, limit) -> list[RetrievedChunk]:
    """Vector nearest neighbours, ordered by cosine similarity."""
    sql = """
        SELECT content, module, lesson, source_url,
               1 - (embedding <=> %(q)s) AS score
        FROM chunks
        {where}
        ORDER BY embedding <=> %(q)s
        LIMIT %(limit)s
    """.format(where="WHERE module = %(module)s" if module else "")
    with conn.cursor() as cur:
        cur.execute(sql, {"q": q_vec, "module": module, "limit": limit})
        return _rows_to_chunks(cur.fetchall())


def _keyword_search(conn, query, q_vec, module, limit) -> list[RetrievedChunk]:
    """Full-text keyword matches, ordered by ts_rank. Still carries each chunk's
    cosine `score` (computed here) so fused results keep a valid floor score."""
    where = "to_tsvector('english', content) @@ websearch_to_tsquery('english', %(query)s)"
    if module:
        where += " AND module = %(module)s"
    sql = f"""
        SELECT content, module, lesson, source_url,
               1 - (embedding <=> %(q)s) AS score
        FROM chunks
        WHERE {where}
        ORDER BY ts_rank(to_tsvector('english', content),
                         websearch_to_tsquery('english', %(query)s)) DESC
        LIMIT %(limit)s
    """
    with conn.cursor() as cur:
        cur.execute(sql, {"q": q_vec, "query": query, "module": module, "limit": limit})
        return _rows_to_chunks(cur.fetchall())


def _rrf_merge(lists: list[list[RetrievedChunk]], rrf_k: int) -> list[RetrievedChunk]:
    """Reciprocal Rank Fusion: combine ranked lists without comparing their score
    scales. A chunk's fused score is sum(1 / (rrf_k + rank)) across the lists it
    appears in; chunks found by both arms rise. Dedup by content."""
    fused: dict[str, list] = {}
    for ranked in lists:
        for rank, chunk in enumerate(ranked):
            entry = fused.setdefault(chunk.content, [0.0, chunk])
            entry[0] += 1.0 / (rrf_k + rank + 1)
    ordered = sorted(fused.values(), key=lambda e: e[0], reverse=True)
    return [chunk for _, chunk in ordered]

"""Optional reranking stage behind retrieval.

Vector similarity is recall-oriented: it gets the right neighbourhood but often
mis-orders within it. A cross-encoder reranker (Cohere or Voyage) reads the query
and each candidate together and scores true relevance, so the best chunks rise to
the top. The flow is: pull ~20 candidates by cosine, rerank, keep the top few.

Config-driven and fail-open: with no provider/key configured (or the SDK absent,
or the API erroring), this is a transparent no-op that returns the candidates in
their original order. Reranking only ever reorders/trims candidates - each chunk
keeps its original cosine score, so the cosine-calibrated refusal/context floors
downstream stay valid.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from app.config import (
    COHERE_API_KEY,
    RERANK_MODEL,
    RERANK_PROVIDER,
    VOYAGE_API_KEY,
)

if TYPE_CHECKING:
    from app.retrieval import RetrievedChunk

_DEFAULT_MODEL = {"cohere": "rerank-english-v3.0", "voyage": "rerank-2"}


def rerank_enabled() -> bool:
    return RERANK_PROVIDER in ("cohere", "voyage")


def rerank(query: str, chunks: list[RetrievedChunk], top_n: int) -> list[RetrievedChunk]:
    """Reorder candidates by cross-encoder relevance and keep the top_n. No-op
    (returns the first top_n unchanged) if reranking is off or anything fails."""
    if not rerank_enabled() or len(chunks) <= 1:
        return chunks[:top_n]
    try:
        order = _provider_rank(query, [c.content for c in chunks])
    except Exception:
        return chunks[:top_n]  # fail open: dense order is still useful
    return [chunks[i] for i in order][:top_n]


def _provider_rank(query: str, docs: list[str]) -> list[int]:
    """Return document indices best-first, per the configured provider."""
    if RERANK_PROVIDER == "cohere":
        return _cohere_rank(query, docs)
    if RERANK_PROVIDER == "voyage":
        return _voyage_rank(query, docs)
    raise RuntimeError(f"unknown rerank provider: {RERANK_PROVIDER!r}")


def _cohere_rank(query: str, docs: list[str]) -> list[int]:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY not set")
    import cohere  # lazy: only needed when this provider is active

    client = cohere.Client(COHERE_API_KEY)
    res = client.rerank(
        model=RERANK_MODEL or _DEFAULT_MODEL["cohere"],
        query=query,
        documents=docs,
        top_n=len(docs),
    )
    return [r.index for r in res.results]


def _voyage_rank(query: str, docs: list[str]) -> list[int]:
    if not VOYAGE_API_KEY:
        raise RuntimeError("VOYAGE_API_KEY not set")
    import voyageai  # lazy: only needed when this provider is active

    client = voyageai.Client(api_key=VOYAGE_API_KEY)
    res = client.rerank(query, docs, model=RERANK_MODEL or _DEFAULT_MODEL["voyage"], top_k=len(docs))
    return [r.index for r in res.results]

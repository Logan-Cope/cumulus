"""Reranker logic, offline. Reordering is verified with a mocked provider so no
API key is required; the no-op and fail-open paths are the safety net that keeps
retrieval working when no reranker is configured.
"""
import app.rerank as rk
from app.retrieval import RetrievedChunk


def _chunks(n):
    # Descending cosine score, like a real candidate pool.
    return [RetrievedChunk(f"doc {i}", "Mod", f"Lesson {i}", "src", 0.9 - i * 0.1) for i in range(n)]


def test_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(rk, "rerank_enabled", lambda: False)
    chunks = _chunks(5)
    assert rk.rerank("q", chunks, top_n=3) == chunks[:3]


def test_reorders_and_trims_when_enabled(monkeypatch):
    monkeypatch.setattr(rk, "rerank_enabled", lambda: True)
    # Pretend the cross-encoder reverses the ranking (last candidate most relevant).
    monkeypatch.setattr(rk, "_provider_rank", lambda q, docs: list(reversed(range(len(docs)))))
    chunks = _chunks(5)
    out = rk.rerank("q", chunks, top_n=3)
    assert [c.content for c in out] == ["doc 4", "doc 3", "doc 2"]
    # Original cosine scores are preserved (floors stay valid).
    assert out[0].score == chunks[4].score


def test_fail_open_on_provider_error(monkeypatch):
    monkeypatch.setattr(rk, "rerank_enabled", lambda: True)

    def boom(q, docs):
        raise RuntimeError("api down")

    monkeypatch.setattr(rk, "_provider_rank", boom)
    chunks = _chunks(5)
    assert rk.rerank("q", chunks, top_n=3) == chunks[:3]


def test_default_disabled_in_this_env():
    # No provider configured -> retrieval stays pure dense vectors.
    assert rk.rerank_enabled() is False

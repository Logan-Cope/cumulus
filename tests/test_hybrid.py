"""Hybrid search: RRF fusion logic (offline) plus an end-to-end check that the
keyword arm surfaces exact terms.
"""
import pytest

from app.retrieval import RetrievedChunk, _rrf_merge


def _c(content, score=0.5):
    return RetrievedChunk(content, "Mod", "Lesson", "src", score)


def test_rrf_rewards_consensus_and_dedups():
    a, b = _c("A"), _c("B")
    dense = [a, b]      # dense prefers A
    keyword = [b]       # keyword only found B
    merged = _rrf_merge([dense, keyword], rrf_k=60)
    # B appears in both arms, so consensus lifts it above A (dense-only top hit).
    assert merged[0].content == "B"
    # Dedup by content: B is not duplicated despite appearing in both lists.
    assert sorted(m.content for m in merged) == ["A", "B"]


def test_rrf_single_list_preserves_order():
    chunks = [_c("A"), _c("B"), _c("C")]
    assert [m.content for m in _rrf_merge([chunks], rrf_k=60)] == ["A", "B", "C"]


@pytest.mark.eval
def test_keyword_arm_surfaces_exact_term():
    """An exact CLI term should retrieve the CLI Commands lesson via hybrid."""
    from app.retrieval import search_curriculum

    chunks = search_curriculum("What does the pwd command do in Linux?")
    assert any((c.lesson or "") == "CLI Commands" for c in chunks)
    assert all(c.source_url and c.module for c in chunks)

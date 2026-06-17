"""The refusal contract: when retrieval finds nothing close enough, Cumulus
refuses instead of guessing — and does so without spending a model call.
Retrieval is monkeypatched so this stays offline and fast.
"""
import app.answer as answer
from app.answer import REFUSAL, answer_question
from app.retrieval import RetrievedChunk


def _chunk(score):
    return RetrievedChunk("body", "Week 1", "Lesson", "src", score)


def test_refuses_when_nothing_retrieved(monkeypatch):
    monkeypatch.setattr(answer, "search_curriculum", lambda *a, **k: [])
    ans = answer_question("anything")
    assert ans.text == REFUSAL
    assert ans.grounded is False
    assert ans.sources == []


def test_refuses_when_top_score_below_floor(monkeypatch):
    monkeypatch.setattr(answer, "search_curriculum", lambda *a, **k: [_chunk(0.2)])
    # If the gate failed and we called the model, this would raise (no real call).
    monkeypatch.setattr(
        answer, "_llm", lambda: (_ for _ in ()).throw(AssertionError("model called"))
    )
    ans = answer_question("off topic")
    assert ans.text == REFUSAL
    assert ans.grounded is False

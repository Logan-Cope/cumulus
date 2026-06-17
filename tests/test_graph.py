"""Corrective-RAG routing is deterministic and score-based — test it offline
without touching the model or the database.
"""
from app.config import RELEVANCE_FLOOR
from app.graph import CONFIDENT, MAX_REQUERY, build_graph, grade_node
from app.retrieval import RetrievedChunk


def _state(score, attempts=0):
    chunks = [RetrievedChunk("body", "Week 1", "Lesson", "src", score)] if score else []
    return {"retrieved": chunks, "attempts": attempts}


def test_confident_retrieval_answers():
    assert grade_node(_state(CONFIDENT + 0.01))["verdict"] == "answer"


def test_nothing_close_refuses():
    assert grade_node(_state(RELEVANCE_FLOOR - 0.01))["verdict"] == "refuse"
    assert grade_node(_state(0))["verdict"] == "refuse"


def test_weak_band_requeries_then_answers_when_capped():
    weak = (RELEVANCE_FLOOR + CONFIDENT) / 2
    assert grade_node(_state(weak, attempts=0))["verdict"] == "requery"
    # Once the retry budget is spent, fall through to answering on weak evidence.
    assert grade_node(_state(weak, attempts=MAX_REQUERY))["verdict"] == "answer"


def test_graph_compiles_with_expected_nodes():
    graph = build_graph()
    nodes = graph.get_graph().nodes
    for name in ("retrieve", "grade", "requery", "answer", "refuse"):
        assert name in nodes

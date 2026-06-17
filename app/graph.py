"""The conversation engine: a LangGraph state machine (the deterministic spine).

Nodes are phases; edges are flow. The model supplies the language; the graph
supplies the structure. Memory/audit come from the Postgres checkpointer.

Flow (corrective RAG):

    START -> retrieve -> grade --good--> answer -> END
                           |  ^
                       weak|  | (capped retry)
                           v  |
                        requery
                           |
                       none|--------> refuse -> END

`grade` is score-based and deterministic: confident retrieval answers, a weak
band rewrites the query and retries once, and nothing close enough refuses.
"""
from __future__ import annotations

from typing import TypedDict

from langchain_anthropic import ChatAnthropic
from langgraph.graph import END, START, StateGraph

from app.answer import REFUSAL, _format_context, _llm, _prompt
from app.config import CHEAP_MODEL, RELEVANCE_FLOOR, RETRIEVAL_K
from app.personas.cloud_mentor import CLOUD_MENTOR
from app.retrieval import search_curriculum

# Above CONFIDENT -> answer; below RELEVANCE_FLOOR -> refuse; in between -> one
# capped query rewrite. In-curriculum questions score ~0.65+, so the band mostly
# catches awkwardly-phrased questions a rewrite can rescue.
CONFIDENT = float(max(RELEVANCE_FLOOR + 0.15, 0.5))
MAX_REQUERY = 1


class State(TypedDict, total=False):
    question: str           # the (possibly rewritten) query driving retrieval
    original: str           # the learner's original question, for generation
    messages: list          # conversation history
    retrieved: list         # chunks pulled this turn (with sources)
    persona: str            # which persona config is active
    attempts: int           # query rewrites used so far (CRAG cap)
    verdict: str            # grade decision: answer | requery | refuse
    answer: str
    sources: list
    grounded: bool


def retrieve_node(state: State) -> State:
    chunks = search_curriculum(state["question"], k=RETRIEVAL_K)
    return {"retrieved": chunks}


def grade_node(state: State) -> State:
    """Corrective-RAG: judge retrieval quality; re-query or refuse if weak (capped)."""
    chunks = state.get("retrieved") or []
    top = chunks[0].score if chunks else 0.0
    if top >= CONFIDENT:
        verdict = "answer"
    elif top >= RELEVANCE_FLOOR:
        verdict = "answer" if state.get("attempts", 0) >= MAX_REQUERY else "requery"
    else:
        verdict = "refuse"
    return {"verdict": verdict}


def requery_node(state: State) -> State:
    """Rewrite the question to improve retrieval, then loop back (capped)."""
    prompt = (
        "Rewrite this cloud-curriculum question to use clearer, more standard "
        "technical terms for search. Return only the rewritten question.\n\n"
        f"{state['original']}"
    )
    new_q = (
        ChatAnthropic(model=CHEAP_MODEL, temperature=0, max_tokens=128)
        .invoke(prompt)
        .content.strip()
    )
    return {"question": new_q, "attempts": state.get("attempts", 0) + 1}


def answer_node(state: State) -> State:
    """Call the chat model with retrieved context + persona prompt; cite sources."""
    chunks = state["retrieved"]
    messages = [
        ("system", CLOUD_MENTOR.system_prompt),
        ("human", _prompt(state["original"], _format_context(chunks))),
    ]
    text = _llm().invoke(messages).content.strip()
    grounded = REFUSAL.rstrip(".") not in text
    return {"answer": text, "sources": chunks if grounded else [], "grounded": grounded}


def refuse_node(state: State) -> State:
    return {"answer": REFUSAL, "sources": [], "grounded": False}


def build_graph(checkpointer=None):
    g = StateGraph(State)
    g.add_node("retrieve", retrieve_node)
    g.add_node("grade", grade_node)
    g.add_node("requery", requery_node)
    g.add_node("answer", answer_node)
    g.add_node("refuse", refuse_node)

    g.add_edge(START, "retrieve")
    g.add_edge("retrieve", "grade")
    g.add_conditional_edges(
        "grade",
        lambda s: s["verdict"],
        {"answer": "answer", "requery": "requery", "refuse": "refuse"},
    )
    g.add_edge("requery", "retrieve")
    g.add_edge("answer", END)
    g.add_edge("refuse", END)
    # PostgresSaver in prod gives durable conversation memory + audit for free.
    return g.compile(checkpointer=checkpointer)


def ask(question: str) -> State:
    """Run one turn through the graph and return the final state."""
    graph = build_graph()
    return graph.invoke(
        {"question": question, "original": question, "attempts": 0}
    )


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "What is Terraform?"
    result = ask(q)
    print("\n" + result["answer"] + "\n")
    seen = set()
    for c in result.get("sources", []):
        key = (c.module, c.lesson)
        if key not in seen:
            seen.add(key)
            print(f"  - {c.module} - {c.lesson}  (similarity {c.score:.2f})")

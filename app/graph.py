"""The conversation engine: a LangGraph state machine (the deterministic spine).

Nodes are phases; edges are flow. The model supplies the language; the graph
supplies the structure. Memory/audit come from the Postgres checkpointer.
"""
from __future__ import annotations
from typing import TypedDict

# from langgraph.graph import StateGraph, START, END
# from langgraph.checkpoint.postgres import PostgresSaver   # durable conversation memory


class State(TypedDict, total=False):
    messages: list          # conversation history
    retrieved: list         # chunks pulled this turn (with sources)
    persona: str            # which persona config is active


def retrieve_node(state: State) -> State:
    """Call search_curriculum() and attach grounded, sourced context."""
    raise NotImplementedError


def answer_node(state: State) -> State:
    """Call the chat model with retrieved context + persona prompt; cite sources."""
    raise NotImplementedError


def grade_node(state: State) -> State:
    """Corrective-RAG: judge retrieval quality; re-query or refuse if weak (capped)."""
    raise NotImplementedError


def build_graph(checkpointer=None):
    """Wire the nodes into a compiled graph.

    TODO:
      g = StateGraph(State)
      g.add_node("retrieve", retrieve_node)
      g.add_node("grade", grade_node)        # CRAG loop (borrowed from AIE cert module 02)
      g.add_node("answer", answer_node)
      g.add_edge(START, "retrieve")
      g.add_edge("retrieve", "grade")
      g.add_conditional_edges("grade", ...)  # good -> answer ; weak -> retrieve (capped) ; none -> refuse
      g.add_edge("answer", END)
      return g.compile(checkpointer=checkpointer)   # PostgresSaver in prod
    """
    raise NotImplementedError

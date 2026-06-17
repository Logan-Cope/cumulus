"""MCP server = the knowledge bridge. Exposes retrieval (and later structured
tools) as callable tools. The agent calls a tool and gets back grounded data;
it neither knows nor cares whether the data came from a vector store or an API.

For Cumulus the curriculum is unstructured, so we start with one RAG tool.
For Transfer Placticas this is where structured tools (College Scorecard,
ASSIST.org) would live alongside the RAG tool.

Run it (stdio transport, the MCP default):

    uv run python -m app.mcp_server
"""
from __future__ import annotations

import asyncio
import json

from mcp.server.fastmcp import FastMCP

from app.config import RETRIEVAL_K
from app.retrieval import RetrievedChunk
from app.retrieval import search_curriculum as _retrieve

mcp = FastMCP("cumulus-curriculum")

TOOL_NAME = "search_curriculum"


@mcp.tool()
def search_curriculum(query: str, module: str | None = None) -> list[dict]:
    """Search the cloud-engineering curriculum for lessons relevant to a question.

    Returns the most relevant chunks, each WITH its source metadata
    (module, lesson, source_url) so the caller can cite the exact lesson behind
    every claim. Optionally restrict the search to a single module.
    """
    chunks = search_curriculum_chunks(query, module=module)
    return [_to_dict(c) for c in chunks]


def search_curriculum_chunks(
    query: str, module: str | None = None, k: int = RETRIEVAL_K
) -> list[RetrievedChunk]:
    """Plain retrieval the tool wraps; kept separate so callers can get typed
    RetrievedChunk objects without going through MCP serialization."""
    return _retrieve(query, module=module, k=k)


def _to_dict(c: RetrievedChunk) -> dict:
    return {
        "content": c.content,
        "module": c.module,
        "lesson": c.lesson,
        "source_url": c.source_url,
        "score": round(c.score, 4),
    }


def _from_dict(d: dict) -> RetrievedChunk:
    return RetrievedChunk(
        content=d["content"],
        module=d.get("module"),
        lesson=d.get("lesson"),
        source_url=d.get("source_url"),
        score=float(d.get("score", 0.0)),
    )


def retrieve_via_mcp(
    query: str, module: str | None = None, k: int = RETRIEVAL_K
) -> list[RetrievedChunk]:
    """Retrieve through the MCP tool dispatch (in-process), returning typed chunks.

    This is how the LangGraph engine reaches the knowledge layer: it exercises
    the real MCP tool rather than calling retrieval directly. If anything about
    the tool dispatch fails, fall back to direct retrieval so the answer loop
    never breaks over the transport.
    """
    try:
        payload = asyncio.run(mcp.call_tool(TOOL_NAME, {"query": query, "module": module}))
        return [_from_dict(d) for d in _parse_tool_result(payload)]
    except Exception:
        return search_curriculum_chunks(query, module=module, k=k)


def _parse_tool_result(payload) -> list[dict]:
    """FastMCP.call_tool returns (content_blocks, structured). Pull the list of
    chunk dicts out of whichever the running mcp version populated."""
    structured = payload[1] if isinstance(payload, tuple) and len(payload) > 1 else None
    if isinstance(structured, dict):
        # Structured output wraps a bare list under "result".
        if isinstance(structured.get("result"), list):
            return structured["result"]
        for v in structured.values():
            if isinstance(v, list):
                return v
    blocks = payload[0] if isinstance(payload, tuple) else payload
    for block in blocks or []:
        text = getattr(block, "text", None)
        if text:
            data = json.loads(text)
            return data if isinstance(data, list) else data.get("result", [])
    return []


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()

"""MCP server tests: the curriculum retrieval tool must return sourced chunks,
both in-process and over a real stdio MCP client session, and the graph's
MCP-backed retrieval must still work end to end.

Marked `eval` (real DB + embedding calls + a subprocess), so the pre-commit hook
(`pytest -m "not eval"`) skips it. Run with `uv run pytest -m eval -q`.
"""
import asyncio
import json
import sys

import pytest

from app.mcp_server import TOOL_NAME, mcp, retrieve_via_mcp

pytestmark = pytest.mark.eval

QUERY = "What is Terraform?"


def _assert_sourced(chunks: list[dict]):
    assert chunks, "tool returned no chunks"
    for c in chunks:
        assert c["content"].strip()
        assert c["module"] and c["lesson"]  # every claim is attributable
        assert "score" in c
    # The Terraform lesson should surface for this question.
    assert any("terraform" in (c["lesson"] or "").lower() for c in chunks)


def test_tool_returns_sourced_chunks_in_process():
    payload = asyncio.run(mcp.call_tool(TOOL_NAME, {"query": QUERY, "module": None}))
    structured = payload[1] if isinstance(payload, tuple) else None
    chunks = structured["result"] if isinstance(structured, dict) else None
    _assert_sourced(chunks)


def test_tool_over_stdio_mcp_client():
    """Spawn the server as a subprocess and call the tool through a real MCP
    client session, exactly as an external agent would."""
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    async def _run():
        params = StdioServerParameters(
            command=sys.executable, args=["-m", "app.mcp_server"]
        )
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await session.list_tools()
                assert TOOL_NAME in {t.name for t in tools.tools}
                result = await session.call_tool(TOOL_NAME, {"query": QUERY})
                assert not result.isError
                if result.structuredContent and "result" in result.structuredContent:
                    return result.structuredContent["result"]
                return json.loads(result.content[0].text)

    _assert_sourced(asyncio.run(_run()))


def test_retrieve_via_mcp_returns_typed_chunks():
    chunks = retrieve_via_mcp(QUERY)
    assert chunks and all(c.source_url and c.module and c.lesson for c in chunks)
    assert any("terraform" in (c.lesson or "").lower() for c in chunks)


def test_full_answer_still_grounded_and_cited():
    """The MCP wiring must not break the grounded, cited answer loop."""
    from app.graph import ask

    result = ask(QUERY)
    assert result["grounded"] is True
    assert result["sources"]
    assert all(s.module and s.lesson for s in result["sources"])

"""MCP server = the knowledge bridge. Exposes retrieval (and later structured
tools) as callable tools. The agent calls a tool and gets back grounded data;
it neither knows nor cares whether the data came from a vector store or an API.

For Cumulus the curriculum is unstructured, so we start with one RAG tool.
For Transfer Placticas this is where structured tools (College Scorecard,
ASSIST.org) would live alongside the RAG tool.
"""
# from mcp.server.fastmcp import FastMCP
# from app.retrieval import search_curriculum

# mcp = FastMCP("cumulus-curriculum")


# @mcp.tool()
def search_curriculum_tool(query: str, module: str | None = None) -> list[dict]:
    """Search the cloud curriculum for relevant lessons. Returns chunks WITH sources."""
    raise NotImplementedError


if __name__ == "__main__":
    # mcp.run()
    pass

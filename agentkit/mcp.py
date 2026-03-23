"""MCP server builder — auto-generate an MCP server from a Coordinator."""

import uuid

from .coordinator import Coordinator
from .messages import AgentContext


class MCPServerBuilder:
    """
    Build a FastMCP server from an agentkit Coordinator.

    Usage:
        builder = MCPServerBuilder(coordinator, name="my-app")
        mcp = builder.build()
        mcp.run()
    """

    def __init__(self, coordinator: Coordinator, name: str):
        self.coordinator = coordinator
        self.name = name

    def build(self):
        from fastmcp import FastMCP

        mcp = FastMCP(self.name)
        coordinator = self.coordinator

        @mcp.tool()
        async def query(
            question: str,
            context_key: str = "",
            session_id: str = "",
        ) -> str:
            """Ask a question using the multi-agent pipeline."""
            ctx = AgentContext(
                session_id=session_id or uuid.uuid4().hex,
                user_query=question,
                metadata={"context_key": context_key} if context_key else {},
            )
            result = await coordinator.run(ctx)
            return result.get("answer", "No answer generated.")

        return mcp


def run_server():
    """Entry point for `agentkit-mcp` CLI command. Override in your app."""
    raise NotImplementedError(
        "To run the MCP server, build a Coordinator and use MCPServerBuilder.build().run()"
    )

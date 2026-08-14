import json
import logging

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent

from mcp2_orchestrator.settings import settings

logger = logging.getLogger(__name__)


class Mcp1Client:
    """Thin async wrapper around the MCP 1 HTTP/SSE server."""

    def __init__(self):
        self._mcp1_url = settings.mcp1_url

    async def search(self, query: str, top_k: int = 3) -> list[dict]:
        logger.info("Calling mcp1 search  query=%r  top_k=%d", query, top_k)
        async with sse_client(f"{self._mcp1_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(
                    "search", {"query": query, "top_k": top_k}
                )
                content = result.content[0]
                if not isinstance(content, TextContent):
                    raise RuntimeError(
                        f"Expected TextContent from search, got {type(content)}"
                    )
                return json.loads(content.text)

    async def list_documents(self) -> list[dict]:
        async with sse_client(f"{self._mcp1_url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("list_documents", {})
                content = result.content[0]
                if not isinstance(content, TextContent):
                    raise RuntimeError(
                        f"Expected TextContent from list_documents, got {type(content)}"
                    )
                return json.loads(content.text)

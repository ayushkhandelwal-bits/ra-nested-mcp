import json
import logging

from mcp import ClientSession
from mcp.client.sse import sse_client
from mcp.types import TextContent

from mcp2_orchestrator.settings import settings

logger = logging.getLogger(__name__)


class Mcp3Client:
    """Thin async wrapper around the MCP 3 (Analytics) HTTP/SSE server.

    Unlike Mcp1Client, this exposes a single generic call_tool() rather than
    one method per tool, since MCP 3 has ~11 tools and they're all invoked
    the same way — the tool name and arguments come straight from the LLM's
    function call.
    """

    def __init__(self):
        self._url = settings.mcp3_url

    async def call_tool(self, name: str, arguments: dict) -> dict | list:
        logger.info("Calling mcp3 tool=%r  args=%r", name, arguments)
        async with sse_client(f"{self._url}/sse") as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                content = result.content[0]
                if not isinstance(content, TextContent):
                    raise RuntimeError(f"Expected TextContent from {name}, got {type(content)}")
                return json.loads(content.text)

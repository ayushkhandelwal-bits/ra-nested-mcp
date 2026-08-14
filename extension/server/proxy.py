"""
stdio <-> HTTP/SSE bridge.

Claude Desktop spawns this script (stdio). It connects to the MCP 2
orchestrator server over HTTP/SSE and forwards messages in both directions.
"""
import asyncio
import os
import sys

import anyio
from mcp.client.sse import sse_client
from mcp.server.stdio import stdio_server

MCP2_URL = os.environ.get("MCP2_URL", "http://127.0.0.1:8002")


async def forward(src, dst):
    async for message in src:
        await dst.send(message)


async def main():
    async with sse_client(f"{MCP2_URL}/sse") as (sse_read, sse_write):
        async with stdio_server() as (stdio_read, stdio_write):
            async with anyio.create_task_group() as tg:
                tg.start_soon(forward, stdio_read, sse_write)
                tg.start_soon(forward, sse_read, stdio_write)


if __name__ == "__main__":
    asyncio.run(main())
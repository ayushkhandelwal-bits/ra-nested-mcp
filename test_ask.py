import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8002/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "ask",
                {"question": "What causes revenue leakage in telecom billing?"},
            )
            for block in result.content:
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
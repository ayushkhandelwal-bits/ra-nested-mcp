import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8002/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "ask",
                {
                    "question": (
                        "Why was revenue leakage abnormally high around June 2025, "
                        "what does our RA documentation say about likely causes, "
                        "and what do you forecast for revenue leakage over the next 30 days?"
                    )
                },
            )
            for block in result.content:
                print(block.text)


if __name__ == "__main__":
    asyncio.run(main())
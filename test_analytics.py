"""Integration smoke test for MCP3 (Analytics), mirroring test_ask.py's style.
Requires the MCP3 server running: `make run-mcp3`
"""
import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8003/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for name, args in [
                ("get_kpi_metrics", {}),
                ("detect_anomalies", {"metric": "revenue_leakage", "method": "zscore"}),
                ("analyze_by_region", {}),
            ]:
                result = await session.call_tool(name, args)
                print(f"--- {name} ---")
                for block in result.content:
                    print(block.text[:500])
                print()


if __name__ == "__main__":
    asyncio.run(main())

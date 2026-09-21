"""Integration smoke test for MCP4 (Forecasting), mirroring test_ask.py's style.
Requires the MCP4 server running: `make run-mcp4`
"""
import asyncio

from mcp import ClientSession
from mcp.client.sse import sse_client


async def main():
    async with sse_client("http://localhost:8004/sse") as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            for name, args in [
                ("compare_forecast_models", {"test_days": 30}),
                ("forecast_revenue_leakage", {"steps": 30}),
            ]:
                result = await session.call_tool(name, args)
                print(f"--- {name} ---")
                for block in result.content:
                    print(block.text[:800])
                print()


if __name__ == "__main__":
    asyncio.run(main())

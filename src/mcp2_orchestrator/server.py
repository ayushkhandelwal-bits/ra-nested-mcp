import logging

from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from mcp2_orchestrator.agent import Agent
from mcp2_orchestrator.mcp1_client import Mcp1Client
from mcp2_orchestrator.mcp3_client import Mcp3Client
from mcp2_orchestrator.mcp4_client import Mcp4Client

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp_app = Server("mcp2-orchestrator")
sse_transport = SseServerTransport("/messages/")


@mcp_app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="ask",
            description=(
                "Ask a question about telecom Revenue Assurance — conceptual "
                "(leakage, reconciliation, CDRs, CRMS/RMS, ETL pipelines, CPI validation), "
                "operational/statistical (KPIs, trends, anomalies, incident impact, hypothesis "
                "tests, correlations, regional/service/network/plan segmentation), forward-looking "
                "(30-day revenue leakage forecasts), or any combination. The agent decomposes the "
                "question, calls the knowledge base and/or the analytics and forecasting data "
                "sources as needed, and synthesizes a business-friendly final answer."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "The question to answer",
                    }
                },
                "required": ["question"],
            },
        )
    ]


@mcp_app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name != "ask":
        raise ValueError(f"Unknown tool: {name}")

    question = arguments["question"]
    logger.info("ask: %r", question)
    mcp1 = Mcp1Client()
    mcp3 = Mcp3Client()
    mcp4 = Mcp4Client()
    agent = Agent(mcp1, mcp3, mcp4)
    answer = await agent.run(question)
    logger.info("answer: %s", answer)
    return [TextContent(type="text", text=answer)]


app = FastAPI()


@app.get("/sse")
async def handle_sse(request: Request):
    async with sse_transport.connect_sse(
        request.scope, request.receive, request._send
    ) as streams:
        await mcp_app.run(
            streams[0], streams[1], mcp_app.create_initialization_options()
        )


async def _messages_app(scope, receive, send):
    await sse_transport.handle_post_message(scope, receive, send)


app.mount("/messages", _messages_app)

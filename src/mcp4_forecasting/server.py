import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from mcp4_forecasting import forecasting
from mcp4_forecasting.data_access import load_daily_leakage_series

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp_app = Server("mcp4-forecasting")
sse_transport = SseServerTransport("/messages/")


@mcp_app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="forecast_revenue_leakage",
            description="Forecast daily revenue_leakage N days into the future using Holt-Winters (trend + weekly seasonality), with 80%/95% prediction intervals and a plain-language direction/limitations summary. This is a projection, not a guaranteed value.",
            inputSchema={"type": "object", "properties": {
                "steps": {"type": "integer", "description": "Number of days to forecast ahead", "default": 30},
                "intervals": {"type": "boolean", "description": "Include prediction intervals", "default": True},
            }},
        ),
        Tool(
            name="compare_forecast_models",
            description="Compare seasonal-naive baseline vs. Holt-Winters on a chronological (never random) holdout, reporting MAE/RMSE/MAPE for each and which model wins.",
            inputSchema={"type": "object", "properties": {
                "test_days": {"type": "integer", "description": "Size of the chronological holdout window in days", "default": 30},
            }},
        ),
        Tool(
            name="get_forecast_accuracy",
            description="Day-by-day accuracy (actual vs predicted, absolute error) for one model over a chronological holdout, plus aggregate MAE/RMSE/MAPE.",
            inputSchema={"type": "object", "properties": {
                "model": {"type": "string", "enum": ["holt_winters", "seasonal_naive"], "default": "holt_winters"},
                "test_days": {"type": "integer", "default": 30},
            }},
        ),
        Tool(
            name="get_forecast_intervals",
            description="Just the forecasted values with 80%/95% prediction intervals for the next N days (no narrative fields) — useful when only the numbers are needed.",
            inputSchema={"type": "object", "properties": {
                "steps": {"type": "integer", "default": 30},
            }},
        ),
    ]


_DISPATCH = {
    "forecast_revenue_leakage": lambda a: forecasting.forecast_revenue_leakage(**a),
    "compare_forecast_models": lambda a: forecasting.compare_forecast_models(**a),
    "get_forecast_accuracy": lambda a: forecasting.get_forecast_accuracy(**a),
    "get_forecast_intervals": lambda a: forecasting.get_forecast_intervals(**a),
}


@mcp_app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in _DISPATCH:
        raise ValueError(f"Unknown tool: {name}")
    logger.info("Invoking forecasting tool: %s  args=%r", name, arguments)
    try:
        result = _DISPATCH[name](arguments or {})
    except ValueError as e:
        result = {"error": str(e)}
    return [TextContent(type="text", text=json.dumps(result, default=str))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing mcp4-forecasting: preloading time series…")
    load_daily_leakage_series()
    logger.info("mcp4-forecasting ready")
    yield


app = FastAPI(lifespan=lifespan)


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


def main():
    import uvicorn
    from mcp4_forecasting.settings import settings
    uvicorn.run(app, host="0.0.0.0", port=settings.mcp4_port)

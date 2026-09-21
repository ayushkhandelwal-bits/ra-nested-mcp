import json
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool

from mcp3_analytics import analytics
from mcp3_analytics.data_access import load_daily, load_events, load_transactions

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp_app = Server("mcp3-analytics")
sse_transport = SseServerTransport("/messages/")

_DATE_ARGS = {
    "start_date": {"type": "string", "description": "ISO date (YYYY-MM-DD), inclusive. Omit for full history."},
    "end_date": {"type": "string", "description": "ISO date (YYYY-MM-DD), inclusive. Omit for full history."},
}
_METRIC_ENUM = ["revenue_leakage", "failed_cdrs", "mediation_failures", "rating_failures", "billing_mismatches", "processing_delay_avg", "duplicate_cdrs"]


@mcp_app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="get_ra_summary",
            description="High-level 'what happened' summary: KPIs, descriptive statistics, and trend direction for revenue leakage over a period.",
            inputSchema={"type": "object", "properties": {**_DATE_ARGS}},
        ),
        Tool(
            name="get_kpi_metrics",
            description="Compute the standard Revenue Assurance KPI set (leakage rate, recovery rate, failure rates, etc.) over a date range, with definitions.",
            inputSchema={"type": "object", "properties": {**_DATE_ARGS}},
        ),
        Tool(
            name="get_revenue_leakage_trend",
            description="Daily revenue_leakage time series with a rolling mean, for trend/chart analysis.",
            inputSchema={"type": "object", "properties": {
                **_DATE_ARGS,
                "rolling_window": {"type": "integer", "description": "Rolling mean window in days", "default": 7},
            }},
        ),
        Tool(
            name="detect_anomalies",
            description="Detect statistically anomalous days on a chosen metric using z-score (rolling baseline) or IQR methods. Returns date, observed vs baseline value, deviation, anomaly score, and severity for each flagged day.",
            inputSchema={"type": "object", "properties": {
                "metric": {"type": "string", "enum": _METRIC_ENUM, "default": "revenue_leakage"},
                "method": {"type": "string", "enum": ["zscore", "iqr"], "default": "zscore"},
                "window": {"type": "integer", "description": "Rolling window (days) for zscore method", "default": 14},
                "threshold": {"type": "number", "description": "|z| threshold for zscore method", "default": 3.0},
                **_DATE_ARGS,
            }},
        ),
        Tool(
            name="analyze_incident",
            description="Pre/during/post-incident analysis for a specific operational_events.csv event_id: metric shifts, a pre-vs-during hypothesis test, estimated extra leakage, and whether metrics recovered to baseline afterward.",
            inputSchema={"type": "object", "properties": {
                "event_id": {"type": "string", "description": "e.g. 'EVT003'"},
                "pre_days": {"type": "integer", "default": 14},
                "post_days": {"type": "integer", "default": 14},
            }, "required": ["event_id"]},
        ),
        Tool(
            name="run_hypothesis_test",
            description="Test whether a metric differs significantly before vs. on/after a given split_date (e.g. an incident start date). Auto-selects Welch's t-test or Mann-Whitney U based on sample size/normality. Reports statistical AND business significance separately.",
            inputSchema={"type": "object", "properties": {
                "metric": {"type": "string", "enum": _METRIC_ENUM, "default": "revenue_leakage"},
                "split_date": {"type": "string", "description": "ISO date to split before/after, e.g. an incident start_date"},
                **_DATE_ARGS,
                "alpha": {"type": "number", "default": 0.05},
            }, "required": ["split_date"]},
        ),
        Tool(
            name="get_correlation_analysis",
            description="Pearson and Spearman correlation of operational metrics (mediation/rating failures, billing mismatches, duplicates, delay, volume) against revenue_leakage. Explicitly notes correlation != causation.",
            inputSchema={"type": "object", "properties": {**_DATE_ARGS}},
        ),
        Tool(
            name="analyze_by_region",
            description="Revenue leakage, leakage rate, and failure counts broken down by region (North/South/East/West).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="analyze_by_service",
            description="Revenue leakage, leakage rate, and failure counts broken down by service_type (Voice/SMS/Data/Roaming).",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="analyze_by_network",
            description="Revenue leakage, leakage rate, and failure counts broken down by network_type (4G/5G) — e.g. to check whether leakage is concentrated in 5G.",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(
            name="analyze_by_plan",
            description="Revenue leakage, leakage rate, and failure counts broken down by plan_type (Prepaid/Postpaid).",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


_DISPATCH = {
    "get_ra_summary": lambda a: analytics.get_ra_summary(**a),
    "get_kpi_metrics": lambda a: analytics.get_kpi_metrics(**a),
    "get_revenue_leakage_trend": lambda a: analytics.get_revenue_leakage_trend(**a),
    "detect_anomalies": lambda a: analytics.detect_anomalies(**a),
    "analyze_incident": lambda a: analytics.analyze_incident(**a),
    "run_hypothesis_test": lambda a: analytics.run_hypothesis_test(**a),
    "get_correlation_analysis": lambda a: analytics.get_correlation_analysis(**a),
    "analyze_by_region": lambda a: analytics.analyze_by_region(),
    "analyze_by_service": lambda a: analytics.analyze_by_service(),
    "analyze_by_network": lambda a: analytics.analyze_by_network(),
    "analyze_by_plan": lambda a: analytics.analyze_by_plan(),
}


@mcp_app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name not in _DISPATCH:
        raise ValueError(f"Unknown tool: {name}")
    logger.info("Invoking analytics tool: %s  args=%r", name, arguments)
    try:
        result = _DISPATCH[name](arguments or {})
    except ValueError as e:
        # business-logic validation errors (bad date range, unknown event_id, etc.)
        # are returned as a structured error rather than raising, so the agent
        # can see what went wrong and retry with corrected arguments.
        result = {"error": str(e)}
    return [TextContent(type="text", text=json.dumps(result, default=str))]


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Initializing mcp3-analytics: preloading datasets…")
    load_daily()
    load_events()
    load_transactions()
    logger.info("mcp3-analytics ready")
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
    from mcp3_analytics.settings import settings
    uvicorn.run(app, host="0.0.0.0", port=settings.mcp3_port)

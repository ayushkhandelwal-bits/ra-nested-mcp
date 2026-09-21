# RA-Nested-MCP

A nested MCP (Model Context Protocol) system demonstrating MCP composition over HTTP/SSE — a server that is simultaneously a client to three other MCP servers.

This started as a fork of [Origin-Digital-LLC/nested-mcps](https://github.com/Origin-Digital-LLC/nested-mcps) (a generic "Acme Robotics" RAG demo, re-plumbed to run without Azure) adapted to **telecom Revenue Assurance**, and has since grown from a pure knowledge agent into a full **Telecom Revenue Assurance Analytics & Intelligence Platform**: structured data + statistics + forecasting sit alongside the original RAG knowledge base, orchestrated by the same agentic loop.

| | Original nested-mcps | This project |
|---|---|---|
| Knowledge base | Acme Robotics (fictional company) | Telecom Revenue Assurance concepts (leakage, CDRs, CRMS/RMS, ETL, CPI) |
| Embeddings | Azure OpenAI (`text-embedding-3-small`) | Local `sentence-transformers` (`all-MiniLM-L6-v2`) — no API key |
| Orchestrator LLM | GPT-4.1 via Azure AI Foundry | Llama 3.3 70B via Groq's free API tier (OpenAI-compatible tool calling) |
| Structured data / stats / forecasting | — | **MCP 3 (Analytics) + MCP 4 (Forecasting)**, backed by pandas/scipy/statsmodels |
| Cost to run | Requires Azure billing | **$0 — entirely free tiers, no credit card needed** |

## Architecture

```
Claude Desktop
     │  stdio
     ▼
[stdio proxy]              ← spawned by Claude Desktop, bridges stdio ↔ HTTP
     │  HTTP/SSE (:8002)
     ▼
MCP 2: Orchestrator        ← FastAPI/uvicorn. Agentic loop (Llama 3.3 70B via Groq).
     │                        Decides which of the three data sources below to call,
     │                        for none/one/many depending on the question.
     ├── HTTP/SSE (:8001) → MCP 1: Vector Store    (RAG knowledge base)
     ├── HTTP/SSE (:8003) → MCP 3: Analytics        (structured RA data: KPIs,
     │                                                anomalies, hypothesis tests,
     │                                                correlation, segmentation)
     └── HTTP/SSE (:8004) → MCP 4: Forecasting      (Holt-Winters revenue_leakage
                                                       forecast, model comparison)
```

**MCP 1 (`mcp1_vectorstore`)** — unchanged from the original. A low-level in-memory vector store: embeds 10 Revenue Assurance documents locally via `sentence-transformers` at startup, serves semantic search via numpy cosine similarity. Internal only, never exposed directly.

**MCP 2 (`mcp2_orchestrator`)** runs the agentic reasoning loop using Llama 3.3 70B via Groq's free API tier. It exposes a single `ask` tool via HTTP/SSE, decomposes questions into tasks, and calls MCP 1/3/4 as needed — the system prompt instructs it to use the *minimum* set of tools for a given question (a "what does CRMS mean" question never touches MCP 3/4; a "why did leakage spike and what's next" question uses all three). Independent tool calls within one turn are dispatched in parallel via `asyncio.gather`.

**MCP 3 (`mcp3_analytics`)** — new. Loads `cdr_transactions.csv` / `daily_ra_metrics.csv` / `operational_events.csv` (or their cleaned/aggregated Parquet equivalents) into pandas at startup and exposes 11 tools: KPI computation, descriptive stats, z-score/IQR anomaly detection, incident pre/during/post analysis, auto-selected hypothesis testing (Welch's t-test or Mann-Whitney U based on sample size/normality), Pearson/Spearman correlation, and segmentation by region/service/network/plan. Internal only.

**MCP 4 (`mcp4_forecasting`)** — new. Fits Holt-Winters (additive damped trend + weekly seasonality) on the `revenue_leakage` series and exposes forecasting, chronological-holdout model comparison against a seasonal-naive baseline, day-by-day accuracy, and Monte-Carlo-simulated prediction intervals. Internal only.

**The proxy** (`extension/server/proxy.py`) is a thin stdio↔HTTP bridge, unchanged. Claude Desktop spawns it locally; it connects to MCP 2 over the network. This is the only piece that runs on client machines.

## Project Structure

```
src/
├── mcp1_vectorstore/
│   ├── documents.py      # Revenue Assurance knowledge base (10 docs)
│   ├── settings.py       # embedding_model, port
│   └── server.py         # FastAPI/SSE: search + list_documents tools
├── mcp2_orchestrator/
│   ├── settings.py       # groq_api_key, groq_model, mcp1_url, mcp3_url, mcp4_url
│   ├── mcp1_client.py    # HTTP/SSE client wrapping MCP 1
│   ├── mcp3_client.py    # HTTP/SSE client wrapping MCP 3 (generic call_tool)
│   ├── mcp4_client.py    # HTTP/SSE client wrapping MCP 4 (generic call_tool)
│   ├── agent.py          # Agentic loop: scratchpad, task planning, parallel tool dispatch
│   └── server.py         # FastAPI/SSE: exposes the ask tool
├── mcp3_analytics/
│   ├── data_access.py    # Loads/caches the three RA datasets
│   ├── analytics.py      # KPIs, anomaly detection, hypothesis tests, correlation, segmentation
│   ├── settings.py       # port
│   └── server.py         # FastAPI/SSE: 11 analytics tools
└── mcp4_forecasting/
    ├── data_access.py    # Loads/caches the revenue_leakage time series
    ├── forecasting.py    # Holt-Winters, seasonal-naive baseline, chronological evaluation
    ├── settings.py       # port
    └── server.py         # FastAPI/SSE: 4 forecasting tools
data/
├── raw/                  # cdr_transactions.csv, daily_ra_metrics.csv, operational_events.csv
└── processed/            # cleaned/aggregated Parquet versions
tests/
├── test_analytics.py     # unit tests for mcp3_analytics.analytics (pytest)
└── test_forecasting.py   # unit tests for mcp4_forecasting.forecasting (pytest)
extension/
├── manifest.json         # Claude Desktop Extension manifest
└── server/
    └── proxy.py          # stdio ↔ HTTP/SSE bridge (runs on client machines)
```

## Server Setup

### 1. Install dependencies

```bash
uv sync
```

The first run of MCP 1 will download the `all-MiniLM-L6-v2` model (~90MB) from Hugging Face automatically — this requires network access once, then it's cached locally and runs fully offline.

### 2. Configure environment

```bash
cp .env.example .env
# Fill in your GROQ_API_KEY (free — get one at https://console.groq.com, no card required)
```

No Azure account, no paid API billing — this project runs entirely on free tiers (Groq's free API + a locally-run embedding model).

### 3. Start the servers

In four separate terminals (MCP 1, 3, 4 must be up before MCP 2 receives a request — MCP 2 lazily connects to each on demand, so exact startup order between 1/3/4 doesn't matter, but all three should be running before you `ask`):

```bash
make run-mcp1   # vector store  (RAG)          on http://0.0.0.0:8001
make run-mcp3   # analytics     (structured RA) on http://0.0.0.0:8003
make run-mcp4   # forecasting   (revenue_leakage) on http://0.0.0.0:8004
make run-mcp2   # orchestrator  (the one exposed to clients) on http://0.0.0.0:8002
```

MCP 3 and MCP 4 read from `data/raw/` and `data/processed/` relative to the repo root — populate those with your `cdr_transactions.csv` / `daily_ra_metrics.csv` / `operational_events.csv` (and optionally the cleaned Parquet versions, which are preferred when present) before starting them.

## Connecting Claude Desktop (local dev)

Run `make claude-config` to print the config block, then paste it into `%APPDATA%\Claude\claude_desktop_config.json` and restart Claude Desktop.

This spawns `proxy.py` via WSL, which connects to MCP 2 over HTTP. Both servers must be running first.

## Enterprise Deployment (claude.ai)

For enterprise claude.ai, no proxy or client-side installation is needed:

1. Deploy MCP 2 on an internal server with a publicly reachable HTTPS URL
2. An org admin adds the URL once: **claude.ai → Settings → Connectors → Add custom connector**
3. Users click to enable it — no URL entry, no configuration

MCP 1 stays internal; only MCP 2 needs to be reachable from Claude's servers (claude.ai's connector infrastructure).

## Distributing via Claude Desktop Extension (.mcpb)

Any Claude Desktop user — not just local dev — needs the proxy to connect to an internal server, since Claude Desktop only speaks stdio. The `.mcpb` packages the proxy and all Python dependencies into a one-click install.

```bash
make pack   # produces ra-orchestrator-proxy.mcpb
```

Before packing, update `MCP2_URL` in `extension/manifest.json` to point at your internal server. Distribute the `.mcpb` to users — they double-click it in Windows Explorer and Claude Desktop installs it automatically.

## Tools

### MCP 1 tools (internal, HTTP only)

| Tool             | Input                          | Output                       |
| ---------------- | ------------------------------ | ---------------------------- |
| `search`         | `query: str`, `top_k: int = 3` | `[{doc_id, content, score}]` |
| `list_documents` | —                              | `[{doc_id, content}]`        |

### MCP 3 tools (internal, HTTP only)

| Tool | Input | Output |
| --- | --- | --- |
| `get_ra_summary` | `start_date?, end_date?` | KPIs + descriptive stats + trend direction |
| `get_kpi_metrics` | `start_date?, end_date?` | leakage/recovery/failure rate KPI set with definitions |
| `get_revenue_leakage_trend` | `start_date?, end_date?, rolling_window=7` | daily series + rolling mean |
| `detect_anomalies` | `metric, method=zscore\|iqr, window=14, threshold=3.0, start_date?, end_date?` | flagged days with observed/baseline/deviation/score/severity |
| `analyze_incident` | `event_id, pre_days=14, post_days=14` | pre/during/post metric windows + hypothesis test + recovery check |
| `run_hypothesis_test` | `metric, split_date, start_date?, end_date?, alpha=0.05` | auto-selected test, p-value, statistical vs business significance |
| `get_correlation_analysis` | `start_date?, end_date?` | Pearson/Spearman vs revenue_leakage + causation caveat |
| `analyze_by_region` / `_service` / `_network` / `_plan` | — | leakage & failure counts per segment |

### MCP 4 tools (internal, HTTP only)

| Tool | Input | Output |
| --- | --- | --- |
| `forecast_revenue_leakage` | `steps=30, intervals=true` | forecast + 80%/95% intervals + direction + limitations |
| `compare_forecast_models` | `test_days=30` | seasonal-naive vs Holt-Winters on a chronological holdout (MAE/RMSE/MAPE) |
| `get_forecast_accuracy` | `model, test_days=30` | day-by-day actual vs predicted |
| `get_forecast_intervals` | `steps=30` | forecast + intervals only, no narrative |

### MCP 2 tool (exposed via HTTP/SSE)

| Tool  | Input           | Output                    |
| ----- | --------------- | ------------------------- |
| `ask` | `question: str` | synthesized answer string, routing across MCP 1/3/4 as needed |

## Agentic Loop

The agent in `agent.py` maintains a per-request scratchpad:

```python
{
  "question": str,
  "tasks": [{"id", "description", "status", "depends_on", "result"}],
  "final_answer": str | None
}
```

The LLM drives the loop using four internal tools: `add_task`, `complete_task`, `search_knowledge`, and `finish`. Tasks with satisfied dependencies are dispatched concurrently. The loop is hard-capped at 10 iterations.

## Test Questions

### Knowledge-only (MCP 1)

These require multi-hop retrieval over the Revenue Assurance knowledge base — not derivable from any LLM's training data alone, since they depend on the specific document set in `documents.py`.

**Sequential (two-hop):**
> "What is the relationship between CRMS and RMS, and how does a mismatch between them lead to revenue leakage?"

**Parallel + synthesis:**
> "Compare where leakage originates in rating/mediation versus what CPI validation is meant to prevent — how do these two failure points differ?"

**Multi-hop stretch:**
> "Walk through how a duplicate CDR moves through the ETL pipeline and which stage should catch it, then explain how that failure would show up in the RA KPIs."

### Data-only (MCP 3 / MCP 4)

> "What was our revenue leakage last month, and how does it compare to the month before?" (descriptive → `get_ra_summary`/`get_kpi_metrics`)

> "Which days in the last quarter were statistically anomalous for revenue leakage?" (anomaly → `detect_anomalies`)

> "Which region contributes the most revenue leakage, and is the problem concentrated in 5G?" (segmentation → `analyze_by_region`, `analyze_by_network`)

> "What do you expect revenue leakage to look like over the next 30 days?" (forecast → `forecast_revenue_leakage`)

### Combined — the main demonstration (MCP 1 + MCP 3 + MCP 4)

> "Revenue leakage increased significantly last month. What caused it, what does our RA documentation say about the likely causes, and what do you forecast for the next 30 days?"

This should route through: `detect_anomalies`/`get_revenue_leakage_trend` (confirm the increase and locate it in time) → `analyze_incident` (check for a coinciding operational event) → `run_hypothesis_test` (statistical significance) → `search_knowledge` (why that failure mode causes leakage, in general) → `forecast_revenue_leakage` (what's next) → a synthesized answer separating observed facts, statistical evidence, retrieved domain knowledge, the forecast, and recommendations.

## Testing

Unit tests cover the analytics and forecasting logic directly (no server required):

```bash
uv run pytest tests/ -v
```

Integration smoke tests exercise each server over real HTTP/SSE (server must be running first):

```bash
make run-mcp1 &   # or mcp3 / mcp4
uv run python test_ask.py         # MCP 2 → MCP 1
uv run python test_analytics.py   # MCP 3 directly
uv run python test_forecast.py    # MCP 4 directly
```

`make check` verifies `.env` is present and all four settings classes load without error.

## Credits

Original architecture and implementation: [Origin-Digital-LLC/nested-mcps](https://github.com/Origin-Digital-LLC/nested-mcps). This fork adapts the domain to telecom Revenue Assurance, swaps Azure OpenAI for local embeddings (`sentence-transformers`) + Groq's free API tier, and extends the original two-layer knowledge agent into a four-server platform combining RAG, structured analytics, and time-series forecasting — the whole project still runs at zero cost.

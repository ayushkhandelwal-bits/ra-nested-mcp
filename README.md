# ra-nested-mcp

A two-layer nested MCP (Model Context Protocol) system demonstrating MCP composition over HTTP/SSE — a server that is simultaneously a client to another MCP server.

This is a fork/extension of [Origin-Digital-LLC/nested-mcps](https://github.com/Origin-Digital-LLC/nested-mcps), adapted from a generic "Acme Robotics" demo into a **telecom Revenue Assurance** knowledge agent, and re-plumbed to run **without Azure**:

| | Original | This fork |
|---|---|---|
| Knowledge base | Acme Robotics (fictional company) | Telecom Revenue Assurance concepts (leakage, CDRs, CRMS/RMS, ETL, CPI) |
| Embeddings | Azure OpenAI (`text-embedding-3-small`) | Local `sentence-transformers` (`all-MiniLM-L6-v2`) — no API key |
| Orchestrator LLM | GPT-4.1 via Azure AI Foundry | Llama 3.3 70B via Groq's free API tier (OpenAI-compatible tool calling) |
| Cost to run | Requires Azure billing | **$0 — entirely free tiers, no credit card needed** |

The architecture, agentic loop, and stdio↔HTTP proxy pattern are unchanged from the original.

## Architecture

```
Claude Desktop
     │  stdio
     ▼
[stdio proxy]              ← spawned by Claude Desktop, bridges stdio ↔ HTTP
     │  HTTP/SSE (:8002)
     ▼
MCP 2: Orchestrator        ← FastAPI/uvicorn, runs on the server
     │  HTTP/SSE (:8001)
     ▼
MCP 1: Vector Store        ← FastAPI/uvicorn, internal only
```

**MCP 1 (`mcp1_vectorstore`)** is a low-level in-memory vector store. On startup it embeds 10 Revenue Assurance documents locally via `sentence-transformers`, then serves semantic search using numpy cosine similarity. It runs as a standalone HTTP service and is never exposed to Claude Desktop directly.

**MCP 2 (`mcp2_orchestrator`)** runs an agentic reasoning loop using Llama 3.3 70B via Groq's free API tier. It exposes a single `ask` tool via HTTP/SSE, decomposes questions into tasks, retrieves against MCP 1 over HTTP, and synthesizes a final answer. Independent tasks are dispatched in parallel via `asyncio.gather`.

**The proxy** (`extension/server/proxy.py`) is a thin stdio↔HTTP bridge. Claude Desktop spawns it locally; it connects to MCP 2 over the network. This is the only piece that runs on client machines.

## Project Structure

```
src/
├── mcp1_vectorstore/
│   ├── documents.py      # Revenue Assurance knowledge base (10 docs)
│   ├── settings.py       # embedding_model, port
│   └── server.py         # FastAPI/SSE: search + list_documents tools
└── mcp2_orchestrator/
    ├── settings.py       # groq_api_key, groq_model, mcp1_url
    ├── mcp1_client.py    # HTTP/SSE client wrapping MCP 1
    ├── agent.py          # Agentic loop: scratchpad, task planning, parallel search
    └── server.py         # FastAPI/SSE: exposes the ask tool
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

In two separate terminals:

```bash
make run-mcp1   # vector store on http://0.0.0.0:8001
make run-mcp2   # orchestrator on http://0.0.0.0:8002
```

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

### MCP 2 tool (exposed via HTTP/SSE)

| Tool  | Input           | Output                    |
| ----- | --------------- | ------------------------- |
| `ask` | `question: str` | synthesized answer string |

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

These questions require multi-hop retrieval over the Revenue Assurance knowledge base. The answers are not derivable from any LLM's training data alone — they depend on the specific document set in `documents.py`.

**Sequential (two-hop):**
> "What is the relationship between CRMS and RMS, and how does a mismatch between them lead to revenue leakage?"

**Parallel + synthesis:**
> "Compare where leakage originates in rating/mediation versus what CPI validation is meant to prevent — how do these two failure points differ?"

**Multi-hop stretch:**
> "Walk through how a duplicate CDR moves through the ETL pipeline and which stage should catch it, then explain how that failure would show up in the RA KPIs."

## Credits

Original architecture and implementation: [Origin-Digital-LLC/nested-mcps](https://github.com/Origin-Digital-LLC/nested-mcps). This fork adapts the domain to telecom Revenue Assurance and swaps Azure OpenAI for local embeddings (`sentence-transformers`) + Groq's free API tier — the whole project runs at zero cost.

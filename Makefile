DIR := $(shell pwd)

.PHONY: fix lint check run-mcp1 run-mcp2 run-mcp3 run-mcp4 claude-config pack

# Auto-format and fix lint issues in all Python files
fix:
	uv run ruff format src
	uv run ruff check --fix src

# Run ruff lint check and ty type check
lint:
	uv run ruff check src
	uv run ty check src

# Verify .env is present and all settings classes load without error
check:
	@test -f .env || (echo "ERROR: .env not found — run 'make init-env' first" && exit 1)
	@uv run python -c "from mcp1_vectorstore.settings import Settings; Settings(); print('MCP1 OK')"
	@uv run python -c "from mcp2_orchestrator.settings import Settings; Settings(); print('MCP2 OK')"
	@uv run python -c "from mcp3_analytics.settings import Settings; Settings(); print('MCP3 OK')"
	@uv run python -c "from mcp4_forecasting.settings import Settings; Settings(); print('MCP4 OK')"

# Start MCP 1 vector store server (HTTP on port 8001 — internal only, RAG knowledge base)
run-mcp1:
	uv run uvicorn mcp1_vectorstore.server:app --host 0.0.0.0 --port 8001

# Start MCP 2 orchestrator server (HTTP on port 8002 — the only server exposed to clients)
run-mcp2:
	uv run uvicorn mcp2_orchestrator.server:app --host 0.0.0.0 --port 8002

# Start MCP 3 analytics server (HTTP on port 8003 — internal only, structured RA data)
run-mcp3:
	uv run uvicorn mcp3_analytics.server:app --host 0.0.0.0 --port 8003

# Start MCP 4 forecasting server (HTTP on port 8004 — internal only, revenue_leakage forecasts)
run-mcp4:
	uv run uvicorn mcp4_forecasting.server:app --host 0.0.0.0 --port 8004

# Bundle Python deps into extension/lib/ and pack into a .mcpb file
pack:
	uv pip install --target extension/lib mcp httpx anyio
	npx @anthropic-ai/mcpb pack extension/ ra-orchestrator-proxy.mcpb
	@echo ""
	@echo "Double-click ra-orchestrator-proxy.mcpb in Windows Explorer to install."

# Print the Claude Desktop config block for local dev (stdio proxy → MCP2 over HTTP)
claude-config:
	@echo 'Paste into %APPDATA%\Claude\claude_desktop_config.json:'
	@echo '{'
	@echo '  "mcpServers": {'
	@echo '    "ra-orchestrator": {'
	@echo '      "command": "wsl",'
	@echo '      "args": ["-e", "bash", "-c", "cd $(DIR) && /home/bbarnett/.local/bin/uv run python extension/server/proxy.py"],'
	@echo '      "env": { "MCP2_URL": "http://127.0.0.1:8002" }'
	@echo '    }'
	@echo '  }'
	@echo '}'
DIR := $(shell pwd)

.PHONY: fix lint check run-mcp1 run-mcp2 claude-config pack

# Auto-format and fix lint issues in all Python files
fix:
	uv run ruff format src
	uv run ruff check --fix src

# Run ruff lint check and ty type check
lint:
	uv run ruff check src
	uv run ty check src

# Verify .env is present and both settings classes load without error
check:
	@test -f .env || (echo "ERROR: .env not found — run 'make init-env' first" && exit 1)
	@uv run python -c "from mcp1_vectorstore.settings import Settings; Settings(); print('MCP1 OK')"
	@uv run python -c "from mcp2_orchestrator.settings import Settings; Settings(); print('MCP2 OK')"

# Start MCP 1 vector store server (HTTP on port 8001 — internal only)
run-mcp1:
	uv run uvicorn mcp1_vectorstore.server:app --host 0.0.0.0 --port 8001

# Start MCP 2 orchestrator server (HTTP on port 8002)
run-mcp2:
	uv run uvicorn mcp2_orchestrator.server:app --host 0.0.0.0 --port 8002

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
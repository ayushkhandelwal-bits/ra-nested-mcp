import json
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, Request
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from mcp.types import TextContent, Tool
from sentence_transformers import SentenceTransformer

from mcp1_vectorstore.documents import DOCUMENTS
from mcp1_vectorstore.settings import settings

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger(__name__)

mcp_app = Server("mcp1-vectorstore")
sse_transport = SseServerTransport("/messages/")

# Populated at startup
_doc_matrix: np.ndarray | None = None
_model: SentenceTransformer | None = None


def embed(texts: list[str]) -> np.ndarray:
    """Embed texts locally via sentence-transformers (no external API call)."""
    if _model is None:
        raise RuntimeError("embed() called before lifespan initialized _model")
    vectors = _model.encode(texts, convert_to_numpy=True, normalize_embeddings=False)
    return np.array(vectors, dtype=np.float32)


@mcp_app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search",
            description="Semantic search over the Revenue Assurance knowledge base.",
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 3,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_documents",
            description="Return all Revenue Assurance documents with their IDs.",
            inputSchema={"type": "object", "properties": {}},
        ),
    ]


@mcp_app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "list_documents":
        results = [{"doc_id": i, "content": doc} for i, doc in enumerate(DOCUMENTS)]
        return [TextContent(type="text", text=json.dumps(results))]

    if name == "search":
        query = arguments["query"]
        top_k = int(arguments.get("top_k", 3))
        logger.info("Searching: %r  (top_k=%d)", query, top_k)

        query_vec = embed([query])[0]
        if _doc_matrix is None:
            raise RuntimeError(
                "call_tool() called before lifespan initialized _doc_matrix"
            )
        norms = np.linalg.norm(_doc_matrix, axis=1)
        scores = np.dot(_doc_matrix, query_vec) / (
            norms * np.linalg.norm(query_vec) + 1e-10
        )
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = [
            {
                "doc_id": int(idx),
                "content": DOCUMENTS[idx],
                "score": float(scores[idx]),
            }
            for idx in top_indices
        ]
        return [TextContent(type="text", text=json.dumps(results))]

    raise ValueError(f"Unknown tool: {name}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _doc_matrix, _model
    logger.info(
        "Initializing mcp1-vectorstore: loading %s and embedding %d documents…",
        settings.embedding_model,
        len(DOCUMENTS),
    )
    _model = SentenceTransformer(settings.embedding_model)
    _doc_matrix = embed(DOCUMENTS)
    logger.info("mcp1-vectorstore ready")
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

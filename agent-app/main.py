"""
FastAPI wrapper around the LangGraph agent.

Public endpoints:
    GET  /             → HTML demo UI (two-user isolation showcase)
    GET  /health       → liveness check
    POST /chat         → run one agent turn for a user
    GET  /authorize    → get Scalekit OAuth link for a user
    GET  /callback     → verify OAuth callback from Scalekit
    POST /admin/seed   → insert a few demo memories (no real OAuth needed)

Environment variables required:
    OPENAI_API_KEY
    SCALEKIT_ENV_URL
    SCALEKIT_CLIENT_ID
    SCALEKIT_CLIENT_SECRET
    SCALEKIT_CONNECTION_NAME   (connector slug, e.g. "github")
    VECTORAI_DB_URL            (default: localhost:6574)

Optional (MCP path):
    SCALEKIT_MCP_CONFIG_ID
    SCALEKIT_MCP_SERVER_URL
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from dotenv import find_dotenv, load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from langchain_mcp_adapters.client import MultiServerMCPClient
from pydantic import BaseModel

from agent import run_agent
from memory import recall_memories
from scalekit_utils import (
    build_mcp_server_config,
    ensure_user_connection,
    get_langchain_tools,
    mcp_is_configured,
    scalekit_is_configured,
    verify_user_connection,
)

load_dotenv(find_dotenv(usecwd=True))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Lifespan: warm up VectorAI DB connection on startup
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    log.info("Startup: warming VectorAI DB connection …")
    try:
        from memory import _get_client
        _get_client()
        log.info("VectorAI DB connected OK")
    except Exception as exc:
        log.warning("VectorAI DB not reachable at startup: %s", exc)

    log.info("Startup: pre-loading embedding model …")
    try:
        from memory import _get_embeddings
        _get_embeddings()
        log.info("Embedding model loaded OK")
    except Exception as exc:
        log.warning("Embedding model failed to load at startup: %s", exc)

    yield


app = FastAPI(
    title="Scalekit × VectorAI Memory Agent",
    description="LangGraph agent with per-user memory isolation via Actian VectorAI DB and Scalekit tool auth.",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    user_id: str
    message: str


class ChatResponse(BaseModel):
    user_id: str
    response: str
    memories_used: int
    tool_path: str


class SeedRequest(BaseModel):
    user_id: str
    memories: list[str]


# ---------------------------------------------------------------------------
# Tool resolution helper
# ---------------------------------------------------------------------------

async def _get_tools_for_user(user_id: str) -> tuple[list, str]:
    """
    Return (tools, path_label) for the given user.

    Tries the MCP path first (if configured), then falls back to the
    direct LangChain adapter, then returns an empty list with a warning.
    The path_label is a short string for the response so callers can see
    which path was used.
    """
    if not scalekit_is_configured():
        log.warning("Scalekit not configured — running memory-only (no external tools)")
        return [], "none"

    if not os.environ.get("SCALEKIT_CONNECTION_NAME"):
        log.warning("SCALEKIT_CONNECTION_NAME not set — running memory-only (no external tools)")
        return [], "none"

    conn_status = ensure_user_connection(user_id)
    if conn_status["status"] != "ACTIVE":
        log.warning(
            "User %s has no active Scalekit connection (%s) — memory-only",
            user_id,
            conn_status["status"],
        )
        return [], "none"

    if mcp_is_configured():
        server_cfg = build_mcp_server_config(user_id)
        # We return a sentinel so main can open the async context manager.
        # The actual tool fetch happens inside the async with block in /chat.
        return server_cfg, "mcp"

    tools = get_langchain_tools(user_id)
    return tools, "direct"


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {"status": "ok", "service": "agent-app"}


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """
    Run one agent turn for the given user.

    The app scopes every VectorAI call to the authenticated user's
    collection — cross-user memory leakage is prevented here in app
    code, not by VectorAI DB.
    """
    user_id = req.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id must not be empty")

    tools_or_cfg, path = await _get_tools_for_user(user_id)

    try:
        if path == "mcp":
            async with MultiServerMCPClient(tools_or_cfg) as mcp_client:
                tools = mcp_client.get_tools()
                result = await run_agent(user_id, req.message, tools)
        else:
            tools = tools_or_cfg if isinstance(tools_or_cfg, list) else []
            result = await run_agent(user_id, req.message, tools)
    except Exception as exc:
        log.exception("Agent run failed for %s", user_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return ChatResponse(
        user_id=user_id,
        response=result["response"],
        memories_used=result["memories_used"],
        tool_path=path,
    )


@app.get("/authorize")
def authorize(user_id: str = Query(..., description="Your unique user identifier")):
    """
    Get a Scalekit OAuth link for a user.  Redirect the user to the
    returned URL; after they authorize, Scalekit calls /callback.
    """
    if not scalekit_is_configured():
        raise HTTPException(status_code=503, detail="Scalekit not configured")
    status = ensure_user_connection(user_id)
    if status["status"] == "ACTIVE":
        return JSONResponse({"status": "ACTIVE", "message": "Already authorized"})
    return JSONResponse({"status": status["status"], "auth_url": status["auth_url"]})


@app.get("/callback")
def callback(
    auth_request_id: str = Query(...),
    user_id: str = Query(...),
):
    """
    Scalekit OAuth callback — verify the user and redirect to the demo UI.
    """
    if not scalekit_is_configured():
        raise HTTPException(status_code=503, detail="Scalekit not configured")
    try:
        redirect_url = verify_user_connection(auth_request_id, user_id)
        return RedirectResponse(redirect_url or "/")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/admin/seed")
async def seed(req: SeedRequest):
    """
    Insert demo memories for a user without requiring a real conversation.

    Useful for the isolation demo: seed User A and User B with different
    content, then query each to confirm their collections never overlap.
    """
    user_id = req.user_id.strip()
    if not user_id:
        raise HTTPException(status_code=400, detail="user_id must not be empty")
    if len(req.memories) > 20:
        raise HTTPException(status_code=400, detail="Max 20 memories per seed call")

    from memory import get_user_store

    store = get_user_store(user_id)
    ids = store.add_texts(req.memories, metadatas=[{"user_id": user_id}] * len(req.memories))
    return {"user_id": user_id, "inserted": len(ids), "ids": ids}


@app.get("/admin/recall")
def recall(user_id: str = Query(...), query: str = Query(...), k: int = Query(5)):
    """
    Inspect what a user would recall for a given query.
    Used in the isolation demo to prove cross-user leakage is zero.
    """
    results = recall_memories(user_id, query, k=k)
    return {"user_id": user_id, "query": query, "results": results}


# ---------------------------------------------------------------------------
# Demo UI
# ---------------------------------------------------------------------------

DEMO_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Scalekit × VectorAI Memory Agent</title>
<style>
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #0f1117;
    color: #e2e8f0;
    min-height: 100vh;
    padding: 24px;
  }
  h1 { font-size: 1.5rem; font-weight: 700; margin-bottom: 4px; }
  .subtitle { color: #94a3b8; font-size: 0.875rem; margin-bottom: 32px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  @media (max-width: 768px) { .grid { grid-template-columns: 1fr; } }
  .panel {
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    overflow: hidden;
    display: flex;
    flex-direction: column;
  }
  .panel-header {
    padding: 16px 20px;
    background: #252840;
    border-bottom: 1px solid #2d3148;
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .avatar {
    width: 36px; height: 36px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 700; font-size: 0.9rem;
  }
  .avatar-a { background: #4f46e5; }
  .avatar-b { background: #0891b2; }
  .panel-title { font-weight: 600; }
  .panel-id { color: #94a3b8; font-size: 0.75rem; font-family: monospace; }
  .messages {
    flex: 1;
    min-height: 360px;
    max-height: 400px;
    overflow-y: auto;
    padding: 16px;
    display: flex;
    flex-direction: column;
    gap: 12px;
  }
  .msg {
    padding: 10px 14px;
    border-radius: 8px;
    max-width: 85%;
    font-size: 0.875rem;
    line-height: 1.5;
    white-space: pre-wrap;
    word-break: break-word;
  }
  .msg-user {
    background: #4f46e5;
    align-self: flex-end;
    border-bottom-right-radius: 2px;
  }
  .msg-assistant {
    background: #252840;
    border: 1px solid #2d3148;
    align-self: flex-start;
    border-bottom-left-radius: 2px;
  }
  .msg-meta { font-size: 0.7rem; color: #64748b; margin-top: 4px; }
  .input-row {
    padding: 16px;
    border-top: 1px solid #2d3148;
    display: flex;
    gap: 10px;
  }
  input[type="text"] {
    flex: 1;
    background: #0f1117;
    border: 1px solid #2d3148;
    border-radius: 8px;
    color: #e2e8f0;
    padding: 10px 14px;
    font-size: 0.875rem;
    outline: none;
    transition: border-color 0.15s;
  }
  input[type="text"]:focus { border-color: #4f46e5; }
  button {
    background: #4f46e5;
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 10px 18px;
    font-size: 0.875rem;
    font-weight: 600;
    cursor: pointer;
    transition: background 0.15s;
    white-space: nowrap;
  }
  button:hover { background: #4338ca; }
  button:disabled { background: #374151; cursor: not-allowed; }
  .panel-b button { background: #0891b2; }
  .panel-b button:hover { background: #0e7490; }
  .isolation-demo {
    margin-top: 32px;
    background: #1e2130;
    border: 1px solid #2d3148;
    border-radius: 12px;
    padding: 24px;
  }
  .isolation-demo h2 { font-size: 1rem; font-weight: 600; margin-bottom: 16px; }
  .isolation-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  @media (max-width: 768px) { .isolation-grid { grid-template-columns: 1fr; } }
  .isolation-result {
    background: #0f1117;
    border: 1px solid #2d3148;
    border-radius: 8px;
    padding: 14px;
    font-size: 0.8rem;
    font-family: monospace;
    white-space: pre-wrap;
    min-height: 80px;
    color: #94a3b8;
  }
  .isolation-controls { display: flex; gap: 10px; margin-bottom: 16px; flex-wrap: wrap; }
  .btn-seed { background: #059669; }
  .btn-seed:hover { background: #047857; }
  .btn-verify { background: #7c3aed; }
  .btn-verify:hover { background: #6d28d9; }
  .tag {
    display: inline-block;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    font-family: monospace;
  }
  .tag-mcp { background: #065f46; color: #6ee7b7; }
  .tag-direct { background: #1e3a5f; color: #93c5fd; }
  .tag-none { background: #292524; color: #a8a29e; }
</style>
</head>
<body>
<h1>Scalekit × VectorAI Memory Agent</h1>
<p class="subtitle">
  Two users · one endpoint · zero cross-user memory leakage ·
  Scalekit tool auth · Actian VectorAI DB
</p>

<div class="grid">
  <!-- User A -->
  <div class="panel" id="panel-a">
    <div class="panel-header">
      <div class="avatar avatar-a">A</div>
      <div>
        <div class="panel-title">User Alice</div>
        <div class="panel-id" id="uid-a">user_alice</div>
      </div>
    </div>
    <div class="messages" id="msgs-a"></div>
    <div class="input-row">
      <input type="text" id="input-a" placeholder="Message as Alice…" />
      <button onclick="sendMessage('a')" id="btn-a">Send</button>
    </div>
  </div>

  <!-- User B -->
  <div class="panel panel-b" id="panel-b">
    <div class="panel-header">
      <div class="avatar avatar-b">B</div>
      <div>
        <div class="panel-title">User Bob</div>
        <div class="panel-id" id="uid-b">user_bob</div>
      </div>
    </div>
    <div class="messages" id="msgs-b"></div>
    <div class="input-row">
      <input type="text" id="input-b" placeholder="Message as Bob…" />
      <button onclick="sendMessage('b')" id="btn-b">Send</button>
    </div>
  </div>
</div>

<!-- Isolation verifier -->
<div class="isolation-demo">
  <h2>Memory isolation verifier</h2>
  <div class="isolation-controls">
    <button class="btn-seed" onclick="seedDemo()">Seed demo memories</button>
    <button class="btn-verify" onclick="verifyIsolation()">Verify isolation</button>
  </div>
  <div class="isolation-grid">
    <div>
      <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:6px;">
        Alice's recall (query: "Bob")
      </div>
      <div class="isolation-result" id="iso-a">Click "Verify isolation"</div>
    </div>
    <div>
      <div style="font-size:0.75rem; color:#94a3b8; margin-bottom:6px;">
        Bob's recall (query: "Alice")
      </div>
      <div class="isolation-result" id="iso-b">Click "Verify isolation"</div>
    </div>
  </div>
</div>

<script>
const users = { a: 'user_alice', b: 'user_bob' };

function appendMessage(side, role, text, meta) {
  const box = document.getElementById('msgs-' + side);
  const div = document.createElement('div');
  div.className = 'msg msg-' + role;
  div.textContent = text;
  if (meta) {
    const m = document.createElement('div');
    m.className = 'msg-meta';
    m.innerHTML = meta;
    div.appendChild(m);
  }
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendMessage(side) {
  const input = document.getElementById('input-' + side);
  const btn = document.getElementById('btn-' + side);
  const msg = input.value.trim();
  if (!msg) return;

  input.value = '';
  btn.disabled = true;
  appendMessage(side, 'user', msg);

  const thinkingEl = document.createElement('div');
  thinkingEl.className = 'msg msg-assistant';
  thinkingEl.textContent = '…';
  thinkingEl.id = 'thinking-' + side;
  document.getElementById('msgs-' + side).appendChild(thinkingEl);

  try {
    const res = await fetch('/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ user_id: users[side], message: msg }),
    });
    const data = await res.json();
    thinkingEl.remove();

    let pathTag = '';
    if (data.tool_path === 'mcp') pathTag = '<span class="tag tag-mcp">MCP tools</span>';
    else if (data.tool_path === 'direct') pathTag = '<span class="tag tag-direct">direct tools</span>';
    else pathTag = '<span class="tag tag-none">memory only</span>';

    const meta = pathTag + ' &nbsp; ' + data.memories_used + ' memories used';
    appendMessage(side, 'assistant', data.response || data.detail, meta);
  } catch (e) {
    thinkingEl.remove();
    appendMessage(side, 'assistant', 'Error: ' + e.message);
  }
  btn.disabled = false;
}

async function seedDemo() {
  const aliceMemories = [
    "User: My cat is named Whiskers.\\nAssistant: That is a sweet name!",
    "User: I work as a software engineer in Berlin.\\nAssistant: Berlin is a great city for tech.",
    "User: My favourite food is sushi.\\nAssistant: Great choice!",
  ];
  const bobMemories = [
    "User: I am training for the Berlin Marathon.\\nAssistant: That is an impressive goal!",
    "User: I have a dog named Rex.\\nAssistant: Rex is a classic dog name!",
    "User: I prefer pizza over sushi any day.\\nAssistant: Hard to argue with a good pizza.",
  ];

  const seedA = fetch('/admin/seed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'user_alice', memories: aliceMemories }),
  });
  const seedB = fetch('/admin/seed', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ user_id: 'user_bob', memories: bobMemories }),
  });
  const [ra, rb] = await Promise.all([seedA, seedB]);
  const da = await ra.json();
  const db = await rb.json();
  appendMessage('a', 'assistant', `Seeded ${da.inserted} memories for Alice.`);
  appendMessage('b', 'assistant', `Seeded ${db.inserted} memories for Bob.`);
}

async function verifyIsolation() {
  const [ra, rb] = await Promise.all([
    fetch('/admin/recall?user_id=user_alice&query=Bob&k=5'),
    fetch('/admin/recall?user_id=user_bob&query=Alice&k=5'),
  ]);
  const da = await ra.json();
  const db = await rb.json();

  const fmtA = da.results.length === 0
    ? '✅ No Bob-related memories found in Alice\\'s collection.'
    : '⚠️  Found ' + da.results.length + ' results:\\n' + da.results.join('\\n---\\n');

  const fmtB = db.results.length === 0
    ? '✅ No Alice-related memories found in Bob\\'s collection.'
    : '⚠️  Found ' + db.results.length + ' results:\\n' + db.results.join('\\n---\\n');

  document.getElementById('iso-a').textContent = fmtA;
  document.getElementById('iso-b').textContent = fmtB;
}

// Enter key support
['a', 'b'].forEach(side => {
  document.getElementById('input-' + side).addEventListener('keydown', e => {
    if (e.key === 'Enter') sendMessage(side);
  });
});
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
def demo_ui():
    return HTMLResponse(content=DEMO_HTML)

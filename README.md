# Scalekit × VectorAI Memory Agent

A LangGraph agent that authenticates users via Scalekit, stores and recalls each user's memory in Actian VectorAI DB, and runs as two connected services on Render — with a single public demo URL.

---

## Architecture

```
Browser → agent-app (Render public) → vectorai-db (Render private, gRPC 6574)
                    ↓
          Scalekit API (OAuth vault + tool execution)
```

| Service | Type | Role |
|---|---|---|
| `vectorai-db` | Render private Docker service | Memory store — runs `actian/vectorai:latest` |
| `agent-app` | Render public Python web service | LangGraph agent + FastAPI demo UI |

**Isolation contract.** VectorAI DB has no per-user access control. Every operation in this app is scoped to the authenticated user's collection (`user-{user_id}-memories`) in application code. The database never enforces the boundary — the app does. Cross-user leakage is impossible by construction, not by database policy.

---

## LangGraph graph

```
START → recall → agent ──┬── tools → agent (loop)
                          └── remember → END
```

- **`recall_node`** — similarity-searches the current user's VectorAI collection with the incoming message as the query. Injects the top-5 results into the LLM system prompt.
- **`agent_node`** — calls the LLM (`gpt-4o-mini`) with memories in context. If the LLM emits tool calls, routes to `ToolNode`.
- **`ToolNode`** — executes Scalekit-authenticated tool calls on behalf of the user. Scalekit injects the user's OAuth token; your code never sees it.
- **`remember_node`** — writes the completed user+assistant turn back to VectorAI DB as a single document.

---

## Prerequisites

- [Render account](https://render.com) (Starter plan or higher for persistent disks)
- [Scalekit account](https://scalekit.com) with at least one connector configured
- OpenAI API key
- Docker (local dev only)

---

## Local dev

### 1. Start VectorAI DB

```bash
docker pull actian/vectorai:latest
docker run -d --name vectorai \
  -v ./local_data:/var/lib/actian-vectorai \
  -p 6573-6575:6573-6575 \
  -e ACTIAN_VECTORAI_ACCEPT_EULA=YES \
  actian/vectorai:latest
```

Or use docker-compose (starts both services):

```bash
cp .env.example .env   # fill in your values
docker-compose up
```

### 2. Install Python dependencies

```bash
cd agent-app
pip install -r requirements.txt
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — fill in OPENAI_API_KEY, SCALEKIT_*, etc.
```

### 4. (Optional) Create Scalekit Virtual MCP Server config

This step is only needed for the MCP tool path. Skip it if you want to use the direct LangChain adapter path instead.

```bash
cd agent-app
python setup_mcp.py --connection-name github
```

Copy the printed `SCALEKIT_MCP_CONFIG_ID` and `SCALEKIT_MCP_SERVER_URL` values into `.env`.

### 5. Run agent-app

```bash
cd agent-app
uvicorn main:app --reload
```

Open [http://localhost:8000](http://localhost:8000) for the demo UI.

---

## Deploy to Render

### Step 1 — Connect repo and apply blueprint

1. Push this repo to GitHub/GitLab.
2. In the Render dashboard, click **New → Blueprint** and point it at your repo.
3. Render reads `render.yaml` and creates both services.

### Step 2 — Set secret env vars

In the Render dashboard, set these for `agent-app` (they are marked `sync: false` in `render.yaml`):

| Variable | Where to find it |
|---|---|
| `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com) |
| `SCALEKIT_ENV_URL` | Scalekit dashboard → Developers → Settings |
| `SCALEKIT_CLIENT_ID` | Scalekit dashboard → Developers → Settings |
| `SCALEKIT_CLIENT_SECRET` | Scalekit dashboard → Developers → Settings |
| `SCALEKIT_CONNECTION_NAME` | The connector slug you configured (e.g. `github`) |
| `SCALEKIT_MCP_CONFIG_ID` | Output of `setup_mcp.py` |
| `SCALEKIT_MCP_SERVER_URL` | Output of `setup_mcp.py` |

### Step 3 — Persistent disk (important)

Render requires Starter plan or higher for persistent disks. The `vectorai-db` service in `render.yaml` declares a 10 GB disk at `/var/lib/actian-vectorai`. Confirm your plan supports it in the Render dashboard before deploying. Without the disk, vector data is lost on container restart.

### Step 4 — Redeploy

After setting env vars, trigger a redeploy of `agent-app`. `vectorai-db` starts automatically via the blueprint.

---

## Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | HTML demo UI — two-user chat side by side |
| `/health` | GET | Liveness check |
| `/chat` | POST | Run one agent turn for a user |
| `/authorize` | GET | Get Scalekit OAuth link for a user |
| `/callback` | GET | Scalekit OAuth callback handler |
| `/admin/seed` | POST | Insert demo memories (no OAuth needed) |
| `/admin/recall` | GET | Inspect what a user would recall for a query |

### `POST /chat`

```json
{
  "user_id": "user_alice",
  "message": "What do you remember about me?"
}
```

Response:

```json
{
  "user_id": "user_alice",
  "response": "...",
  "memories_used": 3,
  "tool_path": "mcp"
}
```

`tool_path` is `"mcp"`, `"direct"`, or `"none"` — indicates which Scalekit path was used.

---

## Proving isolation (definition of done)

### Via the demo UI

1. Click **Seed demo memories** — inserts different memories for Alice and Bob.
2. Click **Verify isolation** — queries each collection for the other user's name. Both results should be empty.

### Via curl

```bash
BASE=https://your-agent-app.onrender.com

# Seed Alice and Bob
curl -s -X POST $BASE/admin/seed \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_alice","memories":["User: My cat is named Whiskers.","User: I work in Berlin."]}'

curl -s -X POST $BASE/admin/seed \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_bob","memories":["User: I am training for a marathon.","User: I have a dog named Rex."]}'

# Alice queries for Bob's content — expect empty results
curl -s "$BASE/admin/recall?user_id=user_alice&query=Rex+marathon+dog"

# Bob queries for Alice's content — expect empty results
curl -s "$BASE/admin/recall?user_id=user_bob&query=Whiskers+cat+Berlin"

# Full agent flow — Alice remembers after each turn
curl -s -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_alice","message":"What do you know about me?"}'

curl -s -X POST $BASE/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id":"user_bob","message":"What do you know about me?"}'
```

Each user's agent should recall only their own prior turns.

---

## Key constraints honoured

- **VectorAI DB Community Edition cap**: max 5,000 vectors. Seed data is intentionally tiny (a few entries per user).
- **No hosted VectorAI DB endpoint**: `api.vectorai.actian.com` is not live. The client always connects to the Docker service on `localhost:6574` (local) or `vectorai-db:6574` (Render).
- **`delete_by_ids` `strict=True`**: if you write test code that deletes by ID, pass `strict=True` to `store.delete()` so silent failures are visible.
- **Unimplemented features avoided**: no grouped search, no collection aliases, no dynamic field-index creation via REST.
- **`is_tenant=True`** is not used — it is a query-performance hint, not an isolation mechanism.

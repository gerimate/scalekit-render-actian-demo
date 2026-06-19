"""
Scalekit identity and MCP session helpers.

Two integration paths are supported:

  MCP path  (recommended, requires SCALEKIT_MCP_CONFIG_ID +
             SCALEKIT_MCP_SERVER_URL env vars):
    - Mint a per-user, short-lived session token.
    - Pass the token as a Bearer header to MultiServerMCPClient.
    - Tools are discovered dynamically from the Virtual MCP Server.

  Direct path (fallback, no MCP config needed):
    - Call actions.langchain.get_tools(identifier=user_id).
    - Returns StructuredTool objects directly usable in LangGraph.

The active path is detected at runtime from env vars.  Either works
for the hackathon demo; the MCP path is what the brief prescribes.
"""

from __future__ import annotations

import logging
import os
from datetime import timedelta
from typing import Any

from scalekit import ScalekitClient

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Client singletons
# ---------------------------------------------------------------------------

_scalekit: ScalekitClient | None = None
_actions: Any = None


def get_scalekit_client() -> ScalekitClient:
    global _scalekit
    if _scalekit is None:
        _scalekit = ScalekitClient(
            env_url=os.environ["SCALEKIT_ENV_URL"],
            client_id=os.environ["SCALEKIT_CLIENT_ID"],
            client_secret=os.environ["SCALEKIT_CLIENT_SECRET"],
        )
    return _scalekit


def get_actions_client() -> Any:
    """
    Return an ActionClient, building it directly from ScalekitClient's
    sub-clients rather than relying on sc.actions.  The sc.actions attribute
    is not reliably set on all Python versions (Python 3.14 on Render).
    """
    global _actions
    if _actions is None:
        sc = get_scalekit_client()
        # Prefer sc.actions if available; fall back to constructing directly.
        if hasattr(sc, "actions") and sc.actions is not None:
            _actions = sc.actions
        else:
            log.warning("sc.actions not available — constructing ActionClient directly")
            from scalekit.actions import ActionClient  # lazy import; may vary by SDK version
            _actions = ActionClient(sc.tools, sc.connected_accounts, sc.mcp)
    return _actions


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

def ensure_user_connection(user_id: str) -> dict:
    """
    Ensure a Scalekit connected account exists for this user.

    Returns:
        {"status": "ACTIVE"} if the account is ready.
        {"status": <other>, "auth_url": "..."} if the user must authorize.
    """
    connection_name = _connection_name()
    actions = get_actions_client()

    account_resp = actions.get_or_create_connected_account(
        connection_name=connection_name,
        identifier=user_id,
    )
    account = account_resp.connected_account

    if account.status == "ACTIVE":
        return {"status": "ACTIVE"}

    link_resp = actions.get_authorization_link(
        connection_name=connection_name,
        identifier=user_id,
    )
    return {"status": account.status, "auth_url": link_resp.link}


def verify_user_connection(auth_request_id: str, user_id: str) -> str:
    """Verify the OAuth callback for a user. Returns post-verify redirect URL."""
    actions = get_actions_client()
    result = actions.verify_connected_account_user(
        auth_request_id=auth_request_id,
        identifier=user_id,
    )
    return result.post_user_verify_redirect_url


# ---------------------------------------------------------------------------
# Tool retrieval — MCP path
# ---------------------------------------------------------------------------

def mcp_is_configured() -> bool:
    """True when both MCP env vars are present."""
    return bool(
        os.environ.get("SCALEKIT_MCP_CONFIG_ID")
        and os.environ.get("SCALEKIT_MCP_SERVER_URL")
    )


def mint_mcp_session_token(user_id: str, expiry_minutes: int = 60) -> str:
    """
    Mint a short-lived bearer token for this user's Virtual MCP Server session.

    Call this once per agent run, immediately before constructing the
    MultiServerMCPClient.  The token is scoped to exactly this user and
    this MCP config; no other user's tools are accessible.
    """
    actions = get_actions_client()
    config_id = os.environ["SCALEKIT_MCP_CONFIG_ID"]
    token_resp = actions.mcp.create_session_token(
        mcp_config_id=config_id,
        identifier=user_id,
        expiry=timedelta(minutes=expiry_minutes),
    )
    return token_resp.token


def get_mcp_server_url() -> str:
    return os.environ["SCALEKIT_MCP_SERVER_URL"]


def build_mcp_server_config(user_id: str) -> dict:
    """
    Return a MultiServerMCPClient-compatible server config for this user.

    Usage:
        async with MultiServerMCPClient(build_mcp_server_config(user_id)) as c:
            tools = c.get_tools()
    """
    token = mint_mcp_session_token(user_id)
    return {
        "scalekit": {
            "transport": "streamable_http",
            "url": get_mcp_server_url(),
            "headers": {"Authorization": f"Bearer {token}"},
        }
    }


# ---------------------------------------------------------------------------
# Tool retrieval — direct LangChain path (fallback)
# ---------------------------------------------------------------------------

def get_langchain_tools(user_id: str) -> list:
    """
    Return native LangChain StructuredTool objects for this user via the
    direct adapter — no MCP server config required.
    """
    connection_name = _connection_name()
    actions = get_actions_client()
    tools = actions.langchain.get_tools(
        identifier=user_id,
        connection_names=[connection_name] if connection_name else None,
        page_size=100,
    )
    log.info("Retrieved %d Scalekit tools for %s (direct path)", len(tools), user_id)
    return tools


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connection_name() -> str:
    """The Scalekit connector slug configured for this deployment."""
    name = os.environ.get("SCALEKIT_CONNECTION_NAME", "")
    if not name:
        raise RuntimeError(
            "SCALEKIT_CONNECTION_NAME is not set. "
            "Set it to the connector slug configured in your Scalekit dashboard "
            "(e.g. 'github', 'gmail', 'slack')."
        )
    return name


def scalekit_is_configured() -> bool:
    """True when the minimum Scalekit env vars are present."""
    return all(
        os.environ.get(k)
        for k in ("SCALEKIT_ENV_URL", "SCALEKIT_CLIENT_ID", "SCALEKIT_CLIENT_SECRET")
    )

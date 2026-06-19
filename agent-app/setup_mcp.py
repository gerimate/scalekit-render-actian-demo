"""
One-time setup: create a Scalekit Virtual MCP Server config.

Run this once before deploying agent-app, then copy the printed env vars
into your Render service settings (or .env file for local dev).

Usage:
    python setup_mcp.py [--connection-name SLUG] [--config-name NAME]

Example:
    python setup_mcp.py --connection-name github --config-name memory-agent

Required env vars (from your Scalekit dashboard → Developers → Settings):
    SCALEKIT_ENV_URL
    SCALEKIT_CLIENT_ID
    SCALEKIT_CLIENT_SECRET

The script is idempotent: if a config with the same name already exists it
prints its details and exits cleanly without creating a duplicate.
"""

from __future__ import annotations

import argparse
import os
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    parser = argparse.ArgumentParser(description="Create Scalekit Virtual MCP Server config")
    parser.add_argument(
        "--connection-name",
        default=os.getenv("SCALEKIT_CONNECTION_NAME", "github"),
        help="Connector slug configured in your Scalekit dashboard (default: github)",
    )
    parser.add_argument(
        "--config-name",
        default="memory-agent",
        help="Human-readable name for this MCP config (default: memory-agent)",
    )
    args = parser.parse_args()

    for var in ("SCALEKIT_ENV_URL", "SCALEKIT_CLIENT_ID", "SCALEKIT_CLIENT_SECRET"):
        if not os.getenv(var):
            print(f"ERROR: {var} is not set.", file=sys.stderr)
            sys.exit(1)

    from scalekit import ScalekitClient
    from scalekit.actions.models.mcp_config import McpConfigConnectionToolMapping

    sc = ScalekitClient(
        env_url=os.environ["SCALEKIT_ENV_URL"],
        client_id=os.environ["SCALEKIT_CLIENT_ID"],
        client_secret=os.environ["SCALEKIT_CLIENT_SECRET"],
    )

    # Check whether a config with this name already exists.
    existing = sc.actions.mcp.list_configs(filter_name=args.config_name)
    if getattr(existing, "configs", None):
        cfg = existing.configs[0]
        print(f"Config '{args.config_name}' already exists.")
        _print_env_vars(cfg, args.connection_name)
        return

    print(f"Creating Virtual MCP Server config '{args.config_name}' …")
    response = sc.actions.mcp.create_config(
        name=args.config_name,
        description=(
            f"Memory agent MCP config — exposes all tools for the "
            f"'{args.connection_name}' connection."
        ),
        connection_tool_mappings=[
            McpConfigConnectionToolMapping(
                connection_name=args.connection_name,
                tools=[],  # empty list = expose all tools for this connection
            )
        ],
    )
    cfg = response.config
    print("Created successfully.")
    _print_env_vars(cfg, args.connection_name)


def _print_env_vars(cfg, connection_name: str) -> None:
    config_id = cfg.id
    # The mcp_server_url is on the config object if the SDK exposes it;
    # fall back to the field name variants the SDK might use.
    mcp_url = (
        getattr(cfg, "mcp_server_url", None)
        or getattr(cfg, "server_url", None)
        or getattr(cfg, "url", None)
        or "<see Scalekit dashboard → AgentKit → Virtual MCP Servers>"
    )

    print()
    print("=" * 60)
    print("Add these to your Render environment (or .env):")
    print("=" * 60)
    print(f"SCALEKIT_MCP_CONFIG_ID={config_id}")
    print(f"SCALEKIT_MCP_SERVER_URL={mcp_url}")
    print("=" * 60)
    print()
    print(
        "NOTE: If SCALEKIT_MCP_SERVER_URL shows a placeholder, find the\n"
        "      server URL in the Scalekit dashboard under:\n"
        "      AgentKit → Virtual MCP Servers → your config → Details."
    )


if __name__ == "__main__":
    main()

"""CLI entry point for the Nextcloud MCP server."""

import argparse

import uvicorn
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette

from .config import Config
from .http_auth import BearerTokenMiddleware, is_loopback_host
from .server import create_server


def check_bind_address(config: Config) -> None:
    """Refuse to expose the HTTP transport beyond the local machine without a bearer token."""
    if not config.auth_token and not is_loopback_host(config.host):
        raise SystemExit(
            f"Refusing to listen on {config.host!r} without authentication. "
            "Set NEXTCLOUD_MCP_AUTH_TOKEN or bind NEXTCLOUD_MCP_HOST to 127.0.0.1."
        )


def build_http_app(mcp: FastMCP, config: Config) -> Starlette:
    """Streamable HTTP app of `mcp`, guarded by the bearer token when one is configured.

    The middleware wraps the whole Starlette app, so it covers every path (not only /mcp) and runs before
    FastMCP's own Host/Origin checks. The app keeps FastMCP's lifespan, which starts the session manager.
    """
    app = mcp.streamable_http_app()
    if config.auth_token:
        app.add_middleware(BearerTokenMiddleware, token=config.auth_token)
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description="Nextcloud MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "http"],
        default="stdio",
        help="Transport mode: stdio (default, for local use) or http (for remote/container)",
    )
    args = parser.parse_args()

    config = Config.from_env()
    if args.transport == "http":
        check_bind_address(config)
    mcp = create_server(config)

    if args.transport == "http":
        app = build_http_app(mcp, config)
        uvicorn.run(app, host=config.host, port=config.port, log_level=mcp.settings.log_level.lower())
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()

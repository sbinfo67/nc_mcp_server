"""Bearer-token protection for the HTTP transport."""

import hmac
import ipaddress

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocketClose


def is_loopback_host(host: str) -> bool:
    """True if `host` only accepts connections from the local machine."""
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


class BearerTokenMiddleware:
    """ASGI middleware that rejects every HTTP or WebSocket request lacking `Authorization: Bearer <token>`.

    Lifespan events pass through so that the MCP session manager still starts. The token is compared in constant
    time and is never logged or echoed back.
    """

    def __init__(self, app: ASGIApp, token: str) -> None:
        if not token:
            raise ValueError("BearerTokenMiddleware needs a non-empty token")
        self.app = app
        self._token = token.encode("ascii")

    def _authorized(self, scope: Scope) -> bool:
        for name, value in scope["headers"]:
            if name == b"authorization":
                scheme, _, supplied = value.partition(b" ")
                return scheme.lower() == b"bearer" and hmac.compare_digest(supplied.strip(), self._token)
        return False

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] not in ("http", "websocket") or self._authorized(scope):
            await self.app(scope, receive, send)
            return
        if scope["type"] == "websocket":
            await WebSocketClose(code=1008)(scope, receive, send)
            return
        response = JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
        await response(scope, receive, send)

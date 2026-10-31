"""Tests for the production hardening: audit log, HTTP bearer token, disabled tools, bind address."""

import asyncio
import json
import sys
import threading
from collections.abc import Iterator, MutableMapping
from pathlib import Path
from typing import Any

import httpx
import pytest

from nc_mcp_server import __main__ as cli
from nc_mcp_server import audit
from nc_mcp_server.audit import close_audit, configure_audit, record_call
from nc_mcp_server.config import Config
from nc_mcp_server.http_auth import BearerTokenMiddleware, is_loopback_host
from nc_mcp_server.permissions import (
    PermissionDeniedError,
    PermissionLevel,
    require_permission,
    set_permission_level,
)
from nc_mcp_server.server import create_server

TOKEN = "t" * 40
MCP_HEADERS = {"Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "0"}},
}
TOOLS_LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list"}


@pytest.fixture(autouse=True)
def _reset() -> Iterator[None]:
    yield
    close_audit()
    set_permission_level(PermissionLevel.READ)


def _config(**overrides: Any) -> Config:
    base: dict[str, Any] = {"nextcloud_url": "http://nc.test", "user": "u", "password": "p"}
    return Config(**{**base, **overrides})


def _clear_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "NEXTCLOUD_MCP_HOST",
        "NEXTCLOUD_MCP_AUTH_TOKEN",
        "NEXTCLOUD_MCP_DISABLED_TOOLS",
        "NEXTCLOUD_MCP_AUDIT_LOG",
        "NEXTCLOUD_MCP_UPLOAD_ROOT",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NEXTCLOUD_URL", "http://nc.test")
    monkeypatch.setenv("NEXTCLOUD_USER", "u")
    monkeypatch.setenv("NEXTCLOUD_PASSWORD", "p")


def _rpc_result(response: httpx.Response) -> dict[str, Any]:
    """Extract the JSON-RPC result from a streamable HTTP reply (SSE framed)."""
    for line in response.text.splitlines():
        if line.startswith("data: "):
            message: dict[str, Any] = json.loads(line.removeprefix("data: "))
            return message["result"]
    raise AssertionError(f"no data line in {response.text!r}")


class TestAuditLog:
    async def test_records_ok_denied_and_error_without_values(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.log"
        configure_audit(str(log))
        set_permission_level(PermissionLevel.WRITE)

        @require_permission(PermissionLevel.WRITE)
        async def fine(path: str) -> str:
            return "ok"

        @require_permission(PermissionLevel.DESTRUCTIVE)
        async def dangerous(path: str) -> str:
            return "x"

        @require_permission(PermissionLevel.READ)
        async def broken(secret: str) -> str:
            raise RuntimeError("boom hunter2")

        await fine(path="/very/private/file.txt")
        with pytest.raises(PermissionDeniedError):
            await dangerous(path="/a")
        with pytest.raises(RuntimeError):
            await broken(secret="hunter2")

        text = log.read_text()
        assert "/very/private" not in text
        assert "hunter2" not in text
        assert "boom" not in text
        entries = [json.loads(line) for line in text.splitlines()]
        assert [(e["tool"], e["outcome"]) for e in entries] == [
            ("fine", "ok"),
            ("dangerous", "denied"),
            ("broken", "error:RuntimeError"),
        ]
        assert entries[0]["args"] == ["path"]
        assert entries[0]["level"] == "write"
        assert entries[0]["ts"].endswith("+00:00")
        assert (log.stat().st_mode & 0o777) == 0o600

    async def test_cancelled_call_is_recorded(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.log"
        configure_audit(str(log))

        @require_permission(PermissionLevel.READ)
        async def slow() -> str:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await slow()
        assert json.loads(log.read_text())["outcome"] == "cancelled"

    async def test_disabled_by_default(self) -> None:
        @require_permission(PermissionLevel.READ)
        async def tool() -> str:
            return "ok"

        assert not audit.audit_enabled()
        assert await tool() == "ok"

    def test_appends_to_existing_file(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.log"
        log.write_text('{"previous": true}\n')
        configure_audit(str(log))
        record_call("t", "read", [], "ok", 0.0)
        assert len(log.read_text().splitlines()) == 2

    def test_reconfigure_closes_previous_file(self, tmp_path: Path) -> None:
        configure_audit(str(tmp_path / "a.log"))
        first = audit._stream
        assert first is not None
        configure_audit(str(tmp_path / "b.log"))
        assert first.closed
        close_audit()
        assert not audit.audit_enabled()

    def test_close_never_closes_stderr(self) -> None:
        configure_audit("stderr")
        close_audit()
        assert not sys.stderr.closed

    def test_stderr_target(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_audit("STDERR")
        record_call("list_files", "read", ["path"], "ok", 0.0)
        captured = capsys.readouterr()
        assert captured.out == ""
        assert json.loads(captured.err)["tool"] == "list_files"

    def test_unwritable_target_fails_fast(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            configure_audit(str(tmp_path / "missing-dir" / "audit.log"))

    def test_write_failure_does_not_propagate(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class Broken:
            def write(self, _: str) -> int:
                raise OSError("disk full")

            def flush(self) -> None:
                pass

        monkeypatch.setattr(audit, "_stream", Broken())
        monkeypatch.setattr(audit, "_owns_stream", False)
        record_call("t", "read", [], "ok", 0.0)

    def test_concurrent_writes_stay_line_atomic(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.log"
        configure_audit(str(log))

        def worker(index: int) -> None:
            for _ in range(200):
                record_call(f"tool_{index}", "read", ["a", "b"], "ok", 0.0)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        lines = log.read_text().splitlines()
        assert len(lines) == 1600
        assert all(json.loads(line)["args"] == ["a", "b"] for line in lines)


class TestBearerMiddleware:
    @staticmethod
    async def _hello(scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"hi"})

    def _client(self) -> httpx.AsyncClient:
        app = BearerTokenMiddleware(self._hello, token=TOKEN)
        return httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1")

    @pytest.mark.parametrize(
        "headers",
        [
            {},
            {"Authorization": "Bearer nope"},
            {"Authorization": f"Basic {TOKEN}"},
            {"Authorization": "Bearer"},
            {"Authorization": f"Bearer {TOKEN}x"},
        ],
    )
    async def test_rejected(self, headers: dict[str, str]) -> None:
        async with self._client() as client:
            response = await client.get("/anything", headers=headers)
        assert response.status_code == 401
        assert response.headers["WWW-Authenticate"] == "Bearer"
        assert TOKEN not in response.text

    @pytest.mark.parametrize("scheme", ["Bearer", "bearer", "BEARER"])
    async def test_accepted(self, scheme: str) -> None:
        async with self._client() as client:
            response = await client.get("/", headers={"Authorization": f"{scheme} {TOKEN}"})
        assert response.status_code == 200

    async def test_websocket_rejected_without_token(self) -> None:
        sent: list[MutableMapping[str, Any]] = []
        called = False

        async def inner(scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
            nonlocal called
            called = True

        async def receive() -> MutableMapping[str, Any]:
            return {"type": "websocket.connect"}

        async def send(message: MutableMapping[str, Any]) -> None:
            sent.append(message)

        middleware = BearerTokenMiddleware(inner, token=TOKEN)
        await middleware({"type": "websocket", "headers": [], "path": "/"}, receive, send)
        assert not called
        assert sent[0]["type"] == "websocket.close"

    async def test_lifespan_passes_through(self) -> None:
        seen: list[str] = []

        async def inner(scope: MutableMapping[str, Any], receive: Any, send: Any) -> None:
            seen.append(scope["type"])

        async def noop() -> MutableMapping[str, Any]:
            return {}

        async def discard(message: MutableMapping[str, Any]) -> None:
            pass

        await BearerTokenMiddleware(inner, token=TOKEN)({"type": "lifespan"}, noop, discard)
        assert seen == ["lifespan"]

    def test_empty_token_refused(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            BearerTokenMiddleware(self._hello, token="")


class TestHttpApp:
    """End-to-end through the real FastMCP streamable HTTP app (no network, no Nextcloud call)."""

    async def test_token_guards_mcp_endpoint_and_other_paths(self) -> None:
        config = _config(auth_token=TOKEN, disabled_tools=frozenset({"delete_user"}))
        mcp = create_server(config)
        app = cli.build_http_app(mcp, config)
        transport = httpx.ASGITransport(app=app)
        auth = {"Authorization": f"Bearer {TOKEN}"}
        async with (
            mcp.session_manager.run(),
            httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1:8100") as client,
        ):
            assert (await client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)).status_code == 401
            wrong = {**MCP_HEADERS, "Authorization": "Bearer wrong"}
            assert (await client.post("/mcp", json=INITIALIZE, headers=wrong)).status_code == 401
            assert (await client.get("/not-mcp")).status_code == 401
            assert (await client.get("/not-mcp", headers=auth)).status_code == 404

            init = await client.post("/mcp", json=INITIALIZE, headers={**MCP_HEADERS, **auth})
            assert init.status_code == 200
            assert _rpc_result(init)["serverInfo"]["name"] == "nc-mcp-server"
            # Stateless: no session id is issued and none is needed for the next request.
            assert "mcp-session-id" not in init.headers
            listed = await client.post("/mcp", json=TOOLS_LIST, headers={**MCP_HEADERS, **auth})
            names = {tool["name"] for tool in _rpc_result(listed)["tools"]}
            assert "list_directory" in names
            assert "delete_user" not in names

    async def test_dns_rebinding_protection_kept_on_loopback(self) -> None:
        config = _config(auth_token=TOKEN)
        mcp = create_server(config)
        app = cli.build_http_app(mcp, config)
        headers = {**MCP_HEADERS, "Authorization": f"Bearer {TOKEN}"}
        async with (
            mcp.session_manager.run(),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://evil.example") as client,
        ):
            response = await client.post("/mcp", json=INITIALIZE, headers=headers)
        assert response.status_code == 421

    async def test_no_middleware_without_token(self) -> None:
        config = _config()
        mcp = create_server(config)
        app = cli.build_http_app(mcp, config)
        async with (
            mcp.session_manager.run(),
            httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://127.0.0.1:8100") as client,
        ):
            response = await client.post("/mcp", json=INITIALIZE, headers=MCP_HEADERS)
        assert response.status_code == 200


class TestBindAddress:
    @pytest.mark.parametrize("host", ["127.0.0.1", "127.0.0.2", "::1", "localhost"])
    def test_loopback(self, host: str) -> None:
        assert is_loopback_host(host)

    @pytest.mark.parametrize("host", ["0.0.0.0", "192.168.30.15", "::", "example.com", ""])
    def test_not_loopback(self, host: str) -> None:
        assert not is_loopback_host(host)

    def test_default_host_is_loopback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        assert Config.from_env().host == "127.0.0.1"

    def test_public_bind_without_token_refused(self) -> None:
        with pytest.raises(SystemExit, match=r"Refusing to listen on '0\.0\.0\.0'"):
            cli.check_bind_address(_config(host="0.0.0.0"))

    def test_public_bind_with_token_allowed(self) -> None:
        cli.check_bind_address(_config(host="0.0.0.0", auth_token=TOKEN))

    def test_loopback_without_token_allowed(self) -> None:
        cli.check_bind_address(_config(host="127.0.0.1"))


class TestMain:
    def _run_main(self, monkeypatch: pytest.MonkeyPatch, *argv: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []

        def fake_run(app: Any, **kwargs: Any) -> None:
            calls.append({"app": app, **kwargs})

        monkeypatch.setattr(cli.uvicorn, "run", fake_run)
        monkeypatch.setattr(sys, "argv", ["nc-mcp-server", *argv])
        cli.main()
        return calls

    def test_http_on_public_host_without_token_exits_before_serving(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_MCP_HOST", "0.0.0.0")
        with pytest.raises(SystemExit, match="NEXTCLOUD_MCP_AUTH_TOKEN"):
            self._run_main(monkeypatch, "--transport", "http")

    def test_http_on_public_host_with_token_serves(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_MCP_HOST", "0.0.0.0")
        monkeypatch.setenv("NEXTCLOUD_MCP_AUTH_TOKEN", TOKEN)
        calls = self._run_main(monkeypatch, "--transport", "http")
        assert len(calls) == 1
        assert calls[0]["host"] == "0.0.0.0"
        assert calls[0]["port"] == 8100
        assert any(m.cls is BearerTokenMiddleware for m in calls[0]["app"].user_middleware)

    def test_http_default_host_serves_without_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        calls = self._run_main(monkeypatch, "--transport", "http")
        assert calls[0]["host"] == "127.0.0.1"
        assert calls[0]["app"].user_middleware == []

    def test_unknown_disabled_tool_exits_before_serving(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_MCP_DISABLED_TOOLS", "list_directory,no_such_tool")
        with pytest.raises(ValueError, match="no_such_tool"):
            self._run_main(monkeypatch, "--transport", "http")


class TestTokenConfig:
    def test_token_hidden_from_repr(self) -> None:
        config = _config(auth_token=TOKEN)
        assert TOKEN not in repr(config)
        assert "'p'" not in repr(config)

    def test_short_token_rejected_without_echo(self) -> None:
        with pytest.raises(ValueError, match="at least 32") as excinfo:
            _config(auth_token="short-secret").validate()
        assert "short-secret" not in str(excinfo.value)

    @pytest.mark.parametrize("token", ["a" * 31 + " b", "é" * 40, "a" * 31 + "\tb"])
    def test_invalid_characters_rejected_without_echo(self, token: str) -> None:
        with pytest.raises(ValueError, match="printable ASCII") as excinfo:
            _config(auth_token=token).validate()
        assert token not in str(excinfo.value)

    def test_token_read_from_env_and_stripped(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_MCP_AUTH_TOKEN", f"  {TOKEN}\n")
        config = Config.from_env()
        assert config.auth_token == TOKEN
        config.validate()


class TestDisabledTools:
    def test_env_parsing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _clear_env(monkeypatch)
        monkeypatch.setenv("NEXTCLOUD_MCP_DISABLED_TOOLS", " delete_user , create_user,,")
        assert Config.from_env().disabled_tools == {"delete_user", "create_user"}

    async def test_tools_removed(self) -> None:
        full = create_server(_config())
        names = {t.name for t in await full.list_tools()}
        victim = min(names)
        trimmed = create_server(_config(disabled_tools=frozenset({victim})))
        assert victim not in {t.name for t in await trimmed.list_tools()}
        assert len(await trimmed.list_tools()) == len(names) - 1

    def test_unknown_name_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="no_such_tool"):
            create_server(_config(disabled_tools=frozenset({"no_such_tool", "list_directory"})))

    def test_unregistered_optional_tool_fails_fast(self) -> None:
        # upload_file_from_path only exists when NEXTCLOUD_MCP_UPLOAD_ROOT is set.
        with pytest.raises(ValueError, match="upload_file_from_path"):
            create_server(_config(disabled_tools=frozenset({"upload_file_from_path"})))

    def test_unknown_name_does_not_open_audit_log(self, tmp_path: Path) -> None:
        log = tmp_path / "audit.log"
        with pytest.raises(ValueError, match="nope"):
            create_server(_config(disabled_tools=frozenset({"nope"}), audit_log=str(log)))
        assert not log.exists()

"""
The "mcp" tool layer on the host side: McpToolbox offers the interface the
agent loop and the Streamlit app use (demo/agent.py's Toolbox: call,
pending, preview, execute_pending, decline_pending, ready_fixes,
set_session...), but every tool runs in demo/mcp_server.py, a separate
process the host launches and talks to over MCP (stdio).

  - The planner's tool definitions are DISCOVERED from the server
    (tools/list); host-only tools ("host_*") are filtered out.
  - audit_manifest runs here, in the host, where the fine-tuned model is
    loaded: the manifest comes from the server (host_audit_input) and the
    result goes back to it (host_record_audit) for apply_fix.
  - The pending change is the server's; `pending` mirrors it (one object per
    change, so the UI's widgets keep their state across reruns).

The MCP SDK is asyncio-based and the agent loop is synchronous: the client
session lives on its own event loop, in a background thread, and each call
waits for its result.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path

from demo.agent import TOOLS, PendingChange

REPO_ROOT = Path(__file__).resolve().parent.parent
HOST_PREFIX = "host_"


class McpConnection:
    """A ClientSession to a stdio MCP server, usable from synchronous code."""

    def __init__(self, command: str, args: list[str], env: dict | None = None, cwd: str | None = None, timeout: float = 120.0):
        from mcp.client.stdio import StdioServerParameters

        self.params = StdioServerParameters(command=command, args=args, env=env, cwd=cwd)
        self.timeout = timeout
        self.session = None
        self.server_info = None
        self._error: BaseException | None = None
        self._ready = threading.Event()
        self._loop = asyncio.new_event_loop()
        self._stop: asyncio.Event | None = None
        self._thread = threading.Thread(target=self._loop.run_until_complete, args=(self._main(),), daemon=True, name="mcp-client")
        self._thread.start()
        if not self._ready.wait(timeout) or self._error is not None:
            self.close()
            raise RuntimeError(f"the MCP server didn't start: {self._error or 'timed out'}")

    async def _main(self) -> None:
        from mcp import ClientSession
        from mcp.client.stdio import stdio_client

        self._stop = asyncio.Event()
        try:
            async with stdio_client(self.params) as (read, write), ClientSession(read, write) as session:
                init = await session.initialize()
                self.session, self.server_info = session, init.server_info
                self._ready.set()
                await self._stop.wait()
        except BaseException as e:  # noqa: BLE001 -- reported to the caller of __init__ / call
            self._error = e
        finally:
            self.session = None
            self._ready.set()

    def _run(self, coroutine):
        if self.session is None:
            coroutine.close()
            raise RuntimeError(f"the MCP server isn't connected: {self._error or 'closed'}")
        return asyncio.run_coroutine_threadsafe(coroutine, self._loop).result(self.timeout)

    def list_tools(self) -> list:
        return self._run(self.session.list_tools()).tools

    def call_tool(self, name: str, arguments: dict) -> tuple[str, bool]:
        result = self._run(self.session.call_tool(name, arguments))
        text = "\n".join(c.text for c in result.content if getattr(c, "type", "") == "text")
        return text, bool(result.is_error)

    def close(self) -> None:
        if self._stop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)


class _AuditLogStatus:
    """What the app needs of the audit log in this layer: is it reachable?
    (The server records the changes; the app still reads the history itself.)"""

    def __init__(self, toolbox: "McpToolbox"):
        self._toolbox = toolbox

    def ping(self) -> bool:
        return bool(self._toolbox._status().get("reachable"))


class McpToolbox:
    def __init__(self, audit, connection: McpConnection | None = None):
        """`audit` runs the fine-tuned model on a manifest (demo/agent.py's
        make_auditor). Without `connection`, launches demo/mcp_server.py."""
        self._audit = audit
        self.connection = connection or McpConnection(
            sys.executable, ["-m", "demo.mcp_server"], env=dict(os.environ), cwd=str(REPO_ROOT)
        )
        discovered = {t.name: t for t in self.connection.list_tools()}
        self.server_tools = {name for name in discovered if not name.startswith(HOST_PREFIX)}
        # The planner sees the server's own definitions, in the usual order,
        # plus audit_manifest, which runs here.
        self.tools = []
        for local in TOOLS:
            name = local["function"]["name"]
            if name == "audit_manifest":
                self.tools.append(local)
            elif name in discovered:
                t = discovered[name]
                self.tools.append({"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}})
        self.can_apply = False
        self._mirror: PendingChange | None = None

    # ---- plumbing -----------------------------------------------------------

    def _host(self, name: str, arguments: dict | None = None) -> str:
        text, is_error = self.connection.call_tool(HOST_PREFIX + name, arguments or {})
        if is_error:
            raise RuntimeError(f"{name}: {text}")
        return text

    def _status(self) -> dict:
        return json.loads(self._host("audit_log_status"))

    def close(self) -> None:
        self.connection.close()

    # ---- what the agent loop uses ---------------------------------------------

    def call(self, name: str, arguments: dict) -> str:
        if name == "audit_manifest":
            try:
                return self._audit_manifest(**arguments)
            except TypeError as e:
                return f"Bad arguments for {name}: {e}"
        if name not in self.server_tools:
            return f"Unknown tool '{name}'. Available: {', '.join(t['function']['name'] for t in self.tools)}."
        text, _ = self.connection.call_tool(name, arguments)
        return text

    def _audit_manifest(self, kind: str, name: str, namespace: str | None = None) -> str:
        target = {"kind": kind, "name": name, **({"namespace": namespace} if namespace else {})}
        got = json.loads(self._host("audit_input", target))
        if got.get("error"):
            return got["error"]
        parsed, errors = self._audit(got["manifest"])
        return self._host("record_audit", {**target, "manifest": got["manifest"], "parsed": parsed, "errors": list(errors)})

    @property
    def pending(self) -> PendingChange | None:
        data = json.loads(self._host("pending_change"))
        if data is None:
            self._mirror = None
        elif self._mirror is None or self._mirror.change_id != data["change_id"]:
            self._mirror = PendingChange.from_dict(data)
        return self._mirror

    def ready_fixes(self) -> list[tuple[str, str, str]]:
        return [tuple(key) for key in json.loads(self._host("ready_fixes"))]

    @property
    def last_request(self) -> str:
        return ""

    @last_request.setter
    def last_request(self, text: str) -> None:
        self._host("set_session", {"last_request": text})

    def execute_pending(self, new_resources: list[str] | None = None, image_overrides: dict | None = None) -> str:
        change = self.pending
        if change is None:
            return "There was no pending change."
        overrides = {str(k): v for k, v in (image_overrides or {}).items()}
        return self._host("confirm_change", {"change_id": change.change_id, "new_resources": new_resources, "image_overrides": overrides})

    def decline_pending(self) -> str:
        change = self.pending
        if change is None:
            return "There was no pending change."
        return self._host("decline_change", {"change_id": change.change_id})

    # ---- what the app uses ----------------------------------------------------

    def preview(self, change: PendingChange, image_overrides: dict | None = None) -> tuple[str | None, str | None]:
        overrides = {str(k): v for k, v in (image_overrides or {}).items()}
        result = json.loads(self._host("preview_change", {"change_id": change.change_id, "image_overrides": overrides}))
        return result["yaml"], result["error"]

    def set_session(self, can_apply: bool, actor=None, adapter: str | None = None, audit_log=None) -> None:
        self.can_apply = can_apply
        session = {"can_apply": can_apply, "adapter": adapter or "", "audit": audit_log is not None}
        if actor is not None:
            session["actor"] = {"subject": actor.subject, "username": actor.username, "email": actor.email, "roles": list(actor.roles)}
        self._host("set_session", session)

    @property
    def audit_log(self):
        return _AuditLogStatus(self) if self._status().get("enabled") else None

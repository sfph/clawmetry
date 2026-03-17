"""Multi-agent workspace discovery and registry for ClawMetry.

Discovers OpenClaw agent workspaces via three methods (tried in priority order):
  1. CLI: ``openclaw agents list --json`` — richest data, includes workspace paths
  2. Gateway: ``agents_list`` tool via /tools/invoke — sparser, may miss agents
  3. Filesystem: scan ``~/.openclaw/agents/*/`` for workspace dirs

Single-agent setups (one workspace) continue to work unchanged.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("clawmetry.workspaces")

_WORKSPACE_MARKERS = ("SOUL.md", "AGENTS.md", "MEMORY.md", "IDENTITY.md")


@dataclass
class AgentWorkspace:
    """Describes one OpenClaw agent and its filesystem paths."""

    agent_id: str
    workspace: str
    sessions_dir: str = ""
    agent_dir: str = ""
    log_dir: str = ""
    model: str = ""
    label: str = ""
    emoji: str = ""
    is_default: bool = False
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.agent_id,
            "workspace": self.workspace,
            "sessionsDir": self.sessions_dir,
            "agentDir": self.agent_dir,
            "logDir": self.log_dir,
            "model": self.model,
            "label": self.label or self.agent_id,
            "emoji": self.emoji,
            "isDefault": self.is_default,
            "workspaceExists": os.path.isdir(self.workspace) if self.workspace else False,
            "sessionsDirExists": os.path.isdir(self.sessions_dir) if self.sessions_dir else False,
        }


class WorkspaceRegistry:
    """Thread-safe registry of discovered agent workspaces."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentWorkspace] = {}
        self._lock = threading.Lock()
        self._discovered_at: float = 0
        self._discovery_source: str = ""

    # ── Access ────────────────────────────────────────────────────────────────

    def add(self, ws: AgentWorkspace) -> None:
        with self._lock:
            self._agents[ws.agent_id] = ws

    def get(self, agent_id: str) -> Optional[AgentWorkspace]:
        with self._lock:
            return self._agents.get(agent_id)

    def default(self) -> Optional[AgentWorkspace]:
        with self._lock:
            for ws in self._agents.values():
                if ws.is_default:
                    return ws
            if self._agents:
                return next(iter(self._agents.values()))
            return None

    def list_all(self) -> List[AgentWorkspace]:
        with self._lock:
            return list(self._agents.values())

    def count(self) -> int:
        with self._lock:
            return len(self._agents)

    def is_multi(self) -> bool:
        return self.count() > 1

    @property
    def discovery_source(self) -> str:
        return self._discovery_source

    @property
    def discovered_at(self) -> float:
        return self._discovered_at

    def clear(self) -> None:
        with self._lock:
            self._agents.clear()
            self._discovered_at = 0
            self._discovery_source = ""

    def to_list(self) -> List[Dict[str, Any]]:
        return [ws.to_dict() for ws in self.list_all()]

    # ── Discovery orchestration ───────────────────────────────────────────────

    def discover(
        self,
        data_dir: str = "",
        gw_invoke: Optional[Callable] = None,
        extra_workspaces: Optional[List[str]] = None,
    ) -> bool:
        """Run discovery in priority order. Returns True if agents were found.

        Args:
            data_dir: OpenClaw data directory (e.g. ``~/.openclaw``).
            gw_invoke: Callable matching ``_gw_invoke(tool, args)`` from dashboard.py.
            extra_workspaces: Manually specified workspace paths (from CLI/env).
        """
        if self.discover_from_cli():
            pass
        elif gw_invoke and self.discover_from_gateway(gw_invoke, data_dir):
            pass
        elif data_dir:
            self.discover_from_filesystem(data_dir)

        if extra_workspaces:
            self._add_manual_workspaces(extra_workspaces)

        self._discovered_at = time.time()
        return self.count() > 0

    # ── CLI discovery ─────────────────────────────────────────────────────────

    def discover_from_cli(self) -> bool:
        """Try ``openclaw agents list --json``. Returns True on success."""
        openclaw_bin = shutil.which("openclaw")
        if not openclaw_bin:
            logger.debug("openclaw binary not in PATH, skipping CLI discovery")
            return False

        try:
            result = subprocess.run(
                [openclaw_bin, "agents", "list", "--json"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode != 0:
                logger.debug("openclaw agents list failed: %s", result.stderr.strip())
                return False

            agents = json.loads(result.stdout)
            if not isinstance(agents, list) or not agents:
                return False

            for entry in agents:
                if not isinstance(entry, dict) or "id" not in entry:
                    continue
                ws = _agent_workspace_from_cli(entry)
                self.add(ws)

            self._discovery_source = "cli"
            logger.info(
                "Discovered %d agent(s) via CLI: %s",
                self.count(),
                [a.agent_id for a in self.list_all()],
            )
            return self.count() > 0

        except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
            logger.debug("CLI discovery failed: %s", exc)
            return False

    # ── Gateway discovery ─────────────────────────────────────────────────────

    def discover_from_gateway(
        self,
        gw_invoke: Callable,
        data_dir: str = "",
    ) -> bool:
        """Try gateway ``agents_list`` tool. Returns True on success.

        The gateway response is sparser (no workspace paths), so we derive
        paths from the conventional OpenClaw directory layout.
        """
        try:
            result = gw_invoke("agents_list", {})
            if not result:
                return False

            agents_list = result if isinstance(result, list) else result.get("agents", [])
            if not agents_list:
                return False

            data_dir = data_dir or os.path.expanduser("~/.openclaw")
            for entry in agents_list:
                if not isinstance(entry, dict) or "id" not in entry:
                    continue
                ws = _agent_workspace_from_gateway(entry, data_dir)
                self.add(ws)

            self._discovery_source = "gateway"
            logger.info(
                "Discovered %d agent(s) via gateway: %s",
                self.count(),
                [a.agent_id for a in self.list_all()],
            )
            return self.count() > 0

        except Exception as exc:
            logger.debug("Gateway discovery failed: %s", exc)
            return False

    # ── Filesystem discovery ──────────────────────────────────────────────────

    def discover_from_filesystem(self, data_dir: str) -> bool:
        """Scan ``{data_dir}/agents/*/`` for workspaces. Returns True on success."""
        agents_base = os.path.join(data_dir, "agents")
        if not os.path.isdir(agents_base):
            return False

        found = False
        for agent_name in sorted(os.listdir(agents_base)):
            agent_path = os.path.join(agents_base, agent_name)
            if not os.path.isdir(agent_path):
                continue

            ws = _agent_workspace_from_filesystem(agent_name, agent_path, data_dir)
            if ws:
                self.add(ws)
                found = True

        if found:
            self._discovery_source = "filesystem"
            logger.info(
                "Discovered %d agent(s) via filesystem: %s",
                self.count(),
                [a.agent_id for a in self.list_all()],
            )
        return found

    # ── Manual workspaces ─────────────────────────────────────────────────────

    def _add_manual_workspaces(self, paths: List[str]) -> None:
        for i, raw_path in enumerate(paths):
            path = os.path.expanduser(raw_path)
            if not os.path.isdir(path):
                logger.warning("Manual workspace path does not exist: %s", path)
                continue
            agent_id = os.path.basename(path) or f"workspace-{i}"
            if self.get(agent_id):
                existing = self.get(agent_id)
                if existing and existing.workspace == path:
                    continue
                agent_id = f"{agent_id}-{i}"
            self.add(AgentWorkspace(
                agent_id=agent_id,
                workspace=path,
                is_default=(i == 0 and self.count() == 0),
            ))


# ── Factory helpers ───────────────────────────────────────────────────────────


def _agent_workspace_from_cli(entry: Dict[str, Any]) -> AgentWorkspace:
    """Build AgentWorkspace from ``openclaw agents list --json`` entry."""
    agent_id = entry["id"]
    agent_dir = entry.get("agentDir", "")
    sessions_dir = ""
    if agent_dir:
        sessions_dir = os.path.join(os.path.dirname(agent_dir), "sessions")

    return AgentWorkspace(
        agent_id=agent_id,
        workspace=entry.get("workspace", ""),
        sessions_dir=sessions_dir,
        agent_dir=agent_dir,
        model=entry.get("model", ""),
        label=entry.get("identityName") or entry.get("name") or agent_id,
        emoji=entry.get("identityEmoji", ""),
        is_default=bool(entry.get("isDefault")),
        extra=entry,
    )


def _agent_workspace_from_gateway(
    entry: Dict[str, Any],
    data_dir: str,
) -> AgentWorkspace:
    """Build AgentWorkspace from gateway ``agents_list`` response entry.

    The gateway only returns ``id``, ``name``, ``configured``, so workspace and
    session paths are derived from the conventional OpenClaw layout.
    """
    agent_id = entry["id"]
    is_default = agent_id == "main" or bool(entry.get("default"))

    if is_default:
        workspace = os.path.join(data_dir, "workspace")
    else:
        workspace = os.path.join(data_dir, f"workspace-{agent_id}")

    agent_dir = os.path.join(data_dir, "agents", agent_id, "agent")
    sessions_dir = os.path.join(data_dir, "agents", agent_id, "sessions")

    return AgentWorkspace(
        agent_id=agent_id,
        workspace=workspace,
        sessions_dir=sessions_dir,
        agent_dir=agent_dir,
        model="",
        label=entry.get("name") or agent_id,
        emoji="",
        is_default=is_default,
        extra=entry,
    )


def _agent_workspace_from_filesystem(
    agent_name: str,
    agent_path: str,
    data_dir: str,
) -> Optional[AgentWorkspace]:
    """Build AgentWorkspace by scanning filesystem conventions."""
    sessions_dir = os.path.join(agent_path, "sessions")
    agent_dir = os.path.join(agent_path, "agent")
    is_default = agent_name == "main"

    if is_default:
        workspace_candidates = [
            os.path.join(data_dir, "workspace"),
        ]
    else:
        workspace_candidates = [
            os.path.join(data_dir, f"workspace-{agent_name}"),
            os.path.join(agent_path, "workspace"),
        ]

    workspace = ""
    for candidate in workspace_candidates:
        if os.path.isdir(candidate) and _has_workspace_markers(candidate):
            workspace = candidate
            break

    if not workspace and not os.path.isdir(sessions_dir):
        return None

    if not workspace:
        for candidate in workspace_candidates:
            if os.path.isdir(candidate):
                workspace = candidate
                break

    return AgentWorkspace(
        agent_id=agent_name,
        workspace=workspace,
        sessions_dir=sessions_dir if os.path.isdir(sessions_dir) else "",
        agent_dir=agent_dir if os.path.isdir(agent_dir) else "",
        is_default=is_default,
    )


def _has_workspace_markers(path: str) -> bool:
    """Check if a directory looks like an OpenClaw workspace."""
    for marker in _WORKSPACE_MARKERS:
        if os.path.exists(os.path.join(path, marker)):
            return True
    return os.path.isdir(os.path.join(path, "memory"))

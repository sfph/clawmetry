"""Multi-workspace provider — aggregates data from multiple LocalDataProviders.

Wraps one LocalDataProvider per agent workspace and exposes the standard
ClawMetryDataProvider interface.  When ``agent_id`` filtering is requested,
delegates to the specific provider; otherwise merges results from all.

Session IDs are namespaced as ``{agent_id}:{original_id}`` to prevent
collisions across agents.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

from clawmetry.providers.base import (
    ClawMetryDataProvider,
    Event,
    MemoryFile,
    Session,
)
from clawmetry.providers.local import LocalDataProvider
from clawmetry.workspaces import AgentWorkspace, WorkspaceRegistry

logger = logging.getLogger("clawmetry.providers.multi")

_SEP = "@@"


def _ns_session_id(agent_id: str, session_id: str) -> str:
    """Namespace a session ID: ``agent_id@@original_id``.

    Uses ``@@`` as separator to avoid collision with ``:`` which appears
    in raw OpenClaw session IDs (e.g. ``agent:main:subagent:xyz``).
    """
    if session_id.startswith(agent_id + _SEP):
        return session_id
    return f"{agent_id}{_SEP}{session_id}"


def _parse_ns_session_id(namespaced_id: str) -> tuple[str, str]:
    """Return ``(agent_id, original_session_id)`` from a namespaced ID.

    If the ID has no namespace prefix, returns ``("", original_id)``.
    """
    if _SEP in namespaced_id:
        agent_id, _, original = namespaced_id.partition(_SEP)
        return agent_id, original
    return "", namespaced_id


class MultiWorkspaceProvider(ClawMetryDataProvider):
    """Aggregates multiple local providers behind the standard interface."""

    def __init__(self, registry: WorkspaceRegistry, log_dir: str = "",
                 metrics_file: str = "", fleet_db: str = "", **kwargs):
        default_ws = registry.default()
        super().__init__(
            sessions_dir=default_ws.sessions_dir if default_ws else "",
            log_dir=log_dir,
            workspace=default_ws.workspace if default_ws else "",
            metrics_file=metrics_file,
            fleet_db=fleet_db,
            **kwargs,
        )
        self._registry = registry
        self._providers: Dict[str, LocalDataProvider] = {}
        self._build_providers(log_dir)

    def _build_providers(self, shared_log_dir: str) -> None:
        for ws in self._registry.list_all():
            self._providers[ws.agent_id] = LocalDataProvider(
                sessions_dir=ws.sessions_dir,
                log_dir=ws.log_dir or shared_log_dir,
                workspace=ws.workspace,
            )

    @property
    def registry(self) -> WorkspaceRegistry:
        return self._registry

    def provider_for(self, agent_id: str) -> Optional[LocalDataProvider]:
        return self._providers.get(agent_id)

    def _iter_providers(
        self, agent_id: Optional[str] = None
    ) -> list[tuple[str, LocalDataProvider]]:
        """Return ``[(agent_id, provider), ...]`` filtered by agent_id if given."""
        if agent_id:
            p = self._providers.get(agent_id)
            return [(agent_id, p)] if p else []
        return list(self._providers.items())

    # ── Sessions ──────────────────────────────────────────────────────────────

    def list_sessions(
        self,
        limit: int = 30,
        include_subagents: bool = True,
        since_ms: Optional[int] = None,
        agent_id: Optional[str] = None,
    ) -> List[Session]:
        merged: List[Session] = []
        for aid, provider in self._iter_providers(agent_id):
            for s in provider.list_sessions(
                limit=None,
                include_subagents=include_subagents,
                since_ms=since_ms,
            ):
                s.agent_id = aid
                s.session_id = _ns_session_id(aid, s.session_id)
                merged.append(s)
        merged.sort(key=lambda s: s.updated_at, reverse=True)
        return merged[:limit] if limit else merged

    def get_session(self, session_id: str) -> Optional[Session]:
        agent_id, original_id = _parse_ns_session_id(session_id)
        if agent_id and agent_id in self._providers:
            s = self._providers[agent_id].get_session(original_id)
            if s:
                s.agent_id = agent_id
                s.session_id = _ns_session_id(agent_id, s.session_id)
            return s
        for aid, provider in self._providers.items():
            s = provider.get_session(original_id or session_id)
            if s:
                s.agent_id = aid
                s.session_id = _ns_session_id(aid, s.session_id)
                return s
        return None

    def get_session_index(self, agent_id: Optional[str] = None) -> Dict[str, Dict]:
        merged: Dict[str, Dict] = {}
        for aid, provider in self._iter_providers(agent_id):
            for key, meta in provider.get_session_index().items():
                ns_key = _ns_session_id(aid, key)
                if isinstance(meta, dict):
                    meta = {**meta, "agent_id": aid}
                    if "sessionId" in meta:
                        meta["sessionId"] = _ns_session_id(aid, meta["sessionId"])
                merged[ns_key] = meta
        return merged

    # ── Events ────────────────────────────────────────────────────────────────

    def get_events(
        self,
        session_id: str,
        limit: int = 500,
        tail_bytes: Optional[int] = None,
    ) -> List[Event]:
        agent_id, original_id = _parse_ns_session_id(session_id)
        if agent_id and agent_id in self._providers:
            return self._providers[agent_id].get_events(original_id, limit, tail_bytes)
        for provider in self._providers.values():
            events = provider.get_events(original_id or session_id, limit, tail_bytes)
            if events:
                return events
        return []

    # ── Logs (shared — all agents use same gateway log dir) ───────────────────

    def get_log_lines(
        self, date_str: Optional[str] = None, limit: int = 1000
    ) -> List[str]:
        default = self._default_provider()
        return default.get_log_lines(date_str, limit) if default else []

    def list_log_dates(self, days_back: int = 31) -> List[str]:
        default = self._default_provider()
        return default.list_log_dates(days_back) if default else []

    # ── Memory / Workspace ────────────────────────────────────────────────────

    def list_memory_files(self, agent_id: Optional[str] = None) -> List[MemoryFile]:
        merged: List[MemoryFile] = []
        for aid, provider in self._iter_providers(agent_id):
            for mf in provider.list_memory_files():
                mf.agent_id = aid
                merged.append(mf)
        return merged

    def read_workspace_file(
        self, relative_path: str, agent_id: Optional[str] = None
    ) -> str:
        if agent_id:
            p = self._providers.get(agent_id)
            return p.read_workspace_file(relative_path) if p else ""
        default = self._default_provider()
        return default.read_workspace_file(relative_path) if default else ""

    # ── Crons (shared — from the data dir, not per-agent) ─────────────────────

    def list_crons(self) -> List[Dict[str, Any]]:
        default = self._default_provider()
        return default.list_crons() if default else []

    # ── Health ────────────────────────────────────────────────────────────────

    def health_check(self) -> Dict[str, Any]:
        agents_health = {}
        for aid, provider in self._providers.items():
            agents_health[aid] = provider.health_check()
        return {
            "provider": "MultiWorkspaceProvider",
            "ok": True,
            "agent_count": len(self._providers),
            "discovery_source": self._registry.discovery_source,
            "agents": agents_health,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _default_provider(self) -> Optional[LocalDataProvider]:
        default_ws = self._registry.default()
        if default_ws:
            return self._providers.get(default_ws.agent_id)
        if self._providers:
            return next(iter(self._providers.values()))
        return None

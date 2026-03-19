"""
Plugin Dependency Graph — automatic ordering and missing-dep warnings.

Plugins can declare ``requires: list[str]`` (plugin IDs they depend on).
``resolve_order`` topologically sorts the plugin list so dependencies run
first.  If a cycle or missing dependency is detected, a descriptive
``DependencyError`` is raised.

Usage:
    from core.plugin_deps import resolve_order, DependencyError
"""

from __future__ import annotations

import warnings
from typing import Any


class DependencyError(Exception):
    """Raised for cyclic or missing plugin dependencies."""


def _get_requires(plugin: Any) -> list[str]:
    return getattr(plugin, "requires", None) or []


def resolve_order(plugins: list) -> list:
    """
    Topologically sort *plugins* so that each plugin's ``requires`` are
    satisfied by plugins earlier in the returned list.

    Returns a new list; the input is not mutated.
    Raises ``DependencyError`` on cycles or missing deps.
    """
    id_map: dict[str, Any] = {}
    for p in plugins:
        pid = p.plugin_id
        if pid in id_map:
            warnings.warn(
                f"Duplicate plugin_id {pid!r}; keeping first occurrence.",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        id_map[pid] = p

    available = set(id_map.keys())
    for p in plugins:
        missing = set(_get_requires(p)) - available
        if missing:
            raise DependencyError(
                f"Plugin {p.plugin_id!r} requires missing plugins: {sorted(missing)}"
            )

    UNVISITED, IN_PROGRESS, DONE = 0, 1, 2
    state: dict[str, int] = {pid: UNVISITED for pid in id_map}
    order: list[str] = []

    def visit(pid: str) -> None:
        if state[pid] == DONE:
            return
        if state[pid] == IN_PROGRESS:
            raise DependencyError(f"Cyclic dependency detected involving {pid!r}")
        state[pid] = IN_PROGRESS
        for dep in _get_requires(id_map[pid]):
            visit(dep)
        state[pid] = DONE
        order.append(pid)

    for pid in id_map:
        visit(pid)

    return [id_map[pid] for pid in order]


def build_graph(plugins: list) -> dict[str, list[str]]:
    """Return adjacency dict {plugin_id: [dependency_ids]}."""
    return {p.plugin_id: list(_get_requires(p)) for p in plugins}


def check_health(plugins: list) -> dict:
    """
    Diagnostic: check for missing deps and cycles without raising.
    Returns {"ok": bool, "warnings": [...], "order": [...]}.
    """
    result: dict = {"ok": True, "warnings": [], "order": []}
    try:
        ordered = resolve_order(plugins)
        result["order"] = [p.plugin_id for p in ordered]
    except DependencyError as e:
        result["ok"] = False
        result["warnings"].append(str(e))
    return result

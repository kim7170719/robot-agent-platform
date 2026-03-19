"""Load optional third-party registrations from setuptools entry points."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from core.registry import PluginRegistry


def load_entrypoint_registers(registry: PluginRegistry) -> None:
    """
    Call every function registered under the ``iraap.register`` entry-point group.

    Each entry point should be callable as ``register(registry: PluginRegistry) -> None``.
    """
    try:
        from importlib.metadata import entry_points
    except ImportError:
        return

    try:
        eps = entry_points(group="iraap.register")
    except TypeError:
        eps = entry_points().select(group="iraap.register")

    for ep in eps:
        fn = ep.load()
        if callable(fn):
            fn(registry)

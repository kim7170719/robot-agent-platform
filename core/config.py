"""
Config — centralized configuration management for IR-AAP.
Loads from YAML/JSON files or plain dicts. Provides typed access
with defaults so no hardcoded values are scattered across modules.
"""

from __future__ import annotations

import json
import os
from typing import Any


_DEFAULTS = {
    "world": {
        "type": "physics",
        "width": 10,
        "height": 10,
        "gui": False,
        "timestep": 1 / 240,
        "gravity": -9.81,
    },
    "sensor": {
        "noise_level": 0.0,
        "noise_seed": None,
        "ray_max_distance": 20.0,
    },
    "camera": {
        "width": 64,
        "height": 64,
        "fov": 60.0,
        "near": 0.1,
        "far": 20.0,
        "noise_level": 0.0,
    },
    "lidar": {
        "num_rays": 36,
        "max_distance": 15.0,
        "height_offset": 0.3,
    },
    "agent": {
        "safe_distance": 1.5,
        "goal_reached_radius": 0.8,
    },
    "engine": {
        "enable_metrics": True,
        "max_history": 10_000,
    },
    "recorder": {
        "enabled": False,
        "output_file": "recording.jsonl",
    },
}


class Config:
    """
    Hierarchical configuration with dot-path access.

    Usage::

        cfg = Config.from_file("config.json")
        timestep = cfg.get("world.timestep")
        noise = cfg.get("sensor.noise_level", 0.0)
    """

    def __init__(self, overrides: dict | None = None) -> None:
        self._data = _deep_merge(_deep_copy(_DEFAULTS), overrides or {})

    @classmethod
    def from_file(cls, path: str) -> Config:
        """Load configuration from a ``.json``, ``.yaml``, or ``.yml`` file."""
        ext = os.path.splitext(path)[1].lower()
        with open(path, "r", encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                try:
                    import yaml
                    data = yaml.safe_load(f) or {}
                except ImportError:
                    raise ImportError("PyYAML is required to load .yaml config files")
            else:
                data = json.load(f)
        return cls(data)

    @classmethod
    def load_file(cls, path: str) -> Config:
        """Alias for :meth:`from_file` (explicit persistence API)."""
        return cls.from_file(path)

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        return cls(data)

    def get(self, dot_path: str, default: Any = None) -> Any:
        keys = dot_path.split(".")
        node = self._data
        for key in keys:
            if isinstance(node, dict) and key in node:
                node = node[key]
            else:
                return default
        return node

    def set(self, dot_path: str, value: Any) -> None:
        keys = dot_path.split(".")
        node = self._data
        for key in keys[:-1]:
            if key not in node or not isinstance(node[key], dict):
                node[key] = {}
            node = node[key]
        node[keys[-1]] = value

    def section(self, name: str) -> dict:
        return dict(self._data.get(name, {}))

    def to_dict(self) -> dict:
        return _deep_copy(self._data)

    def save(self, path: str) -> None:
        """Write config to ``.json`` or ``.yaml`` / ``.yml`` based on extension."""
        ext = os.path.splitext(path)[1].lower()
        with open(path, "w", encoding="utf-8") as f:
            if ext in (".yaml", ".yml"):
                try:
                    import yaml
                except ImportError:
                    raise ImportError("PyYAML is required to save .yaml config files")
                yaml.safe_dump(
                    self._data,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False,
                )
            else:
                json.dump(self._data, f, indent=2, ensure_ascii=False)


def _deep_merge(base: dict, override: dict) -> dict:
    result = {}
    for key in set(base) | set(override):
        if key in override and key in base:
            if isinstance(base[key], dict) and isinstance(override[key], dict):
                result[key] = _deep_merge(base[key], override[key])
            else:
                result[key] = _deep_copy_val(override[key])
        elif key in override:
            result[key] = _deep_copy_val(override[key])
        else:
            result[key] = _deep_copy_val(base[key])
    return result


def _deep_copy_val(v: Any) -> Any:
    if isinstance(v, dict):
        return _deep_copy(v)
    if isinstance(v, list):
        return [_deep_copy_val(item) for item in v]
    return v


def _deep_copy(d: dict) -> dict:
    result = {}
    for k, v in d.items():
        result[k] = _deep_copy_val(v)
    return result

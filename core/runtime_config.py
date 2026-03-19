"""
Runtime configuration from environment variables (and optional ``.env``).

Variables (prefix ``IRAAP_`` to avoid collisions):

  IRAAP_API_HOST   — bind address for FastAPI (default ``0.0.0.0``)
  IRAAP_API_PORT   — REST port (default ``8000``)
  IRAAP_WS_PORT    — WebSocket dashboard port (default ``8765``)

``load_env_file()`` parses a simple ``KEY=value`` file without requiring python-dotenv.
"""

from __future__ import annotations

import os
from pathlib import Path


def load_env_file(path: str | Path | None = None) -> None:
    """
    Load ``KEY=value`` pairs into ``os.environ`` if not already set.
    Skips comments and blank lines. Does nothing if file is missing.
    """
    p = Path(path or ".env")
    if not p.is_file():
        return
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = val


def get_api_host(default: str = "0.0.0.0") -> str:
    return os.environ.get("IRAAP_API_HOST", default).strip() or default


def get_api_port(default: int = 8000) -> int:
    return _parse_port("IRAAP_API_PORT", default)


def get_ws_port(default: int = 8765) -> int:
    return _parse_port("IRAAP_WS_PORT", default)


def _parse_port(name: str, default: int) -> int:
    raw = os.environ.get(name, "")
    if not raw.strip():
        return default
    try:
        p = int(raw.strip(), 10)
        return p if 1 <= p <= 65535 else default
    except ValueError:
        return default

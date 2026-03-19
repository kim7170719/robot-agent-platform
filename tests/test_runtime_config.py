"""Tests for core.runtime_config (.env loading and port helpers)."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from core import runtime_config as rc


def test_parse_port_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("IRAAP_API_PORT", raising=False)
    assert rc.get_api_port(8000) == 8000


def test_parse_port_valid(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRAAP_API_PORT", "9000")
    assert rc.get_api_port(8000) == 9000


def test_parse_port_invalid_falls_back(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRAAP_API_PORT", "not-a-port")
    assert rc.get_api_port(8000) == 8000


def test_parse_port_out_of_range(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRAAP_WS_PORT", "70000")
    assert rc.get_ws_port(8765) == 8765


def test_load_env_file_respects_existing_env(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("IRAAP_API_PORT", "1111")
    p = tmp_path / ".env"
    p.write_text("IRAAP_API_PORT=2222\nIRAAP_WS_PORT=3333\n", encoding="utf-8")
    rc.load_env_file(p)
    assert os.environ["IRAAP_API_PORT"] == "1111"
    assert os.environ["IRAAP_WS_PORT"] == "3333"


def test_load_env_file_strips_quotes(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FOO_BAR_BAZ", raising=False)
    p = tmp_path / ".env"
    p.write_text('FOO_BAR_BAZ="hello"\n', encoding="utf-8")
    rc.load_env_file(p)
    assert os.environ["FOO_BAR_BAZ"] == "hello"

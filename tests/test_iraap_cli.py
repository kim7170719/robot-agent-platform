"""Smoke tests for the iraap CLI package."""

from __future__ import annotations

import os
import subprocess
import sys


def test_iraap_doctor_exit_zero() -> None:
    root = os.path.join(os.path.dirname(__file__), "..")
    env = {**os.environ, "PYTHONPATH": root}
    r = subprocess.run(
        [sys.executable, "-m", "iraap", "doctor"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 0, r.stderr + r.stdout


def test_cli_main_doctor() -> None:
    from iraap.cli import main

    assert main(["doctor"]) == 0


def test_cli_init_list() -> None:
    from iraap.cli import main

    assert main(["init", "--list"]) == 0


def test_cli_init_writes_preset(tmp_path) -> None:
    from iraap.cli import main

    out = tmp_path / "mission.yaml"
    assert main(["init", str(out), "--preset", "simple_robot"]) == 0
    assert out.is_file()
    text = out.read_text(encoding="utf-8")
    assert "plugins:" in text and "agents:" in text


def test_cli_init_forklift_and_agv_presets(tmp_path) -> None:
    from iraap.cli import main

    for preset in ("forklift_minimal", "agv_patrol"):
        out = tmp_path / f"{preset}.yaml"
        assert main(["init", str(out), "--preset", preset]) == 0
        body = out.read_text(encoding="utf-8")
        assert "world:" in body and "agents:" in body

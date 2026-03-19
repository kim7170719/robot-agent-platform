"""
IR-AAP command-line interface.

  iraap init    — copy a commented preset YAML (low-code start)
  iraap serve   — same as ``python run_live.py`` (loads ``.env`` for ports)
  iraap doctor  — import / dependency smoke check
"""

from __future__ import annotations

import argparse
import importlib
import shutil
import sys
from pathlib import Path


def _presets_dir() -> Path:
    return Path(__file__).resolve().parent / "presets"


def _list_presets() -> list[str]:
    d = _presets_dir()
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.yaml"))


def _cmd_init(args: argparse.Namespace) -> int:
    if args.list_presets:
        names = _list_presets()
        if not names:
            print("No presets found.", file=sys.stderr)
            return 1
        print("Available presets:")
        for n in names:
            print(f"  {n}")
        return 0

    preset = args.preset
    src = _presets_dir() / f"{preset}.yaml"
    if not src.is_file():
        print(f"Unknown preset {preset!r}. Use: iraap init --list", file=sys.stderr)
        return 1

    dest = Path(args.output)
    if dest.exists() and not args.force:
        print(f"Refusing to overwrite {dest} (use --force)", file=sys.stderr)
        return 1
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    print(f"Created {dest} from preset {preset!r}")
    print("Edit the YAML, then run:")
    print(f"  iraap serve --profile {dest}")
    return 0


def _doctor() -> int:
    """Verify core imports and optional heavy deps."""
    mods = [
        "core.event_bus",
        "core.engine",
        "core.runtime_config",
        "simulator.physics_world",
        "api.server",
        "fastapi",
        "uvicorn",
        "websockets",
    ]
    failed: list[str] = []
    for name in mods:
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 — surface all import errors
            failed.append(f"  {name}: {exc}")
    if failed:
        print("iraap doctor: FAILED", file=sys.stderr)
        print("\n".join(failed), file=sys.stderr)
        return 1
    print("iraap doctor: OK (imports)")
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="iraap", description="IR-AAP platform CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    init_p = sub.add_parser("init", help="Copy a preset mission YAML to edit (low-code)")
    init_p.add_argument(
        "output",
        nargs="?",
        default="my_robot_mission.yaml",
        help="Output file path (default: ./my_robot_mission.yaml)",
    )
    init_p.add_argument(
        "--preset",
        default="simple_robot",
        metavar="NAME",
        help="Preset name (default: simple_robot); see iraap init --list",
    )
    init_p.add_argument("--list", action="store_true", dest="list_presets", help="List preset names")
    init_p.add_argument("--force", action="store_true", help="Overwrite output if it exists")

    serve_p = sub.add_parser("serve", help="Run API + WebSocket + simulation (run_live)")
    serve_p.add_argument(
        "--profile",
        default=None,
        metavar="PATH",
        help="YAML stack profile (default: config/default_iraap.yaml or IRAAP_PROFILE env)",
    )
    serve_p.add_argument(
        "--mock-world",
        action="store_true",
        help="Force MockPhysicsWorld (no PyBullet)",
    )

    sub.add_parser("doctor", help="Check that core modules import correctly")

    args = p.parse_args(argv)

    if args.cmd == "init":
        return _cmd_init(args)

    if args.cmd == "doctor":
        return _doctor()

    if args.cmd == "serve":
        import run_live  # noqa: PLC0415 — only when serving

        run_live.main(
            getattr(args, "profile", None),
            world_backend="mock" if getattr(args, "mock_world", False) else None,
        )
        return 0

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

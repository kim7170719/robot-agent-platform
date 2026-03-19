# Contributing to IR-AAP

## Setup

```bash
cd robot-agent-platform
make dev
# or: pip install -e ".[dev]"
```

Optional: copy `.env.example` to `.env` and set `IRAAP_API_HOST`, `IRAAP_API_PORT`, `IRAAP_WS_PORT`.

With the stack running, **`/wizard`** serves a browser form to generate a mission YAML (download and pass to `iraap serve --profile`).

## Commands

| Command | Purpose |
|---------|---------|
| `pytest tests/ -v` | Full test suite |
| `ruff check .` | Lint |
| `iraap init` / `iraap init --list` | Copy a commented preset YAML for integrators |
| `python run_live.py` | API + WebSocket + simulation |
| `iraap serve` | Same as `run_live` (after install) |
| `iraap doctor` | Import smoke check |

## Guidelines

- Prefer the **event bus** for cross-module data; avoid tight imports between plugins and agents where possible.
- World / PyBullet mutations should go through paths that respect `Engine.step_lock` (REST handlers and WS commands are serialized with the sim loop).
- Add tests under `tests/` for new behavior; keep `examples/` for demos.

## Docs

- Event type reference: [docs/events.md](docs/events.md)
- Simulation layout & physics (obstacles, dynamic props, gravity): [docs/simulation_physics.md](docs/simulation_physics.md)

## Third-party extensions (entry points)

Register extra plugins/agents without editing core code:

1. In your package `pyproject.toml`:

```toml
[project.entry-points."iraap.register"]
acme = "acme_iraap.ext:register"

# acme_iraap/ext.py
# def register(registry):
#     registry.register_plugin("acme_sensor", AcmeSensor)
```

2. `pip install` your package; `build_live_stack()` calls every `iraap.register` entry point on a fresh `PluginRegistry` after built-ins are registered.

Built-in keys are defined in `core/builtin_entrypoints.py`. YAML profiles reference them via `plugins[].type` / `agents[].type`.

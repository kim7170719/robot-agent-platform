# Presets (`iraap init`)

Built-in YAML templates for **low-code** setup.

| Preset             | Use case                                      |
|--------------------|-----------------------------------------------|
| `simple_robot`     | One named robot + goal + full sensor stack    |
| `forklift_minimal` | Single vehicle (`forklift`) + fewer rays      |
| `agv_patrol`       | Two AGVs (`agv_alpha` / `agv_bravo`) + A*     |

```bash
iraap init --list
iraap init ./my_mission.yaml --preset agv_patrol
iraap serve --profile ./my_mission.yaml
```

**Web wizard (no CLI):** with the API running, open **`/wizard`** — form → download `mission.yaml`.

For **real hardware**, keep the same YAML shape; add a vendor plugin via `iraap.register` (see CONTRIBUTING.md).

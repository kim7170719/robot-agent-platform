# Event catalog

IR-AAP events are published on the `EventBus`. Payload shapes are validated by
`SchemaValidator` using definitions in `core/event_schema.py` (`SCHEMAS`).

## Registered event types

| Type | Summary |
|------|---------|
| `action.move_3d` | Discrete grid / physics move with positions |
| `sensor.distance_3d` | Multi-direction distance readings |
| `sensor.lidar` | Lidar point cloud |
| `sensor.joint` | Joint positions and velocities |
| `engine.metrics` | Per-step timing |
| `goal.target` | Goal pose update |
| `goal.reached` | Robot reached goal |
| `physics.collision` | Contact / collision report |
| `map.occupancy` | Occupancy grid update |
| `bt.tick` | Behavior tree tick metadata |
| `system.plugin_loaded` | Plugin registered |
| `system.plugin_unloaded` | Plugin removed |
| `system.agent_loaded` | Agent registered |
| `system.agent_unloaded` | Agent removed |
| `task.allocated` | Task assignment |
| `action.hardware` | Hardware / real-world action |
| `ws.command.ack` | WebSocket command result |

For required fields and types per event, see `SCHEMAS` in `core/event_schema.py`.

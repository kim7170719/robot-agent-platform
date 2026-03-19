# IR-AAP — Industrial Robot AI Agent Platform

A low-barrier, highly extensible platform for building perception, decision-making and control systems for industrial robots through a modular plugin-based architecture.

## Architecture

```
Perception Plugins --> Event Bus --> Agent Core --> Action Layer
       |                  |               |              |
  NoiseMixin        Config/Registry   PhysicsAgentBase  SimulatedAction/URDFAction
  CollisionDetector   Recorder/Replayer  AStarAgent     LoggingAction
  JointSensor        WebSocketBridge    BTAgent (BT)   RealWorldAdapter
  TaskAllocator      HotReloadManager   SB3Agent(DQN/PPO)
  OccupancyMap       Observability      RobotEnv (Gymnasium)
                     REST API (FastAPI)  IK Solver
                     Docker + CI        RRT* Planner
                                        GraspPlanner
```

All modules communicate exclusively through the **Event Bus** using a unified data protocol. No direct coupling between modules.

## Quick Start

```bash
cd robot-agent-platform

# Install dependencies (editable + dev tools: pytest, ruff)
pip install -e ".[dev]"
# or: pip install -r requirements.txt

# Run 2D demo
python -m examples.run_demo

# Run 3D physics demo
python -m examples.run_physics_demo

# Run RL training demo
python -m examples.run_rl_demo

# Generate futuristic visualization
python -m examples.visualize_physics

# Run all tests
pytest tests/ -v

# Full stack: API + WebSocket + sim loop + interactive dashboard
python run_live.py
# or: iraap serve
# Open http://localhost:8000/dashboard (override host/ports via .env — see .env.example)
```

### 採購整機 / 想「少寫程式」時（低門檻）

1. **一鍵產生可編輯任務檔**（內建註解說明要改哪幾行）  
   `iraap init --list` → `iraap init ./我的任務.yaml`（預設複製 `simple_robot` 範本）  
2. **用記事本改**：機器人名字與出生點、障礙物、目標點、要掛哪些感測（`plugins`）與用哪種行為（`agents`）。  
3. **跑模擬**：`iraap serve --profile ./我的任務.yaml`，瀏覽器開儀表板操作。  

真機時：YAML **結構不變**；由整合商提供一個透過 `iraap.register` 註冊的「驅動 / 視覺」外掛，把硬體資料丟上 **Event Bus**，你仍主要改設定檔即可。詳見 `CONTRIBUTING.md` 與 `iraap/presets/README.md`。

- **更多一鍵範本**：`iraap init --list` 含 `forklift_minimal`、`agv_patrol` 等。  
- **瀏覽器表單**：服務啟動後開 **`http://localhost:8000/wizard`**，填表後下載 `mission.yaml`，再 `iraap serve --profile mission.yaml`。

## Project Structure

```
robot-agent-platform/
├── core/
│   ├── __init__.py        # Public API facade
│   ├── engine.py          # Main execution loop (Config-aware)
│   ├── event_bus.py       # Pub/sub Event Bus
│   ├── config.py          # Centralized YAML/JSON config
│   ├── recorder.py        # Event recording to JSONL
│   ├── replayer.py        # Event replay from JSONL (offline debug)
│   ├── registry.py        # Plugin/Agent discovery registry
│   ├── ws_bridge.py       # WebSocket bridge plugin for dashboard
│   └── observability.py   # Structured logging + MetricsCollector
├── agents/
│   ├── __init__.py
│   ├── base_agent.py          # ABC for all agents
│   ├── physics_agent_base.py  # Shared 3D agent logic (DRY, ActionLayer-aware)
│   ├── simple_agent.py        # 2D grid agent
│   ├── physics_agent.py       # 3D obstacle-avoidance agent
│   ├── goal_agent.py          # 3D goal-directed agent
│   ├── astar_agent.py         # A* global path planning agent
│   ├── bt_nodes.py            # Behavior Tree primitives
│   └── bt_agent.py            # BT-driven composable agent
├── actions/
│   ├── __init__.py
│   ├── base_action.py         # ActionLayer ABC
│   ├── simulated_action.py    # PhysicsWorld movement (default)
│   ├── logging_action.py      # Decorator: publishes action.executed events
│   ├── urdf_action.py         # Joint-space control for URDF robots + IK
│   └── real_action.py         # RealWorldAdapter for hardware bridges
├── plugins/
│   ├── __init__.py
│   ├── base_plugin.py         # ABC for all plugins
│   ├── noise_mixin.py         # Shared noise generation (DRY)
│   ├── mock_camera.py         # 2D mock camera
│   ├── mock_distance.py       # 2D mock distance
│   ├── physics_camera.py      # 3D PyBullet depth camera
│   ├── physics_distance.py    # 3D PyBullet ray-cast distance
│   ├── physics_lidar.py       # 3D LiDAR batch ray-cast
│   ├── goal_publisher.py      # Goal target publisher
│   ├── collision_detector.py  # PyBullet collision event publisher
│   ├── joint_sensor.py        # URDF joint state sensor
│   └── occupancy_map.py       # SLAM-lite 3D occupancy grid
├── simulator/
│   ├── __init__.py
│   ├── world_interface.py     # ABC for 2D/3D worlds
│   ├── world.py               # 2D GridWorld
│   ├── physics_world.py       # 3D PyBullet PhysicsWorld (multi-robot)
│   ├── urdf_robot.py          # URDF articulated robot model
│   └── sensor_sim.py          # 2D sensor simulation
├── rl/
│   ├── __init__.py
│   ├── robot_env.py           # RobotEnv(gymnasium.Env)
│   ├── q_learning_agent.py    # Tabular Q-learning agent
│   └── sb3_agent.py           # DQN/PPO via stable-baselines3
├── planning/
│   ├── __init__.py
│   ├── rrt.py                 # RRT* continuous 3D path planner
│   └── grasp_planner.py       # Pick-and-place grasp trajectory
├── api/
│   ├── __init__.py
│   └── server.py              # FastAPI REST server
├── dashboard/
│   ├── __init__.py
│   ├── server.py              # WebSocket dashboard server
│   └── index.html             # Futuristic real-time monitoring UI
├── examples/
│   ├── run_demo.py            # 2D demo
│   ├── run_physics_demo.py    # 3D physics demo
│   ├── run_full_demo.py       # Multi-feature integrated demo
│   ├── run_rl_demo.py         # RL training demo
│   └── visualize_physics.py   # Futuristic 8-panel visualization
├── tests/
│   ├── conftest.py            # PyBullet cleanup fixtures
│   ├── test_core.py           # 2D module tests
│   ├── test_physics.py        # 3D physics tests
│   └── test_upgrades.py       # Upgrade module tests
├── Dockerfile                 # Production container image
├── docker-compose.yml         # Multi-service orchestration
├── .github/workflows/ci.yml  # GitHub Actions CI pipeline
├── requirements.txt
└── README.md
```

## Unified Data Protocol

Every event flowing through the bus follows this format:

```json
{
  "timestamp": 1710840000000,
  "type": "sensor.camera",
  "source": "mock_camera",
  "data": {}
}
```

## Action Layer

The Action Layer decouples agent decisions from world execution. Agents call `action_layer.execute(command)` instead of directly interacting with the world.

```python
from actions import SimulatedAction, LoggingAction, URDFAction

# Sphere robot (default)
action = SimulatedAction(world, robot_id="scout")
result = action.execute({"direction": "forward"})

# With event logging
logged = LoggingAction(action, bus)
result = logged.execute({"direction": "forward"})  # also publishes action.executed

# URDF joint control
urdf_action = URDFAction(urdf_robot, world.client_id)
result = urdf_action.execute({"joint_positions": [0.5, -0.3]})
result = urdf_action.execute({"direction": "forward"})  # preset mapping
```

## REST API (FastAPI)

Full HTTP interface for remote simulation control. See **Swagger** at `/docs` for the complete list.

**Preferred prefix:** `/api/v1/...` (documented in OpenAPI). The same routes are also mounted at `/api/...` for backward compatibility (hidden from the schema).

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/v1/health` (alias `/api/health`) | GET | Health check (`paused` reflects `engine.sim_paused`) |
| `/api/v1/state` | GET | Robots, obstacles, **dynamics**, plugins, agents |
| `/api/v1/step` | POST | Advance simulation N steps |
| `/api/v1/pause` / `/api/v1/resume` | POST | Pause / resume the **live** sim loop (`run_live.py`) |
| `/api/v1/reset` | POST | Reset world |
| `/api/v1/goal` | POST | Set goal target |
| `/api/v1/robots/*`, `/api/v1/obstacles/*` | POST/DELETE | Add/remove robots and obstacles |
| `/api/v1/robots/position` | POST | Teleport sphere robot (`robot_id`, `x`, `y`, optional `z`) |
| `/api/v1/dynamics/box` | POST | Spawn dynamic test cube (`mass`, `half_extent`) |
| `/api/v1/dynamics` | DELETE | Clear all dynamic bodies |
| `/api/v1/manipulator/state` | GET | URDF arm snapshot (requires profile `manipulator.enabled` + live stack) |
| `/api/v1/manipulator/gripper` | POST | Set prismatic gripper `openness` 0–1 (JSON body) |
| `/api/v1/config` | GET / PATCH | Read / patch config keys |
| `/api/v1/config/save` | POST | Persist config to project dir (`.yaml` / `.json` by extension) |
| `/api/v1/replay/sessions` | GET | List `*.jsonl` recordings in project root |
| `/api/v1/replay/events` | GET | Paginated events from a session file |
| `/api/v1/metrics` | GET | Engine step timing |
| `/api/v1/events` | GET | Recent event history |

Errors return JSON: `{"error": {"code": "...", "message": "..."}}` (and optional `details` for validation).

```python
from api import create_app
app = create_app(engine, bus, world, config=cfg, goal_publisher=gp)
# Use uvicorn or FastAPI TestClient
```

### Config file I/O

```python
cfg = Config.load_file("settings.yaml")   # or Config.from_file(...)
cfg.set("sensor.noise_level", 0.05)
cfg.save("settings.yaml")               # PyYAML required for .yaml/.yml
```

## Real-time Dashboard

WebSocket-based live monitoring (`ws://host:8765` with `run_live.py`). The HTML dashboard is served at **`/dashboard`** when the API is running.

```python
from core.ws_bridge import WebSocketBridge
bridge = WebSocketBridge(bus)
engine.register_plugin(bridge)
```

**Features:** **Three.js 3D scene** (robots, static obstacles, **dynamic test cubes** in green, goal torus, orbit controls; optional **2D** top-down), trails, event stream, REST + optional **WebSocket commands** (`ping`, `step`, `pause`, `resume`, `reset`, `goal` — see `core/ws_commands.py`). **Tabs:** *Sim* (step / goal / move), *World* (add robot with Z, **teleport** `POST /robots/position`, obstacles, **dynamics** spawn/clear, manipulator gripper when backend has URDF), *Params* (sliders + config patch), ***設定*** (edit API origin / prefix / WebSocket URL, save + reload), *Plugins*, *Replay*. **Auto-sync** option polls `/state` every 2s. Defaults to **`/api/v1`**; override with query `?api=...&api_prefix=/api/v1&ws=ws://host:8765` or `localStorage` keys `iraap.api`, `iraap.api_prefix`, `iraap.ws`, `iraap.ws_port`.

## 模擬場景與物理（布置測試物、重力、落體）

場景可用 YAML 描述：**靜態障礙**、**可動剛體（質量方塊，例如自由落體）**、**重力與時間步長**。說明見 **[docs/simulation_physics.md](docs/simulation_physics.md)**。  
`GET /api/v1/state` 回傳 **`dynamics`**（可動物位置與線速度），便於做偵測或記錄。

## Docker

```bash
docker compose up --build
# API + dashboard + sim: http://localhost:8000/dashboard  ·  WS :8765
```

Single-container full stack (`run_live.py`). PyBullet is **not** multi-worker safe — run **one** replica per simulation.

**Engine:** On `start()`, plugins are **topologically sorted** by optional `requires: list[str]` (see `core/plugin_deps.py`).

## Gymnasium RL Environment

Standard Gymnasium wrapper for training RL agents.

```python
from rl import RobotEnv, QLearningAgent

env = RobotEnv(goal=(5.0, 0.0, 0.31), max_steps=200)
obs, info = env.reset()

for _ in range(100):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()

env.close()
```

Compatible with stable-baselines3, RLlib, and any Gymnasium-compliant framework.

## Deep RL (DQN / PPO via stable-baselines3)

Production-ready DQN and PPO training pipelines with built-in save/load and evaluation.

```python
from rl import SB3Agent

agent = SB3Agent(algorithm="DQN", env_kwargs={"max_steps": 200})
agent.train(total_timesteps=10000)
result = agent.evaluate(n_eval_episodes=10)
print(result)  # {'mean_reward': ..., 'std_reward': ..., 'n_episodes': 10}
agent.save("models/my_dqn")

# Load and use
agent2 = SB3Agent(algorithm="DQN")
agent2.load("models/my_dqn")
action = agent2.predict(obs)
```

Also supports PPO: `SB3Agent(algorithm="PPO", model_kwargs={"n_steps": 128})`.

## Behavior Tree Agent

Composable multi-phase missions via a full BT node library.

```python
from agents import BTAgent, Sequence, Selector, Condition, ActionNode, Status

has_target = Condition(lambda: robot_has_goal(), name="has_target?")
navigate  = ActionNode(lambda: move_to_goal(),   name="navigate")
scan      = ActionNode(lambda: scan_area(),       name="scan")

tree = Selector([
    Sequence([has_target, navigate]),
    scan,
])

agent = BTAgent(bus, tree, agent_id="patrol_bot")
engine.register_agent(agent)
engine.run(steps=100)
print(agent.get_summary())
```

Node types: `Sequence`, `Selector`, `Condition`, `ActionNode`, `Inverter`, `RepeatUntilSuccess`, `Parallel`.

## Occupancy Map (SLAM-lite)

Incremental 3D occupancy grid built from distance and LiDAR sensor events. Supports optional cell decay for dynamic environments.

```python
from plugins import OccupancyMap

omap = OccupancyMap(bus, resolution=0.5, decay_enabled=True, decay_steps=50)
engine.register_plugin(omap)
engine.run(steps=20)

print(omap.cell_count())        # {'unknown': 5, 'free': 42, 'occupied': 8, 'total': 55}
print(omap.is_free(3.0, 0, 0))  # True
print(omap.get_occupied_cells()) # [(5, 0, 0), (3, 2, 0), ...]
```

## Observability (Structured Logging + Prometheus Metrics)

Production monitoring with JSON-formatted logs and Prometheus-compatible metrics.

```python
from core import ObservabilityPlugin, MetricsCollector

metrics = MetricsCollector()
obs = ObservabilityPlugin(bus, metrics=metrics)
engine.register_plugin(obs)
engine.run(steps=10)

print(metrics.to_json())            # counters, gauges, histograms
print(metrics.to_prometheus())      # Prometheus text format
```

API endpoints: `GET /api/metrics/prometheus` and `GET /api/metrics/json`.

## RRT* Motion Planner

Sampling-based continuous 3D path planning with obstacle avoidance and path smoothing.

```python
from planning import RRTStar

planner = RRTStar(
    bounds_min=(-10, -10, 0), bounds_max=(10, 10, 5),
    step_size=0.5, goal_radius=0.5,
    collision_fn=lambda x, y, z: occupancy_map.is_occupied(x, y, z),
)
path = planner.plan(start=(0, 0, 0), goal=(8, 3, 0))
smoothed = planner.smooth_path(path)
```

## Grasp Planning (Pick-and-Place)

Full pick-and-place trajectory planning for URDF robot arms.

```python
from planning import GraspPlanner

planner = GraspPlanner(urdf_robot, approach_height=0.15)
plan = planner.plan_pick_and_place(
    pick_target=(0.1, 0.0, 0.2),
    place_target=(0.2, 0.0, 0.2),
)
# 6 phases: approach → grasp → lift → transport → place → retreat

commands = planner.get_action_commands(plan)
for cmd in commands:
    result = urdf_action.execute(cmd)

print(planner.evaluate_plan(plan))
# {'phases': 6, 'feasible': True, 'avg_ik_error': 0.003, ...}
```

## Inverse Kinematics (IK Solver)

URDFRobot includes a built-in IK solver powered by PyBullet's damped least-squares algorithm. Move end-effectors to target Cartesian positions without manual joint calculation.

```python
robot = URDFRobot(world.client_id)

# Low-level: solve for joint angles
joints = robot.solve_ik(target_position=(0.1, 0.0, 0.3))

# High-level: move and verify
result = robot.move_end_effector_to(0.1, 0.0, 0.3, sim_steps=240)
print(result["achieved"])  # [0.098, 0.001, 0.302]
print(result["error"])     # 0.003

# Via ActionLayer
action = URDFAction(robot, world.client_id)
result = action.execute({"target_position": [0.1, 0.0, 0.3]})
```

## Real-World Hardware Adapter

`RealWorldAdapter` bridges the platform's ActionLayer to physical robot hardware through pluggable transports and safety validation.

```python
from actions import RealWorldAdapter, DryRunTransport, CommandValidator

# Dry-run mode (no hardware needed)
transport = DryRunTransport()
validator = CommandValidator(joint_limits=[(-1.57, 1.57), (-2.0, 2.0)], max_velocity=2.0)
adapter = RealWorldAdapter(transport, event_bus=bus, validator=validator)

adapter.connect()
result = adapter.execute({"joint_positions": [0.5, -0.3]})

# Emergency stop
validator.activate_e_stop()
result = adapter.execute({"direction": "forward"})  # blocked

# For real hardware, subclass HardwareTransport:
# class ROSTransport(HardwareTransport): ...
# class SerialTransport(HardwareTransport): ...
```

## URDF Robot Models

Articulated URDF robots with joint-level control, coexisting with sphere robots.

```python
from simulator.urdf_robot import URDFRobot
from plugins import JointSensor
from actions import URDFAction

robot = URDFRobot(world.client_id, position=(0, 0, 0))
print(robot.num_joints)           # 3 (bundled box-arm + gripper_slide)
robot.set_joint_positions([0.5, -0.3])
robot.set_gripper_openness(0.8)   # prismatic jaw (demo)
ee = robot.get_end_effector_position()

# Sensor plugin
sensor = JointSensor(bus, robot)
engine.register_plugin(sensor)    # publishes sensor.joints events

# Action layer
action = URDFAction(robot, world.client_id)
result = action.execute({"joint_positions": [1.0, -0.5]})
```

## Plugin Interface

All plugins extend `BasePlugin`. Only `on_update()` needs implementing; the active-flag lifecycle (`start`/`stop`) is managed by the base class.

| Method       | Purpose                              |
|--------------|--------------------------------------|
| `start()`    | Activate the plugin                  |
| `on_update()`| Read sensor data, publish to bus     |
| `stop()`     | Deactivate the plugin                |

## Agent Interface

All agents extend `BaseAgent`:

| Method      | Purpose                             |
|-------------|-------------------------------------|
| `perceive()`| Collect sensor events from bus      |
| `decide()`  | Choose an action                    |
| `act()`     | Execute via ActionLayer             |
| `cleanup()` | Unsubscribe from bus on teardown    |

## DRY Abstractions

| Module              | Eliminates duplication in                        |
|---------------------|--------------------------------------------------|
| `NoiseMixin`        | PhysicsCamera, PhysicsDistance, PhysicsLidar      |
| `PhysicsAgentBase`  | PhysicsAgent, GoalAgent, AStarAgent              |

## Event Replay System

Record events during simulation, then replay them on a different EventBus for offline debugging and regression testing.

```python
from core import Recorder, Replayer, EventBus

bus = EventBus()
rec = Recorder(bus)
rec.start()
# ... run simulation ...
rec.stop()
rec.save()

bus2 = EventBus()
replayer = Replayer(bus2)
replayer.load("recording.jsonl")
replayer.replay(speed=0)          # instant
replayer.replay(speed=1.0)        # real-time paced
replayer.replay(event_filter="sensor.distance_3d")  # filtered
replayer.step()                   # one event at a time
```

## Multi-Agent Coordination

Multiple robots operate simultaneously in the same PhysicsWorld, each controlled by an independent agent communicating through the shared EventBus.

```python
world = PhysicsWorld(gui=False)
world.add_robot("scout", x=5, y=0)
world.add_robot("worker", x=-3, y=2)

agent_a = PhysicsAgent(bus, world, robot_id="default")
agent_b = GoalAgent(bus, world, robot_id="scout")
agent_c = AStarAgent(bus, world, robot_id="worker")
```

## A* Path Planning

`AStarAgent` builds an occupancy grid from sensor data, plans an optimal global path using A*, then follows waypoints while avoiding obstacles reactively. Re-plans when blocked or a new goal arrives.

```python
from agents import AStarAgent
from plugins import GoalPublisher

agent = AStarAgent(bus, world)
gp = GoalPublisher(bus)
gp.set_target(10.0, 5.0)
engine.run(steps=50)
print(agent.path)              # remaining waypoints
print(agent.blocked_cells)     # discovered obstacles
```

## Collision Detection

`CollisionDetector` publishes `physics.collision` events when robots contact obstacles, ground, or other robots.

```python
from plugins import CollisionDetector

detector = CollisionDetector(bus, world, include_ground=False, force_threshold=0.01)
engine.register_plugin(detector)
```

## Event Bus Features

| Feature | Description |
|---------|-------------|
| Typed subscriptions | Subscribe to specific event types |
| Wildcard (`"*"`) | Receive all events for logging/monitoring |
| Bounded history | `max_history` cap prevents memory leaks; `0` disables recording |
| Type-indexed lookup | `get_history(type)` is O(1) via internal index |
| Exception isolation | A failing subscriber never blocks other subscribers |
| Publish counter | `bus.publish_count` tracks total events without scanning history |

## Engine Features

| Feature | Description |
|---------|-------------|
| Deterministic loop | `plugin.update() -> agent.perceive() -> decide() -> act()` |
| Config-aware | Optionally reads settings from `Config` instance |
| Step metrics | Per-step wall-clock timing, published as `engine.metrics` events |
| `get_metrics()` | Summary: steps, total/avg/min/max ms |
| Exception isolation | Failing plugins/agents are skipped with a warning, loop continues |
| Agent cleanup | `engine.stop()` calls `agent.cleanup()` to prevent callback leaks |

## Config System

Centralized hierarchical configuration with dot-path access, defaults, and JSON/YAML file support.

```python
from core import Config

cfg = Config({"world": {"gui": True}})
gui = cfg.get("world.gui")        # True
cfg.set("sensor.noise_level", 0.3)
cfg.save("my_config.json")
```

## Plugin Registry

Register plugins and agents by name for dynamic discovery and factory creation.

```python
from core import PluginRegistry

registry = PluginRegistry()

@registry.plugin("my_sensor")
class MySensor(BasePlugin): ...

sensor = registry.create_plugin("my_sensor", event_bus=bus)
```

## Docker & CI/CD

### Docker

```bash
# Build and run API server
docker build -t iraap .
docker run -p 8000:8000 iraap

# Full stack (API + Dashboard)
docker-compose up

# Run tests in container
docker-compose --profile test run tests
```

### GitHub Actions CI

The CI pipeline (`.github/workflows/ci.yml`) runs on every push/PR:

1. **Lint** — `ruff check` for code quality
2. **Test** — `pytest` on Python 3.11 + 3.12 matrix
3. **Docker** — Build image + smoke test

## Design Principles

- **Plugin-first** — extend via plugins, not by modifying core
- **Decoupled** — all data flows through the Event Bus
- **Action Layer** — agents are world-agnostic; swap SimulatedAction for real hardware
- **Test-first** — every module has unit, edge case, and mock data tests (516 tests)
- **DRY** — `NoiseMixin` and `PhysicsAgentBase` eliminate cross-module duplication
- **Multi-agent** — multiple robots coexist and coordinate via the EventBus
- **RL-ready** — standard Gymnasium interface for any RL framework
- **REST API** — full HTTP control via FastAPI
- **Real-time dashboard** — WebSocket-based live monitoring
- **URDF support** — articulated robot models with joint-level control
- **Mock-first** — simulate hardware, no real devices required
- **Config-driven** — centralized settings, no hardcoded values scattered in modules
- **IK Solver** — Cartesian end-effector targets via inverse kinematics
- **Deep RL** — DQN/PPO training pipelines via stable-baselines3
- **Hardware-ready** — RealWorldAdapter with e-stop, joint limits, pluggable transports
- **Containerized** — Dockerfile + docker-compose for one-command deployment
- **Behavior Trees** — composable multi-phase missions via BT nodes
- **SLAM-lite** — incremental occupancy mapping from sensor events
- **Observability** — structured JSON logging + Prometheus metrics export
- **RRT\*** — continuous-space sampling-based path planning with rewiring
- **Grasp Planning** — pick-and-place trajectory with IK-solved waypoints
- **CI/CD** — GitHub Actions: lint, test (matrix), Docker smoke test
- **Deterministic** — discrete-step execution, fully reproducible
- **Fault-tolerant** — exceptions are isolated, never crash the loop

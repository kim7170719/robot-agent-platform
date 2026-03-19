"""
run_full_demo.py — comprehensive simulation demo showcasing all IR-AAP features.

Demonstrates:
  - Multi-robot PhysicsWorld (3 robots)
  - A* path planning agent vs GoalAgent vs reactive PhysicsAgent
  - CollisionDetector publishing contact events
  - Sensor noise (distance, camera, LiDAR)
  - Recorder capturing everything
  - Replayer verifying recorded events
  - Engine metrics

Produces a futuristic 8-panel visualization PNG.
"""

from __future__ import annotations

import math
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

from core.event_bus import EventBus
from core.engine import Engine
from core.config import Config
from core.recorder import Recorder
from core.replayer import Replayer
from simulator.physics_world import PhysicsWorld, DIRECTIONS_3D
from plugins.physics_camera import PhysicsCamera
from plugins.physics_distance import PhysicsDistance
from plugins.physics_lidar import PhysicsLidar
from plugins.goal_publisher import GoalPublisher
from plugins.collision_detector import CollisionDetector
from agents.physics_agent import PhysicsAgent
from agents.goal_agent import GoalAgent
from agents.astar_agent import AStarAgent

BG_DARK = "#0a0e17"
BG_PANEL = "#0f1623"
CYAN = "#00f0ff"
CYAN_DIM = "#00607a"
MAGENTA = "#ff00aa"
GREEN = "#00ff88"
ORANGE = "#ff8800"
RED = "#ff2255"
YELLOW = "#ffe033"
WHITE = "#e0e8f0"
GRID_COLOR = "#1a2540"

ROBOT_COLORS_MAP = {"default": CYAN, "scout": GREEN, "planner": MAGENTA}


def collect_data(steps: int = 30) -> dict:
    cfg = Config({
        "engine": {"enable_metrics": True},
        "sensor": {"noise_level": 0.3},
    })
    bus = EventBus(max_history=100_000)
    world = PhysicsWorld(gui=False, timestep=1 / 240)

    world.add_robot("scout", x=0, y=4)
    world.add_robot("planner", x=-3, y=0)

    obstacles = [
        (3, 0, 0.5), (3, 2, 0.5), (-2, 1, 0.5),
        (1, 3, 0.5), (5, -1, 0.5), (-1, -3, 0.5),
        (4, 4, 0.5), (6, 2, 0.5), (2, -2, 0.5),
    ]
    for ox, oy, oz in obstacles:
        world.place_obstacle(ox, oy, oz)

    goal_pub = GoalPublisher(bus)
    goal_pub.set_target(6.0, -3.0, 0.31)

    dist_default = PhysicsDistance(bus, world, noise_level=0.3, seed=42)
    camera = PhysicsCamera(bus, world, width=16, height=16, noise_level=0.02, seed=42)
    lidar = PhysicsLidar(bus, world, num_rays=36, noise_level=0.1, seed=42)
    collision = CollisionDetector(bus, world, include_ground=False, force_threshold=0.1)

    agent_default = GoalAgent(bus, world, robot_id=None)
    agent_scout = PhysicsAgent(bus, world, robot_id="scout")
    agent_planner = AStarAgent(bus, world, robot_id="planner")

    recorder = Recorder(bus)
    recorder.start()

    engine = Engine(bus, config=cfg)
    engine.register_plugin(dist_default)
    engine.register_plugin(camera)
    engine.register_plugin(lidar)
    engine.register_plugin(goal_pub)
    engine.register_plugin(collision)
    engine.register_agent(agent_default)
    engine.register_agent(agent_scout)
    engine.register_agent(agent_planner)

    positions = {
        "default": [world.get_robot_position("default")],
        "scout": [world.get_robot_position("scout")],
        "planner": [world.get_robot_position("planner")],
    }

    def on_step(step: int) -> None:
        for name in positions:
            positions[name].append(world.get_robot_position(name))

    engine.run(steps=steps, on_step=on_step)
    recorder.stop()

    recorder.save()

    replay_bus = EventBus()
    replayer = Replayer(replay_bus)
    replayer.load_events(recorder.get_events())
    replay_count = replayer.replay(speed=0)

    result = {
        "positions": positions,
        "obstacles": obstacles,
        "goal": (6.0, -3.0, 0.31),
        "robot_ids": ["default", "scout", "planner"],
        "distance_events": bus.get_history("sensor.distance_3d"),
        "camera_events": bus.get_history("sensor.camera_3d"),
        "lidar_events": bus.get_history("sensor.lidar"),
        "metric_events": bus.get_history("engine.metrics"),
        "action_events": bus.get_history("action.move_3d"),
        "goal_events": bus.get_history("goal.reached"),
        "collision_events": bus.get_history("physics.collision"),
        "engine_metrics": engine.get_metrics(),
        "recorder_summary": recorder.get_summary(),
        "replay_count": replay_count,
        "astar_path": agent_planner.path,
        "astar_blocked": len(agent_planner.blocked_cells),
        "collision_total": collision.collision_count,
    }
    world.disconnect()
    return result


def _setup_style():
    plt.rcParams.update({
        "figure.facecolor": BG_DARK,
        "axes.facecolor": BG_PANEL,
        "axes.edgecolor": CYAN_DIM,
        "axes.labelcolor": WHITE,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "text.color": WHITE,
        "xtick.color": CYAN_DIM,
        "ytick.color": CYAN_DIM,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.5,
        "font.family": "monospace",
        "font.size": 8,
    })


def _draw_3d_multi(ax, data):
    obstacles = data["obstacles"]
    goal = data["goal"]

    for name in data["robot_ids"]:
        pos = data["positions"][name]
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        zs = [p[2] for p in pos]
        color = ROBOT_COLORS_MAP.get(name, CYAN)
        ax.plot(xs, ys, zs, color=color, linewidth=2, alpha=0.85, label=name.upper())
        ax.scatter(xs[0], ys[0], zs[0], c=color, s=80, marker="^", edgecolors="white", linewidths=0.5)
        ax.scatter(xs[-1], ys[-1], zs[-1], c=color, s=100, marker="*", edgecolors="white", linewidths=0.5)

    ax.scatter(*goal, c=ORANGE, s=200, marker="D", edgecolors="white", linewidths=1, zorder=10, label="GOAL")

    for o in obstacles:
        ax.bar3d(o[0] - 0.4, o[1] - 0.4, 0, 0.8, 0.8, o[2] + 0.5,
                 color=RED, alpha=0.1, edgecolor=RED, linewidth=0.5)

    ax.set_xlabel("X", labelpad=5)
    ax.set_ylabel("Y", labelpad=5)
    ax.set_zlabel("Z", labelpad=5)
    ax.set_title("MULTI-ROBOT  3D  TRAJECTORIES", color=CYAN, pad=10)
    ax.legend(fontsize=6, loc="upper left", facecolor=BG_PANEL, edgecolor=CYAN_DIM, labelcolor=WHITE)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=CYAN_DIM, labelsize=6)


def _draw_topdown_multi(ax, data):
    obstacles = data["obstacles"]
    goal = data["goal"]

    for name in data["robot_ids"]:
        pos = data["positions"][name]
        xs = [p[0] for p in pos]
        ys = [p[1] for p in pos]
        color = ROBOT_COLORS_MAP.get(name, CYAN)
        ax.plot(xs, ys, color=color, linewidth=1.5, alpha=0.8, label=name.upper())
        ax.plot(xs[0], ys[0], "^", color=color, markersize=8, markeredgecolor="white", markeredgewidth=0.5)
        ax.plot(xs[-1], ys[-1], "*", color=color, markersize=10, markeredgecolor="white")

    lidar_events = data["lidar_events"]
    if lidar_events:
        last = lidar_events[-1]["data"]
        rp = last["robot_position"]
        for ray in last.get("scan", []):
            a = ray["angle_rad"]
            d = min(ray["distance"], 8)
            ex = rp[0] + math.cos(a) * d
            ey = rp[1] + math.sin(a) * d
            ax.plot([rp[0], ex], [rp[1], ey], color=RED if ray["hit"] else GREEN, alpha=0.12, linewidth=0.6)

    ax.plot(goal[0], goal[1], "D", color=ORANGE, markersize=10, markeredgecolor="white", zorder=10)
    for o in obstacles:
        r = FancyBboxPatch((o[0]-0.45, o[1]-0.45), 0.9, 0.9, boxstyle="round,pad=0.05",
                           facecolor=RED, alpha=0.2, edgecolor=RED, linewidth=0.8)
        ax.add_patch(r)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("TACTICAL  MAP  (3 ROBOTS)", color=CYAN, pad=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color=GRID_COLOR)
    ax.legend(fontsize=6, loc="lower left", facecolor=BG_PANEL, edgecolor=CYAN_DIM, labelcolor=WHITE)


def _draw_sensor_heatmap(ax, data):
    distance_events = data["distance_events"]
    directions = list(DIRECTIONS_3D.keys())
    n = len(distance_events)
    if n == 0:
        ax.text(0.5, 0.5, "NO DATA", transform=ax.transAxes, ha="center", color=RED, fontsize=12)
        return
    mat = np.zeros((len(directions), n))
    for si, ev in enumerate(distance_events):
        for di, d in enumerate(directions):
            mat[di, si] = ev["data"]["distances"].get(d, {}).get("distance", 0)
    neon = LinearSegmentedColormap.from_list("neon", [
        (0.0, "#ff0044"), (0.15, "#ff4400"), (0.3, "#ff8800"),
        (0.5, "#ffcc00"), (0.7, "#00cc66"), (1.0, "#003322"),
    ])
    im = ax.imshow(mat, aspect="auto", cmap=neon, interpolation="bilinear", vmin=0, vmax=20)
    ax.set_yticks(range(len(directions)))
    ax.set_yticklabels([d.upper() for d in directions], fontsize=7)
    ax.set_xlabel("Step")
    ax.set_title("DISTANCE  SENSOR  HEATMAP", color=CYAN, pad=10)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Dist", color=WHITE, fontsize=7)
    cbar.ax.tick_params(colors=CYAN_DIM, labelsize=6)


def _draw_lidar_polar(ax, data):
    lidar_events = data["lidar_events"]
    if not lidar_events:
        ax.text(0.5, 0.5, "NO LIDAR", transform=ax.transAxes, ha="center", color=RED, fontsize=12)
        return
    last = lidar_events[-1]["data"]
    scan = last.get("scan", [])
    angles = [r["angle_rad"] for r in scan] + [scan[0]["angle_rad"]]
    dists = [r["distance"] for r in scan] + [scan[0]["distance"]]
    hits = [r["hit"] for r in scan] + [scan[0]["hit"]]

    ax.plot(angles, dists, color=CYAN, linewidth=1.5, alpha=0.8)
    ax.fill(angles, dists, alpha=0.06, color=CYAN)
    for a, d, h in zip(angles, dists, hits):
        ax.scatter(a, d, c=RED if h else GREEN, s=10, alpha=0.7, zorder=5)
    ax.set_facecolor(BG_PANEL)
    ax.set_title("LiDAR  360°  SWEEP", color=CYAN, pad=15)
    ax.tick_params(colors=CYAN_DIM, labelsize=6)
    ax.set_rmax(last["max_distance"])
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


def _draw_depth(ax, data):
    cam = data["camera_events"]
    if not cam:
        ax.text(0.5, 0.5, "NO CAMERA", transform=ax.transAxes, ha="center", color=RED, fontsize=12)
        return
    depth = np.array(cam[-1]["data"]["depth"])
    cmap = LinearSegmentedColormap.from_list("d", [
        (0.0, "#ff0044"), (0.3, "#ff8800"), (0.6, "#00ffaa"), (1.0, "#001122"),
    ])
    im = ax.imshow(depth, cmap=cmap, interpolation="nearest")
    ax.set_title("DEPTH  CAMERA", color=CYAN, pad=10)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Depth", color=WHITE, fontsize=7)
    cbar.ax.tick_params(colors=CYAN_DIM, labelsize=6)
    noisy = cam[-1]["data"].get("noisy", False)
    ax.text(0.02, 0.95, "NOISY" if noisy else "CLEAN", transform=ax.transAxes, fontsize=7,
            color=ORANGE if noisy else GREEN, va="top",
            path_effects=[pe.withStroke(linewidth=2, foreground=BG_PANEL)])


def _draw_collisions(ax, data):
    coll = data["collision_events"]
    ax.set_title("COLLISION  EVENTS", color=CYAN, pad=10)

    if not coll:
        ax.text(0.5, 0.6, "0 COLLISIONS", transform=ax.transAxes, ha="center",
                color=GREEN, fontsize=16, fontweight="bold",
                path_effects=[pe.withStroke(linewidth=2, foreground=BG_PANEL)])
        ax.text(0.5, 0.4, "All robots navigated safely", transform=ax.transAxes,
                ha="center", color=CYAN_DIM, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        return

    types: dict[str, int] = {}
    for c in coll:
        t = c["data"]["other_body"]
        types[t] = types.get(t, 0) + 1

    labels = list(types.keys())
    values = list(types.values())
    colors = [
        RED if "obstacle" in lbl else (YELLOW if "ground" in lbl else MAGENTA)
        for lbl in labels
    ]
    bars = ax.barh(labels, values, color=colors, alpha=0.8, edgecolor=WHITE, linewidth=0.5)
    ax.set_xlabel("Count")
    ax.grid(True, axis="x", alpha=0.15, color=GRID_COLOR)
    for bar, v in zip(bars, values):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                str(v), va="center", fontsize=8, color=WHITE)


def _draw_performance(ax, data):
    metrics = data["metric_events"]
    em = data["engine_metrics"]
    times = [ev["data"]["elapsed_ms"] for ev in metrics]
    steps = list(range(1, len(times) + 1))
    avg = em["avg_ms"]

    colors = [RED if t > avg * 2 else (ORANGE if t > avg * 1.3 else CYAN) for t in times]
    edges = [c + "88" if isinstance(c, str) else c for c in colors]
    ax.bar(steps, times, color=colors, alpha=0.85, edgecolor=edges, linewidth=0.3)
    ax.axhline(avg, color=GREEN, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(len(steps)+0.3, avg, f"AVG {avg:.2f}ms", fontsize=6, color=GREEN, va="center")
    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.set_title("ENGINE  PERFORMANCE", color=CYAN, pad=10)
    ax.grid(True, axis="y", alpha=0.15, color=GRID_COLOR)


def _draw_stats(ax, data):
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("SIMULATION  SUMMARY", color=CYAN, pad=10)

    actions = data["action_events"]
    successes = sum(1 for a in actions if a["data"]["success"])
    goal_reached = len(data["goal_events"]) > 0
    em = data["engine_metrics"]
    rs = data["recorder_summary"]

    lines = [
        ("ROBOTS", "3  (GoalAgent · PhysicsAgent · A*Agent)", CYAN),
        ("STEPS", str(em["steps"]), CYAN),
        ("MOVES", f"{successes} success / {len(actions) - successes} blocked", GREEN if successes > 0 else RED),
        ("GOAL", "REACHED" if goal_reached else "IN PROGRESS", GREEN if goal_reached else ORANGE),
        ("COLLISIONS", str(data["collision_total"]), RED if data["collision_total"] > 0 else GREEN),
        ("A* BLOCKED CELLS", str(data["astar_blocked"]), YELLOW),
        ("TOTAL TIME", f"{em['total_ms']:.1f} ms  (avg {em['avg_ms']:.2f} ms/step)", CYAN),
        ("RECORDED EVENTS", str(rs["total_events"]), ORANGE),
        ("REPLAYED EVENTS", str(data["replay_count"]), MAGENTA),
        ("SENSORS", "Distance (noisy) · Camera (noisy) · LiDAR (noisy)", CYAN_DIM),
    ]

    y = 0.92
    for label, value, color in lines:
        ax.text(0.03, y, f"{label}:", transform=ax.transAxes, fontsize=8,
                color=CYAN_DIM, fontweight="bold", va="top")
        ax.text(0.38, y, value, transform=ax.transAxes, fontsize=8,
                color=color, va="top")
        y -= 0.095


def plot_results(data: dict) -> str:
    _setup_style()
    fig = plt.figure(figsize=(24, 16))

    fig.text(0.5, 0.975, "IR-AAP  ·  FULL SYSTEM DEMO", fontsize=24, color=CYAN, ha="center",
             fontfamily="monospace", fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground="#004455")])
    fig.text(0.5, 0.955,
             "Multi-Robot · A* Path Planning · Collision Detection · Event Replay · Sensor Noise",
             fontsize=9, color=CYAN_DIM, ha="center", fontfamily="monospace")

    ax1 = fig.add_subplot(2, 4, 1, projection="3d")
    _draw_3d_multi(ax1, data)

    ax2 = fig.add_subplot(2, 4, 2)
    _draw_topdown_multi(ax2, data)

    ax3 = fig.add_subplot(2, 4, 3)
    _draw_sensor_heatmap(ax3, data)

    ax4 = fig.add_subplot(2, 4, 4, projection="polar")
    _draw_lidar_polar(ax4, data)

    ax5 = fig.add_subplot(2, 4, 5)
    _draw_depth(ax5, data)

    ax6 = fig.add_subplot(2, 4, 6)
    _draw_collisions(ax6, data)

    ax7 = fig.add_subplot(2, 4, 7)
    _draw_performance(ax7, data)

    ax8 = fig.add_subplot(2, 4, 8)
    _draw_stats(ax8, data)

    plt.tight_layout(rect=[0.01, 0.01, 0.99, 0.94])

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "full_demo_results.png"))
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    return out_path


def main() -> None:
    print("=" * 65)
    print("  IR-AAP Full System Demo")
    print("  Multi-Robot · A* · Collision · Replay · Noise")
    print("=" * 65)
    print()

    print("[1/3] Running 30-step simulation with 3 robots...")
    data = collect_data(steps=30)

    em = data["engine_metrics"]
    rs = data["recorder_summary"]
    actions = data["action_events"]
    successes = sum(1 for a in actions if a["data"]["success"])

    print(f"      Done. {em['steps']} steps in {em['total_ms']:.1f}ms")
    print()

    print("[2/3] Simulation Results:")
    print("      Robots:       3 (GoalAgent, PhysicsAgent, AStarAgent)")
    print(f"      Actions:      {len(actions)} ({successes} success, {len(actions)-successes} blocked)")
    print(f"      Collisions:   {data['collision_total']}")
    print(f"      A* blocked:   {data['astar_blocked']} cells discovered")
    print(f"      Goal reached: {'YES' if data['goal_events'] else 'NO'}")
    print(f"      Recorded:     {rs['total_events']} events")
    print(f"      Replayed:     {data['replay_count']} events verified")
    print()

    for name in data["robot_ids"]:
        start = data["positions"][name][0]
        end = data["positions"][name][-1]
        dist = math.dist(start, end)
        print(f"      [{name:>8}] ({start[0]:+.2f},{start[1]:+.2f},{start[2]:.2f}) → "
              f"({end[0]:+.2f},{end[1]:+.2f},{end[2]:.2f})  Δ={dist:.2f}")

    print()
    print("[3/3] Generating 8-panel visualization...")
    path = plot_results(data)
    print(f"      Saved: {path}")
    print()
    print("=" * 65)
    print("  Demo complete.")
    print("=" * 65)


if __name__ == "__main__":
    main()

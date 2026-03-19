"""
visualize_physics.py — futuristic-styled result plots for 3D physics simulation.

Showcases: GoalAgent, LiDAR, sensor noise, trajectory recording.

Produces a 6-panel dark-theme figure:
  1. 3D trajectory with obstacles and goal
  2. Top-down tactical map with LiDAR sweep
  3. Sensor distance heatmap
  4. LiDAR polar plot
  5. Depth camera view
  6. Engine performance timeline

Usage:
    python -m examples.visualize_physics
"""

from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import numpy as np
import math

from core.event_bus import EventBus
from core.engine import Engine
from core.recorder import Recorder
from simulator.physics_world import PhysicsWorld, DIRECTIONS_3D
from plugins.physics_camera import PhysicsCamera
from plugins.physics_distance import PhysicsDistance
from plugins.physics_lidar import PhysicsLidar
from plugins.goal_publisher import GoalPublisher
from agents.goal_agent import GoalAgent

BG_DARK = "#0a0e17"
BG_PANEL = "#0f1623"
CYAN = "#00f0ff"
CYAN_DIM = "#00607a"
MAGENTA = "#ff00aa"
GREEN = "#00ff88"
ORANGE = "#ff8800"
RED = "#ff2255"
WHITE = "#e0e8f0"
GRID_COLOR = "#1a2540"


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


def collect_data(steps: int = 25):
    bus = EventBus(max_history=50_000)
    world = PhysicsWorld(gui=False, timestep=1 / 240)

    goal_target = (5.0, -3.0, 0.31)

    obstacle_defs = [
        (3, 0, 0.5), (3, 2, 0.5), (-2, 1, 0.5),
        (1, 3, 0.5), (5, -1, 0.5), (-1, -3, 0.5),
        (4, 4, 0.5), (6, 2, 0.5), (-3, -1, 0.5),
    ]
    for ox, oy, oz in obstacle_defs:
        world.place_obstacle(ox, oy, oz)

    goal_pub = GoalPublisher(bus)
    goal_pub.set_target(*goal_target)

    camera = PhysicsCamera(bus, world, width=16, height=16, noise_level=0.02, seed=42)
    distance = PhysicsDistance(bus, world, noise_level=0.3, seed=42)
    lidar = PhysicsLidar(bus, world, num_rays=36, noise_level=0.1, seed=42)
    agent = GoalAgent(bus, world)

    recorder = Recorder(bus)
    recorder.start()

    engine = Engine(bus, enable_metrics=True)
    engine.register_plugin(distance)
    engine.register_plugin(camera)
    engine.register_plugin(lidar)
    engine.register_plugin(goal_pub)
    engine.register_agent(agent)

    positions = [world.get_robot_position()]

    def on_step(step: int) -> None:
        positions.append(world.get_robot_position())

    engine.run(steps=steps, on_step=on_step)
    recorder.stop()

    result = {
        "positions": positions,
        "obstacles": obstacle_defs,
        "goal": goal_target,
        "distance_events": bus.get_history("sensor.distance_3d"),
        "camera_events": bus.get_history("sensor.camera_3d"),
        "lidar_events": bus.get_history("sensor.lidar"),
        "metric_events": bus.get_history("engine.metrics"),
        "action_events": bus.get_history("action.move_3d"),
        "goal_events": bus.get_history("goal.reached"),
        "engine_metrics": engine.get_metrics(),
        "recorder_summary": recorder.get_summary(),
    }
    world.disconnect()
    return result


def _draw_3d(ax, data):
    positions = data["positions"]
    obstacles = data["obstacles"]
    goal = data["goal"]

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    zs = [p[2] for p in positions]

    points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = Line3DCollection(segments, colors=plt.cm.cool(np.linspace(0, 1, len(segments))),
                          linewidths=2.5, alpha=0.9)
    ax.add_collection3d(lc)

    ax.scatter(xs[0], ys[0], zs[0], c=GREEN, s=150, marker="^",
              edgecolors="white", linewidths=1, zorder=10, label="START")
    ax.scatter(xs[-1], ys[-1], zs[-1], c=MAGENTA, s=170, marker="*",
              edgecolors="white", linewidths=0.8, zorder=10, label="END")
    ax.scatter(*goal, c=ORANGE, s=200, marker="D",
              edgecolors="white", linewidths=1, zorder=10, label="GOAL")

    for o in obstacles:
        ax.bar3d(o[0] - 0.4, o[1] - 0.4, 0, 0.8, 0.8, o[2] + 0.5,
                color=RED, alpha=0.12, edgecolor=RED, linewidth=0.5)

    ax.set_xlabel("X", labelpad=6)
    ax.set_ylabel("Y", labelpad=6)
    ax.set_zlabel("Z", labelpad=6)
    ax.set_title("3D  TRAJECTORY + GOAL", color=CYAN, pad=10)
    ax.legend(fontsize=6, loc="upper left", facecolor=BG_PANEL, edgecolor=CYAN_DIM, labelcolor=WHITE)
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor(GRID_COLOR)
    ax.tick_params(colors=CYAN_DIM, labelsize=7)


def _draw_topdown(ax, data):
    positions = data["positions"]
    obstacles = data["obstacles"]
    goal = data["goal"]
    lidar_events = data["lidar_events"]

    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    points = np.array([xs, ys]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(segments, colors=plt.cm.cool(np.linspace(0, 1, len(segments))),
                        linewidths=2, alpha=0.9)
    ax.add_collection(lc)

    if lidar_events:
        last_lidar = lidar_events[-1]["data"]
        robot_pos = last_lidar["robot_position"]
        scan = last_lidar.get("scan", [])
        for ray in scan:
            angle = ray["angle_rad"]
            dist = min(ray["distance"], 8)
            ex = robot_pos[0] + math.cos(angle) * dist
            ey = robot_pos[1] + math.sin(angle) * dist
            color = RED if ray["hit"] else GREEN
            ax.plot([robot_pos[0], ex], [robot_pos[1], ey],
                   color=color, alpha=0.15, linewidth=0.8)

    ax.plot(xs[0], ys[0], "^", color=GREEN, markersize=12, markeredgecolor="white", markeredgewidth=1, zorder=10)
    ax.plot(xs[-1], ys[-1], "*", color=MAGENTA, markersize=12, markeredgecolor="white", zorder=10)
    ax.plot(goal[0], goal[1], "D", color=ORANGE, markersize=10, markeredgecolor="white", zorder=10)

    for o in obstacles:
        rect = FancyBboxPatch((o[0] - 0.45, o[1] - 0.45), 0.9, 0.9,
                             boxstyle="round,pad=0.05", facecolor=RED, alpha=0.25,
                             edgecolor=RED, linewidth=1)
        ax.add_patch(rect)

    for i in range(0, len(xs), 4):
        ax.annotate(f"t{i}", (xs[i], ys[i]), textcoords="offset points",
                   xytext=(4, 4), fontsize=5, color=CYAN_DIM, fontfamily="monospace")

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_title("TACTICAL  MAP + LiDAR", color=CYAN, pad=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.15, color=GRID_COLOR)


def _draw_sensor_heatmap(ax, data):
    distance_events = data["distance_events"]
    directions = list(DIRECTIONS_3D.keys())
    n = len(distance_events)
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
    ax.set_yticklabels([d.upper() for d in directions], fontsize=7, fontfamily="monospace")
    ax.set_xlabel("Step")
    ax.set_title("SENSOR  DISTANCE  (6-DIR)", color=CYAN, pad=10)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Dist", color=WHITE, fontsize=7)
    cbar.ax.tick_params(colors=CYAN_DIM, labelsize=6)


def _draw_lidar_polar(ax, data):
    lidar_events = data["lidar_events"]
    if not lidar_events:
        ax.text(0.5, 0.5, "NO LIDAR DATA", transform=ax.transAxes, ha="center",
               color=RED, fontsize=12)
        return

    last = lidar_events[-1]["data"]
    scan = last.get("scan", [])
    angles = [r["angle_rad"] for r in scan]
    dists = [r["distance"] for r in scan]
    hits = [r["hit"] for r in scan]

    angles.append(angles[0])
    dists.append(dists[0])
    hits.append(hits[0])

    ax.plot(angles, dists, color=CYAN, linewidth=1.5, alpha=0.8)
    ax.fill(angles, dists, alpha=0.08, color=CYAN)

    for a, d, h in zip(angles, dists, hits):
        c = RED if h else GREEN
        ax.scatter(a, d, c=c, s=12, alpha=0.7, zorder=5)

    ax.set_facecolor(BG_PANEL)
    ax.set_title("LiDAR  SWEEP  (LAST STEP)", color=CYAN, pad=15)
    ax.tick_params(colors=CYAN_DIM, labelsize=6)
    ax.set_rmax(last["max_distance"])
    ax.grid(True, alpha=0.2, color=GRID_COLOR)


def _draw_depth(ax, data):
    camera_events = data["camera_events"]
    if not camera_events:
        ax.text(0.5, 0.5, "NO CAMERA DATA", transform=ax.transAxes, ha="center",
               color=RED, fontsize=12)
        return

    last = camera_events[-1]["data"]
    depth = np.array(last["depth"])

    neon_depth = LinearSegmentedColormap.from_list("depth_neon", [
        (0.0, "#ff0044"), (0.3, "#ff8800"), (0.6, "#00ffaa"), (1.0, "#001122"),
    ])
    im = ax.imshow(depth, cmap=neon_depth, interpolation="nearest")
    ax.set_title("DEPTH  CAMERA  (LAST STEP)", color=CYAN, pad=10)
    ax.tick_params(colors=CYAN_DIM, labelsize=6)
    cbar = plt.colorbar(im, ax=ax, pad=0.02)
    cbar.set_label("Depth", color=WHITE, fontsize=7)
    cbar.ax.tick_params(colors=CYAN_DIM, labelsize=6)

    noisy_label = "NOISY" if last.get("noisy") else "CLEAN"
    ax.text(0.02, 0.95, noisy_label, transform=ax.transAxes, fontsize=7,
           color=ORANGE if last.get("noisy") else GREEN,
           verticalalignment="top", fontfamily="monospace",
           path_effects=[pe.withStroke(linewidth=2, foreground=BG_PANEL)])


def _draw_performance(ax, data):
    metric_events = data["metric_events"]
    em = data["engine_metrics"]
    step_times = [ev["data"]["elapsed_ms"] for ev in metric_events]
    step_nums = list(range(1, len(step_times) + 1))
    avg = em["avg_ms"]

    bar_colors = [RED if t > avg * 2 else (ORANGE if t > avg * 1.3 else CYAN) for t in step_times]
    ax.bar(step_nums, step_times, color=bar_colors, alpha=0.85,
          edgecolor=[c.replace("ff", "88") if isinstance(c, str) else c for c in bar_colors], linewidth=0.5)

    ax.axhline(avg, color=GREEN, linestyle="--", linewidth=1, alpha=0.8)
    ax.text(len(step_nums) + 0.3, avg, f"AVG {avg:.2f}ms", fontsize=6, color=GREEN, va="center")

    ax.set_xlabel("Step")
    ax.set_ylabel("Time (ms)")
    ax.set_title("ENGINE  PERFORMANCE", color=CYAN, pad=10)
    ax.grid(True, axis="y", alpha=0.15, color=GRID_COLOR)

    rs = data["recorder_summary"]
    ax.text(0.02, 0.95, f"RECORDED: {rs['total_events']} events", transform=ax.transAxes,
           fontsize=6, color=ORANGE, va="top", fontfamily="monospace")


def plot_results(data: dict) -> str:
    _setup_style()
    fig = plt.figure(figsize=(22, 14))

    fig.text(0.5, 0.975, "IR-AAP", fontsize=26, color=CYAN, ha="center",
             fontfamily="monospace", fontweight="bold",
             path_effects=[pe.withStroke(linewidth=3, foreground="#004455")])
    fig.text(0.5, 0.955, "INDUSTRIAL ROBOT AI AGENT PLATFORM  ·  3D PHYSICS + LiDAR + GOAL NAVIGATION",
             fontsize=9, color=CYAN_DIM, ha="center", fontfamily="monospace")

    ax1 = fig.add_subplot(2, 3, 1, projection="3d")
    _draw_3d(ax1, data)

    ax2 = fig.add_subplot(2, 3, 2)
    _draw_topdown(ax2, data)

    ax3 = fig.add_subplot(2, 3, 3)
    _draw_sensor_heatmap(ax3, data)

    ax4 = fig.add_subplot(2, 3, 4, projection="polar")
    _draw_lidar_polar(ax4, data)

    ax5 = fig.add_subplot(2, 3, 5)
    _draw_depth(ax5, data)

    ax6 = fig.add_subplot(2, 3, 6)
    _draw_performance(ax6, data)

    successes = sum(1 for a in data["action_events"] if a["data"]["success"])
    total = len(data["action_events"])
    goal_reached = len(data["goal_events"]) > 0
    goal_txt = "REACHED" if goal_reached else "IN PROGRESS"

    status = (
        f"STEPS: {total}  |  MOVES: {successes}  |  BLOCKED: {total - successes}  |  "
        f"GOAL: {goal_txt}  |  TIME: {data['engine_metrics']['total_ms']:.1f}ms  |  "
        f"NOISE: ON  |  LiDAR: 36-RAY"
    )
    fig.text(0.5, 0.008, status, fontsize=8, color=CYAN_DIM, ha="center", fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.4", facecolor=BG_PANEL, edgecolor=CYAN_DIM, alpha=0.8))

    plt.tight_layout(rect=[0.01, 0.03, 0.99, 0.94])

    out_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "physics_results.png"))
    plt.savefig(out_path, dpi=180, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def main() -> None:
    print("Collecting simulation data (GoalAgent + LiDAR + noise)...")
    data = collect_data(steps=25)
    print(f"Done. {len(data['positions'])} pos, {len(data['lidar_events'])} lidar, "
          f"{len(data['camera_events'])} camera, {len(data['distance_events'])} distance.")
    print(f"Recorder: {data['recorder_summary']}")
    print("Generating futuristic plots...")
    plot_results(data)


if __name__ == "__main__":
    main()

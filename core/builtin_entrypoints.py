"""
Register built-in plugin and agent factories on a ``PluginRegistry``.

Third-party packages can register their own via ``[project.entry-points."iraap.register"]``.
"""

from __future__ import annotations

from core.registry import PluginRegistry

from agents.astar_agent import AStarAgent
from agents.goal_agent import GoalAgent
from agents.physics_agent import PhysicsAgent
from plugins.collision_detector import CollisionDetector
from plugins.goal_publisher import GoalPublisher
from plugins.occupancy_map import OccupancyMap
from plugins.physics_distance import PhysicsDistance
from plugins.physics_lidar import PhysicsLidar


def register_builtin(registry: PluginRegistry) -> None:
    """Register all first-party plugins and agents by stable string keys."""
    registry.register_plugin("physics_distance", PhysicsDistance)
    registry.register_plugin("physics_lidar", PhysicsLidar)
    registry.register_plugin("goal_publisher", GoalPublisher)
    registry.register_plugin("collision_detector", CollisionDetector)
    registry.register_plugin("occupancy_map", OccupancyMap)

    registry.register_agent("physics_agent", PhysicsAgent)
    registry.register_agent("goal_agent", GoalAgent)
    registry.register_agent("astar_agent", AStarAgent)

"""
WorldInterface — abstract base class for all world implementations.
Both 2D GridWorld and 3D PhysicsWorld implement this interface,
allowing SensorSim and Agents to work with either transparently.
"""

from abc import ABC, abstractmethod


class WorldInterface(ABC):

    @abstractmethod
    def get_robot_position(self) -> tuple:
        """Return the robot's current position (2D or 3D tuple)."""

    @abstractmethod
    def move_robot(self, direction: str) -> bool:
        """Attempt to move the robot. Returns True if successful."""

    @abstractmethod
    def is_obstacle(self, x: float, y: float, z: float = 0) -> bool:
        """Check whether the given coordinate is blocked."""

    @abstractmethod
    def get_surroundings(self, x: float, y: float, z: float = 0, radius: int = 1) -> dict:
        """Return obstacle distance info in each direction around (x, y, z)."""

    @abstractmethod
    def step_physics(self) -> None:
        """Advance the physics simulation by one timestep. No-op for grid worlds."""

    @abstractmethod
    def render(self) -> str:
        """Return a human-readable representation of the world state."""

    @abstractmethod
    def reset(self) -> None:
        """Reset the world to its initial state."""

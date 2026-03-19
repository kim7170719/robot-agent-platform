"""Pydantic request/response models for the REST API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StepRequest(BaseModel):
    steps: int = 1


class GoalRequest(BaseModel):
    x: float
    y: float
    z: float = 0.31


class RobotRequest(BaseModel):
    robot_id: str
    x: float = 0.0
    y: float = 0.0
    z: float = 0.31


class RobotTeleportRequest(BaseModel):
    """Instantly move a sphere robot to (x, y, z) in the physics world."""

    robot_id: str | None = None
    x: float
    y: float
    z: float | None = None


class ObstacleRequest(BaseModel):
    x: float
    y: float
    z: float = 0.5


class ConfigPatchRequest(BaseModel):
    key: str
    value: Any


class MoveRobotRequest(BaseModel):
    robot_id: str | None = None
    direction: str = "forward"


class DynamicBoxRequest(BaseModel):
    """Spawn a dynamic cube (mass > 0) for drop / contact tests."""

    x: float
    y: float
    z: float
    mass: float = 1.0
    half_extent: float = 0.1


class GripperRequest(BaseModel):
    """Prismatic gripper openness for bundled URDF ``gripper_slide`` (0 ≈ closed, 1 ≈ open)."""

    openness: float = Field(0.5, ge=0.0, le=1.0)

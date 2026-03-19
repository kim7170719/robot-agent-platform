"""
Event Schema Validation — declarative JSON-like schemas for every IR-AAP
event type, plus a validator plugin that catches malformed events at the bus.

Usage:
    from core.event_schema import SCHEMAS, SchemaValidator, validate_event_data
"""

from __future__ import annotations

from typing import Any


# ── Primitive type checkers ─────────────────────────────

def _check_type(value: Any, spec: str) -> bool:
    mapping = {
        "int": (int,),
        "float": (int, float),
        "str": (str,),
        "bool": (bool,),
        "list": (list, tuple),
        "dict": (dict,),
        "any": (object,),
    }
    return isinstance(value, mapping.get(spec, (object,)))


def _check_list_items(value: list, item_spec: str) -> bool:
    return all(_check_type(v, item_spec) for v in value)


# ── Schema registry ─────────────────────────────────────

class FieldSpec:
    """Describes one field inside an event's ``data`` dict."""
    __slots__ = ("name", "type", "required", "items", "min_val", "max_val")

    def __init__(
        self,
        name: str,
        type: str = "any",
        required: bool = True,
        items: str | None = None,
        min_val: float | None = None,
        max_val: float | None = None,
    ) -> None:
        self.name = name
        self.type = type
        self.required = required
        self.items = items
        self.min_val = min_val
        self.max_val = max_val


def _F(name: str, **kw) -> FieldSpec:
    return FieldSpec(name, **kw)


SCHEMAS: dict[str, list[FieldSpec]] = {
    "action.move_3d": [
        _F("robot_id", type="str"),
        _F("direction", type="str"),
        _F("success", type="bool"),
        _F("old_position", type="list", items="float"),
        _F("new_position", type="list", items="float"),
    ],
    "sensor.distance_3d": [
        _F("robot_id", type="str"),
        _F("distances", type="dict"),
    ],
    "sensor.lidar": [
        _F("robot_id", type="str"),
        _F("points", type="list"),
        _F("num_rays", type="int"),
    ],
    "sensor.joint": [
        _F("robot_id", type="str"),
        _F("joint_positions", type="list", items="float"),
        _F("joint_velocities", type="list", items="float"),
    ],
    "engine.metrics": [
        _F("step", type="int", min_val=0),
        _F("elapsed_ms", type="float", min_val=0),
    ],
    "goal.target": [
        _F("target", type="list", items="float"),
    ],
    "goal.reached": [
        _F("robot_id", type="str"),
        _F("position", type="list", items="float"),
    ],
    "physics.collision": [
        _F("body_a", type="int"),
        _F("body_b", type="int"),
    ],
    "map.occupancy": [
        _F("cells", type="dict"),
        _F("resolution", type="float", min_val=0),
    ],
    "bt.tick": [
        _F("status", type="str"),
        _F("tick_count", type="int", min_val=0),
    ],
    "system.plugin_loaded": [
        _F("plugin_id", type="str"),
        _F("total_plugins", type="int", min_val=0),
    ],
    "system.plugin_unloaded": [
        _F("plugin_id", type="str"),
        _F("total_plugins", type="int", min_val=0),
    ],
    "system.agent_loaded": [
        _F("agent_id", type="str"),
        _F("total_agents", type="int", min_val=0),
    ],
    "system.agent_unloaded": [
        _F("agent_id", type="str"),
        _F("total_agents", type="int", min_val=0),
    ],
    "task.allocated": [
        _F("task_id", type="str"),
        _F("robot_id", type="str"),
    ],
    "action.hardware": [
        _F("command", type="dict"),
        _F("success", type="bool"),
    ],
    "ws.command.ack": [
        _F("ok", type="bool"),
        _F("cmd", type="str"),
    ],
}


# ── Validation ──────────────────────────────────────────

class ValidationError:
    __slots__ = ("field", "message")

    def __init__(self, field: str, message: str) -> None:
        self.field = field
        self.message = message

    def __repr__(self) -> str:
        return f"ValidationError({self.field!r}, {self.message!r})"


def validate_event_data(
    event_type: str,
    data: dict,
    strict: bool = False,
) -> list[ValidationError]:
    """
    Validate *data* against the registered schema for *event_type*.

    Returns a (possibly empty) list of ``ValidationError`` objects.
    If *strict* is True, extra fields not in the schema are also flagged.
    If no schema exists for the event type, returns an empty list.
    """
    schema = SCHEMAS.get(event_type)
    if schema is None:
        return []

    errors: list[ValidationError] = []
    defined_names = set()

    for spec in schema:
        defined_names.add(spec.name)
        value = data.get(spec.name)

        if value is None:
            if spec.required:
                errors.append(ValidationError(spec.name, "required field missing"))
            continue

        if not _check_type(value, spec.type):
            errors.append(ValidationError(spec.name, f"expected {spec.type}, got {type(value).__name__}"))
            continue

        if spec.items and isinstance(value, (list, tuple)):
            if not _check_list_items(value, spec.items):
                errors.append(ValidationError(spec.name, f"list items must be {spec.items}"))

        if spec.min_val is not None and isinstance(value, (int, float)):
            if value < spec.min_val:
                errors.append(ValidationError(spec.name, f"value {value} < min {spec.min_val}"))

        if spec.max_val is not None and isinstance(value, (int, float)):
            if value > spec.max_val:
                errors.append(ValidationError(spec.name, f"value {value} > max {spec.max_val}"))

    if strict:
        extra = set(data.keys()) - defined_names
        for name in sorted(extra):
            errors.append(ValidationError(name, "unexpected field (strict mode)"))

    return errors


# ── Validator Plugin ────────────────────────────────────

class SchemaValidator:
    """
    Subscribes to ``"*"`` on the EventBus and validates every event.
    Violations are counted and optionally logged via a callback.
    """

    def __init__(
        self,
        event_bus,
        strict: bool = False,
        on_violation=None,
    ) -> None:
        self._bus = event_bus
        self._strict = strict
        self._on_violation = on_violation
        self._violations: list[dict] = []
        self._checked = 0
        self._bus.subscribe("*", self._on_event)

    @property
    def violations(self) -> list[dict]:
        return list(self._violations)

    @property
    def violation_count(self) -> int:
        return len(self._violations)

    @property
    def checked_count(self) -> int:
        return self._checked

    def _on_event(self, event: dict) -> None:
        self._checked += 1
        errs = validate_event_data(event["type"], event.get("data", {}), strict=self._strict)
        if errs:
            record = {
                "event_type": event["type"],
                "errors": [{"field": e.field, "message": e.message} for e in errs],
                "source": event.get("source"),
            }
            self._violations.append(record)
            if self._on_violation:
                self._on_violation(record)

    def summary(self) -> dict:
        return {
            "checked": self._checked,
            "violations": len(self._violations),
            "details": self._violations[-10:],
        }

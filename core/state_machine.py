"""
Mission State Machine — FSM for multi-phase robot missions.

Phases: IDLE → NAVIGATE → MANIPULATE → RETURN → IDLE

Each transition can have guard conditions (predicates) and on-enter/exit
callbacks. The FSM publishes ``mission.transition`` events on the EventBus.

Usage:
    from core.state_machine import MissionFSM, Phase
"""

from __future__ import annotations

import enum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from core.event_bus import EventBus


class Phase(enum.Enum):
    IDLE = "IDLE"
    NAVIGATE = "NAVIGATE"
    MANIPULATE = "MANIPULATE"
    RETURN = "RETURN"
    ERROR = "ERROR"


Callback = Callable[[], None]
Guard = Callable[[], bool]


class Transition:
    __slots__ = ("source", "target", "guard", "on_transition")

    def __init__(
        self,
        source: Phase,
        target: Phase,
        guard: Guard | None = None,
        on_transition: Callback | None = None,
    ) -> None:
        self.source = source
        self.target = target
        self.guard = guard
        self.on_transition = on_transition


_DEFAULT_TRANSITIONS = [
    Transition(Phase.IDLE, Phase.NAVIGATE),
    Transition(Phase.NAVIGATE, Phase.MANIPULATE),
    Transition(Phase.MANIPULATE, Phase.RETURN),
    Transition(Phase.RETURN, Phase.IDLE),
    Transition(Phase.IDLE, Phase.ERROR),
    Transition(Phase.NAVIGATE, Phase.ERROR),
    Transition(Phase.MANIPULATE, Phase.ERROR),
    Transition(Phase.RETURN, Phase.ERROR),
    Transition(Phase.ERROR, Phase.IDLE),
]


class MissionFSM:
    """
    Finite State Machine for a single robot's mission lifecycle.

    Guards can prevent transitions; callbacks fire before entering the
    new phase. All transitions publish events on the bus.
    """

    def __init__(
        self,
        robot_id: str,
        event_bus: EventBus | None = None,
        transitions: list[Transition] | None = None,
    ) -> None:
        self._robot_id = robot_id
        self._bus = event_bus
        self._phase = Phase.IDLE
        self._history: list[dict] = []

        self._transitions: dict[tuple[Phase, Phase], Transition] = {}
        for t in (transitions or _DEFAULT_TRANSITIONS):
            self._transitions[(t.source, t.target)] = t

        self._on_enter: dict[Phase, list[Callback]] = {}
        self._on_exit: dict[Phase, list[Callback]] = {}

    @property
    def robot_id(self) -> str:
        return self._robot_id

    @property
    def phase(self) -> Phase:
        return self._phase

    @property
    def history(self) -> list[dict]:
        return list(self._history)

    def on_enter(self, phase: Phase, callback: Callback) -> None:
        self._on_enter.setdefault(phase, []).append(callback)

    def on_exit(self, phase: Phase, callback: Callback) -> None:
        self._on_exit.setdefault(phase, []).append(callback)

    def can_transition(self, target: Phase) -> bool:
        key = (self._phase, target)
        t = self._transitions.get(key)
        if t is None:
            return False
        if t.guard and not t.guard():
            return False
        return True

    def transition(self, target: Phase) -> bool:
        """
        Attempt transition from current phase to *target*.
        Returns True on success, False if guard rejects or path doesn't exist.
        """
        key = (self._phase, target)
        t = self._transitions.get(key)
        if t is None:
            return False
        if t.guard and not t.guard():
            return False

        old = self._phase
        for cb in self._on_exit.get(old, []):
            cb()
        if t.on_transition:
            t.on_transition()

        self._phase = target

        for cb in self._on_enter.get(target, []):
            cb()

        record = {"robot_id": self._robot_id, "from": old.value, "to": target.value}
        self._history.append(record)

        if self._bus is not None:
            import time
            self._bus.publish({
                "timestamp": int(time.time() * 1000),
                "type": "mission.transition",
                "source": f"fsm.{self._robot_id}",
                "data": record,
            })

        return True

    def force(self, phase: Phase) -> None:
        """Force-set phase without guards or callbacks (for resets)."""
        self._phase = phase

    def advance(self) -> bool:
        """
        Auto-advance to the next phase in the default mission cycle:
        IDLE → NAVIGATE → MANIPULATE → RETURN → IDLE.
        """
        cycle = {
            Phase.IDLE: Phase.NAVIGATE,
            Phase.NAVIGATE: Phase.MANIPULATE,
            Phase.MANIPULATE: Phase.RETURN,
            Phase.RETURN: Phase.IDLE,
            Phase.ERROR: Phase.IDLE,
        }
        next_phase = cycle.get(self._phase)
        if next_phase is None:
            return False
        return self.transition(next_phase)

    def error(self) -> bool:
        """Transition to ERROR from any phase."""
        return self.transition(Phase.ERROR)

    def reset(self) -> None:
        """Force back to IDLE and clear history."""
        self._phase = Phase.IDLE
        self._history.clear()

    def get_status(self) -> dict:
        return {
            "robot_id": self._robot_id,
            "phase": self._phase.value,
            "transitions": len(self._history),
            "history": self._history[-5:],
        }

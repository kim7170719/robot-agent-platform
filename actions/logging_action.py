"""
LoggingAction — decorator ActionLayer that publishes ``action.executed``
events to the EventBus before delegating to a wrapped ActionLayer.

Useful for auditing, replay, and dashboard consumption.
"""

from __future__ import annotations

from core.event_bus import EventBus, _make_event
from actions.base_action import ActionLayer


class LoggingAction(ActionLayer):
    """
    Wraps another ``ActionLayer`` and publishes an ``action.executed``
    event for every command that passes through.

    Parameters
    ----------
    inner : ActionLayer
        The action layer to delegate to.
    event_bus : EventBus
        Bus to publish logging events on.
    source : str
        Source identifier for the published events.
    """

    def __init__(
        self,
        inner: ActionLayer,
        event_bus: EventBus,
        source: str = "logging_action",
    ) -> None:
        self._inner = inner
        self._bus = event_bus
        self._source = source
        self._execution_count = 0

    @property
    def inner(self) -> ActionLayer:
        return self._inner

    @property
    def execution_count(self) -> int:
        return self._execution_count

    def execute(self, command: dict) -> dict:
        result = self._inner.execute(command)
        self._execution_count += 1

        self._bus.publish(_make_event("action.executed", self._source, {
            "command": command,
            "result": result,
            "execution_index": self._execution_count,
        }))
        return result

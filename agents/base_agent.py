"""
BaseAgent — abstract interface for all AI agents.
Every agent must implement perceive(), decide(), act() per Whitepaper §4.2.
"""

from abc import ABC, abstractmethod

from core.event_bus import EventBus


class BaseAgent(ABC):
    """
    Agents subscribe to sensor events, make decisions, and publish actions.
    They never import plugins or other agents directly.
    """

    def __init__(self, event_bus: EventBus, agent_id: str) -> None:
        self._bus = event_bus
        self._agent_id = agent_id

    @property
    def agent_id(self) -> str:
        return self._agent_id

    @property
    def event_bus(self) -> EventBus:
        return self._bus

    @abstractmethod
    def perceive(self) -> None:
        """Collect and buffer the latest sensor events from the bus."""

    @abstractmethod
    def decide(self) -> None:
        """Process buffered perceptions and choose an action."""

    @abstractmethod
    def act(self) -> None:
        """Execute the chosen action (publish action event / move in world)."""

    def cleanup(self) -> None:
        """Unsubscribe from bus. Override in subclasses that subscribe."""

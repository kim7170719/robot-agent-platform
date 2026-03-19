"""
bt_nodes — Behavior Tree node primitives for composable agent logic.

Node types:
  - BTNode (ABC)          : base with tick() -> Status
  - Sequence              : runs children left-to-right, fails on first FAILURE
  - Selector (Fallback)   : runs children left-to-right, succeeds on first SUCCESS
  - Condition             : leaf that evaluates a predicate
  - ActionNode            : leaf that executes a callable
  - Inverter              : decorator that flips SUCCESS <-> FAILURE
  - RepeatUntilSuccess    : decorator that re-ticks child until SUCCESS
  - Parallel              : ticks all children, succeeds if >= threshold succeed
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Callable


class Status(Enum):
    SUCCESS = auto()
    FAILURE = auto()
    RUNNING = auto()


class BTNode(ABC):
    """Abstract base for all BT nodes."""

    def __init__(self, name: str = "") -> None:
        self._name = name or self.__class__.__name__
        self._status = Status.FAILURE

    @property
    def name(self) -> str:
        return self._name

    @property
    def status(self) -> Status:
        return self._status

    @abstractmethod
    def tick(self) -> Status:
        """Execute one tick of this node."""

    def reset(self) -> None:
        """Reset the node state for a new tick cycle."""
        self._status = Status.FAILURE

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._name!r})"


class Sequence(BTNode):
    """
    Ticks children in order. Returns FAILURE on the first child that
    fails, SUCCESS when all succeed, RUNNING if a child is running.
    """

    def __init__(self, children: list[BTNode], name: str = "") -> None:
        super().__init__(name or "Sequence")
        self._children = list(children)

    @property
    def children(self) -> list[BTNode]:
        return list(self._children)

    def tick(self) -> Status:
        for child in self._children:
            result = child.tick()
            if result != Status.SUCCESS:
                self._status = result
                return result
        self._status = Status.SUCCESS
        return Status.SUCCESS

    def reset(self) -> None:
        super().reset()
        for child in self._children:
            child.reset()


class Selector(BTNode):
    """
    Ticks children in order. Returns SUCCESS on the first child that
    succeeds, FAILURE when all fail, RUNNING if a child is running.
    """

    def __init__(self, children: list[BTNode], name: str = "") -> None:
        super().__init__(name or "Selector")
        self._children = list(children)

    @property
    def children(self) -> list[BTNode]:
        return list(self._children)

    def tick(self) -> Status:
        for child in self._children:
            result = child.tick()
            if result != Status.FAILURE:
                self._status = result
                return result
        self._status = Status.FAILURE
        return Status.FAILURE

    def reset(self) -> None:
        super().reset()
        for child in self._children:
            child.reset()


class Condition(BTNode):
    """Leaf node that evaluates a predicate. No side effects."""

    def __init__(self, predicate: Callable[[], bool], name: str = "") -> None:
        super().__init__(name or "Condition")
        self._predicate = predicate

    def tick(self) -> Status:
        try:
            result = self._predicate()
        except Exception:
            self._status = Status.FAILURE
            return Status.FAILURE
        self._status = Status.SUCCESS if result else Status.FAILURE
        return self._status


class ActionNode(BTNode):
    """
    Leaf node that executes a callable. The callable should return
    a Status (SUCCESS/FAILURE/RUNNING) or a bool (True→SUCCESS).
    """

    def __init__(self, action: Callable[[], Status | bool], name: str = "") -> None:
        super().__init__(name or "Action")
        self._action = action

    def tick(self) -> Status:
        try:
            result = self._action()
        except Exception:
            self._status = Status.FAILURE
            return Status.FAILURE

        if isinstance(result, Status):
            self._status = result
        elif isinstance(result, bool):
            self._status = Status.SUCCESS if result else Status.FAILURE
        else:
            self._status = Status.SUCCESS
        return self._status


class Inverter(BTNode):
    """Decorator: flips SUCCESS <-> FAILURE. RUNNING passes through."""

    def __init__(self, child: BTNode, name: str = "") -> None:
        super().__init__(name or f"Inverter({child.name})")
        self._child = child

    @property
    def child(self) -> BTNode:
        return self._child

    def tick(self) -> Status:
        result = self._child.tick()
        if result == Status.SUCCESS:
            self._status = Status.FAILURE
        elif result == Status.FAILURE:
            self._status = Status.SUCCESS
        else:
            self._status = Status.RUNNING
        return self._status

    def reset(self) -> None:
        super().reset()
        self._child.reset()


class RepeatUntilSuccess(BTNode):
    """Decorator: re-ticks child up to max_attempts until SUCCESS."""

    def __init__(self, child: BTNode, max_attempts: int = 10, name: str = "") -> None:
        super().__init__(name or f"RepeatUntilSuccess({child.name})")
        self._child = child
        self._max = max_attempts

    @property
    def child(self) -> BTNode:
        return self._child

    def tick(self) -> Status:
        for _ in range(self._max):
            result = self._child.tick()
            if result == Status.SUCCESS:
                self._status = Status.SUCCESS
                return Status.SUCCESS
            if result == Status.RUNNING:
                self._status = Status.RUNNING
                return Status.RUNNING
        self._status = Status.FAILURE
        return Status.FAILURE

    def reset(self) -> None:
        super().reset()
        self._child.reset()


class Parallel(BTNode):
    """
    Ticks all children every tick. Succeeds when at least *threshold*
    children succeed. Fails when it's impossible to reach threshold.
    """

    def __init__(
        self, children: list[BTNode], threshold: int = 1, name: str = ""
    ) -> None:
        super().__init__(name or "Parallel")
        self._children = list(children)
        self._threshold = max(1, threshold)

    @property
    def children(self) -> list[BTNode]:
        return list(self._children)

    def tick(self) -> Status:
        successes = 0
        failures = 0
        for child in self._children:
            result = child.tick()
            if result == Status.SUCCESS:
                successes += 1
            elif result == Status.FAILURE:
                failures += 1

        if successes >= self._threshold:
            self._status = Status.SUCCESS
            return Status.SUCCESS

        max_possible = len(self._children) - failures
        if max_possible < self._threshold:
            self._status = Status.FAILURE
            return Status.FAILURE

        self._status = Status.RUNNING
        return Status.RUNNING

    def reset(self) -> None:
        super().reset()
        for child in self._children:
            child.reset()

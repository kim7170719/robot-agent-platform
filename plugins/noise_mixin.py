"""
NoiseMixin — shared noise generation for physics sensor plugins.
Eliminates repeated noise_level property/setter and RNG setup.
"""

from __future__ import annotations

import random


class NoiseMixin:
    """
    Mixin providing configurable gaussian noise.

    Subclasses get:
      - ``noise_level`` property with clamped setter
      - ``_apply_noise(raw_value)`` helper
      - ``_rng`` seeded Random instance
    """

    def _init_noise(self, noise_level: float = 0.0, seed: int | None = None) -> None:
        self._noise = max(0.0, noise_level)
        self._rng = random.Random(seed)

    @property
    def noise_level(self) -> float:
        return self._noise

    @noise_level.setter
    def noise_level(self, value: float) -> None:
        self._noise = max(0.0, value)

    def _apply_noise(self, raw: float, min_val: float = 0.0) -> float:
        if self._noise <= 0:
            return raw
        return round(max(min_val, raw + self._rng.gauss(0, self._noise)), 3)

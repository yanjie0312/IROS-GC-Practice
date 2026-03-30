# my_project/utils/logging.py
from __future__ import annotations

import numpy as np
from typing import List


class EpisodeLogger:
    """Lightweight per-step data collector for a single episode."""

    def __init__(self):
        self._positions: List[np.ndarray] = []  # (x, y, z) per control step
        self._min_dists: List[float] = []        # nearest obstacle per control step
        self.n_escape_events: int = 0
        self.n_low_z_events: int = 0

    def record_step(self, pos, min_dist: float):
        self._positions.append(np.asarray(pos, dtype=float).copy())
        d = float(min_dist)
        self._min_dists.append(d if np.isfinite(d) else np.nan)

    def record_escape(self):
        self.n_escape_events += 1

    def record_low_z(self):
        self.n_low_z_events += 1

    @property
    def positions(self) -> np.ndarray:
        if not self._positions:
            return np.empty((0, 3))
        return np.stack(self._positions)

    @property
    def min_dists(self) -> np.ndarray:
        return np.array(self._min_dists, dtype=float)

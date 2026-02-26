from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import numpy as np


UNKNOWN = -1
FREE = 0
OCCUPIED = 1


@dataclass(frozen=True)
class GridBounds:
    x_min: float
    x_max: float
    y_min: float
    y_max: float


class OccupancyGrid:
    """
    2D occupancy grid updated by ray measurements.

    Cell states:
    - UNKNOWN: not observed yet
    - FREE: ray passed through
    - OCCUPIED: ray hit obstacle
    """

    def __init__(
        self,
        resolution: float = 0.10,
        bounds: GridBounds = GridBounds(x_min=-9.0, x_max=9.0, y_min=-9.0, y_max=9.0),
        ray_length: float = 2.5,
        hit_epsilon: float = 0.03,
        free_sample_step: float | None = None,
    ):
        if resolution <= 0.0:
            raise ValueError("resolution must be > 0")
        if bounds.x_max <= bounds.x_min or bounds.y_max <= bounds.y_min:
            raise ValueError("invalid grid bounds")

        self.resolution = float(resolution)
        self.bounds = bounds
        self.ray_length = float(ray_length)
        self.hit_epsilon = float(max(0.0, hit_epsilon))
        self.free_sample_step = float(free_sample_step) if free_sample_step is not None else 0.5 * self.resolution
        self.free_sample_step = max(1e-6, self.free_sample_step)

        self.width = int(np.ceil((self.bounds.x_max - self.bounds.x_min) / self.resolution))
        self.height = int(np.ceil((self.bounds.y_max - self.bounds.y_min) / self.resolution))
        self.grid = np.full((self.height, self.width), UNKNOWN, dtype=np.int8)

    def reset(self) -> None:
        self.grid.fill(UNKNOWN)

    def world_to_grid(self, pos: Sequence[float]) -> Tuple[int, int] | None:
        arr = np.asarray(pos, dtype=float).reshape(-1)
        if arr.size < 2:
            raise ValueError("pos must have at least 2 values")
        x, y = float(arr[0]), float(arr[1])
        c = int(np.floor((x - self.bounds.x_min) / self.resolution))
        r = int(np.floor((y - self.bounds.y_min) / self.resolution))
        if not self._in_bounds_idx(r, c):
            return None
        return r, c

    def grid_to_world(self, row: int, col: int) -> np.ndarray:
        x = self.bounds.x_min + (col + 0.5) * self.resolution
        y = self.bounds.y_min + (row + 0.5) * self.resolution
        return np.array([x, y], dtype=float)

    def update(
        self,
        drone_pos: Sequence[float],
        ray_dists: Sequence[float],
        ray_dirs_world: Sequence[Sequence[float]],
    ) -> None:
        origin = np.asarray(drone_pos, dtype=float).reshape(-1)
        if origin.size < 2:
            raise ValueError("drone_pos must have at least 2 values")
        origin_xy = origin[:2]

        dists = np.asarray(ray_dists, dtype=float).reshape(-1)
        dirs = np.asarray(ray_dirs_world, dtype=float)

        # Keep the current drone cell observable even when ray data is unavailable.
        self._mark_free_world(origin_xy)
        # Handle empty-ray case explicitly, regardless of provided dimensionality.
        if dirs.size == 0:
            return
        # Allow a single direction given as a 1D vector of length 2 or 3.
        if dirs.ndim == 1:
            if dirs.size in (2, 3):
                dirs = dirs.reshape(1, -1)
            else:
                raise ValueError(
                    "ray_dirs_world 1D input must have length 2 or 3; "
                    f"got shape {dirs.shape}"
                )
        # At this point, only 2D inputs are allowed.
        if dirs.ndim != 2:
            raise ValueError(
                "ray_dirs_world must be a 2D array with shape (N,2) or (N,3); "
                f"got array with ndim={dirs.ndim} and shape={dirs.shape}"
            )
        if dirs.shape[1] == 2:
            dirs_xy = dirs
        elif dirs.shape[1] >= 3:
            dirs_xy = dirs[:, :2]
        else:
            raise ValueError("ray_dirs_world must have shape (N,2) or (N,3)")

        if dists.shape[0] != dirs_xy.shape[0]:
            raise ValueError(
                f"ray_dists and ray_dirs_world must have the same length; "
                f"got {dists.shape[0]} distances and {dirs_xy.shape[0]} directions"
            )

        n = dists.shape[0]
        for i in range(n):
            dist = float(dists[i])
            if not np.isfinite(dist):
                continue
            dist = float(np.clip(dist, 0.0, self.ray_length))

            ray_dir = dirs_xy[i]
            norm = float(np.linalg.norm(ray_dir))
            if norm < 1e-9:
                continue
            ray_unit = ray_dir / norm

            has_hit = dist < (self.ray_length - self.hit_epsilon)
            free_dist = max(0.0, dist - self.hit_epsilon) if has_hit else dist
            if free_dist > 1e-9:
                num_steps = int(np.ceil(free_dist / self.free_sample_step))
                num_steps = max(1, num_steps)
                for k in range(1, num_steps + 1):
                    s = free_dist * (k / num_steps)
                    p = origin_xy + s * ray_unit
                    self._mark_free_world(p)

            if has_hit:
                hit_pt = origin_xy + dist * ray_unit
                self._mark_occupied_world(hit_pt)

    def is_free(self, pos: Sequence[float]) -> bool:
        idx = self.world_to_grid(pos)
        if idx is None:
            return False
        r, c = idx
        return int(self.grid[r, c]) == FREE

    def get_frontier_indices(self) -> List[Tuple[int, int]]:
        frontiers: List[Tuple[int, int]] = []
        free_cells = np.argwhere(self.grid == FREE)

        for r, c in free_cells:
            for nr, nc in self._neighbors4(int(r), int(c)):
                if int(self.grid[nr, nc]) == UNKNOWN:
                    frontiers.append((int(r), int(c)))
                    break
        return frontiers

    def get_frontiers(self) -> List[np.ndarray]:
        return [self.grid_to_world(r, c) for (r, c) in self.get_frontier_indices()]

    def snapshot(self) -> np.ndarray:
        return self.grid.copy()

    def _in_bounds_idx(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def _mark_free_world(self, p_xy: np.ndarray) -> None:
        idx = self.world_to_grid(p_xy)
        if idx is None:
            return
        r, c = idx
        if int(self.grid[r, c]) != OCCUPIED:
            self.grid[r, c] = FREE

    def _mark_occupied_world(self, p_xy: np.ndarray) -> None:
        idx = self.world_to_grid(p_xy)
        if idx is None:
            return
        r, c = idx
        self.grid[r, c] = OCCUPIED

    def _neighbors4(self, row: int, col: int) -> Iterable[Tuple[int, int]]:
        candidates = ((row - 1, col), (row + 1, col), (row, col - 1), (row, col + 1))
        for nr, nc in candidates:
            if self._in_bounds_idx(nr, nc):
                yield nr, nc

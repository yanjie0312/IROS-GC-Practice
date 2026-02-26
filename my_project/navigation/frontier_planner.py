from __future__ import annotations

from collections import deque
from typing import List, Sequence, Tuple

import numpy as np

from .occupancy_grid import OccupancyGrid


class FrontierPlanner:
    """
    Frontier-based exploration planner:
    1) update occupancy grid
    2) extract and cluster frontier cells
    3) choose nearest cluster representative as next waypoint
    """

    def __init__(
        self,
        occupancy_grid: OccupancyGrid,
        cluster_connectivity: int = 8,
        min_cluster_size: int = 1,
        done_frontier_streak: int = 8,
        waypoint_z: float | None = None,
        verbose: bool = False,
    ):
        if cluster_connectivity not in (4, 8):
            raise ValueError("cluster_connectivity must be 4 or 8")
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be > 0")

        self.grid = occupancy_grid
        self.cluster_connectivity = int(cluster_connectivity)
        self.min_cluster_size = int(min_cluster_size)
        self.done_frontier_streak = int(max(1, done_frontier_streak))
        self.waypoint_z = waypoint_z
        self.verbose = bool(verbose)

        self._clusters: List[List[Tuple[int, int]]] = []
        self._cluster_centers: List[np.ndarray] = []
        self._cluster_representatives: List[np.ndarray] = []
        self._no_frontier_steps = 0
        self._done = False
        self._last_selected_xy: np.ndarray | None = None

    def reset(self) -> None:
        self.grid.reset()
        self._clusters = []
        self._cluster_centers = []
        self._cluster_representatives = []
        self._no_frontier_steps = 0
        self._done = False
        self._last_selected_xy = None

    def update(
        self,
        drone_pos: Sequence[float],
        ray_dists: Sequence[float],
        ray_dirs_world: Sequence[Sequence[float]],
    ) -> None:
        self.grid.update(drone_pos=drone_pos, ray_dists=ray_dists, ray_dirs_world=ray_dirs_world)

        frontier_cells = self.grid.get_frontier_indices()
        if not frontier_cells:
            self._clusters = []
            self._cluster_centers = []
            self._cluster_representatives = []
            self._no_frontier_steps += 1
            self._done = self._no_frontier_steps >= self.done_frontier_streak
            return

        clusters = self._cluster_frontiers(frontier_cells)
        clusters = [c for c in clusters if len(c) >= self.min_cluster_size]
        if not clusters:
            self._clusters = []
            self._cluster_centers = []
            self._cluster_representatives = []
            self._no_frontier_steps += 1
            self._done = self._no_frontier_steps >= self.done_frontier_streak
            return

        self._clusters = clusters
        self._cluster_centers = [self._cluster_center(c) for c in clusters]
        self._cluster_representatives = [self._cluster_representative(c) for c in clusters]
        self._no_frontier_steps = 0
        self._done = False

    def get_next_waypoint(self, drone_pos: Sequence[float]) -> np.ndarray | None:
        if not self._cluster_representatives:
            return None

        drone = np.asarray(drone_pos, dtype=float).reshape(-1)
        if drone.size < 2:
            raise ValueError("drone_pos must have at least 2 values")
        drone_xy = drone[:2]

        dists = [float(np.linalg.norm(rep - drone_xy)) for rep in self._cluster_representatives]
        best_i = int(np.argmin(np.asarray(dists, dtype=float)))
        best_xy = self._cluster_representatives[best_i].copy()

        if self.verbose and self._should_log(best_xy):
            center = self._cluster_centers[best_i]
            print(
                f"[FrontierPlanner] select frontier cluster={best_i} "
                f"center=({center[0]:.2f}, {center[1]:.2f}) "
                f"waypoint=({best_xy[0]:.2f}, {best_xy[1]:.2f})"
            )

        if drone.size >= 3:
            z = float(self.waypoint_z) if self.waypoint_z is not None else float(drone[2])
            return np.array([best_xy[0], best_xy[1], z], dtype=float)
        return best_xy

    def is_exploration_done(self) -> bool:
        return self._done

    def get_frontier_centers(self) -> List[np.ndarray]:
        return [c.copy() for c in self._cluster_centers]

    def _cluster_frontiers(self, frontier_cells: List[Tuple[int, int]]) -> List[List[Tuple[int, int]]]:
        cell_set = set(frontier_cells)
        visited = set()
        clusters: List[List[Tuple[int, int]]] = []

        for cell in frontier_cells:
            if cell in visited:
                continue
            queue = deque([cell])
            visited.add(cell)
            comp: List[Tuple[int, int]] = []

            while queue:
                cur = queue.popleft()
                comp.append(cur)
                for nxt in self._neighbor_cells(cur):
                    if nxt in cell_set and nxt not in visited:
                        visited.add(nxt)
                        queue.append(nxt)

            clusters.append(comp)
        return clusters

    def _neighbor_cells(self, cell: Tuple[int, int]) -> List[Tuple[int, int]]:
        r, c = cell
        if self.cluster_connectivity == 4:
            offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        else:
            offsets = [
                (-1, 0),
                (1, 0),
                (0, -1),
                (0, 1),
                (-1, -1),
                (-1, 1),
                (1, -1),
                (1, 1),
            ]
        out: List[Tuple[int, int]] = []
        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.grid.height and 0 <= nc < self.grid.width:
                out.append((nr, nc))
        return out

    def _cluster_center(self, cluster: List[Tuple[int, int]]) -> np.ndarray:
        pts = np.asarray([self.grid.grid_to_world(r, c) for (r, c) in cluster], dtype=float)
        return np.mean(pts, axis=0)

    def _cluster_representative(self, cluster: List[Tuple[int, int]]) -> np.ndarray:
        pts = [self.grid.grid_to_world(r, c) for (r, c) in cluster]
        center = np.mean(np.asarray(pts, dtype=float), axis=0)
        dists = [float(np.linalg.norm(p - center)) for p in pts]
        return pts[int(np.argmin(np.asarray(dists, dtype=float)))].copy()

    def _should_log(self, selected_xy: np.ndarray) -> bool:
        if self._last_selected_xy is None:
            self._last_selected_xy = selected_xy.copy()
            return True
        moved = float(np.linalg.norm(selected_xy - self._last_selected_xy)) > 0.05
        if moved:
            self._last_selected_xy = selected_xy.copy()
            return True
        return False

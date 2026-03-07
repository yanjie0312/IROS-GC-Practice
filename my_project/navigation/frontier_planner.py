from __future__ import annotations

from collections import deque
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .occupancy_grid import FREE, OCCUPIED, OccupancyGrid


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
        reachability_step: float | None = None,
        path_lookahead_dist: float = 0.8,
        path_connectivity: int = 8,
        verbose: bool = False,
    ):
        if cluster_connectivity not in (4, 8):
            raise ValueError("cluster_connectivity must be 4 or 8")
        if path_connectivity not in (4, 8):
            raise ValueError("path_connectivity must be 4 or 8")
        if min_cluster_size <= 0:
            raise ValueError("min_cluster_size must be > 0")

        self.grid = occupancy_grid
        self.cluster_connectivity = int(cluster_connectivity)
        self.min_cluster_size = int(min_cluster_size)
        self.done_frontier_streak = int(max(1, done_frontier_streak))
        self.waypoint_z = waypoint_z
        self.reachability_step = (
            float(reachability_step)
            if reachability_step is not None
            else max(0.05, 0.5 * float(self.grid.resolution))
        )
        self.path_lookahead_dist = float(max(0.0, path_lookahead_dist))
        self.path_connectivity = int(path_connectivity)
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
            self._handle_no_frontiers()
            return

        clusters = self._cluster_frontiers(frontier_cells)
        clusters = [c for c in clusters if len(c) >= self.min_cluster_size]
        if not clusters:
            self._handle_no_frontiers()
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

        best_i, best_xy = self._select_path_reachable_waypoint(drone_xy)
        if best_xy is None:
            best_i, best_xy = self._select_direct_reachable_waypoint(drone_xy)
        if best_xy is None:
            return None

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

    def get_waypoint_towards_goal(
        self,
        drone_pos: Sequence[float],
        goal_pos: Sequence[float],
        lookahead_dist: float | None = None,
        goal_search_radius: float = 0.6,
    ) -> np.ndarray | None:
        """
        Return a local waypoint toward an arbitrary goal.
        Prefer known-free path; fallback to short direct move if no occupied cell on ray.
        """
        drone = np.asarray(drone_pos, dtype=float).reshape(-1)
        goal = np.asarray(goal_pos, dtype=float).reshape(-1)
        if drone.size < 2 or goal.size < 2:
            raise ValueError("drone_pos and goal_pos must have at least 2 values")

        drone_xy = drone[:2]
        goal_xy = goal[:2]
        lookahead = self.path_lookahead_dist if lookahead_dist is None else float(max(0.05, lookahead_dist))

        start_idx = self.grid.world_to_grid(drone_xy)
        goal_idx = self.grid.world_to_grid(goal_xy)
        if start_idx is not None and goal_idx is not None:
            dist_map, parent = self._compute_free_space_tree(start_idx)
            reachable_goal_idx = self._find_reachable_goal_idx(
                goal_idx=goal_idx,
                dist_map=dist_map,
                goal_search_radius=float(max(0.0, goal_search_radius)),
            )
            if reachable_goal_idx is not None:
                path = self._reconstruct_path(parent, start_idx, reachable_goal_idx)
                if path:
                    lookahead_cells = int(round(lookahead / max(self.grid.resolution, 1e-6)))
                    lookahead_cells = max(1, lookahead_cells)
                    path_i = min(lookahead_cells, len(path) - 1)
                    wp_idx = path[path_i]
                    wp_xy = self.grid.grid_to_world(wp_idx[0], wp_idx[1])
                    return self._pack_xy_with_optional_z(drone, wp_xy)

        # If free-space path is unavailable, only move a short step directly
        # when no occupied cell blocks line-of-sight.
        if self._is_directly_reachable(drone_xy, goal_xy):
            direction = goal_xy - drone_xy
            dist = float(np.linalg.norm(direction))
            if dist < 1e-9:
                return self._pack_xy_with_optional_z(drone, goal_xy)
            step = min(lookahead, dist)
            wp_xy = drone_xy + direction / dist * step
            return self._pack_xy_with_optional_z(drone, wp_xy)

        return None

    def _handle_no_frontiers(self) -> None:
        self._clusters = []
        self._cluster_centers = []
        self._cluster_representatives = []
        self._no_frontier_steps += 1
        self._done = self._no_frontier_steps >= self.done_frontier_streak

    def _find_reachable_goal_idx(
        self,
        goal_idx: Tuple[int, int],
        dist_map: np.ndarray,
        goal_search_radius: float,
    ) -> Tuple[int, int] | None:
        gr, gc = goal_idx
        if (
            0 <= gr < self.grid.height
            and 0 <= gc < self.grid.width
            and int(self.grid.grid[gr, gc]) == FREE
            and np.isfinite(dist_map[gr, gc])
        ):
            return gr, gc

        radius_cells = int(np.ceil(goal_search_radius / max(self.grid.resolution, 1e-6)))
        radius_cells = max(1, radius_cells)

        best_idx: Tuple[int, int] | None = None
        best_geo = float("inf")
        best_path = float("inf")

        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                nr, nc = gr + dr, gc + dc
                if not (0 <= nr < self.grid.height and 0 <= nc < self.grid.width):
                    continue
                if int(self.grid.grid[nr, nc]) != FREE:
                    continue
                path_cost = float(dist_map[nr, nc])
                if not np.isfinite(path_cost):
                    continue
                geo = float(np.hypot(float(dr), float(dc)))
                if geo < best_geo - 1e-9 or (abs(geo - best_geo) <= 1e-9 and path_cost < best_path):
                    best_geo = geo
                    best_path = path_cost
                    best_idx = (nr, nc)

        return best_idx

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

    def _select_path_reachable_waypoint(self, drone_xy: np.ndarray) -> Tuple[int, np.ndarray | None]:
        start_idx = self.grid.world_to_grid(drone_xy)
        if start_idx is None:
            return -1, None

        dist_map, parent = self._compute_free_space_tree(start_idx)
        best_i = -1
        best_goal_idx: Tuple[int, int] | None = None
        best_path_cost = float("inf")

        for i, rep_xy in enumerate(self._cluster_representatives):
            goal_idx = self.grid.world_to_grid(rep_xy)
            if goal_idx is None:
                continue
            path_cost = float(dist_map[goal_idx[0], goal_idx[1]])
            if not np.isfinite(path_cost):
                continue
            if path_cost < best_path_cost:
                best_path_cost = path_cost
                best_i = int(i)
                best_goal_idx = goal_idx

        if best_i < 0 or best_goal_idx is None:
            return -1, None

        path = self._reconstruct_path(parent, start_idx, best_goal_idx)
        if not path:
            return -1, None

        lookahead_cells = int(round(self.path_lookahead_dist / max(self.grid.resolution, 1e-6)))
        lookahead_cells = max(1, lookahead_cells)
        path_i = min(lookahead_cells, len(path) - 1)
        wp_idx = path[path_i]
        wp_xy = self.grid.grid_to_world(wp_idx[0], wp_idx[1])
        return best_i, wp_xy

    def _select_direct_reachable_waypoint(self, drone_xy: np.ndarray) -> Tuple[int, np.ndarray | None]:
        dists = np.asarray(
            [float(np.linalg.norm(rep - drone_xy)) for rep in self._cluster_representatives],
            dtype=float,
        )
        if dists.size == 0:
            return -1, None

        order = np.argsort(dists)
        for idx in order:
            rep = self._cluster_representatives[int(idx)]
            if self._is_directly_reachable(drone_xy, rep):
                return int(idx), rep.copy()

        fallback_i = int(order[0])
        return fallback_i, self._cluster_representatives[fallback_i].copy()

    def _compute_free_space_tree(
        self,
        start_idx: Tuple[int, int],
    ) -> Tuple[np.ndarray, Dict[Tuple[int, int], Tuple[int, int] | None]]:
        h, w = self.grid.height, self.grid.width
        dist = np.full((h, w), np.inf, dtype=np.float32)
        parent: Dict[Tuple[int, int], Tuple[int, int] | None] = {}

        sr, sc = start_idx
        if int(self.grid.grid[sr, sc]) == OCCUPIED:
            return dist, parent

        q = deque([(sr, sc)])
        dist[sr, sc] = 0.0
        parent[(sr, sc)] = None

        while q:
            r, c = q.popleft()
            base = float(dist[r, c])
            for nr, nc in self._navigation_neighbors(r, c):
                if np.isfinite(dist[nr, nc]):
                    continue
                if int(self.grid.grid[nr, nc]) != FREE:
                    continue
                dist[nr, nc] = base + 1.0
                parent[(nr, nc)] = (r, c)
                q.append((nr, nc))

        return dist, parent

    def _navigation_neighbors(self, row: int, col: int) -> List[Tuple[int, int]]:
        if self.path_connectivity == 4:
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
            nr, nc = row + dr, col + dc
            if 0 <= nr < self.grid.height and 0 <= nc < self.grid.width:
                out.append((nr, nc))
        return out

    def _reconstruct_path(
        self,
        parent: Dict[Tuple[int, int], Tuple[int, int] | None],
        start_idx: Tuple[int, int],
        goal_idx: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        if goal_idx not in parent:
            return []
        path: List[Tuple[int, int]] = []
        cur: Tuple[int, int] | None = goal_idx
        while cur is not None:
            path.append(cur)
            cur = parent.get(cur)
        path.reverse()
        if not path or path[0] != start_idx:
            return []
        return path

    def _is_directly_reachable(self, start_xy: np.ndarray, goal_xy: np.ndarray) -> bool:
        direction = goal_xy - start_xy
        dist = float(np.linalg.norm(direction))
        if dist < 1e-6:
            return True

        step = max(1e-3, float(self.reachability_step))
        num_steps = int(np.ceil(dist / step))
        unit = direction / dist

        for k in range(1, num_steps + 1):
            p = start_xy + unit * (k * step)
            idx = self.grid.world_to_grid(p)
            if idx is None:
                return False
            if int(self.grid.grid[idx[0], idx[1]]) == OCCUPIED:
                return False
        return True

    def _pack_xy_with_optional_z(self, drone: np.ndarray, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float).reshape(2)
        if drone.size >= 3:
            z = float(self.waypoint_z) if self.waypoint_z is not None else float(drone[2])
            return np.array([xy[0], xy[1], z], dtype=float)
        return xy

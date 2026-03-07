# my_project/navigation/search_mission.py
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from .base import BaseMission, Command, State
from .frontier_planner import FrontierPlanner
from .occupancy_grid import OCCUPIED
from ..env.targets import TargetManager


class _Phase(Enum):
    TAKEOFF = auto()
    EXPLORE = auto()
    GOTO_TARGET = auto()
    INSPECT = auto()
    DONE = auto()


class SearchMission(BaseMission):
    """
    搜索任务状态机：TAKEOFF → EXPLORE → GOTO_TARGET → INSPECT → DONE

    - TAKEOFF   : 爬升至巡航高度
    - EXPLORE   : Frontier 驱动探索；发现目标时立即切换 GOTO_TARGET
    - GOTO_TARGET: 飞向最近未巡检目标
    - INSPECT   : 悬停在目标附近，等待 TargetManager 计时完成
    - DONE      : 返回起飞点，finished=True

    使用示例（在 main.py 中）：
        planner = FrontierPlanner(OccupancyGrid(...))
        mission = SearchMission(planner, target_manager)
        manager = MissionManager(mission, AvoidanceLayer())
        manager.reset(state)
        ...
        cmd = manager.update(state, pkt)
    """

    def __init__(
        self,
        frontier_planner: FrontierPlanner,
        target_manager: TargetManager,
        takeoff_height: float = 1.0,
        waypoint_reach_dist: float = 0.35,
        inspect_hover_dist: float = 0.4,
        frontier_replan_cooldown_steps: int = 10,
        min_waypoint_delta: float = 0.12,
        target_retry_cooldown_steps: int = 120,
        stuck_window_steps: int = 180,
        stuck_radius: float = 0.25,
        goto_path_lookahead_dist: float = 0.6,
        goto_goal_search_radius: float = 0.8,
        verbose: bool = True,
    ):
        """
        Parameters
        ----------
        frontier_planner   : 已初始化的 FrontierPlanner 实例
        target_manager     : 已初始化的 TargetManager 实例
        takeoff_height     : 巡航高度（m）
        waypoint_reach_dist: 判定"到达 frontier 航点"的水平距离阈值（m）
        inspect_hover_dist : 判定"到达目标"切换 INSPECT 的距离阈值（m）
        verbose            : 是否打印状态机转换日志
        """
        self.planner = frontier_planner
        self.target_manager = target_manager
        self.takeoff_height = float(takeoff_height)
        self.waypoint_reach_dist = float(waypoint_reach_dist)
        self.inspect_hover_dist = float(inspect_hover_dist)
        self.frontier_replan_cooldown_steps = int(max(1, frontier_replan_cooldown_steps))
        self.min_waypoint_delta = float(max(0.0, min_waypoint_delta))
        self.target_retry_cooldown_steps = int(max(1, target_retry_cooldown_steps))
        self.stuck_window_steps = int(max(10, stuck_window_steps))
        self.stuck_radius = float(max(0.05, stuck_radius))
        self.goto_path_lookahead_dist = float(max(0.05, goto_path_lookahead_dist))
        self.goto_goal_search_radius = float(max(0.1, goto_goal_search_radius))
        self.verbose = verbose

        self._phase = _Phase.TAKEOFF
        self._home_pos = np.zeros(3, dtype=float)
        self._current_waypoint = np.zeros(3, dtype=float)
        self._current_target_id: Optional[int] = None
        self._current_target_pos = np.zeros(3, dtype=float)
        self._cur_step = 0
        self._last_frontier_pick_step = -10**9
        self._next_target_attempt_step = 0
        self._stuck_anchor_pos = np.zeros(3, dtype=float)
        self._stuck_anchor_step = 0

    # ------------------------------------------------------------------
    # BaseMission interface
    # ------------------------------------------------------------------

    def reset(self, state: State) -> None:
        self._cur_step = int(state.step)
        self._home_pos = np.asarray(state.xyz, dtype=float).copy()
        self._current_waypoint = np.array(
            [self._home_pos[0], self._home_pos[1], self.takeoff_height],
            dtype=float,
        )
        self._phase = _Phase.TAKEOFF
        self._current_target_id = None
        self._current_target_pos = np.zeros(3, dtype=float)
        self._last_frontier_pick_step = -10**9
        self._next_target_attempt_step = 0
        self._stuck_anchor_pos = self._home_pos.copy()
        self._stuck_anchor_step = self._cur_step
        self.planner.reset()
        self._log(f"reset → TAKEOFF  home={self._home_pos.round(2)}")

    def update(self, state: State, sensors: dict) -> Command:
        self._cur_step = int(state.step)
        pos = np.asarray(state.xyz, dtype=float)
        zero_rpy = np.zeros(3, dtype=float)

        # 每帧更新占据栅格（无论处于哪个阶段，地图要持续更新）
        self._update_grid(pos, sensors)

        if self._phase == _Phase.TAKEOFF:
            return self._handle_takeoff(pos, zero_rpy)
        elif self._phase == _Phase.EXPLORE:
            return self._handle_explore(pos, zero_rpy)
        elif self._phase == _Phase.GOTO_TARGET:
            return self._handle_goto_target(pos, zero_rpy)
        elif self._phase == _Phase.INSPECT:
            return self._handle_inspect(pos, zero_rpy)
        else:  # DONE
            return self._handle_done(pos, zero_rpy)

    # ------------------------------------------------------------------
    # Phase handlers
    # ------------------------------------------------------------------

    def _handle_takeoff(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        target = self._current_waypoint.copy()
        if abs(pos[2] - self.takeoff_height) < 0.1:
            self._log("TAKEOFF done → EXPLORE")
            self._phase = _Phase.EXPLORE
            wp = self._pick_next_frontier(pos)
            if wp is not None:
                self._current_waypoint = wp
                self._last_frontier_pick_step = self._cur_step
                self._reset_stuck_anchor(pos)
            else:
                # 起飞后立刻没有 frontier（理论上不会），直接检查是否有目标
                self._try_transition_to_target_or_done(pos, rpy)
        return Command(target_pos=target, target_rpy=rpy)

    def _handle_explore(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        # 优先检查是否有已发现但未巡检的目标
        discovered = self._get_discovered_uninspected(pos)
        if discovered is not None and self._cur_step >= self._next_target_attempt_step:
            tid, tpos = discovered
            approach_wp = self._get_target_approach_waypoint(pos, tpos)
            if approach_wp is not None:
                self._current_target_id = tid
                self._current_target_pos = tpos
                self._phase = _Phase.GOTO_TARGET
                self._log(f"EXPLORE: target {tid} discovered → GOTO_TARGET  pos={tpos.round(2)}")
                return Command(target_pos=approach_wp, target_rpy=rpy)
            self._next_target_attempt_step = self._cur_step + self.target_retry_cooldown_steps
            guided_wp = self._pick_frontier_towards(pos, tpos)
            if guided_wp is not None and float(np.linalg.norm(guided_wp[:2] - self._current_waypoint[:2])) > 0.20:
                self._current_waypoint = guided_wp
                self._last_frontier_pick_step = self._cur_step
                self._log(f"EXPLORE: guide toward target {tid} → {guided_wp.round(2)}")

        # 判断是否到达当前 frontier 航点（水平距离）
        dist_xy = float(np.linalg.norm(pos[:2] - self._current_waypoint[:2]))
        force_replan = self._is_explore_stuck(pos)
        if dist_xy < self.waypoint_reach_dist or force_replan:
            if (not force_replan) and (
                (self._cur_step - self._last_frontier_pick_step) < self.frontier_replan_cooldown_steps
            ):
                return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

            wp = self._pick_next_frontier(pos)
            if wp is not None:
                if force_replan or float(np.linalg.norm(wp[:2] - self._current_waypoint[:2])) >= self.min_waypoint_delta:
                    self._current_waypoint = wp
                    if force_replan:
                        self._log(f"EXPLORE: stuck, replan frontier → {wp.round(2)}")
                    else:
                        self._log(f"EXPLORE: next frontier → {wp.round(2)}")
                self._last_frontier_pick_step = self._cur_step
                if force_replan:
                    self._reset_stuck_anchor(pos)
            else:
                # 没有更多 frontier，探索结束
                cmd = self._try_transition_to_target_or_done(pos, rpy)
                if cmd is not None:
                    return cmd

        return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

    def _handle_goto_target(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        if self._current_target_id is None:
            self._phase = _Phase.EXPLORE
            return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

        # 在目标正上方巡航高度悬停，不跟随目标的地面 z 坐标
        hover = np.array([
            self._current_target_pos[0],
            self._current_target_pos[1],
            self.takeoff_height,
        ], dtype=float)
        dist_xy = float(np.linalg.norm(pos[:2] - hover[:2]))
        if dist_xy < self.inspect_hover_dist:
            self._log(f"GOTO_TARGET: reached target {self._current_target_id} → INSPECT")
            self._phase = _Phase.INSPECT
            return Command(target_pos=hover, target_rpy=rpy)

        approach_wp = self._get_target_approach_waypoint(pos, self._current_target_pos)
        if approach_wp is not None:
            return Command(target_pos=approach_wp, target_rpy=rpy)

        # 目标当前不可达：先继续探索，避免贴墙振荡
        self._phase = _Phase.EXPLORE
        self._next_target_attempt_step = self._cur_step + self.target_retry_cooldown_steps
        self._reset_stuck_anchor(pos)
        wp = self._pick_frontier_towards(pos, self._current_target_pos)
        if wp is None:
            wp = self._pick_next_frontier(pos)
        if wp is not None:
            self._current_waypoint = wp
            self._last_frontier_pick_step = self._cur_step
            self._log(f"GOTO_TARGET: target {self._current_target_id} 暂不可达，回到 EXPLORE")
            return Command(target_pos=wp, target_rpy=rpy)
        return Command(target_pos=np.array([pos[0], pos[1], self.takeoff_height], dtype=float), target_rpy=rpy)

    def _handle_inspect(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        tid = self._current_target_id
        # 在目标正上方巡航高度悬停等待计时
        tpos = np.array([
            self._current_target_pos[0],
            self._current_target_pos[1],
            self.takeoff_height,
        ], dtype=float)

        # 检查当前目标是否巡检完成（TargetManager 在 main.py 的循环里计时）
        if tid is not None and self.target_manager.targets[tid].inspected:
            inspected, _, total = self.target_manager.get_progress()
            self._log(f"INSPECT done  ({inspected}/{total} targets inspected)")

            # 还有未巡检目标？
            next_t = self.target_manager.get_nearest_discovered_uninspected(pos)
            if next_t is not None:
                ntid, ntpos, _ = next_t
                approach_wp = self._get_target_approach_waypoint(pos, ntpos)
                if approach_wp is not None:
                    self._current_target_id = ntid
                    self._current_target_pos = ntpos
                    self._phase = _Phase.GOTO_TARGET
                    self._log(f"→ GOTO_TARGET {ntid}  pos={ntpos.round(2)}")
                    return Command(target_pos=approach_wp, target_rpy=rpy)
                self._next_target_attempt_step = self._cur_step + self.target_retry_cooldown_steps
                wp = self._pick_frontier_towards(pos, ntpos)
                if wp is not None:
                    self._current_waypoint = wp
                    self._last_frontier_pick_step = self._cur_step
                    self._phase = _Phase.EXPLORE
                    self._reset_stuck_anchor(pos)
                    self._log(f"→ EXPLORE (guide to target {ntid})")
                    return Command(target_pos=wp, target_rpy=rpy)

            # 无目标：恢复探索或结束
            if not self.planner.is_exploration_done():
                wp = self._pick_next_frontier(pos)
                if wp is not None:
                    self._current_waypoint = wp
                    self._last_frontier_pick_step = self._cur_step
                    self._phase = _Phase.EXPLORE
                    self._reset_stuck_anchor(pos)
                    self._log("→ EXPLORE")
                    return Command(target_pos=wp, target_rpy=rpy)

            # 探索完成且无目标 → DONE
            self._phase = _Phase.DONE
            self._current_waypoint = self._home_pos.copy()
            self._log("→ DONE")
            return Command(target_pos=self._home_pos.copy(), target_rpy=rpy)

        # 悬停在目标上方，等待计时完成
        return Command(target_pos=tpos.copy(), target_rpy=rpy)

    def _handle_done(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        home = self._home_pos.copy()
        dist = float(np.linalg.norm(pos - home))
        finished = dist < 0.2
        if finished:
            self._log("DONE: returned home, mission complete")
        return Command(
            target_pos=home,
            target_rpy=rpy,
            finished=finished,
            info="mission_done",
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_grid(self, pos: np.ndarray, sensors: dict) -> None:
        ray_dists = sensors.get("ray_dists")
        ray_dirs_world = sensors.get("ray_dirs_world")
        if ray_dists is not None and ray_dirs_world is not None:
            self.planner.update(pos, ray_dists, ray_dirs_world)

    def _pick_next_frontier(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """
        获取下一个安全航点：
        沿朝向 frontier 的方向逐步检查占据栅格，
        遇到 OCCUPIED 格才停下（UNKNOWN 格可穿越），返回最远安全点。
        """
        inspected, discovered, _ = self.target_manager.get_progress()
        if discovered == 0 and inspected == 0:
            biased = self._pick_rightward_frontier(pos)
            if biased is not None:
                return biased

        wp = self.planner.get_next_waypoint(pos)
        if wp is None:
            return None
        safe = self._clip_waypoint_to_safe(pos, wp[:2])
        if float(np.linalg.norm(safe[:2] - pos[:2])) < 0.12:
            centers = self.planner.get_frontier_centers()
            if centers:
                dists = np.asarray([float(np.linalg.norm(c[:2] - pos[:2])) for c in centers], dtype=float)
                for idx in np.argsort(dists):
                    alt = self._clip_waypoint_to_safe(pos, centers[int(idx)][:2])
                    if float(np.linalg.norm(alt[:2] - pos[:2])) > 0.20:
                        return alt

        return safe

    def _clip_waypoint_to_safe(self, pos: np.ndarray, goal_xy: np.ndarray) -> np.ndarray:
        goal_xy = np.asarray(goal_xy, dtype=float).reshape(2)
        pos_xy = pos[:2]
        direction = goal_xy - pos_xy
        dist = float(np.linalg.norm(direction))

        if dist < 1e-6:
            return np.array([pos_xy[0], pos_xy[1], self.takeoff_height], dtype=float)

        unit = direction / dist
        check_step = 0.1
        max_step = min(dist, 1.6)

        safe_xy = pos_xy.copy()
        d = check_step
        while d <= max_step:
            candidate = pos_xy + unit * d
            idx = self.planner.grid.world_to_grid(candidate)
            if idx is None:
                break
            if self.planner.grid.grid[idx[0], idx[1]] == OCCUPIED:
                break
            safe_xy = candidate
            d += check_step

        return np.array([safe_xy[0], safe_xy[1], self.takeoff_height], dtype=float)

    def _get_discovered_uninspected(self, pos: np.ndarray) -> Optional[Tuple[int, np.ndarray]]:
        nearest = self.target_manager.get_nearest_discovered_uninspected(pos)
        if nearest is None:
            return None
        tid, tpos, _ = nearest
        return int(tid), np.asarray(tpos, dtype=float).copy()

    def _get_target_approach_waypoint(self, pos: np.ndarray, target_pos: np.ndarray) -> Optional[np.ndarray]:
        wp = self.planner.get_waypoint_towards_goal(
            drone_pos=pos,
            goal_pos=target_pos[:2],
            lookahead_dist=self.goto_path_lookahead_dist,
            goal_search_radius=self.goto_goal_search_radius,
        )
        if wp is None:
            return None
        wp = np.asarray(wp, dtype=float).reshape(-1)
        if wp.size < 2:
            return None
        return np.array([wp[0], wp[1], self.takeoff_height], dtype=float)

    def _pick_frontier_towards(self, pos: np.ndarray, goal_pos: np.ndarray) -> Optional[np.ndarray]:
        centers = self.planner.get_frontier_centers()
        if not centers:
            return None
        goal_xy = np.asarray(goal_pos[:2], dtype=float)
        pos_xy = np.asarray(pos[:2], dtype=float)
        v = goal_xy - pos_xy
        nv = float(np.linalg.norm(v))
        if nv < 1e-9:
            return None
        u = v / nv

        best_wp: Optional[np.ndarray] = None
        best_score = -1e9
        for c in centers:
            wp = self._clip_waypoint_to_safe(pos, c[:2])
            move = wp[:2] - pos_xy
            move_dist = float(np.linalg.norm(move))
            if move_dist < 0.18:
                continue
            progress = float(np.dot(move, u))
            lateral = float(np.linalg.norm(move - progress * u))
            score = progress - 0.35 * lateral
            if score > best_score:
                best_score = score
                best_wp = wp

        if best_wp is not None and best_score > 0.02:
            return best_wp
        return None

    def _pick_rightward_frontier(self, pos: np.ndarray) -> Optional[np.ndarray]:
        """
        Early exploration bias: before any target is discovered,
        prefer frontiers that progress to +x (toward right-side rooms).
        """
        centers = self.planner.get_frontier_centers()
        if not centers:
            return None

        pos_xy = np.asarray(pos[:2], dtype=float)
        best_wp: Optional[np.ndarray] = None
        best_score = -1e9

        for c in centers:
            wp = self._clip_waypoint_to_safe(pos, c[:2])
            move = wp[:2] - pos_xy
            move_dist = float(np.linalg.norm(move))
            if move_dist < 0.20:
                continue
            dx = float(move[0])
            dy = abs(float(move[1]))
            if dx <= 0.08:
                continue
            # Favor forward (+x) progress while discouraging large lateral drift.
            score = dx - 0.35 * dy
            if score > best_score:
                best_score = score
                best_wp = wp

        if best_wp is not None and best_score > 0.04:
            return best_wp
        return None

    def _try_transition_to_target_or_done(
        self, pos: np.ndarray, rpy: np.ndarray
    ) -> Optional[Command]:
        """
        探索结束后的后续决策：
        - 有未巡检目标 → GOTO_TARGET，返回 Command
        - 无目标 → DONE，返回 Command
        - 仍有 frontier（不应发生）→ 返回 None（继续 EXPLORE）
        """
        remaining = self.target_manager.get_nearest_discovered_uninspected(pos)
        if remaining is not None:
            tid, tpos, _ = remaining
            approach_wp = self._get_target_approach_waypoint(pos, tpos)
            if approach_wp is not None:
                self._current_target_id = tid
                self._current_target_pos = tpos
                self._phase = _Phase.GOTO_TARGET
                self._log(f"Exploration done → GOTO_TARGET {tid}  pos={tpos.round(2)}")
                return Command(target_pos=approach_wp, target_rpy=rpy)

        self._phase = _Phase.DONE
        self._current_waypoint = self._home_pos.copy()
        self._log("Exploration done, no targets left → DONE")
        return Command(target_pos=self._home_pos.copy(), target_rpy=rpy)

    def _is_explore_stuck(self, pos: np.ndarray) -> bool:
        moved = float(np.linalg.norm(pos[:2] - self._stuck_anchor_pos[:2]))
        if moved > self.stuck_radius:
            self._reset_stuck_anchor(pos)
            return False
        return (self._cur_step - self._stuck_anchor_step) >= self.stuck_window_steps

    def _reset_stuck_anchor(self, pos: np.ndarray) -> None:
        self._stuck_anchor_pos = np.asarray(pos, dtype=float).copy()
        self._stuck_anchor_step = self._cur_step

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SearchMission] {msg}")

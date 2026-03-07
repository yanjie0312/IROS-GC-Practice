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
        self.verbose = verbose

        self._phase = _Phase.TAKEOFF
        self._home_pos = np.zeros(3, dtype=float)
        self._current_waypoint = np.zeros(3, dtype=float)
        self._current_target_id: Optional[int] = None
        self._current_target_pos = np.zeros(3, dtype=float)

    # ------------------------------------------------------------------
    # BaseMission interface
    # ------------------------------------------------------------------

    def reset(self, state: State) -> None:
        self._home_pos = np.asarray(state.xyz, dtype=float).copy()
        self._current_waypoint = np.array(
            [self._home_pos[0], self._home_pos[1], self.takeoff_height],
            dtype=float,
        )
        self._phase = _Phase.TAKEOFF
        self._current_target_id = None
        self._current_target_pos = np.zeros(3, dtype=float)
        self.planner.reset()
        self._log(f"reset → TAKEOFF  home={self._home_pos.round(2)}")

    def update(self, state: State, sensors: dict) -> Command:
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
            else:
                # 起飞后立刻没有 frontier（理论上不会），直接检查是否有目标
                self._try_transition_to_target_or_done(pos, rpy)
        return Command(target_pos=target, target_rpy=rpy)

    def _handle_explore(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        # 优先检查是否有已发现但未巡检的目标
        discovered = self._get_discovered_uninspected()
        if discovered is not None:
            tid, tpos = discovered
            self._current_target_id = tid
            self._current_target_pos = tpos
            self._log(f"EXPLORE: target {tid} discovered → GOTO_TARGET  pos={tpos.round(2)}")
            self._phase = _Phase.GOTO_TARGET
            return Command(target_pos=tpos, target_rpy=rpy)

        # 判断是否到达当前 frontier 航点（水平距离）
        dist_xy = float(np.linalg.norm(pos[:2] - self._current_waypoint[:2]))
        if dist_xy < self.waypoint_reach_dist:
            wp = self._pick_next_frontier(pos)
            if wp is not None:
                self._current_waypoint = wp
                self._log(f"EXPLORE: next frontier → {wp.round(2)}")
            else:
                # 没有更多 frontier，探索结束
                cmd = self._try_transition_to_target_or_done(pos, rpy)
                if cmd is not None:
                    return cmd

        return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

    def _handle_goto_target(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        tpos = self._current_target_pos
        dist = float(np.linalg.norm(pos - tpos))
        if dist < self.inspect_hover_dist:
            self._log(f"GOTO_TARGET: reached target {self._current_target_id} → INSPECT")
            self._phase = _Phase.INSPECT
        return Command(target_pos=tpos.copy(), target_rpy=rpy)

    def _handle_inspect(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        tid = self._current_target_id
        tpos = self._current_target_pos

        # 检查当前目标是否巡检完成（TargetManager 在 main.py 的循环里计时）
        if tid is not None and self.target_manager.targets[tid].inspected:
            inspected, _, total = self.target_manager.get_progress()
            self._log(f"INSPECT done  ({inspected}/{total} targets inspected)")

            # 还有未巡检目标？
            next_t = self.target_manager.get_nearest_unvisited(pos)
            if next_t is not None:
                ntid, ntpos, _ = next_t
                self._current_target_id = ntid
                self._current_target_pos = ntpos
                self._phase = _Phase.GOTO_TARGET
                self._log(f"→ GOTO_TARGET {ntid}  pos={ntpos.round(2)}")
                return Command(target_pos=ntpos.copy(), target_rpy=rpy)

            # 无目标：恢复探索或结束
            if not self.planner.is_exploration_done():
                wp = self._pick_next_frontier(pos)
                if wp is not None:
                    self._current_waypoint = wp
                    self._phase = _Phase.EXPLORE
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
        1. 从 FrontierPlanner 获取目标 frontier
        2. 沿目标方向逐步检查占据栅格，在遇到 OCCUPIED 格之前停下
        3. 这样保证航点始终在已知自由空间内，不穿越墙壁
        """
        wp = self.planner.get_next_waypoint(pos)
        if wp is None:
            return None

        goal_xy = wp[:2]
        pos_xy = pos[:2]
        direction = goal_xy - pos_xy
        dist = float(np.linalg.norm(direction))

        if dist < 1e-6:
            return np.array([pos_xy[0], pos_xy[1], self.takeoff_height], dtype=float)

        unit = direction / dist
        check_step = 0.1          # 每次检查步长（m）
        max_step = min(dist, 1.5) # 单次最远前进距离（m）

        safe_xy = pos_xy.copy()
        d = check_step
        while d <= max_step:
            candidate = pos_xy + unit * d
            # 只在确认是墙（OCCUPIED）时才停下，UNKNOWN 格可以穿越（探索目的）
            idx = self.planner.grid.world_to_grid(candidate)
            if idx is None:
                break  # 超出栅格范围
            if self.planner.grid.grid[idx[0], idx[1]] == OCCUPIED:
                break  # 确认是墙，停在上一步
            safe_xy = candidate
            d += check_step

        return np.array([safe_xy[0], safe_xy[1], self.takeoff_height], dtype=float)

    def _get_discovered_uninspected(self) -> Optional[Tuple[int, np.ndarray]]:
        """返回第一个已发现但未巡检的目标 (id, pos)，没有则返回 None。"""
        for tid, info in self.target_manager.targets.items():
            if info.discovered and not info.inspected:
                return tid, info.position.copy()
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
        remaining = self.target_manager.get_nearest_unvisited(pos)
        if remaining is not None:
            tid, tpos, _ = remaining
            self._current_target_id = tid
            self._current_target_pos = tpos
            self._phase = _Phase.GOTO_TARGET
            self._log(f"Exploration done → GOTO_TARGET {tid}  pos={tpos.round(2)}")
            return Command(target_pos=tpos.copy(), target_rpy=rpy)

        self._phase = _Phase.DONE
        self._current_waypoint = self._home_pos.copy()
        self._log("Exploration done, no targets left → DONE")
        return Command(target_pos=self._home_pos.copy(), target_rpy=rpy)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(f"[SearchMission] {msg}")

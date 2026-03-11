# my_project/navigation/search_mission.py
from __future__ import annotations

from enum import Enum, auto
from typing import Optional, Tuple

import numpy as np

from .base import BaseMission, Command, State
from .frontier_planner import FrontierPlanner
from .occupancy_grid import FREE
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
        # 区域卡住：同一片区域停留过久且仍有未发现目标时，向起飞点撤退以离开房间
        self._region_anchor_pos = np.zeros(3, dtype=float)
        self._region_anchor_step = 0
        self._region_retreat_until_step = -1
        self._region_stuck_radius = 1.0
        self._region_stuck_steps = 400
        self._region_retreat_duration_steps = 150
        self._just_finished_retreat = False
        self._region_anchor_update_dist = 1.5
        self._retreat_waypoint_min_dist = 1.0
        # 同区域内重规划次数超过此次数也视为区域卡住（不必等 400 步）
        self._region_stuck_replan_threshold = 10
        self._region_replan_count = 0
        # 同区域重规划超过此次数后，每次选点都排除当前方向，主动换 cluster
        self._region_exclude_after_replans = 5
        # 无 frontier 时“仍有未发现目标”的日志节流，避免每帧刷屏
        self._last_nofrontier_log_step = -10**9
        # 返航阶段卡住检测：无进展时缩短 lookahead
        self._done_stuck_anchor_dist: float = -1.0
        self._done_stuck_anchor_step: int = -1
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
        self._region_anchor_pos = self._home_pos.copy()
        self._region_anchor_step = self._cur_step
        self._region_retreat_until_step = -1
        self._region_replan_count = 0
        self._last_nofrontier_log_step = -10**9
        self._just_finished_retreat = False
        self._done_stuck_anchor_dist = -1.0
        self._done_stuck_anchor_step = -1
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
        # 全部发现且全部巡检完 → 立即返航，不继续探索
        inspected, discovered, total = self.target_manager.get_progress()
        if total > 0 and discovered >= total and inspected >= total:
            self._phase = _Phase.DONE
            self._current_waypoint = self._home_pos.copy()
            self._log("EXPLORE: all targets done → DONE, return home")
            return Command(target_pos=self._home_pos.copy(), target_rpy=rpy)

        # 优先检查是否有已发现但未巡检的目标
        discovered_uninsp = self._get_discovered_uninspected(pos)
        if discovered_uninsp is not None and self._cur_step >= self._next_target_attempt_step:
            tid, tpos = discovered_uninsp
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

        # 撤退阶段：向起飞点后退一段，离开当前房间
        if self._region_retreat_until_step >= 0 and self._cur_step < self._region_retreat_until_step:
            dist_to_retreat_wp = float(np.linalg.norm(pos[:2] - self._current_waypoint[:2]))
            if dist_to_retreat_wp < self.waypoint_reach_dist:
                self._region_retreat_until_step = -1
                self._region_anchor_pos = np.asarray(pos, dtype=float).copy()
                self._region_anchor_step = self._cur_step
                self._region_replan_count = 0
                self._just_finished_retreat = True  # 下一帧选 frontier 时排除原 cluster，强制尝试其他房间
                self._log("EXPLORE: retreat reached, resuming exploration")
                # 继续执行下面逻辑，选新 frontier
            else:
                return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

        # 区域卡住：在同一片区域停留过久且仍有未发现目标 → 先试换 cluster，否则向起飞点撤退
        if self._region_retreat_until_step < 0:
            _, discovered_count, total = self.target_manager.get_progress()
            dist_to_anchor = float(np.linalg.norm(pos[:2] - self._region_anchor_pos[:2]))
            steps_in_region = self._cur_step - self._region_anchor_step
            replan_stuck = self._region_replan_count >= self._region_stuck_replan_threshold
            time_stuck = steps_in_region >= self._region_stuck_steps
            if (
                total > 0
                and discovered_count < total
                and dist_to_anchor < self._region_stuck_radius
                and (replan_stuck or time_stuck)
            ):
                center = self.planner.get_last_selected_cluster_center()
                exclude_xy = center if center is not None else self._current_waypoint[:2]
                wp_other = self._pick_next_frontier(pos, exclude_xy=exclude_xy)
                # 只有新航点明显在「另一片区域」才采纳：离当前位 >= 1m 且离本区域锚点 >= 1.5m，否则仍会在此处打转
                if wp_other is not None:
                    dist_wp = float(np.linalg.norm(wp_other[:2] - pos[:2]))
                    dist_wp_to_anchor = float(np.linalg.norm(wp_other[:2] - self._region_anchor_pos[:2]))
                    if dist_wp >= self._retreat_waypoint_min_dist and dist_wp_to_anchor >= self._region_anchor_update_dist:
                        self._current_waypoint = wp_other
                        self._last_frontier_pick_step = self._cur_step
                        self._region_anchor_pos = np.asarray(pos, dtype=float).copy()
                        self._region_anchor_step = self._cur_step
                        self._region_replan_count = 0
                        self._reset_stuck_anchor(pos)
                        self._log(f"EXPLORE: region stuck, try other frontier → {wp_other.round(2)}")
                        return Command(target_pos=wp_other, target_rpy=rpy)
                # 无其他区域或新航点仍在同片区域：向起飞点撤退
                retreat_wp = self.planner.get_waypoint_towards_goal(
                    drone_pos=pos,
                    goal_pos=self._home_pos[:2],
                    lookahead_dist=1.0,
                    goal_search_radius=1.2,
                )
                if retreat_wp is None:
                    home_xy = self._home_pos[:2]
                    pos_xy = pos[:2]
                    retreat_xy = pos_xy + 0.7 * (home_xy - pos_xy)
                    retreat_wp = np.array(
                        [retreat_xy[0], retreat_xy[1], self.takeoff_height],
                        dtype=float,
                    )
                self._current_waypoint = retreat_wp
                self._region_retreat_until_step = self._cur_step + self._region_retreat_duration_steps
                self._last_frontier_pick_step = self._cur_step
                self._region_replan_count = 0
                self._reset_stuck_anchor(pos)
                self._log("EXPLORE: region stuck, retreat toward home")
                return Command(target_pos=retreat_wp, target_rpy=rpy)

        # 判断是否到达当前 frontier 航点（水平距离）
        dist_xy = float(np.linalg.norm(pos[:2] - self._current_waypoint[:2]))
        force_replan = self._is_explore_stuck(pos)
        if dist_xy < self.waypoint_reach_dist or force_replan:
            if (not force_replan) and (
                (self._cur_step - self._last_frontier_pick_step) < self.frontier_replan_cooldown_steps
            ):
                return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

            # 卡住时排除当前航点所在 cluster；同区域已重规划多次时也排除当前方向；刚撤退完也排除原 cluster 以换房间
            exclude = None
            if self._just_finished_retreat:
                center = self.planner.get_last_selected_cluster_center()
                exclude = center if center is not None else self._current_waypoint[:2]
                self._just_finished_retreat = False
            elif force_replan or (
                self._region_replan_count >= self._region_exclude_after_replans
                and float(np.linalg.norm(pos[:2] - self._region_anchor_pos[:2])) <= self._region_stuck_radius
            ):
                center = self.planner.get_last_selected_cluster_center()
                exclude = center if center is not None else self._current_waypoint[:2]
            wp = self._pick_next_frontier(pos, exclude_xy=exclude)
            # 排除当前方向后若无其他 cluster 可达，再试一次不排除，避免直接进“无 frontier”而悬停
            if wp is None and exclude is not None:
                wp = self._pick_next_frontier(pos, exclude_xy=None)
            # 同区域已重规划多次且新航点仍在同一片区域（门边打转）：强制向起飞点方向迈一步，愿意“走回走过的路”出房间
            if (
                wp is not None
                and exclude is not None
                and self._region_replan_count >= self._region_exclude_after_replans
            ):
                dist_wp_to_anchor = float(np.linalg.norm(wp[:2] - self._region_anchor_pos[:2]))
                if dist_wp_to_anchor < self._region_anchor_update_dist:
                    detour_wp = self.planner.get_waypoint_towards_goal(
                        drone_pos=pos,
                        goal_pos=self._home_pos[:2],
                        lookahead_dist=0.7,
                        goal_search_radius=1.2,
                    )
                    if detour_wp is not None:
                        wp = detour_wp
                    else:
                        home_xy = self._home_pos[:2]
                        pos_xy = pos[:2]
                        step_back = pos_xy + 0.35 * (home_xy - pos_xy)
                        wp = np.array(
                            [step_back[0], step_back[1], self.takeoff_height],
                            dtype=float,
                        )
                    self._log("EXPLORE: same region, detour toward home to exit")
            if wp is not None:
                if force_replan or float(np.linalg.norm(wp[:2] - self._current_waypoint[:2])) >= self.min_waypoint_delta:
                    self._current_waypoint = wp
                    # 仅当飞机位置已离开当前区域（相对锚点移动 > 1.5m）时更新锚点，避免在房间内振荡时不断重置锚点
                    if float(np.linalg.norm(pos[:2] - self._region_anchor_pos[:2])) > self._region_anchor_update_dist:
                        self._region_anchor_pos = np.asarray(pos, dtype=float).copy()
                        self._region_anchor_step = self._cur_step
                        self._region_replan_count = 0
                    elif float(np.linalg.norm(pos[:2] - self._region_anchor_pos[:2])) <= self._region_stuck_radius:
                        self._region_replan_count += 1
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

            # 探索完成且无“已发现未巡检”目标时，再检查是否还有未发现目标
            inspected2, discovered2, total2 = self.target_manager.get_progress()
            if total2 > 0 and discovered2 < total2:
                # 仍有未发现目标，继续探索而不是结束
                wp = self._pick_next_frontier(pos)
                if wp is not None:
                    self._current_waypoint = wp
                    self._last_frontier_pick_step = self._cur_step
                    self._phase = _Phase.EXPLORE
                    self._reset_stuck_anchor(pos)
                    self._log("→ EXPLORE (仍有未发现目标)")
                    return Command(target_pos=wp, target_rpy=rpy)
                hover = np.array([pos[0], pos[1], self.takeoff_height], dtype=float)
                self._phase = _Phase.EXPLORE
                self._log("→ EXPLORE (仍有未发现目标，悬停)")
                return Command(target_pos=hover, target_rpy=rpy)

            # 探索完成且无目标 → DONE
            # Safety fallback: do not finish while uninspected targets still exist.
            if total2 > 0 and inspected2 < total2:
                wp = self._pick_next_frontier(pos)
                if wp is not None:
                    self._current_waypoint = wp
                    self._last_frontier_pick_step = self._cur_step
                    self._phase = _Phase.EXPLORE
                    self._reset_stuck_anchor(pos)
                    self._log("EXPLORE (still have uninspected target)")
                    return Command(target_pos=wp, target_rpy=rpy)
                hover = np.array([pos[0], pos[1], self.takeoff_height], dtype=float)
                self._phase = _Phase.EXPLORE
                self._log("EXPLORE (uninspected target remains, hold)")
                return Command(target_pos=hover, target_rpy=rpy)

            self._phase = _Phase.DONE
            self._current_waypoint = self._home_pos.copy()
            self._log("→ DONE")
            return Command(target_pos=self._home_pos.copy(), target_rpy=rpy)

        # 悬停在目标上方，等待计时完成
        return Command(target_pos=tpos.copy(), target_rpy=rpy)

    def _handle_done(self, pos: np.ndarray, rpy: np.ndarray) -> Command:
        home = self._home_pos.copy()
        dist = float(np.linalg.norm(pos - home))
        finished = dist < 1.0
        if finished:
            self._log("DONE: returned home, mission complete")
            return Command(
                target_pos=home,
                target_rpy=rpy,
                finished=finished,
                info="mission_done",
            )

        # 返航卡住检测：有进展则更新锚点；超过约 4 秒无进展则缩短 lookahead
        if self._done_stuck_anchor_step < 0:
            self._done_stuck_anchor_dist = dist
            self._done_stuck_anchor_step = self._cur_step
        if dist < self._done_stuck_anchor_dist - 0.15:
            self._done_stuck_anchor_dist = dist
            self._done_stuck_anchor_step = self._cur_step
        stuck_steps = self._cur_step - self._done_stuck_anchor_step
        lookahead = 0.25 if stuck_steps > 80 else 0.45

        # 沿栅格 FREE 路径返航，避免直线穿墙；只选与飞机直线可达的路径点
        wp = self.planner.get_waypoint_towards_goal(
            pos,
            home,
            lookahead_dist=lookahead,
            goal_search_radius=0.8,
        )
        if wp is not None:
            wp = np.asarray(wp, dtype=float).reshape(-1)
            if wp.size >= 3:
                wp[2] = self.takeoff_height
            else:
                wp = np.array([float(wp[0]), float(wp[1]), self.takeoff_height], dtype=float)
            # 若路径点比当前位置更远离 home，说明路径绕远，直接飞 home 避免越飞越远
            dist_wp_to_home = float(np.linalg.norm(wp[:2] - home[:2]))
            if dist_wp_to_home > dist:
                wp = home.copy()
            return Command(
                target_pos=wp,
                target_rpy=rpy,
                finished=False,
                info="mission_done",
            )

        # 无路径时退化为直线（可能撞墙，仅当起飞点不在栅格内等情况）
        return Command(
            target_pos=home,
            target_rpy=rpy,
            finished=False,
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

    def _pick_next_frontier(
        self, pos: np.ndarray, exclude_xy: Optional[np.ndarray] = None
    ) -> Optional[np.ndarray]:
        """
        获取下一个安全航点：
        沿朝向 frontier 的方向逐步检查占据栅格，
        遇到 OCCUPIED 格才停下（UNKNOWN 格可穿越），返回最远安全点。
        exclude_xy: 卡住时传入当前航点，排除该方向上的 cluster，改选其他 frontier。
        """
        inspected, discovered, _ = self.target_manager.get_progress()
        if discovered == 0 and inspected == 0:
            biased = self._pick_rightward_frontier(pos)
            if biased is not None:
                return biased

        wp = self.planner.get_next_waypoint(pos, exclude_xy=exclude_xy)
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
            if self.planner.grid.grid[idx[0], idx[1]] != FREE:
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
        - 无已发现未巡检目标，但仍有未发现目标 → 不结束，保持 EXPLORE 悬停
        - 全部发现且无未巡检目标 → DONE，返回 Command
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

        # 无“已发现且未巡检”目标时，检查是否还有未发现的目标

            # Target is known but currently unreachable: do NOT finish mission.
            self._phase = _Phase.EXPLORE
            self._next_target_attempt_step = self._cur_step + self.target_retry_cooldown_steps
            self._current_target_id = tid
            self._current_target_pos = np.asarray(tpos, dtype=float).copy()

            guided_wp = self._pick_frontier_towards(pos, tpos)
            if guided_wp is None:
                guided_wp = self._pick_next_frontier(pos)
            if guided_wp is not None:
                self._current_waypoint = guided_wp
                self._last_frontier_pick_step = self._cur_step
                self._log(f"Exploration done, target {tid} unreachable -> EXPLORE toward target")
                return Command(target_pos=guided_wp, target_rpy=rpy)

            hold = np.array([pos[0], pos[1], self.takeoff_height], dtype=float)
            self._log(f"Exploration done, target {tid} unreachable and no frontier -> HOLD")
            return Command(target_pos=hold, target_rpy=rpy)

        inspected, discovered, total = self.target_manager.get_progress()
        if total > 0 and inspected < total:
            if discovered >= total:
                nearest_unvisited = self.target_manager.get_nearest_unvisited(pos)
                if nearest_unvisited is not None:
                    _, tpos, _ = nearest_unvisited
                    guided_wp = self._pick_frontier_towards(pos, tpos)
                    if guided_wp is not None:
                        self._current_waypoint = guided_wp
                        self._last_frontier_pick_step = self._cur_step
                        self._phase = _Phase.EXPLORE
                        self._log("Exploration done fallback: uninspected target remains -> EXPLORE")
                        return Command(target_pos=guided_wp, target_rpy=rpy)
                hold = np.array([pos[0], pos[1], self.takeoff_height], dtype=float)
                self._phase = _Phase.EXPLORE
                self._log("Exploration done fallback: uninspected target remains -> HOLD")
                return Command(target_pos=hold, target_rpy=rpy)
            # 仍有未发现目标：不悬停死等，向起飞点撤退一段，换位置后可能重新出现 frontier
            # 仅当当前不在撤退阶段时才发起新一轮撤退，避免每帧重置
            if self._region_retreat_until_step < 0 or self._cur_step >= self._region_retreat_until_step:
                retreat_wp = self.planner.get_waypoint_towards_goal(
                    drone_pos=pos,
                    goal_pos=self._home_pos[:2],
                    lookahead_dist=1.0,
                    goal_search_radius=1.2,
                )
                if retreat_wp is None:
                    home_xy = self._home_pos[:2]
                    pos_xy = pos[:2]
                    retreat_xy = pos_xy + 0.7 * (home_xy - pos_xy)
                    retreat_wp = np.array(
                        [retreat_xy[0], retreat_xy[1], self.takeoff_height],
                        dtype=float,
                    )
                self._current_waypoint = retreat_wp
                self._region_retreat_until_step = self._cur_step + self._region_retreat_duration_steps
                if self._cur_step - self._last_nofrontier_log_step >= 240:
                    self._last_nofrontier_log_step = self._cur_step
                    self._log("Exploration done, undiscovered targets remain → retreat toward home")
            self._phase = _Phase.EXPLORE
            return Command(target_pos=self._current_waypoint.copy(), target_rpy=rpy)

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

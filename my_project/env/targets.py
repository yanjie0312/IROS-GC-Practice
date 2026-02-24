# my_project/env/targets.py
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
import numpy as np
import pybullet as p


@dataclass
class TargetInfo:
    """单个目标的状态"""
    target_id: int
    position: np.ndarray          # 世界坐标 [x, y, z]
    discovered: bool = False      # 是否被发现过
    inspected: bool = False       # 是否巡检完成
    time_in_range: float = 0.0    # 在巡检范围内累计停留时间


class TargetManager:
    """
    管理目标的发现、巡检、进度追踪。

    使用方式：
        manager = TargetManager(arena_handle, pyb_client_id, inspect_range=0.5, inspect_time=2.0)
        ...
        # 每帧调用
        manager.update(pkt)
    """

    def __init__(
        self,
        arena_handle: dict,
        pyb_client_id: int,
        inspect_range: float = 0.5,
        inspect_time: float = 2.0,
    ):
        """
        Parameters
        ----------
        arena_handle : create_arena() 的返回值
        pyb_client_id : PyBullet 客户端 ID
        inspect_range : 巡检判定距离（m），无人机离目标小于此距离算"在巡检范围内"
        inspect_time : 巡检所需时间（s），在范围内累计停留超过此时间算"巡检完成"
        """
        self.client = int(pyb_client_id)
        self.inspect_range = float(inspect_range)
        self.inspect_time = float(inspect_time)

        self.targets: Dict[int, TargetInfo] = {}
        for tid in arena_handle.get("target_ids", []):
            pos, _ = p.getBasePositionAndOrientation(int(tid), physicsClientId=self.client)
            self.targets[int(tid)] = TargetInfo(
                target_id=int(tid),
                position=np.asarray(pos, dtype=float),
            )

    def update(self, pkt: dict) -> None:
        """
        每帧调用，根据传感器数据更新目标状态。

        Parameters
        ----------
        pkt : sensor.sense() 的返回值
        """
        dt = float(pkt.get("dt", 1.0 / 48.0))
        drone_pos = np.asarray(pkt["pos"], dtype=float)
        detected_ids: Set[int] = set(pkt.get("detected_target_ids", []))

        for tid, info in self.targets.items():
            if info.inspected:
                continue

            # 发现
            if tid in detected_ids and not info.discovered:
                info.discovered = True

            # 巡检：计算距离，在范围内累加时间
            dist = float(np.linalg.norm(drone_pos - info.position))
            if dist <= self.inspect_range:
                info.time_in_range += dt
                if info.time_in_range >= self.inspect_time:
                    info.inspected = True
            else:
                # 离开范围，重置累计时间
                info.time_in_range = 0.0

    def is_all_done(self) -> bool:
        """所有目标都巡检完了吗"""
        if not self.targets:
            return True
        return all(info.inspected for info in self.targets.values())

    def get_progress(self) -> Tuple[int, int, int]:
        """
        返回 (已巡检数, 已发现数, 总数)
        """
        total = len(self.targets)
        discovered = sum(1 for info in self.targets.values() if info.discovered)
        inspected = sum(1 for info in self.targets.values() if info.inspected)
        return inspected, discovered, total

    def get_nearest_unvisited(self, drone_pos: np.ndarray) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        返回最近的未巡检目标。

        Returns
        -------
        (target_id, position, distance) 或 None（全部巡检完）
        """
        drone_pos = np.asarray(drone_pos, dtype=float)
        best: Optional[Tuple[int, np.ndarray, float]] = None

        for tid, info in self.targets.items():
            if info.inspected:
                continue
            dist = float(np.linalg.norm(drone_pos - info.position))
            if best is None or dist < best[2]:
                best = (tid, info.position.copy(), dist)

        return best

    def get_all_status(self) -> List[dict]:
        """返回所有目标的状态，用于调试打印"""
        result = []
        for tid, info in self.targets.items():
            result.append({
                "id": tid,
                "pos": info.position.tolist(),
                "discovered": info.discovered,
                "inspected": info.inspected,
                "time_in_range": round(info.time_in_range, 2),
            })
        return result

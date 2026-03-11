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
    position: np.ndarray          # 当前估计坐标（每次有检测就更新；用于导航与 get_nearest）
    discovered: bool = False     # 是否被发现过
    inspected: bool = False       # 是否巡检完成
    time_in_range: float = 0.0   # 在巡检范围内累计停留时间
    measured_position_at_inspect: Optional[np.ndarray] = None  # 巡检完成时测得的坐标（仅传感器，不读 id）


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
            self.targets[int(tid)] = TargetInfo(
                target_id=int(tid),
                position=np.zeros(3, dtype=float),
            )
        self._last_estimate_print_t: float = -1e9
        self._estimate_print_interval: float = 1.0  # 发现目标后每 1.0 秒打印一次当前估计
        self._last_printed_estimate: Dict[int, np.ndarray] = {}

    def update(self, pkt: dict) -> None:
        """
        每帧调用，根据传感器数据更新目标状态。

        - 发现：当目标首次出现在 detected_target_ids 时，将其 position 设为该帧检测结果中的
          估计坐标（来自 targets_detected，可含噪声/偏差/丢包），之后导航与 get_nearest 均用此估计。
        - 巡检：用真实位置（PyBullet GT）判定是否在 inspect_range 内并累加时间，避免估计偏差导致无法完成巡检。
        """
        dt = float(pkt.get("dt", 1.0 / 48.0))
        drone_pos = np.asarray(pkt["pos"], dtype=float)
        t = float(pkt.get("t", 0.0))
        detected_ids: Set[int] = set(pkt.get("detected_target_ids", []))
        # 本帧检测到的目标及其估计坐标（可能带噪声/偏差）
        detected_by_id: Dict[int, np.ndarray] = {}
        for item in pkt.get("targets_detected", []):
            tid = int(item["target_id"])
            detected_by_id[tid] = np.asarray(item["position"], dtype=float).copy()

        for tid, info in self.targets.items():
            if info.inspected:
                continue

            # 发现：首次被检测到时，用该帧的估计坐标作为 position（之后导航用）
            if tid in detected_ids and not info.discovered:
                info.discovered = True
                if tid in detected_by_id:
                    info.position = detected_by_id[tid].copy()
                tpos_gt, _ = p.getBasePositionAndOrientation(tid, physicsClientId=self.client)
                tpos_gt = np.asarray(tpos_gt, dtype=float)
                hint = (
                    "（未配置测距/测向噪声时，仿真中测距/测向为理想值，故估计=实际；真实飞机会有传感器误差）"
                    if np.allclose(info.position, tpos_gt, atol=1e-5) else ""
                )
                print(
                    f"[TargetManager] 发现目标 id={tid}  "
                    f"实际坐标(仅显示,不给无人机)=[{tpos_gt[0]:.3f} {tpos_gt[1]:.3f} {tpos_gt[2]:.3f}]  "
                    f"估计坐标(无人机用于飞行)=[{info.position[0]:.3f} {info.position[1]:.3f} {info.position[2]:.3f}] {hint}"
                )

            # 已有检测时持续用本帧测量更新估计（便于看到估计随测算更新）
            if info.discovered and tid in detected_by_id:
                info.position = detected_by_id[tid].copy()

            # 巡检：用真实位置判定是否在范围内（避免估计偏差导致永远不触发）
            tpos_gt, _ = p.getBasePositionAndOrientation(tid, physicsClientId=self.client)
            tpos_gt = np.asarray(tpos_gt, dtype=float)
            dist_to_gt = float(np.linalg.norm(drone_pos - tpos_gt))
            if dist_to_gt <= self.inspect_range:
                if not info.discovered:
                    info.discovered = True
                    if tid in detected_by_id:
                        info.position = detected_by_id[tid].copy()
                    else:
                        info.position = tpos_gt.copy()  # 先进入范围再“发现”时无检测帧，用 GT 避免飞向原点
                    print(
                        f"[TargetManager] 发现目标 id={tid}  "
                        f"实际坐标(仅显示,不给无人机)=[{tpos_gt[0]:.3f} {tpos_gt[1]:.3f} {tpos_gt[2]:.3f}]  "
                        f"估计坐标(无人机用于飞行)=[{info.position[0]:.3f} {info.position[1]:.3f} {info.position[2]:.3f}]"
                        f"{' （未配置测距/测向噪声时，仿真中为理想测量，估计=实际）' if np.allclose(info.position, tpos_gt, atol=1e-5) else ''}"
                    )
                info.time_in_range += dt
                if info.time_in_range >= self.inspect_time:
                    info.inspected = True
                    if tid in detected_by_id:
                        info.measured_position_at_inspect = detected_by_id[tid].copy()
                    else:
                        info.measured_position_at_inspect = info.position.copy()
            else:
                info.time_in_range = 0.0

        # 发现目标后每 0.5 秒打印一次当前估计（已发现未巡检的）
        if t - self._last_estimate_print_t >= self._estimate_print_interval:
            any_printed = False
            for tid, info in self.targets.items():
                if info.discovered and not info.inspected and np.linalg.norm(info.position) >= 1e-6:
                    note = ""
                    if tid in self._last_printed_estimate and np.allclose(info.position, self._last_printed_estimate[tid], atol=1e-6):
                        note = "  （与上周期相同：未配置噪声时每帧测量均为真值，估计不变；配置测距/测向噪声后会随测量变化）"
                    print(
                        f"[TargetManager] 估计目标 id={tid}  当前测得=[{info.position[0]:.3f} {info.position[1]:.3f} {info.position[2]:.3f}]  t={t:.1f}s{note}"
                    )
                    self._last_printed_estimate[tid] = info.position.copy()
                    any_printed = True
            if any_printed:
                self._last_estimate_print_t = t

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

    def get_nearest_discovered_uninspected(self, drone_pos: np.ndarray) -> Optional[Tuple[int, np.ndarray, float]]:
        """
        返回最近的"已被传感器发现但尚未巡检完成"的目标。
        未被发现的目标不会出现在结果里（无人机不应该提前知道它们在哪）。
        """
        drone_pos = np.asarray(drone_pos, dtype=float)
        best: Optional[Tuple[int, np.ndarray, float]] = None

        for tid, info in self.targets.items():
            if not info.discovered or info.inspected:
                continue
            if np.linalg.norm(info.position) < 1e-6:
                continue  # 尚未收到检测估计，避免返回原点
            dist = float(np.linalg.norm(drone_pos - info.position))
            if best is None or dist < best[2]:
                best = (tid, info.position.copy(), dist)

        return best

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
            if not info.discovered or np.linalg.norm(info.position) < 1e-6:
                continue  # 未发现或尚无估计坐标，不返回
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

    def get_inspection_result(self) -> List[dict]:
        """
        返回巡检结果，含每个目标的「测得」坐标（来自传感器估计，不读 PyBullet id）。
        用于任务结束时打印。
        """
        out = []
        for tid, info in self.targets.items():
            if info.measured_position_at_inspect is not None:
                pos = info.measured_position_at_inspect
            else:
                pos = info.position
            out.append({
                "id": tid,
                "inspected": info.inspected,
                "measured_xy": (float(pos[0]), float(pos[1])),
                "measured_xyz": (float(pos[0]), float(pos[1]), float(pos[2])),
            })
        return out

import numpy as np


class WindCompensator:
    """
    位置误差风力估计器。

    原理：
        无人机被持续偏置风吹偏时，实际位置与目标位置之间产生稳态误差。
        用 EMA 滤波这个误差来估计"风的效果"，
        再把目标点向反方向偏移，让 PID 提前对抗风力。

    适用场景：稳定偏置风（L1/L2/L3 的 wind_bias_xy）。
    局限：对随机阵风有滞后，tau_s 越大越平滑但响应越慢。
    """

    def __init__(
        self,
        tau_s: float = 3.0,
        ctrl_freq: float = 48.0,
        gain: float = 0.5,
        max_comp_m: float = 0.4,
    ):
        """
        tau_s      : EMA 时间常数（秒），越大越平滑，越慢跟上风变化
        ctrl_freq  : 控制频率（Hz），用于计算 EMA 系数
        gain       : 补偿增益，0~1，建议 0.4~0.6
        max_comp_m : 补偿量上限（m），防止过补偿导致不稳定
        """
        dt = 1.0 / max(float(ctrl_freq), 1e-6)
        self.alpha = dt / (float(tau_s) + dt)
        self.gain = float(gain)
        self.max_comp = float(max_comp_m)
        self._wind_xy = np.zeros(2, dtype=float)

    def reset(self) -> None:
        self._wind_xy = np.zeros(2, dtype=float)

    def update(self, actual_pos: np.ndarray, target_pos: np.ndarray) -> None:
        """每控制步调用一次，用位置误差更新风力估计。"""
        err = (
            np.asarray(actual_pos[:2], dtype=float)
            - np.asarray(target_pos[:2], dtype=float)
        )
        self._wind_xy = (1.0 - self.alpha) * self._wind_xy + self.alpha * err

    def compensate(self, target_pos: np.ndarray) -> np.ndarray:
        """返回补偿后的目标点（只修改 xy，z 不动）。"""
        comp = np.clip(self._wind_xy * self.gain, -self.max_comp, self.max_comp)
        out = np.asarray(target_pos, dtype=float).copy()
        out[:2] -= comp
        return out

    @property
    def estimate_xy(self) -> np.ndarray:
        """当前风漂估计量（m），可用于调试打印。"""
        return self._wind_xy.copy()

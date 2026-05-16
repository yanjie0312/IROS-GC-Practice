import numpy as np


class WindCompensator:
    """
    双信号风力补偿器：位置误差 EMA（慢）+ 速度误差 EMA（快）。

    位置误差 EMA：
        actual_pos - target_pos 的慢速滤波，估计稳态风漂偏置。
        tau 越大越平滑，越慢收敛，适合恒定风偏。

    速度误差 EMA（vel_gain > 0 时启用）：
        "意外速度" = actual_vel - expected_vel
        expected_vel = (target_pos - actual_pos) * speed_scale
        当无人机悬停时 expected_vel≈0，意外速度 ≈ 实际速度（风导致）→ 正确补偿。
        当无人机主动飞行时 expected_vel 较大，意外速度≈0 → 不干扰导航。
        tau 越小响应越快，适合抵消阵风。

    L0（无风）：vel_gain=0，退化为原始位置误差补偿，与原版行为完全相同。
    L1+（有风）：vel_gain>0，双信号联合，兼顾稳态风和阵风。
    """

    def __init__(
        self,
        tau_s: float = 3.0,
        ctrl_freq: float = 48.0,
        gain: float = 0.5,
        max_comp_m: float = 0.4,
        vel_gain: float = 0.0,
        vel_tau_s: float = 0.5,
        speed_scale: float = 0.5,
    ):
        """
        tau_s       : 位置误差 EMA 时间常数（s）
        ctrl_freq   : 控制频率（Hz）
        gain        : 位置误差补偿增益
        max_comp_m  : 总补偿上限（m）
        vel_gain    : 速度误差补偿增益，0=禁用（L0）
        vel_tau_s   : 速度误差 EMA 时间常数（s），越小越灵敏
        speed_scale : 期望速度估算系数，调节"主动飞行"的剔除程度
        """
        dt = 1.0 / max(float(ctrl_freq), 1e-6)
        self.alpha      = dt / (float(tau_s)     + dt)
        self.vel_alpha  = dt / (float(vel_tau_s) + dt)
        self.gain       = float(gain)
        self.vel_gain   = float(vel_gain)
        self.max_comp   = float(max_comp_m)
        self.speed_scale = float(speed_scale)
        self._wind_xy   = np.zeros(2, dtype=float)  # 位置误差估计
        self._vel_err_xy = np.zeros(2, dtype=float) # 速度误差估计

    def reset(self) -> None:
        self._wind_xy    = np.zeros(2, dtype=float)
        self._vel_err_xy = np.zeros(2, dtype=float)

    def update(
        self,
        actual_pos: np.ndarray,
        target_pos: np.ndarray,
        actual_vel: np.ndarray | None = None,
    ) -> None:
        """每控制步调用一次，更新风力估计。"""
        pos = np.asarray(actual_pos[:2], dtype=float)
        tgt = np.asarray(target_pos[:2], dtype=float)

        # 位置误差 EMA（慢速，捕捉稳态风漂）
        err = pos - tgt
        self._wind_xy = (1.0 - self.alpha) * self._wind_xy + self.alpha * err

        # 速度误差 EMA（快速，捕捉阵风），仅 vel_gain > 0 时启用
        if self.vel_gain > 0.0 and actual_vel is not None:
            vel = np.asarray(actual_vel[:2], dtype=float)
            # 期望速度：无人机主动飞向目标时应该有的速度方向和量级
            delta = tgt - pos
            dist = float(np.linalg.norm(delta))
            if dist > 1e-6:
                expected_vel = delta / dist * min(dist * self.speed_scale, 1.0)
            else:
                expected_vel = np.zeros(2, dtype=float)
            # 意外速度 = 实际速度 - 期望速度（剩余部分是风造成的）
            vel_err = vel - expected_vel
            self._vel_err_xy = (
                (1.0 - self.vel_alpha) * self._vel_err_xy
                + self.vel_alpha * vel_err
            )

    def compensate(self, target_pos: np.ndarray) -> np.ndarray:
        """返回补偿后的目标点（只修改 xy，z 不动）。"""
        comp = self._wind_xy * self.gain
        if self.vel_gain > 0.0:
            comp = comp + self._vel_err_xy * self.vel_gain
        comp = np.clip(comp, -self.max_comp, self.max_comp)
        out = np.asarray(target_pos, dtype=float).copy()
        out[:2] -= comp
        return out

    @property
    def estimate_xy(self) -> np.ndarray:
        """当前风漂估计量（m），可用于调试打印。"""
        return self._wind_xy.copy()

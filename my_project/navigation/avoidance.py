# my_project/navigation/avoidance.py
import numpy as np


class AvoidanceLayer:
    """
    势场法避障层：
    - 上层 mission 给出一个全局 target_pos（世界坐标）
    - 本层根据射线信息对 target_pos 做局部修正，推离近距离障碍
    """

    def __init__(
        self,
        k_att: float = 1.0,
        k_rep: float = 0.8,
        d0: float = 1.0,
        step_len: float = 0.7,
        alpha: float = 0.5,
        min_dist_emergency: float = 0.15,
        keep_z: bool = True,
        horizontal_only: bool = True,
    ):
        """
        k_att: 吸引力系数
        k_rep: 排斥力系数
        d0: 产生排斥力的距离阈值（m）
        step_len: 从当前位置沿合力方向“看向前方”的步长（m）
        alpha: 与原始 target_pos 的插值权重，越大越偏向避障修正方向
        min_dist_emergency: 小于该距离认为是紧急情况，排斥力会更强
        keep_z: 是否保持 target 的高度不被避障层改动（推荐 True）
        horizontal_only: 是否只在水平面（xy）做避障（推荐 True）
        """
        self.k_att = float(k_att)
        self.k_rep = float(k_rep)
        self.d0 = float(d0)
        self.step_len = float(step_len)
        self.alpha = float(alpha)
        self.min_dist_emergency = float(min_dist_emergency)
        self.keep_z = bool(keep_z)
        self.horizontal_only = bool(horizontal_only)

    def reset(self):
        # 当前实现没有内部状态，预留接口
        pass

    def filter_target(self, state, sensors: dict, target_pos: np.ndarray) -> np.ndarray:
        """
        根据传感器信息对 target_pos 做局部避障修正。

        sensors 期望至少包含:
            - "ray_dists": ndarray[N]
            - "ray_dirs_world": ndarray[N,3]  (推荐)
              或 "ray_dirs_body": ndarray[N,3]/[N,2] (兜底)
            - "min_dist": float
            - "collision": bool

        返回：修正后的 target_pos（世界坐标）
        """
        if state is None or sensors is None or target_pos is None:
            return target_pos

        pos = np.asarray(state.xyz, dtype=float).reshape(3)
        target_pos = np.asarray(target_pos, dtype=float).reshape(3)

        # ---------------- 吸引力（指向原始 target_pos） ----------------
        dir_to_target = target_pos - pos
        dist_to_target = float(np.linalg.norm(dir_to_target))
        if dist_to_target < 1e-9:
            return target_pos

        f_att = self.k_att * dir_to_target / max(dist_to_target, 1e-9)

        # ---------------- 排斥力（由近距离障碍产生） ----------------
        ray_dists = sensors.get("ray_dists", None)
        ray_dirs = sensors.get("ray_dirs_world", None)  # ✅ 优先 world
        if ray_dirs is None:
            ray_dirs = sensors.get("ray_dirs_body", None)  # 兜底（老格式）

        min_dist = float(sensors.get("min_dist", np.inf))
        collision = bool(sensors.get("collision", False))

        if ray_dists is None or ray_dirs is None:
            # 没有射线信息就不做修正
            return target_pos

        ray_dists = np.asarray(ray_dists, dtype=float).reshape(-1)
        ray_dirs = np.asarray(ray_dirs, dtype=float)

        # 兼容 [N,2] 的射线方向，补成 [N,3]
        if ray_dirs.ndim == 2 and ray_dirs.shape[1] == 2:
            zeros = np.zeros((ray_dirs.shape[0], 1), dtype=float)
            ray_dirs = np.concatenate([ray_dirs, zeros], axis=1)

        # 保底形状检查
        if ray_dirs.ndim != 2 or ray_dirs.shape[1] != 3:
            return target_pos

        f_rep = np.zeros(3, dtype=float)

        # 紧急情况时放大排斥力
        rep_scale = 3.0 if (collision or min_dist < self.min_dist_emergency) else 1.0

        for dist, d in zip(ray_dists, ray_dirs):
            if not np.isfinite(dist) or dist <= 0.0:
                continue
            if dist >= self.d0:
                continue

            # 单位方向
            norm_dir = float(np.linalg.norm(d))
            if norm_dir < 1e-9:
                continue
            dir_vec = d / norm_dir

            # 只做水平面避障：避免 tilt rays 影响高度
            if self.horizontal_only:
                dir_vec = dir_vec.copy()
                dir_vec[2] = 0.0
                n2 = float(np.linalg.norm(dir_vec))
                if n2 < 1e-9:
                    continue
                dir_vec /= n2

            # 势场排斥力（经典形式）
            inv_d = 1.0 / max(float(dist), 1e-6)
            inv_d0 = 1.0 / max(self.d0, 1e-6)
            strength = self.k_rep * (inv_d - inv_d0) * (inv_d ** 2)

            # 障碍在 dir_vec 方向上，所以排斥力是 -dir_vec
            f_rep += rep_scale * strength * (-dir_vec)

        # ---------------- 合力 -> 生成局部安全点 ----------------
        f_total = f_att + f_rep

        if self.horizontal_only:
            f_total = f_total.copy()
            f_total[2] = 0.0

        norm_total = float(np.linalg.norm(f_total))
        if norm_total < 1e-9:
            return target_pos

        dir_total = f_total / norm_total
        target_safe = pos + self.step_len * dir_total

        # 与原始 target_pos 插值（保留全局意图 + 局部避障）
        target_final = (1.0 - self.alpha) * target_pos + self.alpha * target_safe

        # 保持高度不变（推荐）
        if self.keep_z:
            target_final = target_final.copy()
            target_final[2] = target_pos[2]

        return target_final
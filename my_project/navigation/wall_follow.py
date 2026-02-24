# my_project/navigation/wall_follow.py
import numpy as np
from .base import BaseMission, State, Command
from my_project.utils.geometry import clamp, dist2

class WallFollowMission(BaseMission):
    """
    直飞 -> 撞墙 -> 顺时针绕一圈 -> 回原点 -> finish
    """
    # 状态定义
    GO_STRAIGHT = 0
    WALL_FOLLOW = 1
    RETURN_HOME = 2
    DONE = 3

    # 初始化
    def __init__(self, arena_half_size: float, flight_height: float, speed: float = 0.35, margin: float = 0.12):
        self.ARENA = arena_half_size
        self.H = flight_height
        self.V = speed
        self.margin = margin

        # 初始状态
        self.state = self.GO_STRAIGHT
        self.target_xy = np.array([0.0, 0.0], dtype=float)

        # 顺时针角点序列
        c = self.ARENA - self.margin
        self.corners = [
            np.array([ c, -c], dtype=float),
            np.array([ c,  c], dtype=float),
            np.array([-c,  c], dtype=float),
            np.array([-c, -c], dtype=float),
            np.array([ c, -c], dtype=float),
        ]
        self.corner_idx = 0
        self.lap_completed = False

        # 直飞方向：+X
        self.vel_dir = np.array([1.0, 0.0], dtype=float)
        self.vel_dir = self.vel_dir / (np.linalg.norm(self.vel_dir) + 1e-9)

    # 重置
    def reset(self, state: State):
        self.state = self.GO_STRAIGHT
        self.target_xy = np.array([0.0, 0.0], dtype=float)
        self.corner_idx = 0
        self.lap_completed = False

    # 更新
    def update(self, state: State, sensors: dict) -> Command:
        dt = sensors["dt"]
        hit_wall = sensors.get("hit_wall", False)

        cur_xy = state.xyz[:2]

        # GO_STRAIGHT 状态下，直飞前进
        if self.state == self.GO_STRAIGHT:
            self.target_xy = self.target_xy + self.vel_dir * self.V * dt
            self.target_xy[0] = clamp(self.target_xy[0], -self.ARENA + self.margin, self.ARENA - self.margin)
            self.target_xy[1] = clamp(self.target_xy[1], -self.ARENA + self.margin, self.ARENA - self.margin)

            if hit_wall:
                self.state = self.WALL_FOLLOW
                # 先用“最近角点”起步（你若想更稳，后面可改为“按墙面决定起点”）
                best_k, best_d = 0, 1e9
                for k in range(4):
                    d = dist2(cur_xy, self.corners[k])
                    if d < best_d:
                        best_d, best_k = d, k
                self.corner_idx = best_k
                self.lap_completed = False

        # WALL_FOLLOW 状态下，顺时针绕墙飞行
        elif self.state == self.WALL_FOLLOW:
            wp = self.corners[self.corner_idx]
            to_wp = wp - self.target_xy
            d = np.linalg.norm(to_wp)
            if d > 1e-9:
                self.target_xy = self.target_xy + (to_wp / d) * self.V * dt

            # 速度快时可以把 0.05 改大些
            if dist2(self.target_xy, wp) < (0.05 * 0.05):
                self.corner_idx = min(self.corner_idx + 1, len(self.corners) - 1)

            if self.corner_idx == len(self.corners) - 1 and dist2(self.target_xy, self.corners[-1]) < (0.08 * 0.08):
                self.lap_completed = True

            if self.lap_completed:
                self.state = self.RETURN_HOME

        # RETURN_HOME 状态下，飞回原点
        elif self.state == self.RETURN_HOME:
            home = np.array([0.0, 0.0], dtype=float)
            to_home = home - self.target_xy
            d = np.linalg.norm(to_home)
            if d > 1e-9:
                self.target_xy = self.target_xy + (to_home / d) * self.V * dt

            if dist2(self.target_xy, home) < (0.05 * 0.05) and dist2(cur_xy, home) < (0.08 * 0.08):
                self.state = self.DONE

        # DONE 状态下，悬停在原点上方
        if self.state == self.DONE:
            return Command(
                target_pos=np.array([0.0, 0.0, self.H], dtype=float),
                target_rpy=np.array([0.0, 0.0, 0.0], dtype=float),
                finished=True,
                info="FINISH"
            )

        # 返回当前目标位置指令
        return Command(
            target_pos=np.array([self.target_xy[0], self.target_xy[1], self.H], dtype=float),
            target_rpy=np.array([0.0, 0.0, 0.0], dtype=float),
            finished=False
        )

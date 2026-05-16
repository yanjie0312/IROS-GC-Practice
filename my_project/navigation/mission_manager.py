# my_project/navigation/mission_manager.py
import numpy as np
from .base import Command


class MissionManager:
    def __init__(self, mission, avoidance_layer=None, avoidance_ema: float = 0.5):
        self.mission = mission
        self.avoidance = avoidance_layer
        self.avoidance_ema = float(np.clip(avoidance_ema, 0.0, 0.95))
        self._smooth_target = None

    def reset(self, state):
        self.mission.reset(state)
        if self.avoidance:
            self.avoidance.reset()
        self._smooth_target = None

    def update(self, state, sensors) -> Command:
        cmd = self.mission.update(state, sensors)
        if self.avoidance and not cmd.finished and cmd.info != "inspect":
            raw = self.avoidance.filter_target(state, sensors, cmd.target_pos)
            if (
                self._smooth_target is None
                or float(np.linalg.norm(raw - self._smooth_target)) > 1.5
            ):
                # 目标大幅跳变时重置，保持响应速度
                self._smooth_target = raw.copy()
            else:
                a = self.avoidance_ema
                self._smooth_target = a * self._smooth_target + (1.0 - a) * raw
            cmd.target_pos = self._smooth_target.copy()
        else:
            self._smooth_target = None
        return cmd

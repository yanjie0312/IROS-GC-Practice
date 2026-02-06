# my_project/navigation/mission_manager.py
from .base import Command

class MissionManager:
    def __init__(self, mission, avoidance_layer=None):
        self.mission = mission
        self.avoidance = avoidance_layer

    def reset(self, state):
        self.mission.reset(state)
        if self.avoidance:
            self.avoidance.reset()

    def update(self, state, sensors) -> Command:
        cmd = self.mission.update(state, sensors)
        if self.avoidance and not cmd.finished:
            cmd.target_pos = self.avoidance.filter_target(state, sensors, cmd.target_pos)
        return cmd

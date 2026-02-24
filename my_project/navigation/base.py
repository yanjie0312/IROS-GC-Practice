# my_project/navigation/base.py
from dataclasses import dataclass
import numpy as np

@dataclass
class State:
    xyz: np.ndarray   # [x,y,z]
    vel: np.ndarray   # [vx,vy,vz] (可先不用)
    step: int
    t: float

@dataclass
class Command:
    target_pos: np.ndarray   # [x,y,z]
    target_rpy: np.ndarray   # [r,p,y]
    finished: bool = False
    info: str = ""

class BaseMission:
    def reset(self, state: State):
        pass

    def update(self, state: State, sensors: dict) -> Command:
        raise NotImplementedError

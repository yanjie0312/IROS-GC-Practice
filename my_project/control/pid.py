# my_project/control/pid.py
import numpy as np
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl

class PIDController:
    def __init__(self, drone_model):
        self.ctrl = DSLPIDControl(drone_model=drone_model)

    def compute(self, control_timestep, state, target_pos, target_rpy):
        action, _, _ = self.ctrl.computeControlFromState(
            control_timestep=control_timestep,
            state=state,
            target_pos=np.asarray(target_pos, dtype=float),
            target_rpy=np.asarray(target_rpy, dtype=float),
        )
        return action

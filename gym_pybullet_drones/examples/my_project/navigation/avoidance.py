# my_project/navigation/avoidance.py
import numpy as np

class AvoidanceLayer:
    """
    避障层：输入 mission 给的 target_pos，再根据传感器修正成安全 target_pos。
    先做空壳：现在不改 target_pos，后续你加 ray lidar / 最近距离后再实现。
    """
    def __init__(self):
        pass

    def reset(self):
        pass

    def filter_target(self, state, sensors: dict, target_pos: np.ndarray) -> np.ndarray:
        # TODO: 下一任务在这里实现避障规则
        return target_pos

import numpy as np

from my_project.navigation.avoidance import AvoidanceLayer
from my_project.navigation.base import State


def run_case(name: str, sensors: dict, target_pos: np.ndarray):
    layer = AvoidanceLayer(
        k_att=1.0,
        k_rep=0.8,
        d0=1.0,
        step_len=0.7,
        alpha=0.5,
        keep_z=True,
        horizontal_only=True,
    )

    state = State(
        xyz=np.array([0.0, 0.0, 0.5]),
        vel=np.zeros(3),
        step=0,
        t=0.0,
    )

    target_filtered = layer.filter_target(state, sensors, target_pos)

    print("\n==============================")
    print("CASE:", name)
    print("原始 target_pos:", target_pos)
    print("避障后 target_pos:", target_filtered)

    dist_before = np.linalg.norm(target_pos[:2] - state.xyz[:2])
    dist_after = np.linalg.norm(target_filtered[:2] - state.xyz[:2])
    print("水平距离: before =", dist_before, ", after =", dist_after)

    # 一些简单的正确性检查（不是严格 assert，先用 print）
    print("检查1：高度是否保持不变 (z):", target_filtered[2], "(应当=)", target_pos[2])
    print("检查2：避障后是否不再“穿墙方向”继续前进（至少 x/y 方向要被削弱/偏转）")


def main():
    # ---------------- CASE 1：老格式（只有 ray_dirs_body） ----------------
    # 一根射线：机体“前方” 0.5m 有墙
    sensors_body_only = {
        "ray_dists": np.array([0.5]),
        "ray_dirs_body": np.array([[1.0, 0.0, 0.0]]),
        "min_dist": 0.5,
        "collision": False,
    }
    target_pos = np.array([1.0, 0.0, 0.5])  # 想往 +x 飞
    run_case("Body-only (fallback)", sensors_body_only, target_pos)

    # ---------------- CASE 2：新格式（优先 ray_dirs_world） ----------------
    # 模拟：无人机 yaw=90°，机体前方(body x) 实际指向世界 +y
    # 墙在世界 +y 方向 0.5m
    sensors_world = {
        "ray_dists": np.array([0.5]),
        # body前方
        "ray_dirs_body": np.array([[1.0, 0.0, 0.0]]),
        # ✅ world前方其实是 +y
        "ray_dirs_world": np.array([[0.0, 1.0, 0.0]]),
        "min_dist": 0.5,
        "collision": False,
    }
    # 目标也是往世界 +y 飞（正好在墙后）
    target_pos2 = np.array([0.0, 1.0, 0.5])
    run_case("World-aware (yaw=90deg example)", sensors_world, target_pos2)


if __name__ == "__main__":
    main()
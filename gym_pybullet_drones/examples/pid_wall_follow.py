"""
pid_wall_follow.py
==================

实现目标（按你的要求）：
1) 环境里只有四面墙（正方形、原点中心对称），不加载其他障碍物
2) 无人机从 (0,0,H) 出发
3) 水平恒定速度直飞（默认朝 +X）
4) 真正“撞到墙”（PyBullet contact检测）后，沿墙顺时针绕一圈（走矩形角点）
5) 绕完一圈后回到原点并悬停
6) 每 0.5 秒输出一次当前坐标和状态

运行：
    python pid_wall_follow.py --duration_sec 60

说明：
- 这里使用的是位置 PID（DSLPIDControl）。为了实现“恒定速度飞”，
  我们不是直接控制速度，而是让“目标位置 target_xy”以恒定速度移动。
  PID 会追踪这个移动目标，于是无人机看起来就是匀速飞。
"""

import time
import argparse
import numpy as np
import pybullet as p

from gym_pybullet_drones.utils.enums import DroneModel, Physics
from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.control.DSLPIDControl import DSLPIDControl
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool


# -----------------------
# 默认参数
# -----------------------
DEFAULT_DRONES = DroneModel("cf2x")
DEFAULT_NUM_DRONES = 1
DEFAULT_PHYSICS = Physics("pyb")
DEFAULT_GUI = True
DEFAULT_RECORD_VISION = False
DEFAULT_PLOT = True
DEFAULT_USER_DEBUG_GUI = False

# 关键：禁用环境自带障碍物
DEFAULT_OBSTACLES = False

DEFAULT_SIMULATION_FREQ_HZ = 240
DEFAULT_CONTROL_FREQ_HZ = 48
DEFAULT_DURATION_SEC = 60
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False


# -----------------------
# 工具函数
# -----------------------
def clamp(v, lo, hi):
    """把 v 限制在 [lo, hi]"""
    return max(lo, min(hi, v))


def dist2(xy, target_xy):
    """二维距离平方（更快，不用开方）"""
    dx = xy[0] - target_xy[0]
    dy = xy[1] - target_xy[1]
    return dx * dx + dy * dy


def create_four_walls(pyb_client_id: int,
                      half_size: float = 0.6,
                      wall_thickness: float = 0.05,
                      wall_height: float = 0.8):
    """
    创建四面墙围成一个正方形，中心在原点。

    内部可飞区域约为：x,y in [-half_size, +half_size]
    墙厚 wall_thickness、墙高 wall_height。

    返回：
    - wall_ids：四面墙的 body id 列表，用于之后做碰撞检测
    """
    walls = []
    wall_z_center = wall_height / 2.0

    def _create_box(pos, half_extents):
        """创建静态盒子（墙）"""
        col = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=pyb_client_id
        )
        vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=[0.8, 0.8, 0.8, 1.0],
            physicsClientId=pyb_client_id
        )
        bid = p.createMultiBody(
            baseMass=0,  # 0 表示静态物体
            baseCollisionShapeIndex=col,
            baseVisualShapeIndex=vis,
            basePosition=pos,
            physicsClientId=pyb_client_id
        )
        return bid

    # +X 墙：放在 x = +(half_size + wall_thickness)
    walls.append(_create_box(
        pos=[half_size + wall_thickness, 0.0, wall_z_center],
        half_extents=[wall_thickness, half_size + wall_thickness, wall_height / 2.0]
    ))

    # -X 墙
    walls.append(_create_box(
        pos=[-half_size - wall_thickness, 0.0, wall_z_center],
        half_extents=[wall_thickness, half_size + wall_thickness, wall_height / 2.0]
    ))

    # +Y 墙
    walls.append(_create_box(
        pos=[0.0, half_size + wall_thickness, wall_z_center],
        half_extents=[half_size + wall_thickness, wall_thickness, wall_height / 2.0]
    ))

    # -Y 墙
    walls.append(_create_box(
        pos=[0.0, -half_size - wall_thickness, wall_z_center],
        half_extents=[half_size + wall_thickness, wall_thickness, wall_height / 2.0]
    ))

    return walls


# -----------------------
# 主逻辑
# -----------------------
def run(
        drone=DEFAULT_DRONES,
        num_drones=DEFAULT_NUM_DRONES,
        physics=DEFAULT_PHYSICS,
        gui=DEFAULT_GUI,
        record_video=DEFAULT_RECORD_VISION,
        plot=DEFAULT_PLOT,
        user_debug_gui=DEFAULT_USER_DEBUG_GUI,
        obstacles=DEFAULT_OBSTACLES,
        simulation_freq_hz=DEFAULT_SIMULATION_FREQ_HZ,
        control_freq_hz=DEFAULT_CONTROL_FREQ_HZ,
        duration_sec=DEFAULT_DURATION_SEC,
        output_folder=DEFAULT_OUTPUT_FOLDER,
        colab=DEFAULT_COLAB
):
    # -----------------------
    # 1) 无人机初始位置：从原点起飞
    # -----------------------
    H = 0.3
    INIT_XYZS = np.array([[0.0, 0.0, H] for _ in range(num_drones)])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(num_drones)])

    # -----------------------
    # 2) 创建环境（禁用默认障碍物）
    # -----------------------
    env = CtrlAviary(
        drone_model=drone,
        num_drones=num_drones,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=physics,
        neighbourhood_radius=10,
        pyb_freq=simulation_freq_hz,
        ctrl_freq=control_freq_hz,
        gui=gui,
        record=record_video,
        obstacles=False,  # 强制 False：不加载默认障碍物
        user_debug_gui=user_debug_gui
    )

    # PyBullet client id
    PYB_CLIENT = env.getPyBulletClient()

    # 获取无人机在 PyBullet 里的 body id（用来做碰撞检测）
    DRONE_ID = env.getDroneIds()[0]

    # -----------------------
    # 3) 创建四面墙（你让我定，我定 ARENA=0.6）
    # -----------------------
    ARENA = 1.5        # 墙在 x,y=±1.5 附近
    WALL_THICK = 0.005
    WALL_H = 0.8
    wall_ids = create_four_walls(
        pyb_client_id=PYB_CLIENT,
        half_size=ARENA,
        wall_thickness=WALL_THICK,
        wall_height=WALL_H
    )

    # -----------------------
    # 4) Logger
    # -----------------------
    logger = Logger(
        logging_freq_hz=control_freq_hz,
        num_drones=num_drones,
        output_folder=output_folder,
        colab=colab
    )

    # -----------------------
    # 5) PID 控制器
    # -----------------------
    ctrl = [DSLPIDControl(drone_model=drone) for _ in range(num_drones)]

    # -----------------------
    # 6) 状态机
    # -----------------------
    GO_STRAIGHT = 0   # 匀速直飞
    WALL_FOLLOW = 1   # 沿墙顺时针绕一圈（矩形角点）
    RETURN_HOME = 2   # 回原点
    DONE = 3          # 悬停

    state = GO_STRAIGHT

    # -----------------------
    # 7) 飞行参数
    # -----------------------
    V = 0.35                 # 匀速（m/s）
    dt = env.CTRL_TIMESTEP   # 控制周期

    # 直飞方向：朝 +X
    vel_dir = np.array([1.0, 0.0], dtype=float)
    vel_dir = vel_dir / (np.linalg.norm(vel_dir) + 1e-9)

    # 目标点 target_xy（每步更新）
    target_xy = np.array([0.0, 0.0], dtype=float)

    # -----------------------
    # 8) 沿墙顺时针绕一圈：走角点（更稳定）
    #    顺时针： (c,-c)->(c,c)->(-c,c)->(-c,-c)->(c,-c)
    # -----------------------
    margin = 0.12  # 离墙稍微留点距离，避免一直撞墙抖动
    c = ARENA - margin

    corners = [
        np.array([ c, -c], dtype=float),
        np.array([ c,  c], dtype=float),
        np.array([-c,  c], dtype=float),
        np.array([-c, -c], dtype=float),
        np.array([ c, -c], dtype=float)  # 回到起点角点表示绕完一圈
    ]

    corner_idx = 0
    lap_completed = False

    # -----------------------
    # 9) 输出频率：每 1.0 秒输出一次
    # -----------------------
    PRINT_EVERY_SEC = 1.0
    PRINT_EVERY_STEPS = max(1, int(PRINT_EVERY_SEC * env.CTRL_FREQ))
    # 渲染频率：每 1.0 秒调用一次 env.render()
    RENDER_EVERY_SEC = 1.0
    RENDER_EVERY_STEPS = max(1, int(RENDER_EVERY_SEC * env.CTRL_FREQ))

    # -----------------------
    # 10) 主循环
    # -----------------------
    action = np.zeros((num_drones, 4))
    START = time.time()
    steps = int(duration_sec * env.CTRL_FREQ)

    for i in range(steps):
        # step 仿真
        obs, reward, terminated, truncated, info = env.step(action)

        # 当前坐标
        cur_xyz = obs[0][0:3]
        cur_xy = np.array([cur_xyz[0], cur_xyz[1]], dtype=float)

        # 每 1 秒输出一次（防止刷屏）
        if i % PRINT_EVERY_STEPS == 0:
            print(f"状态[t={i/env.CTRL_FREQ:6.2f}s] pos=({cur_xyz[0]: .3f}, {cur_xyz[1]: .3f}, {cur_xyz[2]: .3f}) state={state}")

        # -----------------------
        # A) 碰撞检测：是否真的撞到墙
        # -----------------------
        hit_wall = False
        for wid in wall_ids:
            contacts = p.getContactPoints(
                bodyA=DRONE_ID,
                bodyB=wid,
                physicsClientId=PYB_CLIENT
            )
            if len(contacts) > 0:
                hit_wall = True
                break

        # -----------------------
        # B) 状态机：更新 target_xy
        # -----------------------
        if state == GO_STRAIGHT:
            # 目标点匀速前进
            target_xy = target_xy + vel_dir * V * dt

            # 把目标限制在安全区（避免目标跑到墙里面）
            target_xy[0] = clamp(target_xy[0], -ARENA + margin, ARENA - margin)
            target_xy[1] = clamp(target_xy[1], -ARENA + margin, ARENA - margin)

            # 真正撞墙才切换
            if hit_wall:
                state = WALL_FOLLOW

                # 选择离当前位置最近的角点作为起点
                best_k = 0
                best_d = 1e9
                for k in range(4):
                    d = dist2(cur_xy, corners[k])
                    if d < best_d:
                        best_d = d
                        best_k = k

                corner_idx = best_k
                lap_completed = False

        elif state == WALL_FOLLOW:
            # 当前要去的角点
            wp = corners[corner_idx]

            # 让目标点匀速朝角点走
            to_wp = wp - target_xy
            d = np.linalg.norm(to_wp)
            if d > 1e-9:
                step_dir = to_wp / d
                target_xy = target_xy + step_dir * V * dt

            # 到达角点后切换下一个角点
            if dist2(target_xy, wp) < (0.05 * 0.05):
                corner_idx += 1
                if corner_idx >= len(corners):
                    corner_idx = len(corners) - 1

            # 到达最后一个重复角点（回到起点）则表示绕圈完成
            if corner_idx == len(corners) - 1 and dist2(target_xy, corners[-1]) < (0.08 * 0.08):
                lap_completed = True

            if lap_completed:
                state = RETURN_HOME

        elif state == RETURN_HOME:
            home = np.array([0.0, 0.0], dtype=float)
            to_home = home - target_xy
            d = np.linalg.norm(to_home)

            if d > 1e-9:
                target_xy = target_xy + (to_home / d) * V * dt

            # 目标点接近原点 + 实际位置也接近原点 => DONE
            if dist2(target_xy, home) < (0.05 * 0.05) and dist2(cur_xy, home) < (0.08 * 0.08):
                state = DONE

        elif state == DONE:
            print("✅ FINISH: mission completed, returning home.")
            break

        # -----------------------
        # C) PID 控制计算
        # -----------------------
        target_pos = np.array([target_xy[0], target_xy[1], H], dtype=float)
        target_rpy = np.array([0.0, 0.0, 0.0], dtype=float)

        action[0, :], _, _ = ctrl[0].computeControlFromState(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=target_rpy
        )

        # -----------------------
        # D) 日志记录
        # -----------------------
        logger.log(
            drone=0,
            timestamp=i / env.CTRL_FREQ,
            state=obs[0],
            control=np.hstack([target_pos, target_rpy, np.zeros(6)])
        )

        # 渲染 + 实时同步
        # 每 1 秒调用一次 env.render()（INFO 也就 1 秒一次）
        if i % RENDER_EVERY_STEPS == 0:
            env.render()
        if gui:
            sync(i, START, env.CTRL_TIMESTEP)

    # -----------------------
    # 结束
    # -----------------------
    env.close()
    logger.save()
    logger.save_as_csv("wall_follow")
    if plot:
        logger.plot()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Wall follow + return home using CtrlAviary and DSLPIDControl')

    parser.add_argument('--drone', default=DroneModel("cf2x"), type=DroneModel,
                        help='Drone model (default: CF2X)', metavar='', choices=DroneModel)

    parser.add_argument('--num_drones', default=DEFAULT_NUM_DRONES, type=int,
                        help='Number of drones (default: 1)', metavar='')

    parser.add_argument('--physics', default=DEFAULT_PHYSICS, type=Physics,
                        help='Physics updates (default: PYB)', metavar='', choices=Physics)

    parser.add_argument('--gui', default=DEFAULT_GUI, type=str2bool,
                        help='Whether to use PyBullet GUI (default: True)', metavar='')

    parser.add_argument('--record_video', default=DEFAULT_RECORD_VISION, type=str2bool,
                        help='Whether to record a video (default: False)', metavar='')

    parser.add_argument('--plot', default=DEFAULT_PLOT, type=str2bool,
                        help='Whether to plot the simulation results (default: True)', metavar='')

    parser.add_argument('--user_debug_gui', default=DEFAULT_USER_DEBUG_GUI, type=str2bool,
                        help='Whether to add debug lines and parameters to the GUI (default: False)', metavar='')

    parser.add_argument('--obstacles', default=DEFAULT_OBSTACLES, type=str2bool,
                        help='Keep as False (default: False)', metavar='')

    parser.add_argument('--simulation_freq_hz', default=DEFAULT_SIMULATION_FREQ_HZ, type=int,
                        help='Simulation frequency in Hz (default: 240)', metavar='')

    parser.add_argument('--control_freq_hz', default=DEFAULT_CONTROL_FREQ_HZ, type=int,
                        help='Control frequency in Hz (default: 48)', metavar='')

    parser.add_argument('--duration_sec', default=DEFAULT_DURATION_SEC, type=int,
                        help='Duration of the simulation in seconds (default: 60)', metavar='')

    parser.add_argument('--output_folder', default=DEFAULT_OUTPUT_FOLDER, type=str,
                        help='Folder where to save logs (default: "results")', metavar='')

    parser.add_argument('--colab', default=DEFAULT_COLAB, type=bool,
                        help='Whether example is being run by a notebook (default: False)', metavar='')

    ARGS = parser.parse_args()
    run(**vars(ARGS))

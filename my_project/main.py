# my_project/main.py
import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync

from my_project.config import CFG
from my_project.env.build_arena import create_four_walls
from my_project.env.sensors import hit_any_wall
from my_project.control.pid import PIDController
from my_project.utils.rate import RateLimiter

from my_project.navigation.base import State
from my_project.navigation.wall_follow import WallFollowMission
from my_project.navigation.avoidance import AvoidanceLayer
from my_project.navigation.mission_manager import MissionManager


def parse_state(obs, step: int, t: float) -> State:
    # CtrlAviary 的 obs[0:3] 通常是 xyz，obs[10:13] 常见是速度，但不同版本可能略有差异
    xyz = np.array(obs[0:3], dtype=float)

    # 速度字段不稳（版本差异），先给占位，后续需要再精确取
    vel = np.zeros(3, dtype=float)
    return State(xyz=xyz, vel=vel, step=step, t=t)


def main():
    # 1) 初始化 env
    H = CFG["flight_height"]
    INIT_XYZS = np.array([[0.0, 0.0, H] for _ in range(CFG["num_drones"])])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0] for _ in range(CFG["num_drones"])])

    env = CtrlAviary(
        drone_model=CFG["drone"],
        num_drones=CFG["num_drones"],
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=CFG["physics"],
        neighbourhood_radius=10,
        pyb_freq=CFG["simulation_freq_hz"],
        ctrl_freq=CFG["control_freq_hz"],
        gui=CFG["gui"],
        record=CFG["record_video"],
        obstacles=False,
        user_debug_gui=CFG["user_debug_gui"],
    )

    PYB_CLIENT = env.getPyBulletClient()
    DRONE_ID = env.getDroneIds()[0]

    # 2) 建墙（环境模块）
    wall_ids = create_four_walls(
        pyb_client_id=PYB_CLIENT,
        half_size=CFG["arena_half_size"],
        wall_thickness=CFG["wall_thickness"],
        wall_height=CFG["wall_height"],
    )

    # 3) logger / controller
    logger = Logger(logging_freq_hz=CFG["control_freq_hz"], num_drones=CFG["num_drones"], output_folder=CFG["output_folder"], colab=False)
    pid = PIDController(drone_model=CFG["drone"])

    # 4) mission + avoidance + manager
    mission = WallFollowMission(
        arena_half_size=CFG["arena_half_size"],
        flight_height=CFG["flight_height"],
        speed=0.35,      # 速度你以后放 CFG 也行
        margin=0.12
    )
    avoidance = AvoidanceLayer()  # 下一任务实现它
    manager = MissionManager(mission, avoidance_layer=avoidance)

    # 5) 降频器：print/render
    print_rate = RateLimiter(int(CFG["print_every_sec"] * CFG["control_freq_hz"]))
    render_rate = RateLimiter(int(CFG["render_every_sec"] * CFG["control_freq_hz"]))

    # 6) loop
    action = np.zeros((CFG["num_drones"], 4))
    START = time.time()
    steps = int(CFG["duration_sec"] * env.CTRL_FREQ)

    # reset mission with initial state
    obs, _, _, _, _ = env.step(action)
    init_state = parse_state(obs[0], 0, 0.0)
    manager.reset(init_state)

    for i in range(steps):
        obs, reward, terminated, truncated, info = env.step(action)
        t = i / env.CTRL_FREQ

        state = parse_state(obs[0], i, t)

        sensors = {
            "dt": env.CTRL_TIMESTEP,
            "hit_wall": hit_any_wall(PYB_CLIENT, DRONE_ID, wall_ids),
        }

        cmd = manager.update(state, sensors)

        if print_rate.hit(i):
            print(f"[t={t:6.2f}s] pos=({state.xyz[0]: .3f}, {state.xyz[1]: .3f}, {state.xyz[2]: .3f}) finished={cmd.finished} info={cmd.info}")

        # PID 输出电机指令
        action[0, :] = pid.compute(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=cmd.target_pos,
            target_rpy=cmd.target_rpy,
        )

        # env.render() 也降频
        if render_rate.hit(i):
            env.render()

        if CFG["gui"]:
            sync(i, START, env.CTRL_TIMESTEP)

        if cmd.finished:
            print("✅ FINISH")
            break

        logger.log(drone=0, timestamp=t, state=obs[0], control=np.hstack([cmd.target_pos, cmd.target_rpy, np.zeros(6)]))

    env.close()
    logger.save()
    logger.save_as_csv("my_project")
    if CFG["plot"]:
        logger.plot()


if __name__ == "__main__":
    main()

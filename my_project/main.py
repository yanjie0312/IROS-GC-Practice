# my_project/main.py
import time
import numpy as np

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

from my_project.config import CFG
from my_project.env.build_arena import create_arena, get_layout
from my_project.experiments.scenarios import make_scenario
from my_project.env.sensors import SensorSuite
from my_project.control.pid import PIDController


def main():
    # 1) 创建场景 & 初始化 env
    scenario = make_scenario("easy", seed=42)
    layout = get_layout(scenario.layout_name)

    H = CFG["flight_height"]
    INIT_XYZS = np.array([[0.0, 0.0, H]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

    env = CtrlAviary(
        drone_model=CFG["drone"],
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=CFG["physics"],
        pyb_freq=CFG["simulation_freq_hz"],
        ctrl_freq=CFG["control_freq_hz"],
        gui=True,
        obstacles=False,
    )

    PYB_CLIENT = env.getPyBulletClient()
    DRONE_ID = env.getDroneIds()[0]

    # 2) 建公寓场景
    arena_handle = create_arena(scenario, client_id=PYB_CLIENT)

    # 3) 传感器 & PID 控制器
    sensor = SensorSuite(
        pyb_client_id=PYB_CLIENT,
        drone_id=DRONE_ID,
        arena_handle=arena_handle,
    )
    pid = PIDController(drone_model=CFG["drone"])

    # 4) 主循环：原地悬停 + 传感器感知
    target_pos = np.array([0.0, 0.0, H])
    target_rpy = np.array([0.0, 0.0, 0.0])

    action = np.zeros((1, 4))
    START = time.time()
    steps = int(CFG["duration_sec"] * env.CTRL_FREQ)

    for i in range(steps):
        obs, _, _, _, _ = env.step(action)
        t = i / env.CTRL_FREQ

        # 传感器感知
        pkt = sensor.sense(obs=obs[0], step=i, t=t)

        # 每 5 秒打印一次全部射线数据
        if i % (CFG["control_freq_hz"] * 5) == 0:
            print(f"\n[t={t:5.1f}s] pos=({pkt['pos'][0]:.2f}, {pkt['pos'][1]:.2f}, {pkt['pos'][2]:.2f})")
            print(f"  {'射线':>4s}  {'方向 (x, y, z)':>22s}  {'距离':>6s}")
            print(f"  {'----':>4s}  {'---------------':>22s}  {'----':>6s}")
            for j in range(len(pkt["ray_dists"])):
                d = pkt["ray_dirs_body"][j]
                dist = pkt["ray_dists"][j]
                label = "水平" if j < 16 else "倾斜"
                print(f"  [{j:2d}]  ({d[0]:+.2f}, {d[1]:+.2f}, {d[2]:+.2f})  {dist:5.2f}m  {label}")

        # PID 控制
        action[0, :] = pid.compute(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=target_pos,
            target_rpy=target_rpy,
        )

        sync(i, START, env.CTRL_TIMESTEP)

    env.close()


if __name__ == "__main__":
    main()

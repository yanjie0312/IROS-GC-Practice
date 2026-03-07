# my_project/main.py
import time
import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

from my_project.config import CFG
from my_project.env.build_arena import create_arena, get_layout
from my_project.experiments.scenarios import make_scenario
from my_project.env.sensors import SensorSuite
from my_project.env.targets import TargetManager
from my_project.control.pid import PIDController
from my_project.navigation.base import State
from my_project.navigation.occupancy_grid import OccupancyGrid, GridBounds
from my_project.navigation.frontier_planner import FrontierPlanner
from my_project.navigation.avoidance import AvoidanceLayer
from my_project.navigation.search_mission import SearchMission
from my_project.navigation.mission_manager import MissionManager


CRUISE_HEIGHT = 0.5    # 巡航高度（m），低于墙顶 1.0m
MAX_DURATION_SEC = 300  # 最长运行时间


def main():
    # 1) 创建场景 & 初始化 env
    scenario = make_scenario("easy", seed=42)
    layout = get_layout(scenario.layout_name)

    INIT_XYZS = np.array([[0.5, 0.0, CFG["flight_height"]]])
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

    # 3) 传感器
    sensor = SensorSuite(
        pyb_client_id=PYB_CLIENT,
        drone_id=DRONE_ID,
        arena_handle=arena_handle,
    )

    # 4) 目标管理
    target_manager = TargetManager(
        arena_handle=arena_handle,
        pyb_client_id=PYB_CLIENT,
        inspect_range=0.5,
        inspect_time=2.0,
    )

    # 5) 导航栈
    W = float(layout["W"])
    H_layout = float(layout["H"])
    inset = 0.05
    grid = OccupancyGrid(
        resolution=0.10,
        bounds=GridBounds(
            x_min=0.0 + inset, x_max=W - inset,
            y_min=-H_layout + inset, y_max=H_layout - inset,
        ),
        ray_length=2.5,
    )
    planner = FrontierPlanner(grid, verbose=True)
    mission = SearchMission(
        frontier_planner=planner,
        target_manager=target_manager,
        takeoff_height=CRUISE_HEIGHT,
        waypoint_reach_dist=0.25,
        frontier_replan_cooldown_steps=10,
        min_waypoint_delta=0.12,
        target_retry_cooldown_steps=120,
        stuck_window_steps=180,
        stuck_radius=0.22,
        goto_path_lookahead_dist=0.55,
        goto_goal_search_radius=0.9,
        verbose=True,
    )
    manager = MissionManager(
        mission=mission,
        avoidance_layer=AvoidanceLayer(
            d0=0.4,    # 只在 0.4m 内才产生排斥力，不干扰正常巡航
            k_rep=0.5, # 排斥力系数保持温和
            alpha=0.3, # 仅 30% 权重给安全方向，主要信任 frontier 航点
            min_dist_emergency=0.15,  # 紧急情况（<0.15m）自动放大排斥力
        ),
    )

    # 6) PID
    pid = PIDController(drone_model=CFG["drone"])

    # 7) 初始化：先 step 一次拿到观测，再 reset 任务状态机
    action = np.zeros((1, 4))
    obs, _, _, _, _ = env.step(action)
    init_pkt = sensor.sense(obs=obs[0], step=0, t=0.0)
    init_state = State(xyz=init_pkt["pos"], vel=init_pkt["vel"], step=0, t=0.0)
    manager.reset(init_state)

    # 8) 主循环
    START = time.time()
    steps = int(MAX_DURATION_SEC * env.CTRL_FREQ)
    print("\n=== 任务开始 ===")

    for i in range(1, steps + 1):
        obs, _, _, _, _ = env.step(action)
        t = i / env.CTRL_FREQ

        # 感知
        try:
            pkt = sensor.sense(obs=obs[0], step=i, t=t)
        except p.error:
            print("\n=== 物理引擎连接断开，提前结束仿真 ===")
            break

        # 目标巡检计时
        target_manager.update(pkt)

        # 状态机 + 避障 → 目标点
        state = State(xyz=pkt["pos"], vel=pkt["vel"], step=i, t=t)
        cmd = manager.update(state, pkt)

        # 限制水平步长，防止 PID 过大倾斜（目标点离当前位置过远会导致大倾角）
        if not cmd.finished:
            # 避免任何阶段给出过低高度目标导致掉高
            cmd.target_pos[2] = max(float(cmd.target_pos[2]), CRUISE_HEIGHT)

            min_dist = float(pkt.get("min_dist", np.inf))
            collision = bool(pkt.get("collision", False))

            if collision or min_dist < 0.10:
                # 紧急情况：按近距离射线合成反推方向脱困，并略微抬高
                escape_xy = np.zeros(2, dtype=float)
                ray_dists = np.asarray(pkt.get("ray_dists", []), dtype=float).reshape(-1)
                ray_dirs_world = np.asarray(pkt.get("ray_dirs_world", []), dtype=float)
                if (
                    ray_dists.size > 0
                    and np.isfinite(ray_dists).any()
                    and ray_dirs_world.ndim == 2
                    and ray_dirs_world.shape[0] == ray_dists.size
                ):
                    dirs_xy = np.asarray(ray_dirs_world[:, :2], dtype=float)
                    norms = np.linalg.norm(dirs_xy, axis=1)
                    valid = norms > 1e-9
                    near = np.isfinite(ray_dists) & (ray_dists < 0.35) & valid
                    if np.any(near):
                        dirs_u = dirs_xy[near] / norms[near][:, None]
                        weights = (0.35 - ray_dists[near]) / 0.35
                        rep = (-dirs_u * weights[:, None]).sum(axis=0)
                        nrep = float(np.linalg.norm(rep))
                        if nrep > 1e-9:
                            escape_xy = rep / nrep * 0.30
                    else:
                        i_min = int(np.nanargmin(ray_dists))
                        dxy = np.asarray(ray_dirs_world[i_min, :2], dtype=float)
                        nxy = float(np.linalg.norm(dxy))
                        if nxy > 1e-9:
                            escape_xy = (-dxy / nxy) * 0.24

                if float(np.linalg.norm(escape_xy)) < 1e-9:
                    # 兜底：沿当前命令方向给一个小步，避免原地不动
                    desired = np.asarray(cmd.target_pos[:2] - pkt["pos"][:2], dtype=float)
                    nd = float(np.linalg.norm(desired))
                    if nd > 1e-9:
                        escape_xy = desired / nd * 0.18
                cmd.target_pos = np.array(
                    [pkt["pos"][0] + escape_xy[0], pkt["pos"][1] + escape_xy[1], CRUISE_HEIGHT + 0.05],
                    dtype=float,
                )

            delta_xy = cmd.target_pos[:2] - pkt["pos"][:2]
            dist_xy = float(np.linalg.norm(delta_xy))
            if collision or min_dist < 0.18:
                max_xy_step = 0.14
            elif min_dist < 0.30:
                max_xy_step = 0.24
            else:
                max_xy_step = 0.45

            if dist_xy > max_xy_step:
                limited = cmd.target_pos.copy()
                limited[:2] = pkt["pos"][:2] + delta_xy / dist_xy * max_xy_step
                cmd.target_pos = limited

            # Hard-bound commands to stay inside apartment envelope.
            cmd.target_pos[0] = float(np.clip(cmd.target_pos[0], 0.10, W - 0.10))
            cmd.target_pos[1] = float(np.clip(cmd.target_pos[1], -H_layout + 0.10, H_layout - 0.10))

        # 每 5 秒打印进度
        if i % (env.CTRL_FREQ * 5) == 0:
            inspected, discovered, total = target_manager.get_progress()
            print(
                f"[t={t:6.1f}s] pos=({pkt['pos'][0]:.2f}, {pkt['pos'][1]:.2f}, {pkt['pos'][2]:.2f})"
                f"  目标：{inspected}/{total} 巡检完, {discovered}/{total} 已发现"
            )

        # 坠毁检测：高度过低 或 姿态严重倾斜（翻转）
        z = float(pkt["pos"][2])
        roll, pitch = float(pkt["rpy"][0]), float(pkt["rpy"][1])
        crashed = z < 0.08  # 高度低于 8cm 视为落地坠毁
        if crashed and i > env.CTRL_FREQ:  # 跳过最初 1 秒的起飞抖动
            inspected, discovered, total = target_manager.get_progress()
            print(f"\n=== 无人机坠毁！t={t:.1f}s  pos=({pkt['pos'][0]:.2f}, {pkt['pos'][1]:.2f}, {z:.2f}) ===")
            print(f"巡检结果：{inspected}/{total} 个目标完成巡检")
            break

        # 任务完成
        if cmd.finished:
            print(f"\n=== 任务完成！用时 {t:.1f}s ===")
            inspected, discovered, total = target_manager.get_progress()
            print(f"巡检结果：{inspected}/{total} 个目标完成巡检")
            break

        # PID 计算电机输出
        action[0, :] = pid.compute(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=cmd.target_pos,
            target_rpy=cmd.target_rpy,
        )

        sync(i, START, env.CTRL_TIMESTEP)

    else:
        print(f"\n=== 超时（{MAX_DURATION_SEC}s）===")
        inspected, discovered, total = target_manager.get_progress()
        print(f"巡检结果：{inspected}/{total} 个目标完成巡检")

    try:
        env.close()
    except p.error:
        pass


if __name__ == "__main__":
    main()

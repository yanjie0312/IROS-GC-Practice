# my_project/main.py
import os
import json
import time
import numpy as np
import pybullet as p
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from collections import deque
from datetime import datetime

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.utils import sync

from my_project.config import CFG
from my_project.env.build_arena import create_arena, get_layout
from my_project.env.disturbances import DisturbanceInjector
from my_project.experiments.scenarios import make_scenario
from my_project.env.sensors import SensorSuite
from my_project.env.targets import TargetManager
from my_project.control.pid import PIDController
from my_project.navigation.base import State
from my_project.navigation.occupancy_grid import OCCUPIED, FREE, UNKNOWN, OccupancyGrid, GridBounds
from my_project.navigation.frontier_planner import FrontierPlanner
from my_project.navigation.avoidance import AvoidanceLayer
from my_project.navigation.search_mission import SearchMission
from my_project.navigation.mission_manager import MissionManager
from my_project.experiments.scenarios import list_difficulty_profiles
from my_project.ui.difficulty_picker import resolve_difficulty_profile


CRUISE_HEIGHT = 0.5    # 巡航高度（m），低于墙顶 1.0m
MAX_DURATION_SEC = 500  # 最长运行时间
# 在 PyBullet 窗口用黄色点画出已探索区域（FREE 格）
SHOW_EXPLORED_OVERLAY = True
EXPLORED_OVERLAY_SUBSAMPLE = 2  # 每 N 格画一个点
EXPLORED_OVERLAY_Z = 0.02
EXPLORED_OVERLAY_INTERVAL = 2   # 每 N 步更新一次
# 黄色含义：栅格 FREE = 射线曾穿过该格（累积地图），不是「当前射线范围」。
# 另一房间有黄 = 曾到过该房或 2D 投影下射线经门洞穿过。目标「已发现」= 本帧传感器 LOS，与 FREE 无关。

def _get_wall_draw_segments(layout: dict) -> list:
    """将 layout 的 outer/inner 墙段扣除门洞后，返回可直接绘制的线段列表 [(x1,y1,x2,y2),...]。"""
    outer = layout["outer"]
    inner = layout["inner"]
    gap_map: dict = {}
    for which, idx, center, width in layout.get("gaps", []):
        gap_map.setdefault((which, idx), []).append((center, width))

    segs = []
    for which, wall_list in (("outer", outer), ("inner", inner)):
        for idx, (x1, y1, x2, y2) in enumerate(wall_list):
            gaps = gap_map.get((which, idx), [])
            if not gaps:
                segs.append((x1, y1, x2, y2))
                continue
            is_vert = abs(x2 - x1) < 1e-6
            if is_vert:
                lo, hi = sorted([y1, y2])
                cuts = sorted((c - w / 2, c + w / 2) for c, w in gaps)
                cur = lo
                for gs, ge in cuts:
                    if cur < gs:
                        segs.append((x1, cur, x1, min(gs, hi)))
                    cur = max(cur, ge)
                if cur < hi:
                    segs.append((x1, cur, x1, hi))
            else:
                lo, hi = sorted([x1, x2])
                cuts = sorted((c - w / 2, c + w / 2) for c, w in gaps)
                cur = lo
                for gs, ge in cuts:
                    if cur < gs:
                        segs.append((cur, y1, min(gs, hi), y1))
                    cur = max(cur, ge)
                if cur < hi:
                    segs.append((cur, y1, hi, y1))
    return segs


def _draw_trajectory(ax, trajectory: list) -> None:
    """在 ax 上画渐深蓝色轨迹。"""
    if not trajectory:
        return
    traj = np.array(trajectory)
    n = len(traj)
    for i in range(n - 1):
        alpha = 0.3 + 0.7 * i / max(n - 1, 1)
        ax.plot(traj[i:i+2, 0], traj[i:i+2, 1], "-",
                color="royalblue", alpha=alpha, linewidth=1.0)
    ax.plot(traj[-1, 0], traj[-1, 1], "s",
            color="crimson", markersize=8, zorder=6)


def _save_flight_data(
    base_path: str,
    trajectory: list,
    grid: OccupancyGrid,
    target_manager,
    targets_gt: dict,
    obstacles_gt: list,
    nofly_gt: list,
    home_pos: np.ndarray,
    result: dict,
    layout: dict,
) -> None:
    """将飞行数据保存为 JSON，与图片同名不同后缀。"""
    b = grid.bounds
    data = {
        "result": result,
        "trajectory_xy": [pt.tolist() for pt in trajectory],
        "home_pos": home_pos[:3].tolist(),
        "occupancy_grid": {
            "resolution": float(grid.resolution),
            "bounds": {"x_min": b.x_min, "x_max": b.x_max,
                       "y_min": b.y_min, "y_max": b.y_max},
            "data": grid.grid.tolist(),
        },
        "targets": [
            {
                "id": int(tid),
                "discovered": bool(info.discovered),
                "inspected": bool(info.inspected),
                "estimated_pos": info.position.tolist(),
                "true_pos": targets_gt.get(int(tid)),
                "measured_pos_at_inspect": (
                    info.measured_position_at_inspect.tolist()
                    if info.measured_position_at_inspect is not None else None
                ),
            }
            for tid, info in target_manager.targets.items()
        ],
        "obstacles_true_pos": obstacles_gt,
        "nofly_true_pos": nofly_gt,
        "layout": {
            "outer": layout["outer"],
            "inner": layout["inner"],
            "gaps": [list(g) for g in layout.get("gaps", [])],
        },
    }
    json_path = base_path + ".json"
    os.makedirs(os.path.dirname(os.path.abspath(json_path)), exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[Data]  saved → {json_path}")


def _plot_flight_result(
    base_path: str,
    trajectory: list,
    grid: OccupancyGrid,
    target_manager,
    targets_gt: dict,
    obstacles_gt: list,
    nofly_gt: list,
    home_pos: np.ndarray,
    result: dict,
    layout: dict,
) -> None:
    """生成左（无人机感知）右（上帝视角）双面板对比图，保存为 PNG。"""
    fig, axes = plt.subplots(1, 2, figsize=(22, 11))
    b = grid.bounds
    extent = [b.x_min, b.x_max, b.y_min, b.y_max]

    # ── 左面板：无人机感知视角 ────────────────────────────────────────
    ax = axes[0]
    g = grid.grid
    rgb = np.ones((*g.shape, 3), dtype=float) * 0.82
    rgb[g == FREE]     = [0.97, 0.97, 0.88]
    rgb[g == OCCUPIED] = [0.22, 0.22, 0.22]
    ax.imshow(rgb, origin="lower", extent=extent, aspect="equal", interpolation="nearest")
    _draw_trajectory(ax, trajectory)
    ax.plot(home_pos[0], home_pos[1], "^", color="limegreen", markersize=13,
            zorder=7, markeredgecolor="darkgreen", markeredgewidth=1.2)
    for tid, info in target_manager.targets.items():
        pos = info.position
        if np.linalg.norm(pos) < 1e-6:
            continue
        if info.inspected:
            ax.plot(pos[0], pos[1], "*", color="gold", markersize=18, zorder=8,
                    markeredgecolor="darkorange", markeredgewidth=1.5)
            ax.annotate(f"T{tid} OK", (pos[0], pos[1]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8,
                        color="darkorange", fontweight="bold")
        elif info.discovered:
            ax.plot(pos[0], pos[1], "*", color="orange", markersize=15, zorder=8,
                    markeredgecolor="saddlebrown", markeredgewidth=1.2)
            ax.annotate(f"T{tid}", (pos[0], pos[1]), xytext=(5, 5),
                        textcoords="offset points", fontsize=8, color="saddlebrown")
    ax.set_title("Drone Perception\n(gray=unknown, cream=explored, dark=wall/obstacle)",
                 fontsize=11, fontweight="bold")
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
    ax.grid(True, alpha=0.2, linestyle="--")

    # ── Right panel: ground truth ─────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor("#f5f5f0")
    ax2.set_xlim(b.x_min - 0.3, b.x_max + 0.3)
    ax2.set_ylim(b.y_min - 0.3, b.y_max + 0.3)
    ax2.set_aspect("equal")

    # 禁飞区
    for pos in nofly_gt:
        circ = mpatches.Circle(pos, radius=0.45, color="tomato", alpha=0.25, zorder=1)
        ax2.add_patch(circ)

    # 墙壁（带门洞）
    wall_segs = _get_wall_draw_segments(layout)
    for x1, y1, x2, y2 in wall_segs:
        ax2.plot([x1, x2], [y1, y2], "-", color="#333333", linewidth=2.5, zorder=3)

    # 障碍物
    for pos in obstacles_gt:
        circ = mpatches.Circle(pos, radius=0.22, color="slategray", alpha=0.7, zorder=4)
        ax2.add_patch(circ)

    # 真实目标位置
    for tid, info in target_manager.targets.items():
        gt_pos = targets_gt.get(int(tid))
        if gt_pos is None:
            continue
        if info.inspected:
            ax2.plot(gt_pos[0], gt_pos[1], "*", color="gold", markersize=18,
                     zorder=8, markeredgecolor="darkorange", markeredgewidth=1.5)
            ax2.annotate(f"T{tid} OK", (gt_pos[0], gt_pos[1]), xytext=(5, 5),
                         textcoords="offset points", fontsize=8,
                         color="darkorange", fontweight="bold")
        elif info.discovered:
            ax2.plot(gt_pos[0], gt_pos[1], "*", color="orange", markersize=15,
                     zorder=8, markeredgecolor="saddlebrown", markeredgewidth=1.2)
            ax2.annotate(f"T{tid}", (gt_pos[0], gt_pos[1]), xytext=(5, 5),
                         textcoords="offset points", fontsize=8, color="saddlebrown")
        else:
            ax2.plot(gt_pos[0], gt_pos[1], "*", color="lightgray", markersize=15,
                     zorder=8, markeredgecolor="gray", markeredgewidth=1.0)

    # 轨迹与起点
    _draw_trajectory(ax2, trajectory)
    ax2.plot(home_pos[0], home_pos[1], "^", color="limegreen", markersize=13,
             zorder=9, markeredgecolor="darkgreen", markeredgewidth=1.2)

    ax2.set_title("Ground Truth\n(true walls, obstacles, target positions)",
                  fontsize=11, fontweight="bold")
    ax2.set_xlabel("X (m)"); ax2.set_ylabel("Y (m)")
    ax2.grid(True, alpha=0.2, linestyle="--")

    # ── 共用图例 ─────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], color="royalblue", linewidth=2, label="Trajectory"),
        Line2D([0], [0], marker="^", color="limegreen", markersize=10,
               linestyle="None", markeredgecolor="darkgreen", label="Home"),
        Line2D([0], [0], marker="*", color="gold", markersize=12,
               linestyle="None", markeredgecolor="darkorange", label="Inspected target"),
        Line2D([0], [0], marker="*", color="orange", markersize=11,
               linestyle="None", markeredgecolor="saddlebrown", label="Discovered (not inspected)"),
        Line2D([0], [0], marker="*", color="lightgray", markersize=11,
               linestyle="None", markeredgecolor="gray", label="Undiscovered target (right)"),
        mpatches.Patch(color="slategray", alpha=0.7, label="Obstacle (right)"),
        mpatches.Patch(color="tomato",    alpha=0.25, label="No-fly zone (right)"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=4,
               fontsize=9, bbox_to_anchor=(0.5, 0.01))

    status = "SUCCESS" if result["success"] else result["termination_reason"]
    fig.suptitle(
        f"Result: {status}  |  Time: {result['flight_time_sec']:.1f}s  |  "
        f"Inspected: {result['targets_inspected']}/{result['targets_total']} targets",
        fontsize=13, fontweight="bold", y=0.99,
    )
    fig.tight_layout(rect=[0, 0.06, 1, 0.97])

    img_path = base_path + ".png"
    os.makedirs(os.path.dirname(os.path.abspath(img_path)), exist_ok=True)
    fig.savefig(img_path, dpi=150, bbox_inches="tight")
    print(f"[Plot]  saved → {img_path}")
    plt.close(fig)


def main():
    # 1) 创建场景 & 初始化 env
    scenario = make_scenario(
        CFG.get("difficulty_profile", "L0_easy"),
        seed=int(CFG.get("scenario_seed", 42)),
    )
    profile_name = str(CFG.get("difficulty_profile", "L0_easy"))
    # Hardening patch:
    # - L3: full hardening
    # - L2: light hardening (reduce stochastic crash probability)
    hardening_enabled = profile_name.startswith("L3")
    light_hardening_enabled = profile_name.startswith("L2") or hardening_enabled
    l1_enabled = profile_name.startswith("L1")
    l2_enabled = profile_name.startswith("L2")
    layout = get_layout(scenario.layout_name)

    # 从 layout 的入口 gap 计算起飞点（门洞内侧 0.3m）
    _entrance = next(
        (g for g in layout.get("gaps", []) if g[0] == "outer"), None
    )
    if _entrance is not None:
        _which, _idx, _center, _width = _entrance
        _wx1, _wy1, _wx2, _wy2 = layout["outer"][_idx]
        if abs(_wx2 - _wx1) < 1e-6:          # 竖墙：门中心是 y_center
            _sx, _sy = _wx1 + 0.3, float(_center)
        else:                                  # 横墙：门中心是 x_center
            _sx, _sy = float(_center), _wy1 + 0.3
    else:
        _sx, _sy = 0.3, 0.0

    INIT_XYZS = np.array([[_sx, _sy, CFG["flight_height"]]])
    INIT_RPYS = np.array([[0.0, 0.0, 0.0]])

    env = CtrlAviary(
        drone_model=CFG["drone"],
        num_drones=1,
        initial_xyzs=INIT_XYZS,
        initial_rpys=INIT_RPYS,
        physics=CFG["physics"],
        pyb_freq=CFG["simulation_freq_hz"],
        ctrl_freq=CFG["control_freq_hz"],
        gui=bool(CFG.get("gui", True)),
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
        state_noise_std_pos=float(scenario.pos_noise_std),
        state_noise_std_rpy=(0.0, 0.0, float(scenario.yaw_noise_std)),
        ray_noise_std=float(scenario.ray_noise_std),
        measurement_delay_steps=int(scenario.delay_steps),
        packet_drop_prob=float(scenario.dropout_prob),
        target_pos_noise_std=getattr(scenario, "target_pos_noise_std", 0.0),
        target_pos_bias=getattr(scenario, "target_pos_bias", None),
        target_range_noise_std=getattr(scenario, "target_range_noise_std", 0.0),
        target_bearing_noise_std=getattr(scenario, "target_bearing_noise_std", 0.0),
        target_range_bias=getattr(scenario, "target_range_bias", 0.0),
        target_false_negative_prob=getattr(scenario, "target_false_negative_prob", 0.0),
        rng_seed=int(scenario.seed),
    )

    # 3.1) 扰动注入器：L0_easy 下所有参数为 0，不影响当前可运行行为
    disturbance_cfg = dict(CFG.get("disturbance", {}))
    disturbance_cfg.update(
        {
            "wind_std": float(scenario.wind_std),
            "wind_bias_xy": tuple(scenario.wind_bias_xy),
            "gust_prob": float(scenario.gust_prob),
            "gust_strength": float(scenario.gust_strength),
            "state_noise_std_pos": float(scenario.pos_noise_std),
            "state_noise_std_rpy": (0.0, 0.0, float(scenario.yaw_noise_std)),
            "ray_noise_std": float(scenario.ray_noise_std),
            "packet_dropout_prob": float(scenario.dropout_prob),
            "measurement_delay_steps": int(scenario.delay_steps),
            "payload_mass_delta": float(scenario.payload_mass_delta),
        }
    )
    disturbance = DisturbanceInjector(
        pyb_client_id=PYB_CLIENT,
        drone_id=DRONE_ID,
        config=disturbance_cfg,
    )
    disturbance.reset(seed=int(scenario.seed) + 101)
    try:
        base_mass = float(p.getDynamicsInfo(DRONE_ID, -1, physicsClientId=PYB_CLIENT)[0])
        payload_delta = disturbance.sample_payload_variation(
            base_mass=base_mass,
            apply_to_simulation=True,
        )
        if abs(payload_delta) > 1e-9:
            print(f"[disturbance] payload delta mass = {payload_delta:+.5f} kg")
    except p.error:
        pass

    # 4) 目标管理
    # 巡检判定是 3D 距离；L3 在保高度策略下常以 ~0.58m 悬停，
    # 若 inspect_range 过小会出现“已发现但长期不完成巡检”的假卡死。
    inspect_range = 0.5
    if light_hardening_enabled:
        # L2/L3 都有保高度策略；巡检判定是 3D 距离，需要与悬停高度对齐
        inspect_range = max(inspect_range, CRUISE_HEIGHT + (0.10 if profile_name.startswith("L2") else 0.12))

    target_manager = TargetManager(
        arena_handle=arena_handle,
        pyb_client_id=PYB_CLIENT,
        inspect_range=inspect_range,
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
    # L3：固定路径规划器输出的 z，避免低飞/贴地时 get_waypoint_towards_goal 把目标高度锁在 drone[2]≈0
    planner = FrontierPlanner(
        grid,
        verbose=True,
        waypoint_z=CRUISE_HEIGHT if hardening_enabled else None,
        inflation_cells=1,
    )
    # L3 下：对“目标被障碍半封堵”的情况更保守，避免 GOTO_TARGET<->EXPLORE 高频抖动
    target_retry_cooldown_steps = 60
    goto_goal_search_radius = 1.40
    inspect_hover_dist = 0.4
    if hardening_enabled:
        # 60@48Hz 约 1.25s 太短，容易马上再次撞同一条不可达路径
        target_retry_cooldown_steps = 360   # 约 7.5s
        # 允许在目标周围更大范围找可达接近点，提升绕障成功率
        goto_goal_search_radius = 2.20
        # 稍放宽“到达目标上方”判定，减少在障碍边缘贴墙逼近
        inspect_hover_dist = 0.55

    mission = SearchMission(
        frontier_planner=planner,
        target_manager=target_manager,
        takeoff_height=CRUISE_HEIGHT,
        waypoint_reach_dist=0.25,
        frontier_replan_cooldown_steps=10,
        min_waypoint_delta=0.20,
        target_retry_cooldown_steps=target_retry_cooldown_steps,
        stuck_window_steps=120,
        stuck_radius=0.30,
        goto_path_lookahead_dist=0.35,
        goto_goal_search_radius=goto_goal_search_radius,
        inspect_hover_dist=inspect_hover_dist,
        l1_explore_hardening=l1_enabled,
        l2_explore_hardening=l2_enabled,
        l3_explore_hardening=hardening_enabled,
        enable_delivery=bool(CFG.get("enable_delivery_mission", False)),
        deliver_hover_secs=float(CFG.get("delivery_hover_secs", 2.0)),
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
    trajectory: list = []   # 记录飞行轨迹，用于结束后绘图
    START = time.time()
    # 运行时长以 MAX_DURATION_SEC 为主，再按难度缩放（与场景默认策略一致）
    if profile_name.startswith("L3"):
        timeout_scale = 0.80
    elif profile_name.startswith("L2"):
        timeout_scale = 0.95
    elif profile_name.startswith("L1"):
        timeout_scale = 0.95
    else:
        timeout_scale = 1.00
    steps = int(max(1, round(MAX_DURATION_SEC * timeout_scale * env.CTRL_FREQ)))
    # 防坠补丁参数（L2 轻量，L3 全量）
    z_guard = CRUISE_HEIGHT - (0.16 if light_hardening_enabled and not hardening_enabled else 0.18)
    z_guard_target = CRUISE_HEIGHT + (0.06 if light_hardening_enabled and not hardening_enabled else 0.08)
    reduced_xy_step_scale = 0.72 if light_hardening_enabled and not hardening_enabled else 0.6
    # 近坠毁短恢复窗口：避免悬停末期瞬间掉高直接结束
    # L2：曾出现低高度恢复窗口用尽仍无法拉起，导致误触发“坠毁退出”。
    # 适当延长恢复窗口，给姿态恢复/脱离障碍更多时间。
    # L2：拉长窗口；L3：也需要避免“trigger 后立刻坠毁”。
    # L3：本轮在 low-z recovery 结束时仍未拉起，因此把 recovery 步数加大到 96，并提高触发提前量。
    # L2：恢复窗口很关键；L3：同样需要更长一点避免 trigger 后还没拉起来就触发坠毁
    # （low-z recovery 阶段同时会把目标 z 保到 0.70，并更强抑制横向机动）
    low_z_recovery_steps = 144 if light_hardening_enabled and not hardening_enabled else 180
    low_z_trigger = 0.20 if light_hardening_enabled and not hardening_enabled else 0.18
    low_z_guard_target = max(CRUISE_HEIGHT + (0.10 if light_hardening_enabled and not hardening_enabled else 0.12), 0.60)
    # L2：低高度恢复窗口内仍可能拉不起来；提高保高度并进一步减少横向机动可帮助更快恢复
    if light_hardening_enabled and not hardening_enabled:
        low_z_guard_target = max(low_z_guard_target, 0.65)
    # L3：同样提高保高度，让 low-z recovery 时更快建立起升力
    if hardening_enabled:
        low_z_guard_target = max(low_z_guard_target, 0.85)
    low_z_recovery_remaining = 0
    # 离开低高度带后才允许再次进入恢复窗口；否则每帧 max 续期会导致 remaining 永不为 0，坠毁无法结束
    low_z_recovery_may_restart = True
    # L3：低空 recovery 若一直无法脱离（例如贴地/受力），允许最多再续航几轮
    low_z_recovery_restart_left = 2 if hardening_enabled else 0
    # L2/L3：主循环层面的“局部卡死”检测与短时脱困
    stuck_window = 72 if light_hardening_enabled and not hardening_enabled else 96   # 约 1.5s / 2.0s
    stuck_disp_thresh = 0.35 if light_hardening_enabled and not hardening_enabled else 0.28
    stuck_escape_steps = 30 if light_hardening_enabled and not hardening_enabled else 24
    stuck_escape_remaining = 0
    stuck_escape_cooldown_remaining = 0
    stuck_escape_cooldown_steps = 240 if light_hardening_enabled and not hardening_enabled else 120
    pos_window = deque(maxlen=stuck_window)
    escape_event_steps = deque(maxlen=20)
    macro_escape_remaining = 0
    macro_escape_target = None
    low_z_crash_frames = 0
    # L2 使用更宽容阈值；L3 从 1 帧放宽到 2 帧，减少临界点抖动造成的快速坠毁
    # L3：进一步放宽到 4 帧，减少临界下落的快速坠毁
    low_z_crash_frames_thresh = 8 if light_hardening_enabled and not hardening_enabled else 5
    # Episode 评估用（不改变控制逻辑）
    success = False
    episode_collision = False
    timeout = False
    termination_reason = "unknown"
    crash_pos = None
    last_i = 0
    last_t = 0.0
    last_pos = None
    print("\n=== 任务开始 ===")
    # 探索 overlay：在弹窗里用黄色点标出已探索区域（FREE），无 frontier 时多半是“已探完当前可见范围”但目标在未探到的地方）
    _explored_debug_id = None

    for i in range(1, steps + 1):
        obs, _, _, _, _ = env.step(action)
        t = i / env.CTRL_FREQ

        # 感知
        try:
            pkt = sensor.sense(obs=obs[0], step=i, t=t)
        except p.error:
            print("\n=== 物理引擎连接断开，提前结束仿真 ===")
            termination_reason = "physics_disconnect"
            break

        # 目标巡检计时
        try:
            target_manager.update(pkt)
        except p.error:
            print("\n=== 物理引擎连接断开，提前结束仿真 ===")
            termination_reason = "physics_disconnect"
            break
        episode_collision = episode_collision or bool(pkt.get("collision", False))
        last_i = i
        last_t = t
        last_pos = np.asarray(pkt["pos"], dtype=float).copy()
        if i % 2 == 0:
            trajectory.append(last_pos[:2].copy())

        # 物理扰动注入（L0_easy 时输出零力）
        try:
            disturbance.apply_physics_disturbance(drone_pos=pkt["pos"], apply_to_simulation=True)
        except p.error:
            pass

        # 状态机 + 避障 → 目标点
        state = State(xyz=pkt["pos"], vel=pkt["vel"], step=i, t=t)
        cmd = manager.update(state, pkt)

        # 在 PyBullet 窗口标出已探索区域（黄色 = 栅格 FREE，累积已探可通行区，非当前射线范围）。
        # 无 frontier 原因见下；目标发现需传感器本帧 LOS，与是否 FREE 无关。
        if SHOW_EXPLORED_OVERLAY and (i % EXPLORED_OVERLAY_INTERVAL == 0):
            if _explored_debug_id is not None:
                try:
                    p.removeUserDebugItem(_explored_debug_id, physicsClientId=PYB_CLIENT)
                except p.error:
                    pass
                _explored_debug_id = None
            g = grid.grid
            subsample = max(1, int(EXPLORED_OVERLAY_SUBSAMPLE))
            points = []
            for r in range(0, grid.height, subsample):
                for c in range(0, grid.width, subsample):
                    if int(g[r, c]) == FREE:
                        xy = grid.grid_to_world(r, c)
                        points.append([float(xy[0]), float(xy[1]), EXPLORED_OVERLAY_Z])
            if points:
                _explored_debug_id = p.addUserDebugPoints(
                    points,
                    pointColorsRGB=[[1.0, 1.0, 0.0]] * len(points),
                    pointSize=4.0,
                    lifeTime=0.0,
                    physicsClientId=PYB_CLIENT,
                )

        # 限制水平步长，防止 PID 过大倾斜（目标点离当前位置过远会导致大倾角）
        if not cmd.finished:
            # 避免任何阶段给出过低高度目标导致掉高
            cmd.target_pos[2] = max(float(cmd.target_pos[2]), CRUISE_HEIGHT)

            min_dist = float(pkt.get("min_dist", np.inf))
            collision = bool(pkt.get("collision", False))
            pos_window.append(np.asarray(pkt["pos"][:2], dtype=float).copy())
            if stuck_escape_cooldown_remaining > 0:
                stuck_escape_cooldown_remaining -= 1
            if (
                light_hardening_enabled
                and i > env.CTRL_FREQ * 5
                and len(pos_window) == pos_window.maxlen
                and stuck_escape_remaining <= 0
                and stuck_escape_cooldown_remaining <= 0
            ):
                disp = float(np.linalg.norm(pos_window[-1] - pos_window[0]))
                target_visible = bool(pkt.get("detected_target_ids", [])) or bool(pkt.get("targets_detected", []))
                obstacle_near = np.isfinite(min_dist) and (min_dist < 0.45)
                if disp < stuck_disp_thresh and (not target_visible) and obstacle_near:
                    stuck_escape_remaining = stuck_escape_steps
                    stuck_escape_cooldown_remaining = stuck_escape_cooldown_steps
                    escape_event_steps.append(i)
                    tag = "L3" if hardening_enabled else "L2"
                    print(f"[{tag}-escape] local stuck detected at t={t:.1f}s, disp={disp:.2f}m, steps={stuck_escape_remaining}")
                    # 若短时间内反复触发，执行一次更大的“侧向脱困”目标，跳出门口/障碍边缘极小值
                    recent_events = [s for s in escape_event_steps if (i - s) <= int(env.CTRL_FREQ * 12)]
                    if len(recent_events) >= 3:
                        ray_dists = np.asarray(pkt.get("ray_dists", []), dtype=float).reshape(-1)
                        ray_dirs_world = np.asarray(pkt.get("ray_dirs_world", []), dtype=float)
                        escape_xy = np.zeros(2, dtype=float)
                        if (
                            ray_dists.size > 0
                            and np.isfinite(ray_dists).any()
                            and ray_dirs_world.ndim == 2
                            and ray_dirs_world.shape[0] == ray_dists.size
                        ):
                            i_min = int(np.nanargmin(ray_dists))
                            nvec = np.asarray(ray_dirs_world[i_min, :2], dtype=float)
                            nn = float(np.linalg.norm(nvec))
                            if nn > 1e-9:
                                u = nvec / nn
                                side = np.array([-u[1], u[0]], dtype=float)
                                # 交替侧向方向，减少反复朝同一侧失败
                                if (len(recent_events) % 2) == 0:
                                    side = -side
                                escape_xy = 0.35 * (-u) + 0.90 * side
                        if float(np.linalg.norm(escape_xy)) < 1e-9:
                            to_home = np.asarray([0.5, 0.0], dtype=float) - np.asarray(pkt["pos"][:2], dtype=float)
                            nt = float(np.linalg.norm(to_home))
                            if nt > 1e-9:
                                escape_xy = to_home / nt * 0.8
                        if float(np.linalg.norm(escape_xy)) > 1e-9:
                            macro_escape_target = np.array(
                                [
                                    float(pkt["pos"][0] + escape_xy[0]),
                                    float(pkt["pos"][1] + escape_xy[1]),
                                    float(max(CRUISE_HEIGHT + 0.12, z_guard_target + 0.10)),
                                ],
                                dtype=float,
                            )
                            macro_escape_remaining = int(env.CTRL_FREQ * 2.0)
                            stuck_escape_cooldown_remaining = max(stuck_escape_cooldown_remaining, int(env.CTRL_FREQ * 8.0))
                            print(f"[{tag}-escape] macro escape armed → {macro_escape_target.round(2)}")

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

            # L2/L3：低高度或近障碍时进一步保守，优先保高度和稳定
            if light_hardening_enabled:
                current_z = float(pkt["pos"][2])
                if current_z < z_guard:
                    cmd.target_pos[2] = max(float(cmd.target_pos[2]), z_guard_target)
                    max_xy_step *= reduced_xy_step_scale
                # 目标接近/巡检阶段，L3 更激进；L2 轻量限制
                if (
                    bool(pkt.get("detected_target_ids", []))
                    or bool(pkt.get("targets_detected", []))
                    or float(np.linalg.norm(np.asarray(cmd.target_pos[:2], dtype=float) - np.asarray(pkt["pos"][:2], dtype=float))) < 1.2
                ):
                    cmd.target_pos[2] = max(float(cmd.target_pos[2]), z_guard_target)
                    max_xy_step = min(max_xy_step, 0.10 if hardening_enabled else 0.16)
                # HOLD/不可达附近容易抖动，近障碍再收紧一步长
                if min_dist < 0.35:
                    max_xy_step = min(max_xy_step, 0.18 if hardening_enabled else 0.22)
                # 局部卡死脱困：短时抬高并收敛横向机动，优先姿态稳定和脱离障碍边缘
                if stuck_escape_remaining > 0:
                    stuck_escape_remaining -= 1
                    cmd.target_pos[2] = max(float(cmd.target_pos[2]), z_guard_target + (0.10 if hardening_enabled else 0.08))
                    max_xy_step = min(max_xy_step, 0.12 if hardening_enabled else 0.16)
                if macro_escape_remaining > 0 and macro_escape_target is not None:
                    macro_escape_remaining -= 1
                    cmd.target_pos = macro_escape_target.copy()
                    max_xy_step = min(max_xy_step, 0.20 if hardening_enabled else 0.24)
                    if float(np.linalg.norm(np.asarray(pkt["pos"][:2], dtype=float) - np.asarray(macro_escape_target[:2], dtype=float))) < 0.25:
                        macro_escape_remaining = 0
                        macro_escape_target = None

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

        if light_hardening_enabled and z >= low_z_trigger + 0.03:
            low_z_recovery_may_restart = True

        if light_hardening_enabled and z < low_z_trigger and i > env.CTRL_FREQ:
            # 仅在「允许新一轮」且当前窗口已耗尽时启动恢复，避免每帧 max(remaining,12) 续期导致永远无法坠毁退出
            if low_z_recovery_remaining <= 0 and (
                low_z_recovery_may_restart or (hardening_enabled and low_z_recovery_restart_left > 0)
            ):
                is_restart = not low_z_recovery_may_restart
                low_z_recovery_remaining = low_z_recovery_steps
                low_z_recovery_may_restart = False
                if hardening_enabled and is_restart:
                    low_z_recovery_restart_left -= 1
                tag = "L3" if hardening_enabled else "L2"
                # 首次触发仍打印；续航重启为静默，避免刷屏
                if not is_restart:
                    print(
                        f"[{tag}-recovery] low-z trigger at t={t:.1f}s, z={z:.2f}, steps={low_z_recovery_remaining}"
                    )

        if light_hardening_enabled and low_z_recovery_remaining > 0 and not cmd.finished:
            # 保高度 + 极小横向步长，给姿态恢复时间
            low_z_recovery_remaining -= 1
            cmd.target_pos[2] = max(float(cmd.target_pos[2]), low_z_guard_target)
            # L3：强制把姿态（roll/pitch）拉平，避免低空阶段仍保持倾角导致持续下坠
            if hardening_enabled:
                cmd.target_rpy[0] = 0.0
                cmd.target_rpy[1] = 0.0
            dxy = np.asarray(cmd.target_pos[:2], dtype=float) - np.asarray(pkt["pos"][:2], dtype=float)
            nd = float(np.linalg.norm(dxy))
            # L2：低高度恢复窗口内进一步收紧横向机动，减少倾角导致的失高
            if light_hardening_enabled and not hardening_enabled:
                xy_cap = 0.05
            else:
                xy_cap = 0.03 if hardening_enabled else 0.10
            if nd > xy_cap:
                cmd.target_pos[:2] = np.asarray(pkt["pos"][:2], dtype=float) + dxy / nd * xy_cap
            if hardening_enabled:
                # L3：恢复阶段强制原地抬升，避免横向误差导致持续下坠
                cmd.target_pos[:2] = np.asarray(pkt["pos"][:2], dtype=float).copy()

        crashed = z < 0.08  # 高度低于 8cm 视为落地坠毁
        if light_hardening_enabled:
            if crashed:
                low_z_crash_frames += 1
            elif z > 0.12:
                low_z_crash_frames = 0
            crashed = low_z_crash_frames >= low_z_crash_frames_thresh

        if crashed and i > env.CTRL_FREQ and (not light_hardening_enabled or low_z_recovery_remaining <= 0):  # 跳过最初 1 秒的起飞抖动
            inspected, discovered, total = target_manager.get_progress()
            print(f"\n=== 无人机坠毁！t={t:.1f}s  pos=({pkt['pos'][0]:.2f}, {pkt['pos'][1]:.2f}, {z:.2f}) ===")
            print(f"巡检结果：{inspected}/{total} 个目标完成巡检")
            for r in target_manager.get_inspection_result():
                if r["inspected"]:
                    print(f"  目标 id={r['id']}  测得坐标(传感器)=[{r['measured_xyz'][0]:.3f}, {r['measured_xyz'][1]:.3f}, {r['measured_xyz'][2]:.3f}]")
            termination_reason = "crash"
            crash_pos = np.asarray(pkt["pos"], dtype=float).copy()
            break

        # 任务完成
        if cmd.finished:
            print(f"\n=== 任务完成！用时 {t:.1f}s ===")
            inspected, discovered, total = target_manager.get_progress()
            print(f"巡检结果：{inspected}/{total} 个目标完成巡检")
            for r in target_manager.get_inspection_result():
                if r["inspected"]:
                    print(f"  目标 id={r['id']}  测得坐标(传感器)=[{r['measured_xyz'][0]:.3f}, {r['measured_xyz'][1]:.3f}, {r['measured_xyz'][2]:.3f}]")
            termination_reason = "mission_complete"
            success = True
            break

        # PID 计算电机输出
        action[0, :] = pid.compute(
            control_timestep=env.CTRL_TIMESTEP,
            state=obs[0],
            target_pos=cmd.target_pos,
            target_rpy=cmd.target_rpy,
        )

        if CFG.get("gui_realtime", True):
            sync(i, START, env.CTRL_TIMESTEP)
        elif CFG.get("gui", True) and i % 10 == 0:
            time.sleep(0.002)  # GUI开启时给OpenGL喘息，防止Mac上渲染过载崩溃

    else:
        timeout = True
        termination_reason = "timeout"
        timeout_sec = float(steps) / float(env.CTRL_FREQ)
        print(f"\n=== 超时（{timeout_sec:.1f}s）===")
        inspected, discovered, total = target_manager.get_progress()
        print(f"巡检结果：{inspected}/{total} 个目标完成巡检")
        for r in target_manager.get_inspection_result():
            if r["inspected"]:
                print(f"  目标 id={r['id']}  测得坐标(传感器)=[{r['measured_xyz'][0]:.3f}, {r['measured_xyz'][1]:.3f}, {r['measured_xyz'][2]:.3f}]")

    inspected, discovered, total = target_manager.get_progress()
    targets_inspected = int(inspected)
    targets_total = int(total)
    step_count = int(last_i)
    flight_time_sec = float(last_t) if last_i > 0 else 0.0
    if last_pos is not None:
        final_pos = [float(last_pos[0]), float(last_pos[1]), float(last_pos[2])]
    else:
        final_pos = None
    if crash_pos is not None:
        crash_pos_out = [float(crash_pos[0]), float(crash_pos[1]), float(crash_pos[2])]
    else:
        crash_pos_out = None

    # collision：与 success 对齐——成功则 False；失败时若全程曾出现传感器 contact/极近障碍则为 True。
    # had_obstacle_contact：任一时间步曾报告接触（贴墙擦过也会 True），与是否成功无关，供细查。
    result = {
        "success": success,
        "collision": (not success) and episode_collision,
        "had_obstacle_contact": episode_collision,
        "timeout": timeout,
        "termination_reason": termination_reason,
        "targets_inspected": targets_inspected,
        "targets_total": targets_total,
        "steps": step_count,
        "flight_time_sec": flight_time_sec,
        "final_pos": final_pos,
        "crash_pos": crash_pos_out,
    }
    print("\n=== 评估结果 (result) ===")
    print(result)

    # 关闭仿真前收集真实坐标（env.close() 后 PyBullet 连接断开）
    obstacles_gt, nofly_gt, targets_gt = [], [], {}
    try:
        for oid in arena_handle.get("obstacle_ids", []):
            pos, _ = p.getBasePositionAndOrientation(oid, physicsClientId=PYB_CLIENT)
            obstacles_gt.append([float(pos[0]), float(pos[1])])
        for nid in arena_handle.get("nofly_ids", []):
            pos, _ = p.getBasePositionAndOrientation(nid, physicsClientId=PYB_CLIENT)
            nofly_gt.append([float(pos[0]), float(pos[1])])
        for tid in arena_handle.get("target_ids", []):
            pos, _ = p.getBasePositionAndOrientation(tid, physicsClientId=PYB_CLIENT)
            targets_gt[int(tid)] = [float(pos[0]), float(pos[1]), float(pos[2])]
    except p.error:
        pass

    try:
        env.close()
    except p.error:
        pass

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_path = os.path.join(str(CFG.get("output_folder", "results")), f"run_{ts}")

    _save_flight_data(
        base_path=base_path,
        trajectory=trajectory,
        grid=grid,
        target_manager=target_manager,
        targets_gt=targets_gt,
        obstacles_gt=obstacles_gt,
        nofly_gt=nofly_gt,
        home_pos=np.array(INIT_XYZS[0], dtype=float),
        result=result,
        layout=layout,
    )
    _plot_flight_result(
        base_path=base_path,
        trajectory=trajectory,
        grid=grid,
        target_manager=target_manager,
        targets_gt=targets_gt,
        obstacles_gt=obstacles_gt,
        nofly_gt=nofly_gt,
        home_pos=np.array(INIT_XYZS[0], dtype=float),
        result=result,
        layout=layout,
    )

    return result


if __name__ == "__main__":
    import argparse

    _profiles = list_difficulty_profiles()
    _parser = argparse.ArgumentParser(description="Indoor drone search simulation")
    _parser.add_argument(
        "-d", "--difficulty",
        choices=_profiles,
        help="Set difficulty and skip the startup picker",
    )
    _parser.add_argument(
        "--no-prompt",
        action="store_true",
        help="Skip picker; use difficulty_profile from config.py",
    )
    _args = _parser.parse_args()

    CFG["difficulty_profile"] = resolve_difficulty_profile(
        cli_difficulty=_args.difficulty,
        no_prompt=_args.no_prompt,
        prompt_enabled=bool(CFG.get("prompt_difficulty_at_start", True)),
        config_default=str(CFG.get("difficulty_profile", "L0_easy")),
    )
    print(f"[Difficulty] Running profile: {CFG['difficulty_profile']}")

    r = main()

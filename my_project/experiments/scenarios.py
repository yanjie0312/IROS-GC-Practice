# my_project/experiments/scenarios.py
# -*- coding: utf-8 -*-
"""
目标：
- 不改 sensors，不改 arena_handle 接口
- 同 seed 可复现，换 seed 会变化
- 扩展为“研究级配置对象”：支持扰动、噪声、控制器/规划器对比等
- 支持“正交设计”：地图复杂度（easy/med/hard）与退化强度（l0~l3）可独立组合

使用方式（全部兼容）：
1) 旧用法（仍然支持）：
   scenario = make_scenario("med", seed=123)
   scenario = make_scenario("l2",  seed=123)

2) 正交用法（推荐用于研究实验）：
   scenario = make_scenario(map_level="med", deg_level="l2", seed=123)

说明：
- map_level 控制：户型/障碍密度/目标数量/no-fly（结构复杂度）
- deg_level 控制：风扰/噪声/丢包/延迟/物理退化（退化强度）
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional


# ============================================================
# 1) Scenario 数据结构（这是“配置接口”）
# ============================================================

@dataclass
class Scenario:
    # ========================
    # 基本信息
    # ========================
    name: str                     # 场景名称（可写 "med+l2" 这种组合名，方便日志）
    seed: int                     # 随机种子（保证可复现）

    # 户型名：由 build_arena.get_layout(layout_name) 使用
    layout_name: str = "apt_2room1hall"

    # 起飞点（给上层/调试用，不影响 sensors）
    spawn_xy: Tuple[float, float] = (0.0, 0.0)

    # 物体生成时抬高一点，避免穿模
    spawn_z: float = 0.05

    # 墙参数
    wall_height: float = 1.0

    # ========================
    # 几何复杂度（Geometry）——主要被 build_arena.py 使用
    # ========================
    num_obstacles: int = 0
    obstacle_min_size: float = 0.20
    obstacle_max_size: float = 0.50
    obstacle_min_height: float = 0.20
    obstacle_max_height: float = 0.80

    num_nofly: int = 0
    nofly_min_size: float = 0.40
    nofly_max_size: float = 1.00
    nofly_is_solid: bool = True

    num_targets: int = 1
    target_radius: float = 0.07
    target_height: float = 0.14

    # 采样最大尝试次数（防止密度太高时死循环）
    max_sample_tries: int = 3000

    # ========================
    # 扰动相关（Disturbance）——主要被 env/disturbances.py 使用
    # ========================
    wind_std: float = 0.0                              # 风力随机扰动标准差（越大越难）
    wind_bias_xy: Tuple[float, float] = (0.0, 0.0)     # 持续风偏置（x,y）
    gust_prob: float = 0.0                             # 阵风触发概率（每 step）
    gust_strength: float = 0.0                         # 阵风强度（额外叠加的 force 幅度）

    physics_mode: str = "nominal"                      # 物理退化模式：nominal / drag / ground_effect / downwash（可选扩展）
    payload_mass_delta: float = 0.0                    # 载荷变化（质量偏移，kg 或相对量，看你们实现）

    # ========================
    # 传感器不确定性（Sensor Uncertainty）——主要被 env/sensors.py 使用
    # ========================
    pos_noise_std: float = 0.0                         # 位置噪声标准差（m）
    yaw_noise_std: float = 0.0                         # 偏航角噪声标准差（rad）
    ray_noise_std: float = 0.0                         # raycast 测距噪声标准差（m）

    dropout_prob: float = 0.0                          # 观测丢包概率（0~1）
    delay_steps: int = 0                               # 观测延迟步数（0 表示无延迟）

    # ========================
    # 实验控制参数 —— 主要被 main.py / metrics.py 使用
    # ========================
    timeout_steps: int = 2000                          # 最大仿真步数（超过视为 fail/timeout）

    # ========================
    # 对比实验开关 —— 主要被 main.py 使用
    # ========================
    controller_type: str = "PID"                       # PID / MRAC / ADAPTIVE（你们 main.py 里切换）
    planner_type: str = "POTENTIAL"                    # POTENTIAL / GLOBAL_LOCAL（你们 planning 里切换）


# ============================================================
# 2) 正交设计：把“地图复杂度”和“退化强度”拆开
# ============================================================

def _apply_map_level(base: Scenario, map_level: str) -> Scenario:
    """根据 map_level 设置户型与几何复杂度参数（结构复杂度轴）"""
    ml = map_level.lower().strip()

    if ml == "easy":
        # 两室一厅 + 少障碍 + 少目标
        base.layout_name = "apt_2room1hall"
        base.num_obstacles = 4
        base.num_nofly = 0
        base.num_targets = 2
        return base

    if ml == "med":
        # 三室两厅 + 中等障碍 + 适中目标 + 少量 no-fly
        base.layout_name = "apt_3room2hall"
        base.num_obstacles = 8
        base.num_nofly = 1
        base.num_targets = 3
        return base

    if ml == "hard":
        # 三室两厅 + 高障碍 + 更多目标 + 更多 no-fly
        base.layout_name = "apt_3room2hall"
        base.num_obstacles = 12
        base.num_nofly = 2
        base.num_targets = 4
        return base

    raise ValueError(f"Unknown map_level: {map_level}")


def _apply_deg_level(base: Scenario, deg_level: str) -> Scenario:
    """根据 deg_level 设置扰动/噪声等（退化强度轴）"""
    dl = deg_level.lower().strip()

    # L0：理想环境（无噪声、无风）
    if dl == "l0":
        base.wind_std = 0.0
        base.wind_bias_xy = (0.0, 0.0)
        base.gust_prob = 0.0
        base.gust_strength = 0.0
        base.physics_mode = "nominal"
        base.payload_mass_delta = 0.0

        base.pos_noise_std = 0.0
        base.yaw_noise_std = 0.0
        base.ray_noise_std = 0.0
        base.dropout_prob = 0.0
        base.delay_steps = 0
        return base

    # L1：轻微退化
    if dl == "l1":
        base.wind_std = 0.10
        base.wind_bias_xy = (0.03, 0.00)    # 轻微持续风（可改）
        base.gust_prob = 0.00
        base.gust_strength = 0.00
        base.physics_mode = "nominal"

        base.pos_noise_std = 0.02
        base.yaw_noise_std = 0.02
        base.ray_noise_std = 0.05
        base.dropout_prob = 0.00
        base.delay_steps = 0
        return base

    # L2：中等退化（阵风 + 噪声 + 少量丢包）
    if dl == "l2":
        base.wind_std = 0.20
        base.wind_bias_xy = (0.05, 0.00)
        base.gust_prob = 0.10
        base.gust_strength = 0.30
        base.physics_mode = "drag"          # 可选：让 L2 开始启用 drag

        base.pos_noise_std = 0.05
        base.yaw_noise_std = 0.05
        base.ray_noise_std = 0.10
        base.dropout_prob = 0.05
        base.delay_steps = 0
        return base

    # L3：强退化（强阵风 + 强噪声 + 丢包 + 延迟）
    if dl == "l3":
        base.wind_std = 0.30
        base.wind_bias_xy = (0.08, 0.02)
        base.gust_prob = 0.20
        base.gust_strength = 0.50
        base.physics_mode = "ground_effect"  # 或 "drag"，按你们实验需求选

        base.pos_noise_std = 0.10
        base.yaw_noise_std = 0.10
        base.ray_noise_std = 0.15
        base.dropout_prob = 0.10
        base.delay_steps = 2
        base.timeout_steps = 2500            # 可适当放宽超时
        return base

    raise ValueError(f"Unknown deg_level: {deg_level}")


# ============================================================
# 3) 场景工厂函数（兼容旧用法 + 支持正交）
# ============================================================

def make_scenario(
    name: str = "easy",
    seed: int = 0,
    map_level: Optional[str] = None,
    deg_level: Optional[str] = None,
) -> Scenario:
    """
    兼容两种调用方式：

    A) 旧方式（只传 name）：
       make_scenario("easy", seed)
       make_scenario("med",  seed)
       make_scenario("hard", seed)
       make_scenario("l2",   seed)   # 也能用，但注意它会使用默认 map_level（见下）

    B) 正交方式（推荐）：
       make_scenario(map_level="med", deg_level="l2", seed=123)

    注意：
    - 如果只传 name="l2"，我们会默认 map_level="med"（你也可以按需改默认）
    """
    n = name.lower().strip()

    # -------------------------
    # 情况 1：显式正交输入（最推荐）
    # -------------------------
    if map_level is not None or deg_level is not None:
        ml = (map_level or "med").lower().strip()
        dl = (deg_level or "l0").lower().strip()

        sc = Scenario(name=f"{ml}+{dl}", seed=seed)
        sc = _apply_map_level(sc, ml)
        sc = _apply_deg_level(sc, dl)
        return sc

    # -------------------------
    # 情况 2：旧 name 用法（easy/med/hard）
    # -------------------------
    if n in ("easy", "med", "hard"):
        sc = Scenario(name=n, seed=seed)
        sc = _apply_map_level(sc, n)
        # 旧场景默认无退化（相当于 l0）
        sc = _apply_deg_level(sc, "l0")
        return sc

    # -------------------------
    # 情况 3：只给退化等级（l0~l3）
    #         默认搭配一个中等地图（med），避免 l2/l3 用 easy 地图太简单
    # -------------------------
    if n in ("l0", "l1", "l2", "l3"):
        sc = Scenario(name=f"med+{n}", seed=seed)
        sc = _apply_map_level(sc, "med")
        sc = _apply_deg_level(sc, n)
        return sc

    raise ValueError(f"Unknown scenario name: {name}")

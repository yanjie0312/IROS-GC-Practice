# my_project/experiments/scenarios.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple


DEFAULT_TIMEOUT_STEPS = 300 * 48  # Keep baseline mission budget unchanged.


@dataclass
class Scenario:
    # Basic metadata
    name: str
    seed: int
    difficulty: str = "L0_easy"

    # Layout / geometry complexity
    layout_name: str = "apt_2room1hall"
    spawn_xy: Tuple[float, float] = (0.0, 0.0)
    spawn_z: float = 0.05
    wall_height: float = 1.0

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
    max_sample_tries: int = 3000

    # Physics disturbance
    wind_std: float = 0.0
    wind_bias_xy: Tuple[float, float] = (0.0, 0.0)
    gust_prob: float = 0.0
    gust_strength: float = 0.0
    physics_mode: str = "nominal"
    payload_mass_delta: float = 0.0

    # Sensor/communication degradation
    pos_noise_std: float = 0.0
    yaw_noise_std: float = 0.0
    ray_noise_std: float = 0.0
    target_pos_noise_std: float = 0.0
    target_pos_bias: float = 0.0
    target_range_noise_std: float = 0.0
    target_bearing_noise_std: float = 0.0
    target_range_bias: float = 0.0
    target_false_negative_prob: float = 0.0
    dropout_prob: float = 0.0
    delay_steps: int = 0

    # Mission constraint
    timeout_steps: int = DEFAULT_TIMEOUT_STEPS

    # Experiment switches
    controller_type: str = "PID"
    planner_type: str = "POTENTIAL"


# Explicit proposal-aligned profiles.
# Each level includes wind, noise, delay, dropout, payload, timeout,
# plus obstacle/layout complexity.
PROFILE_LIBRARY: Dict[str, Dict] = {
    "L0_easy": {
        "layout_name": "apt_2room1hall",
        "num_obstacles": 4,
        "num_nofly": 0,
        "num_targets": 2,
        "wind_std": 0.0,
        "wind_bias_xy": (0.0, 0.0),
        "gust_prob": 0.0,
        "gust_strength": 0.0,
        "physics_mode": "nominal",
        "pos_noise_std": 0.0,
        "yaw_noise_std": 0.0,
        "ray_noise_std": 0.0,
        "target_pos_noise_std": 0.0,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.0,
        "target_bearing_noise_std": 0.0,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.0,
        "dropout_prob": 0.0,
        "delay_steps": 0,
        "timeout_steps": DEFAULT_TIMEOUT_STEPS,
    },
    "L1_mild": {
        "layout_name": "apt_2room1hall",
        "num_obstacles": 6,
        "num_nofly": 0,
        "num_targets": 2,
        "wind_std": 0.08,
        "wind_bias_xy": (0.02, 0.00),
        "gust_prob": 0.03,
        "gust_strength": 0.10,
        "physics_mode": "nominal",
        "pos_noise_std": 0.01,
        "yaw_noise_std": 0.01,
        "ray_noise_std": 0.01,
        "target_pos_noise_std": 0.01,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.01,
        "target_bearing_noise_std": 0.01,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.02,
        "dropout_prob": 0.01,
        "delay_steps": 0,
        "timeout_steps": int(DEFAULT_TIMEOUT_STEPS * 0.95),
    },
    "L2_medium": {
        "layout_name": "apt_3room2hall",
        "num_obstacles": 8,
        "num_nofly": 1,
        "num_targets": 3,
        "wind_std": 0.10,
        "wind_bias_xy": (0.03, 0.01),
        "gust_prob": 0.04,
        "gust_strength": 0.12,
        "physics_mode": "drag",
        "pos_noise_std": 0.01,
        "yaw_noise_std": 0.01,
        "ray_noise_std": 0.01,
        "target_pos_noise_std": 0.01,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.01,
        "target_bearing_noise_std": 0.01,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.02,
        "dropout_prob": 0.01,
        "delay_steps": 1,
        "timeout_steps": int(DEFAULT_TIMEOUT_STEPS * 0.90),
    },
    "L3_hard": {
        "layout_name": "apt_3room2hall",
        "num_obstacles": 12,
        "num_nofly": 2,
        "num_targets": 4,
        "wind_std": 0.15,
        "wind_bias_xy": (0.04, 0.01),
        "gust_prob": 0.06,
        "gust_strength": 0.15,
        "physics_mode": "ground_effect",
        "pos_noise_std": 0.01,
        "yaw_noise_std": 0.01,
        "ray_noise_std": 0.01,
        "target_pos_noise_std": 0.01,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.01,
        "target_bearing_noise_std": 0.01,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.02,
        "dropout_prob": 0.01,
        "delay_steps": 1,
        "timeout_steps": int(DEFAULT_TIMEOUT_STEPS * 0.80),
    },
}


MAP_LEVEL_LIBRARY: Dict[str, Dict] = {
    "easy": {"layout_name": "apt_2room1hall", "num_obstacles": 4, "num_nofly": 0, "num_targets": 2},
    "med": {"layout_name": "apt_3room2hall", "num_obstacles": 8, "num_nofly": 1, "num_targets": 3},
    "hard": {"layout_name": "apt_3room2hall", "num_obstacles": 12, "num_nofly": 2, "num_targets": 4},
}


DEG_LEVEL_LIBRARY: Dict[str, Dict] = {
    "l0": {
        "wind_std": 0.0,
        "wind_bias_xy": (0.0, 0.0),
        "gust_prob": 0.0,
        "gust_strength": 0.0,
        "physics_mode": "nominal",
        "pos_noise_std": 0.0,
        "yaw_noise_std": 0.0,
        "ray_noise_std": 0.0,
        "target_pos_noise_std": 0.0,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.0,
        "target_bearing_noise_std": 0.0,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.0,
        "dropout_prob": 0.0,
        "delay_steps": 0,
        "payload_mass_delta": 0.0,
        "timeout_steps": DEFAULT_TIMEOUT_STEPS,
    },
    "l1": {
        "wind_std": 0.08,
        "wind_bias_xy": (0.02, 0.00),
        "gust_prob": 0.03,
        "gust_strength": 0.10,
        "physics_mode": "nominal",
        "pos_noise_std": 0.01,
        "yaw_noise_std": 0.01,
        "ray_noise_std": 0.03,
        "target_pos_noise_std": 0.01,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.01,
        "target_bearing_noise_std": 0.01,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.02,
        "dropout_prob": 0.01,
        "delay_steps": 0,
        "payload_mass_delta": 0.0015,
        "timeout_steps": int(DEFAULT_TIMEOUT_STEPS * 0.95),
    },
    "l2": {
        "wind_std": 0.18,
        "wind_bias_xy": (0.04, 0.01),
        "gust_prob": 0.10,
        "gust_strength": 0.25,
        "physics_mode": "drag",
        "pos_noise_std": 0.04,
        "yaw_noise_std": 0.04,
        "ray_noise_std": 0.08,
        "target_pos_noise_std": 0.03,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.03,
        "target_bearing_noise_std": 0.03,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.05,
        "dropout_prob": 0.05,
        "delay_steps": 1,
        "payload_mass_delta": 0.0030,
        "timeout_steps": int(DEFAULT_TIMEOUT_STEPS * 0.90),
    },
    "l3": {
        "wind_std": 0.30,
        "wind_bias_xy": (0.08, 0.02),
        "gust_prob": 0.20,
        "gust_strength": 0.45,
        "physics_mode": "ground_effect",
        "pos_noise_std": 0.08,
        "yaw_noise_std": 0.08,
        "ray_noise_std": 0.14,
        "target_pos_noise_std": 0.06,
        "target_pos_bias": 0.0,
        "target_range_noise_std": 0.06,
        "target_bearing_noise_std": 0.05,
        "target_range_bias": 0.0,
        "target_false_negative_prob": 0.10,
        "dropout_prob": 0.10,
        "delay_steps": 2,
        "payload_mass_delta": 0.0050,
        "timeout_steps": int(DEFAULT_TIMEOUT_STEPS * 0.80),
    },
}


def list_difficulty_profiles() -> Tuple[str, ...]:
    return tuple(PROFILE_LIBRARY.keys())


def _make_from_profile(profile_name: str, seed: int) -> Scenario:
    if profile_name not in PROFILE_LIBRARY:
        raise ValueError(f"Unknown profile: {profile_name}")
    scenario = Scenario(name=profile_name, seed=int(seed), difficulty=profile_name)
    for k, v in PROFILE_LIBRARY[profile_name].items():
        setattr(scenario, k, v)
    return scenario


def _make_from_axes(map_level: str, deg_level: str, seed: int) -> Scenario:
    ml = map_level.lower().strip()
    dl = deg_level.lower().strip()
    if ml not in MAP_LEVEL_LIBRARY:
        raise ValueError(f"Unknown map_level: {map_level}")
    if dl not in DEG_LEVEL_LIBRARY:
        raise ValueError(f"Unknown deg_level: {deg_level}")

    scenario = Scenario(name=f"{ml}+{dl}", seed=int(seed), difficulty=f"{ml}+{dl}")
    for k, v in MAP_LEVEL_LIBRARY[ml].items():
        setattr(scenario, k, v)
    for k, v in DEG_LEVEL_LIBRARY[dl].items():
        setattr(scenario, k, v)
    return scenario


def make_scenario(
    name: str = "L0_easy",
    seed: int = 0,
    map_level: Optional[str] = None,
    deg_level: Optional[str] = None,
) -> Scenario:
    """
    Supported usages:
    1) make_scenario("L0_easy", seed=42)
    2) make_scenario(map_level="med", deg_level="l2", seed=42)
    3) Backward compatible:
       - "easy"/"med"/"hard"
       - "l0"/"l1"/"l2"/"l3"
    """
    if map_level is not None or deg_level is not None:
        ml = (map_level or "easy").lower().strip()
        dl = (deg_level or "l0").lower().strip()
        return _make_from_axes(ml, dl, seed)

    raw = str(name).strip()
    if raw in PROFILE_LIBRARY:
        return _make_from_profile(raw, seed)

    n = raw.lower()
    if n in MAP_LEVEL_LIBRARY:
        # Legacy map-only call means map complexity with no degradation.
        return _make_from_axes(n, "l0", seed)
    if n in DEG_LEVEL_LIBRARY:
        # Legacy degradation-only call defaults to medium map complexity.
        return _make_from_axes("med", n, seed)

    raise ValueError(
        f"Unknown scenario name: {name}. Supported profiles: {', '.join(PROFILE_LIBRARY.keys())}"
    )
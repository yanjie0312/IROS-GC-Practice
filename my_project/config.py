# my_project/config.py
from gym_pybullet_drones.utils.enums import DroneModel, Physics

CFG = {
    "drone": DroneModel("cf2x"),
    "num_drones": 1,
    "physics": Physics("pyb"),
    "gui": True,
    "record_video": False,
    "plot": False,
    "user_debug_gui": False,

    "simulation_freq_hz": 240,
    "control_freq_hz": 48,
    "duration_sec": 60,
    "output_folder": "results",
    # Difficulty profile switch (primary runtime knob).
    # Supported: L0_easy, L1_mild, L2_medium, L3_hard.
    "difficulty_profile": "L0_easy",
    "scenario_seed": 42,

    # 场地/墙参数
    "arena_half_size": 1.5,
    "wall_thickness": 0.005,
    "wall_height": 0.8,

    # 飞行高度
    "flight_height": 0.3,

    # 日志/渲染降频
    "print_every_sec": 1.0,
    "render_every_sec": 1.0,

    # Disturbance knobs (scenario profile values are the default source of truth).
    # Keep these at zero / False to preserve baseline behavior in L0_easy.
    "disturbance": {
        # Physics disturbance
        "enabled": True,
        "wind_std": 0.0,
        "wind_bias_xy": (0.0, 0.0),
        "gust_prob": 0.0,
        "gust_strength": 0.0,
        "external_force_std": 0.0,
        "external_force_bias": (0.0, 0.0, 0.0),

        # Sensor uncertainty (optional extra corruption in DisturbanceInjector)
        "state_noise_std_pos": (0.0, 0.0, 0.0),
        "state_noise_std_rpy": (0.0, 0.0, 0.0),
        "state_noise_std_vel": (0.0, 0.0, 0.0),
        "ray_noise_std": 0.0,
        "ray_bias": 0.0,

        # Communication degradation
        "measurement_delay_steps": 0,
        "packet_dropout_prob": 0.0,

        # Payload / mass variation
        "payload_mass_delta": 0.0,
    },
}

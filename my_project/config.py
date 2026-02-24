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

    # 场地/墙参数
    "arena_half_size": 1.5,
    "wall_thickness": 0.005,
    "wall_height": 0.8,

    # 飞行高度
    "flight_height": 0.3,

    # 日志/渲染降频
    "print_every_sec": 1.0,
    "render_every_sec": 1.0,
}

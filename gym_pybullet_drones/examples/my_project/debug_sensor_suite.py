from __future__ import annotations

import math
import traceback

import numpy as np
import pybullet as p

from gym_pybullet_drones.envs.CtrlAviary import CtrlAviary
from gym_pybullet_drones.utils.enums import DroneModel, Physics

from my_project.env.build_arena import create_four_walls
from my_project.env.sensors import SensorSuite


def _create_target(pyb_client_id: int, pos=(0.8, 0.0, 0.3), radius: float = 0.08) -> int:
    col = p.createCollisionShape(
        p.GEOM_SPHERE,
        radius=radius,
        physicsClientId=pyb_client_id,
    )
    vis = p.createVisualShape(
        p.GEOM_SPHERE,
        radius=radius,
        rgbaColor=[1.0, 0.2, 0.2, 1.0],
        physicsClientId=pyb_client_id,
    )
    tid = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=col,
        baseVisualShapeIndex=vis,
        basePosition=list(pos),
        physicsClientId=pyb_client_id,
    )
    return int(tid)


def run_sensor_smoke_test() -> None:
    init_xyzs = np.array([[0.0, 0.0, 0.3]], dtype=float)
    init_rpys = np.array([[0.0, 0.0, 0.0]], dtype=float)

    env = CtrlAviary(
        drone_model=DroneModel("cf2x"),
        num_drones=1,
        initial_xyzs=init_xyzs,
        initial_rpys=init_rpys,
        physics=Physics("pyb"),
        pyb_freq=240,
        ctrl_freq=48,
        gui=False,
        record=False,
        obstacles=False,
        user_debug_gui=False,
    )

    try:
        pyb_client = env.getPyBulletClient()
        drone_id = int(env.getDroneIds()[0])

        wall_ids = create_four_walls(
            pyb_client_id=pyb_client,
            half_size=1.5,
            wall_thickness=0.005,
            wall_height=0.8,
        )
        target_id = _create_target(pyb_client)

        arena_handle = {
            "wall_ids": list(map(int, wall_ids)),
            "obstacle_ids": [],
            "nofly_ids": [],
            "target_ids": [target_id],
        }

        sensors = SensorSuite(
            pyb_client_id=pyb_client,
            drone_id=drone_id,
            arena_handle=arena_handle,
            target_mode="hybrid",
            enable_debug_rays=False,
        )

        action = np.zeros((1, 4), dtype=float)
        hover = float(env.HOVER_RPM)
        action[0, :] = hover

        obs = None
        for _ in range(20):
            obs, _, _, _, _ = env.step(action)

        assert obs is not None, "env.step did not return observation"
        packet = sensors.sense(obs=obs[0], step=20, t=20.0 / env.CTRL_FREQ)

        required_keys = [
            "pos",
            "quat",
            "rpy",
            "vel",
            "ang_vel",
            "ray_dists",
            "ray_dirs_body",
            "ray_hit_ids",
            "min_dist",
            "collision",
            "targets_detected",
            "detected_target_ids",
        ]
        for k in required_keys:
            assert k in packet, f"missing key in sensor packet: {k}"

        ray_dists = packet["ray_dists"]
        ray_dirs = packet["ray_dirs_body"]
        ray_hits = packet["ray_hit_ids"]

        assert isinstance(ray_dists, np.ndarray) and ray_dists.ndim == 1, "ray_dists must be 1D ndarray"
        assert isinstance(ray_dirs, np.ndarray) and ray_dirs.ndim == 2 and ray_dirs.shape[1] == 3, "ray_dirs_body must be (K,3)"
        assert ray_dists.shape[0] == ray_dirs.shape[0], "ray_dists length must match ray_dirs_body"
        assert ray_hits.shape[0] == ray_dirs.shape[0], "ray_hit_ids length must match ray_dirs_body"
        assert np.all(ray_dists >= 0.0), "ray distances must be non-negative"

        wall_hit_ok = any(int(h) in set(map(int, wall_ids)) for h in ray_hits.tolist() if int(h) >= 0)
        assert wall_hit_ok, "expected at least one ray to hit a wall"

        assert target_id in packet["detected_target_ids"], (
            "expected target to be detected in forward-facing setup"
        )

        pos, _ = p.getBasePositionAndOrientation(drone_id, physicsClientId=pyb_client)
        quat_yaw_pi = p.getQuaternionFromEuler([0.0, 0.0, math.pi])
        p.resetBasePositionAndOrientation(
            drone_id,
            posObj=pos,
            ornObj=quat_yaw_pi,
            physicsClientId=pyb_client,
        )

        packet_back = sensors.sense(obs=None)
        assert target_id not in packet_back["detected_target_ids"], (
            "expected target not to be detected when drone faces opposite direction"
        )

        print("PASS: SensorSuite smoke test passed")
        print(f"  rays={ray_dists.shape[0]}, min_dist={packet['min_dist']:.3f}")
        print(f"  detected_target_ids(front)={packet['detected_target_ids']}")
        print(f"  detected_target_ids(back)={packet_back['detected_target_ids']}")

    finally:
        env.close()


if __name__ == "__main__":
    try:
        run_sensor_smoke_test()
    except AssertionError as exc:
        print(f"FAIL: {exc}")
        raise
    except Exception:
        print("FAIL: unexpected exception")
        traceback.print_exc()
        raise

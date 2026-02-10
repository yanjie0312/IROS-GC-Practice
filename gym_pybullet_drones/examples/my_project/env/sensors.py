# my_project/env/sensors.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pybullet as p


def parse_ctrlaviary_obs(obs: np.ndarray) -> Dict[str, np.ndarray]:
    """Parse CtrlAviary observation (shape: [20])."""
    arr = np.asarray(obs, dtype=float).reshape(-1)
    if arr.size < 16:
        raise ValueError(f"Expected obs with >=16 elements, got {arr.size}")
    return {
        "pos": arr[0:3].copy(),
        "quat": arr[3:7].copy(),
        "rpy": arr[7:10].copy(),
        "vel": arr[10:13].copy(),
        "ang_vel": arr[13:16].copy(),
    }


def build_default_ray_dirs() -> np.ndarray:
    """
    Body-frame unit directions for obstacle sensing.
    16 horizontal + 6 tilted rays.
    """
    dirs: List[np.ndarray] = []

    # 16 rays on horizontal plane
    for deg in range(0, 360, 22):
        rad = np.deg2rad(deg)
        dirs.append(np.array([np.cos(rad), np.sin(rad), 0.0], dtype=float))

    # Tilted rays (up/down, front/left/back/right)
    for sign in (-1.0, 1.0):
        dirs.append(np.array([1.0, 0.0, 0.40 * sign], dtype=float))
        dirs.append(np.array([0.0, 1.0, 0.40 * sign], dtype=float))
        dirs.append(np.array([-1.0, 0.0, 0.40 * sign], dtype=float))

    ray_dirs = np.vstack(dirs)
    norms = np.linalg.norm(ray_dirs, axis=1, keepdims=True) + 1e-9
    return ray_dirs / norms


@dataclass
class SelfState:
    pos: np.ndarray
    quat: np.ndarray
    rpy: np.ndarray
    vel: np.ndarray
    ang_vel: np.ndarray


class SensorSuite:
    """
    Unified perception layer:
    - self state
    - obstacle rays
    - collision / closest distance
    - target detection (GT / camera segmentation / hybrid)
    """

    def __init__(
        self,
        pyb_client_id: int,
        drone_id: int,
        arena_handle: Optional[Any] = None,
        ray_dirs_body: Optional[np.ndarray] = None,
        ray_length: float = 2.5,
        ray_start_offset: float = 0.06,
        closest_query_distance: float = 0.8,
        collision_distance: float = 0.02,
        target_mode: str = "hybrid",  # "gt" | "camera" | "hybrid"
        target_detect_range: float = 2.5,
        target_fov_deg: float = 100.0,
        cam_width: int = 96,
        cam_height: int = 96,
        cam_fov_deg: float = 90.0,
        cam_near: float = 0.03,
        cam_far: float = 5.0,
        enable_debug_rays: bool = False,
    ):
        self.client = int(pyb_client_id)
        self.drone_id = int(drone_id)

        if target_mode not in ("gt", "camera", "hybrid"):
            raise ValueError(f"Invalid target_mode={target_mode}")

        self.ray_dirs_body = (
            build_default_ray_dirs() if ray_dirs_body is None else np.asarray(ray_dirs_body, dtype=float)
        )
        if self.ray_dirs_body.ndim != 2 or self.ray_dirs_body.shape[1] != 3:
            raise ValueError("ray_dirs_body must be shape (K,3)")
        norms = np.linalg.norm(self.ray_dirs_body, axis=1, keepdims=True) + 1e-9
        self.ray_dirs_body = self.ray_dirs_body / norms

        self.ray_length = float(ray_length)
        self.ray_start_offset = float(ray_start_offset)
        self.closest_query_distance = float(closest_query_distance)
        self.collision_distance = float(collision_distance)

        self.target_mode = target_mode
        self.target_detect_range = float(target_detect_range)
        self.target_fov_rad = np.deg2rad(float(target_fov_deg))

        self.cam_width = int(cam_width)
        self.cam_height = int(cam_height)
        self.cam_fov_deg = float(cam_fov_deg)
        self.cam_near = float(cam_near)
        self.cam_far = float(cam_far)

        self.enable_debug_rays = bool(enable_debug_rays)
        self._ray_debug_line_ids: List[int] = [-1] * len(self.ray_dirs_body)

        self.wall_ids: Set[int] = set()
        self.obstacle_ids: Set[int] = set()
        self.nofly_ids: Set[int] = set()
        self.target_ids: Set[int] = set()
        self.ray_blocker_ids: Set[int] = set()
        self.los_blocker_ids: Set[int] = set()

        if arena_handle is not None:
            self.update_arena_handle(arena_handle)

    def update_arena_handle(self, arena_handle: Any) -> None:
        """
        Accepts:
        - dict-like handle
        - object with attributes
        - list/set/tuple (interpreted as walls)
        """
        if arena_handle is None:
            return

        if isinstance(arena_handle, (list, tuple, set, np.ndarray)):
            walls = {int(x) for x in arena_handle}
            self.wall_ids = walls
            self.obstacle_ids = set()
            self.nofly_ids = set()
            self.target_ids = set()
            self.ray_blocker_ids = set(walls)
            self.los_blocker_ids = set(walls)
            return

        getter = arena_handle.get if isinstance(arena_handle, dict) else lambda k, d=None: getattr(arena_handle, k, d)

        self.wall_ids = {int(x) for x in getter("wall_ids", [])}
        self.obstacle_ids = {int(x) for x in getter("obstacle_ids", [])}
        self.nofly_ids = {int(x) for x in getter("nofly_ids", [])}
        self.target_ids = {int(x) for x in getter("target_ids", [])}

        default_blockers = self.wall_ids | self.obstacle_ids | self.nofly_ids
        self.ray_blocker_ids = {int(x) for x in getter("ray_blocker_ids", list(default_blockers))}
        self.los_blocker_ids = {int(x) for x in getter("los_blocker_ids", list(default_blockers))}

    def sense(self, obs: Optional[np.ndarray] = None, step: Optional[int] = None, t: Optional[float] = None) -> Dict[str, Any]:
        state = self._read_state(obs)
        rays = self._sense_rays(state)
        collision = self._sense_collision()
        targets_gt, targets_detected = self._sense_targets(state)

        packet: Dict[str, Any] = {
            "step": step,
            "t": t,
            "pos": state.pos,
            "quat": state.quat,
            "rpy": state.rpy,
            "vel": state.vel,
            "ang_vel": state.ang_vel,
            "ray_dists": rays["ray_dists"],
            "ray_dirs_body": self.ray_dirs_body.copy(),
            "ray_hit_ids": rays["ray_hit_ids"],
            "ray_hit_points": rays["ray_hit_points"],
            "min_dist": float(rays["min_dist"]),
            "min_dir_body": rays["min_dir_body"],
            "collision": bool(collision["collision"]),
            "contact_body_ids": collision["contact_body_ids"],
            "closest_obstacle_dist": float(collision["closest_obstacle_dist"]),
            "targets_gt": targets_gt,
            "targets_detected": targets_detected,
            "detected_target_ids": [item["target_id"] for item in targets_detected],
        }

        if self.enable_debug_rays:
            self._draw_debug_rays(rays["ray_from"], rays["ray_to"], rays["ray_dists"])

        return packet

    def _read_state(self, obs: Optional[np.ndarray]) -> SelfState:
        if obs is not None:
            s = parse_ctrlaviary_obs(obs)
            return SelfState(
                pos=s["pos"],
                quat=s["quat"],
                rpy=s["rpy"],
                vel=s["vel"],
                ang_vel=s["ang_vel"],
            )

        pos, quat = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
        vel, ang_vel = p.getBaseVelocity(self.drone_id, physicsClientId=self.client)
        rpy = p.getEulerFromQuaternion(quat)
        return SelfState(
            pos=np.asarray(pos, dtype=float),
            quat=np.asarray(quat, dtype=float),
            rpy=np.asarray(rpy, dtype=float),
            vel=np.asarray(vel, dtype=float),
            ang_vel=np.asarray(ang_vel, dtype=float),
        )

    def _sense_rays(self, state: SelfState) -> Dict[str, Any]:
        rot = np.array(p.getMatrixFromQuaternion(state.quat), dtype=float).reshape(3, 3)

        ray_from: List[np.ndarray] = []
        ray_to: List[np.ndarray] = []
        for d_body in self.ray_dirs_body:
            d_world = rot @ d_body
            start = state.pos + self.ray_start_offset * d_world
            end = state.pos + self.ray_length * d_world
            ray_from.append(start)
            ray_to.append(end)

        results = p.rayTestBatch(ray_from, ray_to, physicsClientId=self.client)

        dists = np.full((len(results),), self.ray_length, dtype=float)
        hit_ids = np.full((len(results),), -1, dtype=int)
        hit_points = np.asarray(ray_to, dtype=float)

        for i, res in enumerate(results):
            body_id = int(res[0])
            hit_fraction = float(res[2])
            hit_pos = np.asarray(res[3], dtype=float)

            if body_id == self.drone_id:
                continue

            if self.ray_blocker_ids and body_id not in self.ray_blocker_ids:
                continue

            if body_id >= 0:
                dists[i] = max(0.0, min(self.ray_length, hit_fraction * self.ray_length))
                hit_ids[i] = body_id
                hit_points[i] = hit_pos

        if dists.size > 0:
            min_i = int(np.argmin(dists))
            min_dist = float(dists[min_i])
            min_dir_body = self.ray_dirs_body[min_i].copy()
        else:
            min_dist = float("inf")
            min_dir_body = np.zeros(3, dtype=float)

        return {
            "ray_dists": dists,
            "ray_hit_ids": hit_ids,
            "ray_hit_points": hit_points,
            "min_dist": min_dist,
            "min_dir_body": min_dir_body,
            "ray_from": np.asarray(ray_from, dtype=float),
            "ray_to": np.asarray(ray_to, dtype=float),
        }

    def _sense_collision(self) -> Dict[str, Any]:
        query_ids = set(self.ray_blocker_ids) if self.ray_blocker_ids else set(self.wall_ids | self.obstacle_ids | self.nofly_ids)

        contacts = p.getContactPoints(bodyA=self.drone_id, physicsClientId=self.client)
        contact_ids: Set[int] = set()
        for cp in contacts:
            other = int(cp[2])
            if other == self.drone_id:
                continue
            if query_ids and other not in query_ids:
                continue
            contact_ids.add(other)

        collision = len(contact_ids) > 0
        closest = float("inf")

        for bid in query_ids:
            cps = p.getClosestPoints(
                bodyA=self.drone_id,
                bodyB=int(bid),
                distance=self.closest_query_distance,
                physicsClientId=self.client,
            )
            for cp in cps:
                d = float(cp[8])  # contact distance (<0 means penetration)
                if d < closest:
                    closest = d
                if d <= self.collision_distance:
                    collision = True

        return {
            "collision": collision,
            "contact_body_ids": sorted(contact_ids),
            "closest_obstacle_dist": closest,
        }

    def _sense_targets(self, state: SelfState) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        if not self.target_ids:
            return [], []

        rot = np.array(p.getMatrixFromQuaternion(state.quat), dtype=float).reshape(3, 3)
        rot_world_to_body = rot.T

        camera_visible_ids: Set[int] = set()
        if self.target_mode in ("camera", "hybrid"):
            camera_visible_ids = self._camera_detect_target_ids(state)

        targets_gt: List[Dict[str, Any]] = []
        targets_detected: List[Dict[str, Any]] = []

        for tid in sorted(self.target_ids):
            tpos, _ = p.getBasePositionAndOrientation(tid, physicsClientId=self.client)
            tpos = np.asarray(tpos, dtype=float)

            vec_w = tpos - state.pos
            dist = float(np.linalg.norm(vec_w))
            vec_b = rot_world_to_body @ vec_w

            forward = vec_b[0] > 1e-6
            yaw = float(np.arctan2(vec_b[1], max(vec_b[0], 1e-6)))
            pitch = float(np.arctan2(vec_b[2], max(np.linalg.norm(vec_b[:2]), 1e-6)))
            in_fov = bool(forward and abs(yaw) <= 0.5 * self.target_fov_rad and abs(pitch) <= 0.5 * self.target_fov_rad)

            los = self._line_of_sight(state.pos, tpos, tid)
            visible_gt = bool(dist <= self.target_detect_range and in_fov and los)
            visible_cam = bool(tid in camera_visible_ids)

            if self.target_mode == "gt":
                visible = visible_gt
                source = "gt"
            elif self.target_mode == "camera":
                visible = visible_cam
                source = "camera"
            else:
                visible = visible_gt or visible_cam
                source = "hybrid"

            info = {
                "target_id": tid,
                "position": tpos,
                "distance": dist,
                "bearing_body": vec_b,
                "yaw": yaw,
                "pitch": pitch,
                "in_fov": in_fov,
                "line_of_sight": los,
                "visible_gt": visible_gt,
                "visible_cam": visible_cam,
                "visible": visible,
                "source": source,
            }
            targets_gt.append(info)
            if visible:
                targets_detected.append(info)

        return targets_gt, targets_detected

    def _line_of_sight(self, start_pos: np.ndarray, target_pos: np.ndarray, target_id: int) -> bool:
        direction = target_pos - start_pos
        norm = float(np.linalg.norm(direction))
        if norm < 1e-9:
            return True
        start = start_pos + (direction / norm) * self.ray_start_offset
        res = p.rayTest(start, target_pos, physicsClientId=self.client)[0]
        hit_id = int(res[0])

        if hit_id == target_id:
            return True
        if hit_id == -1:
            return True
        if hit_id == self.drone_id:
            return True
        if self.los_blocker_ids and hit_id not in self.los_blocker_ids:
            return True
        return False

    def _camera_detect_target_ids(self, state: SelfState) -> Set[int]:
        if not self.target_ids:
            return set()

        rot = np.array(p.getMatrixFromQuaternion(state.quat), dtype=float).reshape(3, 3)
        cam_eye = state.pos + rot @ np.array([0.06, 0.0, 0.02], dtype=float)
        cam_target = cam_eye + rot @ np.array([self.cam_far, 0.0, 0.0], dtype=float)
        cam_up = rot @ np.array([0.0, 0.0, 1.0], dtype=float)

        view = p.computeViewMatrix(
            cameraEyePosition=cam_eye.tolist(),
            cameraTargetPosition=cam_target.tolist(),
            cameraUpVector=cam_up.tolist(),
        )
        proj = p.computeProjectionMatrixFOV(
            fov=self.cam_fov_deg,
            aspect=float(self.cam_width) / float(self.cam_height),
            nearVal=self.cam_near,
            farVal=self.cam_far,
        )

        _, _, _, _, seg = p.getCameraImage(
            width=self.cam_width,
            height=self.cam_height,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=p.ER_TINY_RENDERER,
            flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
            physicsClientId=self.client,
        )

        seg_flat = np.asarray(seg).reshape(-1)
        visible: Set[int] = set()
        for sid in np.unique(seg_flat):
            sid = int(sid)
            if sid < 0:
                continue
            body_id = sid & ((1 << 24) - 1)
            if body_id in self.target_ids:
                visible.add(body_id)
        return visible

    def _draw_debug_rays(self, ray_from: np.ndarray, ray_to: np.ndarray, ray_dists: np.ndarray) -> None:
        for i in range(len(ray_from)):
            ratio = float(ray_dists[i] / max(self.ray_length, 1e-6))
            color = [1.0 - ratio, ratio, 0.0]
            self._ray_debug_line_ids[i] = p.addUserDebugLine(
                lineFromXYZ=ray_from[i].tolist(),
                lineToXYZ=ray_to[i].tolist(),
                lineColorRGB=color,
                lineWidth=1.0,
                lifeTime=0.0,
                replaceItemUniqueId=self._ray_debug_line_ids[i],
                physicsClientId=self.client,
            )


def hit_any_wall(pyb_client_id: int, drone_id: int, wall_ids: Sequence[int]) -> bool:
    """Backward-compatible helper used by current main.py."""
    for wid in wall_ids:
        contacts = p.getContactPoints(bodyA=drone_id, bodyB=int(wid), physicsClientId=pyb_client_id)
        if contacts:
            return True
    return False

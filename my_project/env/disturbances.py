from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Deque, Dict, Mapping, MutableMapping, Optional, Sequence

import numpy as np

try:
    import pybullet as p
except Exception:  # pragma: no cover - allows import in environments without pybullet.
    p = None


def _clone_obj(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.copy()
    if isinstance(obj, dict):
        return {k: _clone_obj(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clone_obj(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_clone_obj(v) for v in obj)
    return obj


def _as_vec3(value: float | Sequence[float] | None, default_scalar: float = 0.0) -> np.ndarray:
    if value is None:
        return np.array([default_scalar, default_scalar, default_scalar], dtype=float)
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == 1:
        return np.array([arr[0], arr[0], arr[0]], dtype=float)
    if arr.size != 3:
        raise ValueError("Expected scalar or 3-vector")
    return arr.copy()


@dataclass
class DisturbanceConfig:
    # Physics disturbance
    enabled: bool = True
    wind_std: float = 0.0
    wind_bias_xy: tuple[float, float] = (0.0, 0.0)
    gust_prob: float = 0.0
    gust_strength: float = 0.0
    external_force_std: float = 0.0
    external_force_bias: tuple[float, float, float] = (0.0, 0.0, 0.0)

    # State/sensor corruption
    state_noise_std_pos: float | Sequence[float] = 0.0
    state_noise_std_rpy: float | Sequence[float] = 0.0
    state_noise_std_vel: float | Sequence[float] = 0.0
    ray_noise_std: float = 0.0
    ray_bias: float = 0.0
    packet_dropout_prob: float = 0.0
    measurement_delay_steps: int = 0

    # Payload disturbance
    payload_mass_delta: float = 0.0


class DisturbanceInjector:
    """
    Unified disturbance injector for physics and sensor channels.

    Required API:
    - reset(seed)
    - apply_physics_disturbance(...)
    - corrupt_state(...)
    - corrupt_sensor_packet(...)
    - sample_payload_variation(...)
    """

    def __init__(
        self,
        pyb_client_id: Optional[int] = None,
        drone_id: Optional[int] = None,
        config: Optional[DisturbanceConfig | Mapping[str, Any]] = None,
    ):
        self.client = None if pyb_client_id is None else int(pyb_client_id)
        self.drone_id = None if drone_id is None else int(drone_id)

        if config is None:
            self.cfg = DisturbanceConfig()
        elif isinstance(config, DisturbanceConfig):
            self.cfg = config
        else:
            self.cfg = DisturbanceConfig(**dict(config))

        self.rng = np.random.default_rng()
        self._delay_buffer: Deque[Dict[str, Any]] = deque()
        self._last_output_packet: Optional[Dict[str, Any]] = None
        self._base_mass: Optional[float] = None
        self._payload_delta: float = 0.0

        self._state_noise_std_pos = _as_vec3(self.cfg.state_noise_std_pos, 0.0)
        self._state_noise_std_rpy = _as_vec3(self.cfg.state_noise_std_rpy, 0.0)
        self._state_noise_std_vel = _as_vec3(self.cfg.state_noise_std_vel, 0.0)
        self._force_bias = _as_vec3(self.cfg.external_force_bias, 0.0)

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = np.random.default_rng(seed)
        self._delay_buffer.clear()
        self._last_output_packet = None
        self._payload_delta = 0.0

    def sample_payload_variation(
        self,
        base_mass: Optional[float] = None,
        apply_to_simulation: bool = False,
    ) -> float:
        """
        Sample one payload/mass variation from [-payload_mass_delta, +payload_mass_delta].
        Returns the sampled delta mass (kg).
        """
        amp = abs(float(self.cfg.payload_mass_delta))
        if amp <= 0.0:
            self._payload_delta = 0.0
            return 0.0

        self._payload_delta = float(self.rng.uniform(-amp, amp))
        if base_mass is not None:
            self._base_mass = float(base_mass)

        if apply_to_simulation and self.client is not None and self.drone_id is not None:
            if p is None:
                raise RuntimeError("pybullet is not available, cannot apply payload variation")
            if self._base_mass is None:
                self._base_mass = float(
                    p.getDynamicsInfo(self.drone_id, -1, physicsClientId=self.client)[0]
                )
            new_mass = max(1e-4, self._base_mass + self._payload_delta)
            p.changeDynamics(self.drone_id, -1, mass=new_mass, physicsClientId=self.client)

        return self._payload_delta

    def apply_physics_disturbance(
        self,
        drone_pos: Optional[Sequence[float]] = None,
        external_force_world: Optional[Sequence[float]] = None,
        apply_to_simulation: bool = True,
    ) -> np.ndarray:
        """
        Build and (optionally) apply world-frame external force disturbance.
        """
        if not bool(self.cfg.enabled):
            return np.zeros(3, dtype=float)

        wind_std = max(0.0, float(self.cfg.wind_std))
        wind_bias_xy = np.asarray(self.cfg.wind_bias_xy, dtype=float).reshape(2)
        wind = np.array(
            [
                wind_bias_xy[0] + self.rng.normal(0.0, wind_std),
                wind_bias_xy[1] + self.rng.normal(0.0, wind_std),
                self.rng.normal(0.0, 0.35 * wind_std),
            ],
            dtype=float,
        )

        gust = np.zeros(3, dtype=float)
        if float(self.cfg.gust_prob) > 0.0 and self.rng.random() < float(self.cfg.gust_prob):
            theta = self.rng.uniform(0.0, 2.0 * np.pi)
            gust_mag = float(self.cfg.gust_strength)
            gust[:2] = gust_mag * np.array([np.cos(theta), np.sin(theta)], dtype=float)

        ext = self._force_bias.copy()
        force_std = max(0.0, float(self.cfg.external_force_std))
        if force_std > 0.0:
            ext += self.rng.normal(0.0, force_std, size=3)

        if external_force_world is not None:
            ext += np.asarray(external_force_world, dtype=float).reshape(3)

        total_force = wind + gust + ext

        if apply_to_simulation and self.client is not None and self.drone_id is not None:
            if p is None:
                raise RuntimeError("pybullet is not available, cannot apply physics disturbance")

            if drone_pos is None:
                pos, _ = p.getBasePositionAndOrientation(self.drone_id, physicsClientId=self.client)
                pos_arr = np.asarray(pos, dtype=float)
            else:
                pos_arr = np.asarray(drone_pos, dtype=float).reshape(-1)[:3]

            p.applyExternalForce(
                objectUniqueId=self.drone_id,
                linkIndex=-1,
                forceObj=total_force.tolist(),
                posObj=pos_arr.tolist(),
                flags=p.WORLD_FRAME,
                physicsClientId=self.client,
            )

        return total_force

    def corrupt_state(self, state: Any) -> Any:
        """
        Corrupt state-like objects (dict or object with xyz/vel/rpy attributes).
        """
        if state is None:
            return state

        if isinstance(state, MutableMapping):
            out = _clone_obj(state)
            if "pos" in out:
                out["pos"] = np.asarray(out["pos"], dtype=float) + self.rng.normal(
                    0.0, self._state_noise_std_pos, size=3
                )
            if "xyz" in out:
                out["xyz"] = np.asarray(out["xyz"], dtype=float) + self.rng.normal(
                    0.0, self._state_noise_std_pos, size=3
                )
            if "vel" in out:
                out["vel"] = np.asarray(out["vel"], dtype=float) + self.rng.normal(
                    0.0, self._state_noise_std_vel, size=3
                )
            if "rpy" in out:
                out["rpy"] = np.asarray(out["rpy"], dtype=float) + self.rng.normal(
                    0.0, self._state_noise_std_rpy, size=3
                )
            return out

        # Dataclass/object style fallback for navigation.State.
        out = state
        if hasattr(out, "xyz"):
            out.xyz = np.asarray(out.xyz, dtype=float) + self.rng.normal(
                0.0, self._state_noise_std_pos, size=3
            )
        if hasattr(out, "vel"):
            out.vel = np.asarray(out.vel, dtype=float) + self.rng.normal(
                0.0, self._state_noise_std_vel, size=3
            )
        return out

    def corrupt_sensor_packet(self, packet: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply packet-level corruption:
        - additive state/ray noise
        - packet dropout
        - measurement delay (stale replay until queue is full)
        """
        if packet is None:
            return packet

        out = _clone_obj(packet)

        if "pos" in out:
            out["pos"] = np.asarray(out["pos"], dtype=float) + self.rng.normal(
                0.0, self._state_noise_std_pos, size=3
            )
        if "vel" in out:
            out["vel"] = np.asarray(out["vel"], dtype=float) + self.rng.normal(
                0.0, self._state_noise_std_vel, size=3
            )
        if "rpy" in out:
            out["rpy"] = np.asarray(out["rpy"], dtype=float) + self.rng.normal(
                0.0, self._state_noise_std_rpy, size=3
            )

        if "ray_dists" in out:
            dists = np.asarray(out["ray_dists"], dtype=float).copy()
            if dists.size > 0:
                dists += float(self.cfg.ray_bias)
                dists += self.rng.normal(0.0, max(0.0, float(self.cfg.ray_noise_std)), size=dists.shape)
                ray_len = float(np.max(dists)) if np.isfinite(dists).any() else 10.0
                out["ray_dists"] = np.clip(dists, 0.0, max(0.0, ray_len))

        # Dropout first: return last packet if available.
        drop_prob = float(np.clip(self.cfg.packet_dropout_prob, 0.0, 1.0))
        dropped = bool(self.rng.random() < drop_prob)
        if dropped and self._last_output_packet is not None:
            replay = _clone_obj(self._last_output_packet)
            replay["packet_dropped"] = True
            replay["packet_stale"] = True
            replay["delay_buffer_size"] = len(self._delay_buffer)
            return replay

        self._delay_buffer.append(_clone_obj(out))
        delay_steps = int(max(0, self.cfg.measurement_delay_steps))

        if delay_steps <= 0:
            emitted = self._delay_buffer.popleft()
            stale = False
        elif len(self._delay_buffer) > delay_steps:
            emitted = self._delay_buffer.popleft()
            stale = False
        else:
            emitted = _clone_obj(self._delay_buffer[0])
            stale = True

        emitted["packet_dropped"] = False
        emitted["packet_stale"] = bool(stale)
        emitted["delay_buffer_size"] = len(self._delay_buffer)
        self._last_output_packet = _clone_obj(emitted)
        return emitted

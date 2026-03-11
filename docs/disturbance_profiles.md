# Disturbance and Difficulty Profiles

This document summarizes disturbance-related parameters, defaults, and how
`L0_easy` to `L3_hard` differ.

## Runtime Switch

- Primary switch: `my_project/config.py -> CFG["difficulty_profile"]`
- Supported values:
  - `L0_easy`
  - `L1_mild`
  - `L2_medium`
  - `L3_hard`

## Parameter Meaning and Defaults

| Parameter | Meaning | Default (`L0_easy`) |
|---|---|---|
| `wind_std` | Random wind force standard deviation | `0.0` |
| `wind_bias_xy` | Constant XY wind bias | `(0.0, 0.0)` |
| `gust_prob` | Gust trigger probability per step | `0.0` |
| `gust_strength` | Gust force magnitude | `0.0` |
| `pos_noise_std` | Position noise std injected in sensing | `0.0` |
| `yaw_noise_std` | Yaw noise std injected in sensing | `0.0` |
| `ray_noise_std` | Ray distance noise std | `0.0` |
| `target_range_noise_std` | Target range measurement noise std | `0.0` |
| `target_bearing_noise_std` | Target bearing measurement noise std | `0.0` |
| `target_false_negative_prob` | Probability to miss a true target detection | `0.0` |
| `dropout_prob` | Packet dropout probability | `0.0` |
| `delay_steps` | Sensor packet delay in control steps | `0` |
| `payload_mass_delta` | Max absolute payload mass variation (kg) | `0.0` |
| `timeout_steps` | Mission timeout budget in control steps | `14400` |
| `layout_name` | Map/layout complexity | `apt_2room1hall` |
| `num_obstacles` | Obstacle density | `4` |
| `num_nofly` | Number of no-fly zones | `0` |

## Difficulty Profiles

| Profile | Layout / obstacles | Wind | Sensor noise | Delay / dropout | Payload | Timeout |
|---|---|---|---|---|---|---|
| `L0_easy` | `apt_2room1hall`, obstacles=4, nofly=0 | very low (`0.0`) | none | none | `0.0 kg` | `14400` |
| `L1_mild` | `apt_2room1hall`, obstacles=6, nofly=0 | mild (`wind_std=0.08`) | low | `delay=0`, `dropout=0.01` | `+/-0.0015 kg` | `13680` |
| `L2_medium` | `apt_3room2hall`, obstacles=8, nofly=1 | medium (`wind_std=0.18`) + gust | medium | `delay=1`, `dropout=0.05` | `+/-0.0030 kg` | `12960` |
| `L3_hard` | `apt_3room2hall`, obstacles=12, nofly=2 | high (`wind_std=0.30`) + frequent gust | high | `delay=2`, `dropout=0.10` | `+/-0.0050 kg` | `11520` |

## Implementation Notes

- Physics disturbance injection is implemented in:
  - `my_project/env/disturbances.py`
  - method: `apply_physics_disturbance(...)`
- Sensor uncertainty, delay, and packet dropout are wired through:
  - `my_project/env/sensors.py` constructor arguments in `main.py`
- Payload variation is sampled via:
  - `sample_payload_variation(...)` in `my_project/env/disturbances.py`
  - and applied through PyBullet dynamics update.

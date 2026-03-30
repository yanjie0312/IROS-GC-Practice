# my_project/utils/metrics.py
from __future__ import annotations

import csv
import json
import os
import numpy as np
from typing import Any, Dict


def compute_metrics(logger, result: Dict[str, Any], scenario, cfg: Dict) -> Dict[str, Any]:
    """Compute all performance metrics from one episode."""
    pos = logger.positions   # (N, 3)
    dists = logger.min_dists  # (N,)

    # --- Path & altitude ---
    if len(pos) >= 2:
        path_length = float(np.sum(np.linalg.norm(np.diff(pos, axis=0), axis=1)))
    else:
        path_length = 0.0

    if len(pos) > 0:
        altitudes = pos[:, 2]
        mean_altitude = float(np.mean(altitudes))
        altitude_variance = float(np.var(altitudes))
    else:
        mean_altitude = float("nan")
        altitude_variance = float("nan")

    # --- Safety ---
    valid = dists[np.isfinite(dists)]
    if len(valid) > 0:
        min_obs_dist = float(np.min(valid))
        mean_obs_dist = float(np.mean(valid))
        frac_below_safe = float(np.mean(valid < 0.3))
    else:
        min_obs_dist = mean_obs_dist = frac_below_safe = float("nan")

    # --- Mission ---
    n_insp = int(result.get("targets_inspected", 0))
    n_total = int(result.get("targets_total", 1))
    inspection_rate = n_insp / n_total if n_total > 0 else 0.0
    flight_time = float(result.get("flight_time_sec", 0.0))
    time_per_insp = flight_time / n_insp if n_insp > 0 else float("nan")
    insp_per_sec = n_insp / flight_time if flight_time > 0 else float("nan")

    fp = result.get("final_pos") or [float("nan")] * 3
    cp = result.get("crash_pos") or [float("nan")] * 3

    def _f(v, decimals=3):
        """Format float; return empty string for NaN."""
        return round(float(v), decimals) if np.isfinite(float(v)) else ""

    metrics: Dict[str, Any] = {
        # Scenario context
        "difficulty":       str(cfg.get("difficulty_profile", "")),
        "seed":             int(cfg.get("scenario_seed", 0)),
        "layout":           str(getattr(scenario, "layout_name", "")),
        "num_targets":      n_total,
        "num_obstacles":    int(getattr(scenario, "num_obstacles", 0)),
        # Outcome
        "success":              int(bool(result.get("success", False))),
        "termination_reason":   str(result.get("termination_reason", "")),
        "timeout":              int(bool(result.get("timeout", False))),
        "collision":            int(bool(result.get("collision", False))),
        "had_obstacle_contact": int(bool(result.get("had_obstacle_contact", False))),
        # Targets
        "targets_inspected": n_insp,
        "targets_total":     n_total,
        "inspection_rate":   round(inspection_rate, 4),
        # Efficiency
        "flight_time_sec":          round(flight_time, 2),
        "steps":                    int(result.get("steps", 0)),
        "path_length_m":            round(path_length, 3),
        "time_per_inspection_sec":  _f(time_per_insp, 2),
        "inspections_per_sec":      _f(insp_per_sec, 4),
        # Safety
        "min_obstacle_dist_m":    _f(min_obs_dist),
        "mean_obstacle_dist_m":   _f(mean_obs_dist),
        "frac_steps_below_03m":   _f(frac_below_safe, 4),
        "n_escape_events":        int(logger.n_escape_events),
        "n_low_z_events":         int(logger.n_low_z_events),
        # Control quality
        "mean_altitude_m":    _f(mean_altitude),
        "altitude_variance":  _f(altitude_variance, 5),
        # Final positions
        "final_pos_x": _f(fp[0]),
        "final_pos_y": _f(fp[1]),
        "final_pos_z": _f(fp[2]),
        "crash_pos_x": _f(cp[0]),
        "crash_pos_y": _f(cp[1]),
        "crash_pos_z": _f(cp[2]),
    }
    return metrics


def save_metrics_csv(path: str, metrics: Dict[str, Any]):
    """Append one row to metrics.csv; write header if file is new."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    file_exists = os.path.isfile(path)
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(metrics)


def save_run_config(path: str, cfg: Dict, scenario):
    """Save run_config.json with CFG + key scenario parameters."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    def _serialize(v):
        if hasattr(v, "value"):   # enum → its primitive value
            return v.value
        if isinstance(v, (list, tuple)):
            return [_serialize(x) for x in v]
        if isinstance(v, dict):
            return {k: _serialize(vv) for k, vv in v.items()}
        return v

    run_cfg = {
        "cfg": {k: _serialize(v) for k, v in cfg.items()},
        "scenario": {
            k: _serialize(v)
            for k, v in vars(scenario).items()
        },
    }
    with open(path, "w") as f:
        json.dump(run_cfg, f, indent=2)

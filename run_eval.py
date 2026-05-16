"""
run_eval.py — 批量评估脚本
每个难度等级运行 N_RUNS 次，统计 Rs / Rc / T_avg / ICR。
结果保存到 results/eval_YYYYMMDD_HHMMSS/

用法：
    conda run -n drones python run_eval.py
"""
from __future__ import annotations

import json
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

# ── 配置 ────────────────────────────────────────────────────────────────────
LEVELS   = ["L0_easy", "L1_mild", "L2_medium", "L3_hard"]
N_RUNS   = 10
BASE_SEED = 0       # 每个 level 使用 seed = BASE_SEED + run_idx

# ── 输出目录 ─────────────────────────────────────────────────────────────────
EVAL_DIR = Path("results") / f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
EVAL_DIR.mkdir(parents=True, exist_ok=True)


def run_one(level: str, seed: int) -> dict:
    """运行一次仿真并返回 result dict。"""
    from my_project.config import CFG
    CFG["difficulty_profile"] = level
    CFG["scenario_seed"]      = seed
    CFG["gui"]                = False
    CFG["gui_realtime"]       = False
    CFG["output_folder"]      = str(EVAL_DIR / level)  # JSON + PNG 存到对应 level 子目录

    from my_project.main import main
    return main()


def compute_metrics(results: list[dict]) -> dict:
    """
    Rs  : 成功率（全部目标在时限内巡检完）
    Rc  : 碰撞率（曾与任何障碍接触）
    T_avg: 成功 episode 的平均任务时间（s）
    ICR : 每 episode 平均目标巡检完成率
    """
    n = len(results)
    if n == 0:
        return {"Rs": 0.0, "Rc": 0.0, "T_avg": float("nan"), "ICR": 0.0, "n": 0}

    Rs  = sum(1 for r in results if r["success"]) / n
    Rc  = sum(1 for r in results if r["had_obstacle_contact"]) / n

    success_times = [r["flight_time_sec"] for r in results if r["success"]]
    T_avg = float(np.mean(success_times)) if success_times else float("nan")

    ICR = float(np.mean([
        r["targets_inspected"] / r["targets_total"]
        for r in results if r.get("targets_total", 0) > 0
    ]))

    return {"Rs": Rs, "Rc": Rc, "T_avg": T_avg, "ICR": ICR, "n": n}


def main_eval():
    all_summary: dict[str, dict] = {}

    for level in LEVELS:
        print(f"\n{'='*60}")
        print(f"  Level: {level}  ({N_RUNS} runs)")
        print(f"{'='*60}")

        level_dir = EVAL_DIR / level
        level_dir.mkdir(exist_ok=True)

        results: list[dict] = []

        for run_idx in range(N_RUNS):
            seed = BASE_SEED + run_idx
            print(f"  [{run_idx+1:02d}/{N_RUNS}] seed={seed} ... ", end="", flush=True)
            t0 = time.time()

            try:
                result = run_one(level, seed)
                elapsed = time.time() - t0

                status = "✓" if result["success"] else "✗"
                print(f"{status}  t={result['flight_time_sec']:.1f}s  "
                      f"inspected={result['targets_inspected']}/{result['targets_total']}  "
                      f"[wall={elapsed:.0f}s]")

                results.append(result)

            except Exception as e:
                print(f"ERROR: {e}")
                traceback.print_exc()
                # 记录失败的 run（不影响其他 run）
                results.append({
                    "success": False, "collision": False,
                    "had_obstacle_contact": False, "timeout": False,
                    "targets_inspected": 0, "targets_total": 1,
                    "flight_time_sec": 0.0, "termination_reason": f"exception: {e}",
                })

        # 计算本 level 的指标
        metrics = compute_metrics(results)
        all_summary[level] = metrics

        print(f"\n  ── {level} 指标 ──")
        print(f"     Rs   = {metrics['Rs']:.3f}  ({int(metrics['Rs']*N_RUNS)}/{N_RUNS} 成功)")
        print(f"     Rc   = {metrics['Rc']:.3f}")
        print(f"     T_avg= {metrics['T_avg']:.1f}s")
        print(f"     ICR  = {metrics['ICR']:.3f}")

        # 保存本 level 汇总
        with open(level_dir / "metrics.json", "w") as f:
            json.dump({"level": level, "n_runs": N_RUNS, "metrics": metrics}, f, indent=2)

    # ── 最终汇总 ────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  最终汇总")
    print(f"{'='*60}")
    print(f"  {'Level':<12} {'Rs':>6} {'Rc':>6} {'T_avg':>8} {'ICR':>6}")
    print(f"  {'-'*45}")
    for level, m in all_summary.items():
        t_str = f"{m['T_avg']:.1f}s" if not np.isnan(m['T_avg']) else "  N/A"
        print(f"  {level:<12} {m['Rs']:>6.3f} {m['Rc']:>6.3f} {t_str:>8} {m['ICR']:>6.3f}")

    # 保存汇总 JSON
    summary_path = EVAL_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "eval_time": datetime.now().isoformat(),
            "n_runs": N_RUNS,
            "levels": all_summary,
        }, f, indent=2)

    # 保存汇总 CSV
    csv_path = EVAL_DIR / "summary.csv"
    with open(csv_path, "w") as f:
        f.write("level,Rs,Rc,T_avg,ICR,n_runs\n")
        for level, m in all_summary.items():
            t_val = f"{m['T_avg']:.2f}" if not np.isnan(m['T_avg']) else ""
            f.write(f"{level},{m['Rs']:.4f},{m['Rc']:.4f},{t_val},{m['ICR']:.4f},{m['n']}\n")

    print(f"\n  结果已保存至: {EVAL_DIR}")
    print(f"  汇总: {summary_path}")
    print(f"  CSV:  {csv_path}")


if __name__ == "__main__":
    main_eval()

# my_project/experiments/run_experiment.py
"""
批量实验脚本：3 map levels × 4 deg levels = 12 种正交组合。
结果逐行追加到 results/batch_metrics.csv。

用法：
    python -m my_project.experiments.run_experiment            # 全部 12 组
    python -m my_project.experiments.run_experiment --gui      # 带 GUI（调试用）
    python -m my_project.experiments.run_experiment --seed 7   # 指定随机种子
"""
from __future__ import annotations

import argparse
import os
import sys
import traceback

from my_project.config import CFG
from my_project.main import main

# 正交轴
MAP_LEVELS = ["easy", "med", "hard"]
DEG_LEVELS = ["l0", "l1", "l2", "l3"]

# 地图/扰动的可读描述（仅用于打印）
MAP_DESC = {
    "easy": "两室一厅 4障",
    "med":  "三室两厅 8障+1禁飞",
    "hard": "三室两厅 12障+2禁飞",
}
DEG_DESC = {
    "l0": "无扰动",
    "l1": "轻度风+噪声",
    "l2": "中度风+延迟+drag",
    "l3": "强风+地面效应+严重噪声",
}


def run_all(seed: int = 42, gui: bool = False, output_csv: str = "results/batch_metrics.csv"):
    total = len(MAP_LEVELS) * len(DEG_LEVELS)
    results_summary = []

    # 全局 CFG 覆盖（批量模式）
    CFG["gui"] = gui
    CFG["scenario_seed"] = seed
    CFG["output_folder"] = os.path.dirname(os.path.abspath(output_csv))

    # 确保输出目录存在；若 CSV 已有旧数据则备份
    os.makedirs(CFG["output_folder"], exist_ok=True)
    if os.path.isfile(output_csv):
        backup = output_csv.replace(".csv", "_backup.csv")
        os.replace(output_csv, backup)
        print(f"[run_experiment] 已有旧结果，备份至 {backup}")

    done = 0
    for map_level in MAP_LEVELS:
        for deg_level in DEG_LEVELS:
            done += 1
            combo = f"{map_level}+{deg_level}"
            print(f"\n{'='*65}")
            print(f"[{done:02d}/{total}]  {combo}  |  {MAP_DESC[map_level]}  ×  {DEG_DESC[deg_level]}")
            print(f"{'='*65}")

            # 每次覆盖 CFG 中与场景相关的字段
            CFG["map_level"] = map_level
            CFG["deg_level"] = deg_level
            CFG["difficulty_profile"] = combo  # 写入 metrics difficulty 列

            try:
                result = main()
                status = "✓ success" if result.get("success") else f"✗ {result.get('termination_reason', '?')}"
            except Exception:
                result = {}
                status = "✗ EXCEPTION"
                traceback.print_exc()

            results_summary.append((combo, status))
            print(f"  → {status}")

    # 汇总打印
    print(f"\n{'='*65}")
    print(f"批量实验完成  (seed={seed}, gui={gui})")
    print(f"{'='*65}")
    print(f"{'组合':<18}  结果")
    for combo, status in results_summary:
        print(f"  {combo:<16}  {status}")
    print(f"\n完整指标 → {os.path.abspath(output_csv)}")


def _parse_args():
    parser = argparse.ArgumentParser(description="IROS-GC 批量实验")
    parser.add_argument("--seed",   type=int,  default=42,    help="随机种子 (默认 42)")
    parser.add_argument("--gui",    action="store_true",       help="开启 PyBullet GUI (默认关闭)")
    parser.add_argument("--output", type=str,  default="results/batch_metrics.csv",
                        help="输出 CSV 路径 (默认 results/batch_metrics.csv)")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run_all(seed=args.seed, gui=args.gui, output_csv=args.output)

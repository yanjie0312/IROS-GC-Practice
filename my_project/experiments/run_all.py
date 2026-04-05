"""
run_all.py — batch experiment runner
Usage:
    python -m my_project.experiments.run_all
    python -m my_project.experiments.run_all --runs 5 --timeout 600 --no-gui
"""

import argparse
import csv
import os
import subprocess
import sys
import time

PROFILES = ["L0_easy", "L1_mild", "L2_medium", "L3_hard"]
DEFAULT_RUNS = 5
DEFAULT_TIMEOUT = 600  # seconds


def run_one(profile: str, seed: int, timeout: int, gui: bool) -> dict:
    cmd = [
        sys.executable, "-m", "my_project.main",
        "--profile", profile,
        "--seed", str(seed),
    ]
    if not gui:
        cmd.append("--no-gui")

    print(f"  [run] profile={profile} seed={seed}  cmd={' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, timeout=timeout)
        elapsed = time.time() - t0
        status = "ok" if proc.returncode == 0 else f"exit_{proc.returncode}"
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        status = "timeout"
        print(f"  [TIMEOUT] profile={profile} seed={seed} exceeded {timeout}s — marked as FAILED")

    return {"profile": profile, "seed": seed, "status": status, "elapsed_s": round(elapsed, 1)}


def main():
    parser = argparse.ArgumentParser(description="Batch experiment runner")
    parser.add_argument("--profiles", nargs="+", default=PROFILES,
                        choices=PROFILES, help="Profiles to run (default: all 4)")
    parser.add_argument("--runs", type=int, default=DEFAULT_RUNS,
                        help=f"Number of repetitions per profile (default: {DEFAULT_RUNS})")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT,
                        help=f"Per-run wall-clock timeout in seconds (default: {DEFAULT_TIMEOUT})")
    parser.add_argument("--seed-start", type=int, default=1,
                        help="First seed; seeds are seed_start .. seed_start+runs-1 (default: 1)")
    parser.add_argument("--no-gui", action="store_true", help="Disable PyBullet GUI for all runs")
    parser.add_argument("--out", default="results/run_all_summary.csv",
                        help="Summary CSV path (default: results/run_all_summary.csv)")
    args = parser.parse_args()

    total = len(args.profiles) * args.runs
    print(f"=== run_all: {len(args.profiles)} profile(s) x {args.runs} run(s) = {total} total ===")
    print(f"    timeout per run : {args.timeout}s")
    print(f"    seeds           : {args.seed_start} .. {args.seed_start + args.runs - 1}")
    print(f"    gui             : {not args.no_gui}")
    print()

    results = []
    n = 0
    t_all = time.time()

    for profile in args.profiles:
        for i in range(args.runs):
            seed = args.seed_start + i
            n += 1
            print(f"--- [{n}/{total}] {profile}  run {i+1}/{args.runs} ---")
            rec = run_one(profile, seed, args.timeout, gui=not args.no_gui)
            results.append(rec)
            flag = "PASS" if rec["status"] == "ok" else "FAIL"
            print(f"  => {flag}  status={rec['status']}  elapsed={rec['elapsed_s']}s\n")

    elapsed_total = round(time.time() - t_all, 1)

    # Summary
    print("=" * 60)
    print(f"DONE — total wall time: {elapsed_total}s")
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_to = sum(1 for r in results if r["status"] == "timeout")
    n_err = total - n_ok - n_to
    print(f"  PASS   : {n_ok}/{total}")
    print(f"  TIMEOUT: {n_to}/{total}")
    print(f"  ERROR  : {n_err}/{total}")
    print()

    for profile in args.profiles:
        sub = [r for r in results if r["profile"] == profile]
        ok = sum(1 for r in sub if r["status"] == "ok")
        print(f"  {profile:<12}  {ok}/{len(sub)} passed")

    # Save CSV
    out_dir = os.path.dirname(args.out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["profile", "seed", "status", "elapsed_s"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nSummary saved -> {args.out}")


if __name__ == "__main__":
    main()

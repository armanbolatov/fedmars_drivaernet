#!/usr/bin/env python
"""Read-only progress report: live runs, tuning ladders, benchmark coverage.

  python scripts/status.py
"""
import json
import re
import subprocess
from pathlib import Path

from tune_full import SCORE_ROUNDS, score

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
SPLITS = ("single_class", "dirichlet_0.5", "dirichlet_10", "dirichlet_1000")


def live_runs():
    out = subprocess.run(["pgrep", "-af", "src/train.py"], capture_output=True,
                         text=True).stdout
    dirs = sorted(set(re.findall(r"--output-dir (\S+)", out)))
    print("== live runs")
    for d in dirs:
        stats = Path(d) / "round_stats.csv"
        r = stats.read_text().splitlines()[-1].split(",")[2] if stats.exists() else "?"
        print(f"  {Path(d).relative_to(RUNS) if Path(d).is_relative_to(RUNS) else d}"
              f"  round {r}")
    if not dirs:
        print("  (none)")


def ladders():
    print("== learning-rate ladders (results/tune_full_*.json)")
    for f in sorted(ROOT.glob("results/tune_full_*.json")):
        cells = json.loads(f.read_text())
        moved = sum(c["start"] != c["best"] for c in cells)
        print(f"  {f.stem[10:]:<16} {len(cells)} splits, {moved} moved off the start")


def epoch_ladders():
    base = RUNS / "full30" / "single_class"
    if not base.is_dir():
        return
    print(f"== local-epoch ladder (full30/single_class, mean of rounds {SCORE_ROUNDS})")
    cells = {}
    for d in base.iterdir():
        m = re.fullmatch(r"(\w+?)_lr([\d.e-]+?)_single_class_E(\d+)_s42", d.name)
        if m and (v := score(d)) is not None:
            cells.setdefault(m.group(1), {})[(int(m.group(3)), float(m.group(2)))] = v
    for method, rungs in sorted(cells.items()):
        row = "  ".join(f"E{e}@{lr:g}={v:.3e}" for (e, lr), v in sorted(rungs.items()))
        print(f"  {method:<10} {row}")


def benchmark(setting="full"):
    print(f"== benchmark rows finished ({setting})")
    for split in SPLITS:
        base = RUNS / setting / split
        if not base.is_dir():
            continue
        done = sorted(d.name for d in base.iterdir() if (d / "summary.csv").exists())
        print(f"  {split:<16} {len(done):2d}: {' '.join(done)}")


if __name__ == "__main__":
    live_runs()
    ladders()
    epoch_ladders()
    benchmark("full30")
    benchmark("full")

#!/usr/bin/env python
"""Read-only experiment status: live runs, tuning ladders, benchmark completion.

  python scripts/status.py
"""
import csv
import json
import re
import statistics
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"


def window_mean(run_dir, lo=20, hi=30, metric="c_d_mae"):
    """Tuning convention: mean over evals in rounds [lo, hi]. None if incomplete."""
    if not (run_dir / "summary.csv").exists():
        return None
    rows = [r for r in csv.DictReader((run_dir / "history.csv").open())
            if r.get("body") == "all" and lo <= int(r["round"]) <= hi
            and r.get(metric) not in (None, "", "nan")]
    return statistics.mean(float(r[metric]) for r in rows) if rows else None


def live_runs():
    out = subprocess.run(["pgrep", "-af", "src/train.py"], capture_output=True,
                         text=True).stdout
    dirs = sorted(set(re.findall(r"--output-dir (\S+)", out)))
    print("== live runs")
    for d in dirs:
        stats = Path(d) / "round_stats.csv"
        r = stats.read_text().splitlines()[-1].split(",")[2] if stats.exists() else "?"
        name = str(Path(d)).replace(str(RUNS) + "/", "").replace(str(ROOT) + "/", "")
        print(f"  {name}  round {r}")
    if not dirs:
        print("  (none)")


def ladders():
    print("== LR ladders (results/tune_full_*.json)")
    for f in sorted(ROOT.glob("results/tune_full_*.json")):
        cells = json.loads(f.read_text())
        moved = sum(c["changed"] for c in cells)
        print(f"  {f.stem[10:]:<12} {len(cells)} cells, {moved} moved")


def e_table(setting="full30"):
    base = RUNS / setting / "single_class"
    if not base.is_dir():
        return
    print(f"== E ladder ({setting}/single_class, window 20-30)")
    runs = {}
    for d in base.iterdir():
        m = re.fullmatch(r"(\w+?)_lr([\d.e-]+?)_single_class_E(\d+)_s42", d.name)
        if m and (v := window_mean(d)) is not None:
            runs.setdefault(m.group(1), {})[(float(m.group(2)), int(m.group(3)))] = v
    for method, cells in sorted(runs.items()):
        row = "  ".join(f"E{e}@{lr:g}={v:.3e}" for (lr, e), v in sorted(cells.items(),
                                                                        key=lambda kv: kv[0][1]))
        print(f"  {method:<10} {row}")


def bench(setting="full"):
    print(f"== benchmark rows ({setting}, *_bench tags)")
    for sp in ("single_class", "dirichlet_0.5", "dirichlet_10", "dirichlet_1000"):
        base = RUNS / setting / sp
        if not base.is_dir():
            continue
        done = [d.name for d in base.iterdir()
                if "_bench" in d.name and (d / "summary.csv").exists()]
        print(f"  {sp:<16} {len(done)}: {' '.join(sorted(done))}")


if __name__ == "__main__":
    live_runs()
    ladders()
    e_table()
    bench("full30")
    bench("full")

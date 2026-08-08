#!/usr/bin/env python
"""Aggregate results/runs into the paper tables: mean +/- std over seeds.

  python scripts/report.py                 # all settings
  python scripts/report.py --setting full
"""
import argparse
import csv
import re
import statistics
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ("c_d_mae", "c_l_mae", "c_s_mae", "c_d_r2", "c_l_r2",
           "rel_l2_p", "rel_l2_wss")
STATS = ("drift_ratio",)   # training dynamics from round_stats.csv, not an eval metric
COLUMNS = METRICS + STATS
N_TAIL = 5  # per-round c_d is noisy: score a run by the mean of its last 5 evals,
            # matching the tuning objective (never a single final round)


def round_stats_means(rundir):
    """Whole-run means of the per-round statistics -- averaged over every round, not
    the eval tail, since drift is a property of training and is largest early."""
    f = rundir / "round_stats.csv"
    rows = list(csv.DictReader(f.open())) if f.exists() else []
    out = {}
    for s in STATS:
        vals = [float(r[s]) for r in rows
                if r.get(s) not in (None, "", "nan") and float(r[s]) == float(r[s])]
        out[s] = statistics.mean(vals) if vals else float("nan")
    return out


def run_rows(setting):
    """One aggregated row per completed run: last-N_TAIL-eval means of each metric."""
    for summary in sorted((ROOT / "results" / "runs" / setting).rglob("summary.csv")):
        rundir = summary.parent
        with (rundir / "history.csv").open() as f:
            rows = [r for r in csv.DictReader(f) if r.get("body") == "all"]
        tail = sorted({int(r["round"]) for r in rows if int(r["round"]) > 0})[-N_TAIL:]
        sel = [r for r in rows if int(r["round"]) in tail]
        agg = {}
        for m in METRICS:
            vals = [float(r[m]) for r in sel
                    if r.get(m) not in (None, "", "nan") and float(r[m]) == float(r[m])]
            agg[m] = statistics.mean(vals) if vals else float("nan")
        agg.update(round_stats_means(rundir))
        yield rundir, agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--setting", choices=["proxy", "full"], default=None)
    a = ap.parse_args()
    settings = [a.setting] if a.setting else ["proxy", "full"]

    for setting in settings:
        groups = defaultdict(list)  # (split, method-tag minus seed) -> [row dicts]
        for rundir, agg in run_rows(setting):
            split = rundir.parent.name
            tag = re.sub(r"_s\d+", "", rundir.name)   # drop the seed -> group key
            groups[(split, tag)].append(agg)
        if not groups:
            continue

        out = ROOT / "results" / "tables"
        out.mkdir(parents=True, exist_ok=True)
        table_path = out / f"{setting}.csv"
        with table_path.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["split", "config", "n_seeds"]
                       + [x for m in COLUMNS for x in (f"{m}_mean", f"{m}_std")])
            print(f"\n== {setting} ==")
            for (split, tag), rows in sorted(groups.items()):
                line = [split, tag, len(rows)]
                shown = [f"{split:14s} {tag:24s} n={len(rows)}"]
                for m in COLUMNS:
                    vals = [r[m] for r in rows if r[m] == r[m]]  # drop NaN
                    mean = statistics.mean(vals) if vals else float("nan")
                    std = statistics.stdev(vals) if len(vals) > 1 else 0.0
                    line += [f"{mean:.4e}", f"{std:.1e}"]
                    if m in ("c_d_mae", "c_l_mae", "drift_ratio"):
                        shown.append(f"{m}={mean:.3e}±{std:.0e}")
                w.writerow(line)
                print("  ".join(shown))
        print(f"wrote {table_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

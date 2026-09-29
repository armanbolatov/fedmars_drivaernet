#!/usr/bin/env python
"""Integrity check over every completed run. Prints one line per problem found.

Checks that every round the cadence requires was evaluated, NaNs in reported
metrics, per-round stats consistency, side-force sanity (c_s ~ 0 by symmetry), and
that every run's config matches its method yaml.

  python scripts/validate.py
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
CFG = ROOT / "configs" / "method"
# fields that must equal the canonical method yaml in every run; the learning rate
# is excluded because ladder rungs vary it by construction
STRUCTURAL = {"fedavg": ["fed_beta", "weight_decay"], "scaffold": ["fed_beta"],
              "fedlion": ["fed_beta", "lion_beta2", "weight_decay"],
              "fedmuon": ["fed_beta", "scalar_lr", "mars_gamma", "muon_wd"],
              "fedmars": ["fed_beta", "scalar_lr", "mars_gamma", "muon_wd"],
              "fedolion": ["fed_beta", "scalar_lr", "mars_gamma", "muon_wd"],
              "centralized": ["fed_beta", "weight_decay"],
              "centralized_muon": ["fed_beta", "scalar_lr", "muon_wd"]}
METRICS = ("c_d_mae", "c_l_mae", "c_s_mae")


def eval_rounds(run):
    rows = [r for r in csv.DictReader((run / "history.csv").open())
            if r.get("body") == "all"]
    return [int(r["round"]) for r in rows], rows


def check(run, problems):
    cfg = json.loads((run / "config.json").read_text())
    rounds, rows = eval_rounds(run)
    want_last, every = cfg.get("rounds"), cfg.get("eval_every", 1)
    tail = cfg.get("eval_tail", 0)

    dupes = [r for r, n in Counter(rounds).items() if n > 1]
    if dupes:
        problems.append(f"{run}: duplicate eval rounds {sorted(dupes)[:5]}")
    if rounds and want_last and max(rounds) != want_last:
        problems.append(f"{run}: last eval round {max(rounds)}, config says {want_last}")
    if want_last:   # every round the cadence asks for must be there (a run resumed
        need = {want_last}   # under a denser cadence may carry extra rounds: fine)
        need |= {r for r in range(1, want_last + 1) if every and r % every == 0}
        need |= set(range(want_last - tail + 1, want_last + 1))
        if missing := sorted(need - set(rounds)):
            problems.append(f"{run}: evaluations missing at rounds {missing[:5]}")

    for m in METRICS:
        vals = [r[m] for r in rows if r.get(m) not in (None, "")]
        if any(v != v for v in map(float, vals)):
            problems.append(f"{run}: NaN in {m}")

    stats = run / "round_stats.csv"
    if stats.exists():
        srounds = [int(r["round"]) for r in csv.DictReader(stats.open())]
        sdupes = [r for r, n in Counter(srounds).items() if n > 1]
        if sdupes:
            problems.append(f"{run}: round_stats has duplicate rounds "
                            f"{sorted(sdupes)[:5]} (inconsistent per-round stats)")
        if srounds and want_last and max(srounds) != want_last:
            problems.append(f"{run}: round_stats stops at {max(srounds)} of {want_last}")

    # side force ~0 by symmetry, judged against the coefficient scale (~0.26 at full
    # resolution, ~0.0121 at proxy); a diverged run never learned the field, so skip it
    scale = 0.259 if any(p.startswith("full") for p in run.parts) else 0.0121
    cd = [float(r["c_d_mae"]) for r in rows if r.get("c_d_mae") not in (None, "")]
    if cd and cd[-1] > 0.25 * scale:
        return
    cs = [float(r["c_s_mae"]) for r in rows if r.get("c_s_mae") not in (None, "")]
    if cs and cs[-1] > 0.15 * scale:
        problems.append(f"{run}: c_s_mae {cs[-1]:.2e} is >15% of the {scale:.3g} "
                        f"coefficient scale (side force should be ~0)")

    if "_s42_g" in run.name:  # gamma-ablation arms deliberately override gamma
        return
    method = run.name.split("_lr")[0].split("_E")[0]
    if method not in STRUCTURAL:
        problems.append(f"{run}: unknown method '{method}'")
        return
    spec = yaml.safe_load((CFG / f"{method}.yaml").read_text())
    want = {k.replace("-", "_"): v for k, v in spec["args"].items()}
    for k in STRUCTURAL[method]:
        if k in want and cfg.get(k) is not None:
            if abs(float(want[k]) - float(cfg[k])) > 1e-12:
                problems.append(f"{run}: {k} = {cfg[k]}, canonical value is {want[k]}")


def main():
    problems, n = [], 0
    for summary in sorted(RUNS.rglob("summary.csv")):
        run = summary.parent
        if any(part.endswith(("_prelim", "_v1")) for part in run.parts):
            continue          # quarantined superseded runs are not checked
        if not (run / "history.csv").exists() or not (run / "config.json").exists():
            problems.append(f"{run}: summary.csv without history/config")
            continue
        n += 1
        try:
            check(run, problems)
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"{run}: check failed -- {exc!r}")

    partial = [d for d in RUNS.rglob("round_stats.csv")
               if not (d.parent / "summary.csv").exists()
               and not any(p.endswith(("_prelim", "_v1")) for p in d.parts)]
    print(f"checked {n} completed runs; {len(partial)} partial (no summary.csv)")
    for d in partial:
        print(f"  partial: {d.parent.relative_to(ROOT)}")
    for p in problems:
        print(f"  PROBLEM {p}")
    print("OK -- no problems found" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

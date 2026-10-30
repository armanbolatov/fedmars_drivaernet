#!/usr/bin/env python
"""Bracketed selection at evaluation scale: 50k points, 30 communication rounds.

The learning-rate ladder walks outward from the start until the best rung is worse
on both sides; ladder ends are recorded, never silently accepted. The local-epoch
ladder evaluates every rung.

  python scripts/tune_full.py --methods fedmuon --gpu 0              # LR per split
  python scripts/tune_full.py --methods fedmuon --epochs --gpu 0     # local epochs
  python scripts/tune_full.py --methods fedmuon --rebracket --gpu 0  # LR at E*

A rung scores as mean c_d MAE over rounds 20, 25, 30 (NaN = diverged). Selections land
in results/tune_*_<method>.json; recorded splits are skipped, so runs resume.
"""
import argparse
import csv
import json
import statistics
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
CFG = ROOT / "configs" / "method"
PY = sys.executable
MODEL = "transolver"   # --model; another model keeps its own runs and records


def suffix():
    return "" if MODEL == "transolver" else f"-{MODEL}"


def runs_root():
    return ROOT / "results" / "runs" / f"full30{suffix()}"


def record_path(stem, method):
    return ROOT / "results" / f"{stem}_{method}{suffix()}.json"

LR_LADDER = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
# largest local budget studied: an E=32 round takes ~2 wall-clock hours, outside the
# compute envelope; a method still improving at E=16 is reported as such
E_LADDER = [1, 2, 4, 8, 16]
SPLITS = ["single_class", "dirichlet_0.5", "dirichlet_10", "dirichlet_1000"]
REFERENCE_SPLIT = "single_class"
# the scale-sensitive learning rate, by method
LR_KEY = {"fedavg": "lr", "scaffold": "scaffold-lr", "centralized": "lr",
          "fedmuon": "muon-lr", "fedmars": "muon-lr", "fedlion": "lion-lr",
          "fedolion": "muon-lr", "centralized_muon": "muon-lr",
          "fedavg_cosine": "lr", "scaffold_cosine": "scaffold-lr",
          "fedlion_cosine": "lion-lr", "centralized_cosine": "lr",
          "fedlion_cosine2": "lion-lr"}
# single-learning-rate methods drive both parameter groups from one value
SINGLE_LR = ("fedavg", "scaffold", "centralized",
             "fedavg_cosine", "scaffold_cosine", "centralized_cosine")
# hold all the data, so one walk on the reference split covers every distribution
SPLIT_INDEPENDENT = ("centralized", "centralized_muon", "centralized_cosine")
SCORE_ROUNDS = (20, 25, 30)   # explicit, so a denser evaluation cadence scores the same
DIVERGED = float("inf")


def method_spec(method, split):
    """The method's arguments for this split: base args plus any per-split block."""
    spec = yaml.safe_load((CFG / f"{method}.yaml").read_text())
    args = dict(spec["args"])
    args.update(spec.get("per_split", {}).get(split, {}) or {})
    return args


def tag_for(method, split, lr):
    """One configuration name per (method, split, rung); runs append the epoch count."""
    return f"{method}_lr{lr:g}_{split}"


def rundir(method, split, lr, epochs):
    return runs_root() / split / f"{tag_for(method, split, lr)}_E{epochs}_s42"


def stopped_nonfinite(run_dir):
    """True if training stopped on non-finite gradients: the FedAvg path's gradient
    clipping raises instead of logging NaN, and a retry resumes into the same crash."""
    log = run_dir / "run.log"
    return log.exists() and "is non-finite" in log.read_text(errors="ignore")[-4000:]


def score(run_dir):
    """Mean c_d MAE over SCORE_ROUNDS. None if the run has not finished."""
    if not (run_dir / "summary.csv").exists():
        return DIVERGED if stopped_nonfinite(run_dir) else None
    rows = [r for r in csv.DictReader((run_dir / "history.csv").open())
            if r.get("body") == "all" and int(r["round"]) in SCORE_ROUNDS]
    values = [float(r["c_d_mae"]) for r in rows]
    if not values or any(v != v for v in values):
        return DIVERGED
    return statistics.mean(values)


def write_config(method, split, lr):
    """Materialise the method yaml for one rung of the learning-rate ladder."""
    args = method_spec(method, split)
    args[LR_KEY[method]] = lr
    args["eval-every"] = 5          # the score only needs rounds 20, 25 and 30
    if method in SINGLE_LR:
        args["scalar-lr"] = lr
    (CFG / f"{tag_for(method, split, lr)}.yaml").write_text(yaml.safe_dump(
        {"method": method_spec(method, split)["methods"], "args": args},
        sort_keys=False))


def run(method, split, lr, epochs, gpu, slots):
    """Train one rung to completion, waiting out a busy or crowded card."""
    write_config(method, split, lr)
    cmd = [PY, str(ROOT / "scripts" / "benchmark.py"),
           "--methods", tag_for(method, split, lr), "--splits", split,
           "--seeds", "42", "--gpu", gpu, "--setting", "full30",
           "--local-epochs", str(epochs), "--min-free-gb", "35",
           "--slots", str(slots)]
    if MODEL != "transolver":
        cmd += ["--model", MODEL]
    for attempt in range(1, 101):
        print(f"    {tag_for(method, split, lr)} E{epochs} (attempt {attempt})",
              flush=True)
        if subprocess.run(cmd, cwd=str(ROOT)).returncode == 0:
            return
        if stopped_nonfinite(rundir(method, split, lr, epochs)):
            print("    diverged (non-finite gradients)", flush=True)
            return
        subprocess.run(["sleep", "600"])
    raise RuntimeError(f"{tag_for(method, split, lr)} E{epochs}: exhausted retries")


def climb(ladder, start, measure):
    """Walk outward from `start` until the best rung has a worse neighbour on both
    sides. `measure(value)` returns a score, or None once the run budget is spent."""
    first = measure(start)
    if first is None:
        raise RuntimeError("no budget for the starting rung")
    scores, best = {start: first}, start
    for step in (+1, -1):
        while True:
            i = ladder.index(best) + step
            if not 0 <= i < len(ladder):
                break                       # ladder end, recorded in `evaluated`
            nxt = ladder[i]
            if nxt in scores:
                break
            value = measure(nxt)
            if value is None:
                break                       # budget spent mid-walk
            scores[nxt] = value
            if value < scores[best]:
                best = nxt                  # improving: keep walking this way
            else:
                break                       # neighbour worse: this side is bracketed
    return best, scores


def select(method, split, gpu, ladder, start, budget, epochs_of, lr_of, slots=1,
           full=False):
    """Run one bracketing walk (or, with `full`, every rung) and return its record."""
    spent = [0]

    def measure(value):
        d = rundir(method, split, lr_of(value), epochs_of(value))
        known = score(d)
        if known is not None:
            return known
        if spent[0] >= budget:
            return None
        spent[0] += 1
        run(method, split, lr_of(value), epochs_of(value), gpu, slots)
        return score(d) or DIVERGED

    if full:                 # every rung: a 2% score difference can stop a walk early
        scores = {}
        for value in ladder:
            if (s := measure(value)) is None:
                break
            scores[value] = s
        best = min(scores, key=scores.get)
    else:
        best, scores = climb(ladder, start, measure)
    evaluated = {f"{k:g}": (v if v != DIVERGED else "diverged")
                 for k, v in scores.items()}
    print(f"  {method} @ {split}: best {best:g} ({scores[best]:.4e}) "
          f"of {evaluated}", flush=True)
    return {"split": split, "start": f"{start:g}", "best": f"{best:g}",
            "best_c_d_mae": scores[best], "evaluated": evaluated}


def record(out, row):
    """Append one split's record, preserving records written by parallel walks."""
    rows = {r["split"]: r for r in
            (json.loads(out.read_text()) if out.exists() else [])}
    rows[row["split"]] = row
    order = {s: i for i, s in enumerate(SPLITS)}
    out.write_text(json.dumps(sorted(rows.values(), key=lambda r: order[r["split"]]),
                              indent=2))


def selected(stem, method, split, what):
    out = record_path(stem, method)
    if not out.exists():
        sys.exit(f"missing {out.name}: run the {what} ladder first")
    for row in json.loads(out.read_text()):
        if row["split"] == split:
            return row["best"]
    sys.exit(f"{out.name} has no record for {split}")


def selected_lr(method, split):
    """The learning rate chosen by the learning-rate ladder for this split."""
    return float(selected("tune_full", method, split, "learning-rate"))


def selected_epochs(method):
    """The local-epoch budget chosen on the reference split."""
    return int(selected("tune_epochs", method, REFERENCE_SPLIT, "local-epoch"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True, choices=sorted(LR_KEY))
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--epochs", action="store_true",
                    help="evaluate every rung of the local-epoch ladder instead of the "
                         "learning rate")
    ap.add_argument("--rebracket", action="store_true",
                    help="re-walk the learning rate at the selected local-epoch budget")
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--slots", type=int, default=1,
                    help="how many of our runs may share one GPU (default 1)")
    ap.add_argument("--model", default="transolver", help="configs/model/<name>.yaml")
    a = ap.parse_args()
    global MODEL
    MODEL = a.model

    for method in a.methods:
        if a.epochs:
            # local epochs are selected once, on the reference split
            splits, stem, budget = [REFERENCE_SPLIT], "tune_epochs", 8
        elif a.rebracket:
            splits, stem, budget = [REFERENCE_SPLIT], "tune_rebracket", 4
        elif method in SPLIT_INDEPENDENT:
            splits, stem, budget = [REFERENCE_SPLIT], "tune_full", 6
        else:
            splits, stem, budget = a.splits or SPLITS, "tune_full", 6
        out = record_path(stem, method)
        done = {r["split"] for r in
                (json.loads(out.read_text()) if out.exists() else [])}
        for split in splits:
            if split in done:
                continue
            if a.epochs:
                lr = selected_lr(method, split)
                row = select(method, split, a.gpu, E_LADDER, 1, budget,
                             epochs_of=lambda e: e, lr_of=lambda e: lr, slots=a.slots,
                             full=True)
            elif a.rebracket:
                # LR and E interact: confirm the rate at the selected E
                epochs = selected_epochs(method)
                row = select(method, split, a.gpu, LR_LADDER,
                             selected_lr(method, split), budget,
                             epochs_of=lambda lr: epochs, lr_of=lambda lr: lr, slots=a.slots)
            else:
                row = select(method, split, a.gpu, LR_LADDER,
                             float(method_spec(method, split)[LR_KEY[method]]),
                             budget, epochs_of=lambda lr: 1, lr_of=lambda lr: lr,
                             slots=a.slots)
            record(out, row)
        print(f"{method}: wrote {out.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()

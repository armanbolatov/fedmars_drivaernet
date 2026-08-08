#!/usr/bin/env python
"""Optuna tuning entry point. Protocol: docs/tuning.md.

Idempotent: --n-trials is the study's TOTAL budget; completed work is counted and
only the remainder runs, so re-running any stage or launcher is safe.

  python scripts/tune.py --stage fedmuon --n-trials 24
  python scripts/tune.py --stage gamma --n-trials 5            # uses fedmuon's best
  python scripts/tune.py --stage fedmuon --alpha 0.5 --n-trials 24
"""
import argparse
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import optuna
from optuna.pruners import HyperbandPruner, MedianPruner, NopPruner
from optuna.samplers import GridSampler, TPESampler
from optuna.storages import RDBStorage

ROOT = Path(__file__).resolve().parent.parent
PY = str(Path.home() / "anaconda3" / "envs" / "cfd_transformer" / "bin" / "python")
SCRIPT = str(ROOT / "src" / "train.py")
OPTDIR = ROOT / "results" / "optuna"
BAD = 1.0            # divergence sentinel (finite, so TPE learns the region is bad)
TRIAL_TIMEOUT = 4 * 3600
N_TAIL = 5           # objective = mean c_d of the last N evaluated rounds
AUDIT = ("c_l_mae", "rel_l2_p", "rel_l2_wss")

# 1e-x/3e-x ladders; *_EXT probe one step beyond an edge, certified via EXT_UNION.
MUON_LR = [3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
MUON_SCALAR = [3e-6, 1e-5, 3e-5, 1e-4, 3e-4]
SCAF_LR = [1e-6, 3e-6, 1e-5, 3e-5, 1e-4]
FEDAVG_LR = [1e-5, 3e-5, 1e-4, 3e-4, 1e-3]
BETA = [0.7, 0.8, 0.9, 0.95, 0.99]
GAMMA = [0.01, 0.025, 0.04, 0.1, 1.0]
SCAF_LR_EXT, FEDAVG_LR_EXT = [3e-4, 1e-3], [3e-3, 1e-2]
GAMMA_EXT, BETA_EXT, BETA_LOW = [0.001, 0.003], [0.8, 0.9], [0.6, 0.7]

ALPHA = None         # --alpha: Dirichlet concentration; None = natural single_class split
PROBE = False        # --probe-edges: bracket a study's edge params one rung outward

# one rung beyond every base ladder, for edge probes
WIDE = {
    "muon_lr": [1e-4] + MUON_LR + [1e-1],
    "scalar_lr": [1e-6] + MUON_SCALAR + [1e-3],
    "scaffold_lr": [3e-7] + SCAF_LR + [3e-4, 1e-3],
    "lr": [3e-6] + FEDAVG_LR + [3e-3, 1e-2],
    "fed_beta": [0.6] + BETA + [0.995],
    "mars_gamma": [0.003] + GAMMA + [3.0],
}

SPACES = {
    "fedmuon": {"muon_lr": MUON_LR, "scalar_lr": MUON_SCALAR, "fed_beta": BETA},
    "scaffold": {"scaffold_lr": SCAF_LR, "fed_beta": BETA},
    "fedavg": {"lr": FEDAVG_LR, "fed_beta": BETA},
    "centralized": {"lr": FEDAVG_LR, "fed_beta": BETA},
    "centralized_muon": {"muon_lr": MUON_LR, "scalar_lr": MUON_SCALAR, "fed_beta": BETA},
    "gamma": {"mars_gamma": GAMMA},
    "scaffold_ext": {"scaffold_lr": SCAF_LR_EXT, "fed_beta": BETA_EXT},
    "fedavg_ext": {"lr": FEDAVG_LR_EXT, "fed_beta": BETA_EXT},
    "gamma_ext": {"mars_gamma": GAMMA_EXT},
    "fedmuon_ext": {"fed_beta": BETA_LOW},
    "smoke": {"muon_lr": MUON_LR, "scalar_lr": MUON_SCALAR, "fed_beta": BETA},
}
EXT_UNION = {  # an ext winner is interior/edge relative to base+ext combined
    "scaffold_ext": {"scaffold_lr": sorted(SCAF_LR + SCAF_LR_EXT), "fed_beta": BETA},
    "fedavg_ext": {"lr": sorted(FEDAVG_LR + FEDAVG_LR_EXT), "fed_beta": BETA},
    "gamma_ext": {"mars_gamma": sorted(GAMMA + GAMMA_EXT)},
    "fedmuon_ext": {"fed_beta": sorted(set(BETA + BETA_LOW))},
}

PROXY = ["--downsample-size", "6000", "--rounds", "20", "--eval-every", "1",
         "--max-test-samples", "150", "--test-downsample-size", "20000",
         "--num-workers", "8", "--local-epochs", "1", "--batch-size", "32",
         "--seed", "42", "--server-lr", "1.0", "--skip-round0-eval"]
MUON_FIXED = ["--cv-rule", "mean", "--muon-scale", "shape", "--muon-scale-const", "0.2",
              "--muon-exclude", "preprocess.linear_pre,mlp_out.2",
              "--muon-lr-decay", "cosine", "--muon-wd", "0.1", "--grad-clip", "1.0"]
ADAMW = ["--grad-clip", "1.0", "--weight-decay", "0.01",
         "--muon-exclude", "preprocess.linear_pre,mlp_out.2"]
ONE_CLIENT = ["--num-clients", "1", "--client-split-mode", "dirichlet",
              "--alphas", "1000000"]


def suggest(trial, stage):
    space = ({k: WIDE[k] for k in SPACES[stage]} if PROBE else SPACES[stage])
    return {k: trial.suggest_categorical(k, lad) for k, lad in space.items()}


def probe_configs(stage, tag):
    """One config per edge warning: the study's best with that param one rung outward."""
    best = load_best(f"{stage}_{tag}" if tag != "b32" else stage)
    base = {k: best[k] for k in SPACES[stage]}
    out = []
    for warn in best.get("edge_warnings", []):
        m = re.match(r"EDGE HIT: (\w+)=([0-9.eE+-]+)", warn)
        if not m:
            continue
        k, v = m.group(1), float(m.group(2))
        lad = WIDE[k]
        i = min(range(len(lad)), key=lambda j: abs(lad[j] - v))
        j = i - 1 if v == min(SPACES[stage][k]) else i + 1
        if 0 <= j < len(lad):
            out.append({**base, k: lad[j]})
    return out


def load_best(stage):
    path = OPTDIR / f"{stage}_best.json"
    if not path.exists():
        sys.exit(f"missing {path} - run its tuning stage first")
    return json.loads(path.read_text())


def reference_params(stage):
    return {k: v for k, v in load_best(stage).items() if k in SPACES[stage]}


def build_cmd(stage, p, outdir, rounds=0):
    proxy = list(PROXY)
    if rounds:
        proxy[proxy.index("--rounds") + 1] = str(rounds)
    cmd = [PY, SCRIPT, "--output-dir", str(outdir)] + proxy
    if ALPHA is not None:
        cmd += ["--client-split-mode", "dirichlet", "--alphas", repr(ALPHA)]
    key = {"scaffold_ext": "scaffold", "fedavg_ext": "fedavg", "gamma_ext": "gamma",
           "smoke": "fedmuon"}.get(stage, stage)
    if key == "scaffold":  # single lr: same value for both param groups
        cmd += ["--methods", "scaffold_adam", "--grad-clip", "1.0",
                "--scaffold-lr", repr(p["scaffold_lr"]), "--scalar-lr", repr(p["scaffold_lr"]),
                "--fed-beta", repr(p["fed_beta"])]
    elif key in ("fedavg", "centralized"):  # single-lr AdamW; beta1 = fed_beta
        cmd += ["--methods", "fedavg", "--lr", repr(p["lr"]), "--scalar-lr", repr(p["lr"]),
                "--fed-beta", repr(p["fed_beta"])] + ADAMW
        if key == "centralized":
            cmd += ONE_CLIENT
    elif key == "centralized_muon":  # Muon ceiling: one client, all data, cv cancels
        cmd += ["--methods", "fedmuon", "--mars-gamma", "0"] + ONE_CLIENT + MUON_FIXED + [
            "--muon-lr", repr(p["muon_lr"]), "--scalar-lr", repr(p["scalar_lr"]),
            "--fed-beta", repr(p["fed_beta"])]
    else:  # fedmuon / gamma / fedmuon_ext share the Muon recipe
        ref = "fedmuon" if ALPHA is None else f"fedmuon_dir{ALPHA:g}"
        base = load_best(ref) if key in ("gamma", "fedmuon_ext") else p
        cmd += ["--methods", "fedmuon"] + MUON_FIXED + [
            "--muon-lr", repr(base["muon_lr"]), "--scalar-lr", repr(base["scalar_lr"]),
            "--fed-beta", repr(p.get("fed_beta", base["fed_beta"])),
            "--mars-gamma", repr(p.get("mars_gamma", 0.0))]
    return cmd


def parse_metric(log_path, name):
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        if "body=all" in line:
            m = re.search(rf"{name}:\s*([0-9.eE+-]+|nan)", line)
            if m:
                try:
                    out.append(float(m.group(1)))
                except ValueError:
                    out.append(float("nan"))
    return out


def parse_cd(log_path):
    """(round, c_d, train_loss) per eval line."""
    try:
        text = log_path.read_text(errors="ignore")
    except OSError:
        return []
    out = []
    for line in text.splitlines():
        if "c_d_mae:" not in line or "body=all" not in line:
            continue
        m_r = re.search(r"round=(\d+)/", line)
        m_v = re.search(r"c_d_mae:\s*([0-9.eE+-]+|nan)", line)
        m_t = re.search(r"train_loss:\s*([0-9.eE+-]+|nan)", line)
        if m_r and m_v:
            def f(m):
                try:
                    return float(m.group(1))
                except (ValueError, AttributeError):
                    return float("nan")
            out.append((int(m_r.group(1)), f(m_v), f(m_t)))
    return out


def objective(trial, stage, rounds=0):
    params = suggest(trial, stage)
    for prev in trial.study.trials:  # TPE repeats configs in small spaces: reuse result
        if (prev.number != trial.number and prev.state == optuna.trial.TrialState.COMPLETE
                and prev.params == trial.params):
            trial.set_user_attr("duplicate_of", prev.number)
            return prev.value
    run_tag = stage if ALPHA is None else f"{stage}_dir{ALPHA:g}"
    outdir = ROOT / "log" / "optuna" / run_tag / f"t{trial.number:03d}"
    shutil.rmtree(outdir, ignore_errors=True)  # run.log appends; start clean
    outdir.mkdir(parents=True, exist_ok=True)
    cmd = build_cmd(stage, params, outdir, rounds)
    (outdir / "cmd.txt").write_text(" ".join(cmd))
    proc = subprocess.Popen(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL, start_new_session=True)
    log, seen, cds, tl, t0 = outdir / "run.log", set(), [], float("nan"), time.time()
    try:
        while True:
            finished = proc.poll() is not None
            for rnd, val, tloss in parse_cd(log):
                if rnd in seen:
                    continue
                seen.add(rnd)
                if math.isnan(val):
                    return BAD
                cds.append(val)
                tl = tloss
                trial.report(val, rnd)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            if finished:
                if not cds:
                    return BAD
                trial.set_user_attr("cd_evals", [round(v, 8) for v in cds])
                trial.set_user_attr("cd_final", cds[-1])
                trial.set_user_attr("train_loss", tl)
                for m in AUDIT:
                    vals = parse_metric(log, m)
                    if vals:
                        trial.set_user_attr(m, sum(vals[-N_TAIL:]) / len(vals[-N_TAIL:]))
                tail = cds[-N_TAIL:]
                return sum(tail) / len(tail)
            if time.time() - t0 > TRIAL_TIMEOUT:
                raise optuna.TrialPruned()
            time.sleep(20)
    finally:
        if proc.poll() is None:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def edge_check(study):
    stage = study.study_name.rsplit("_", 1)[0]
    ladders = EXT_UNION.get(stage) or SPACES.get(stage, {})
    warns = [f"EDGE HIT: {k}={v} is a ladder endpoint - extend {k} and rerun"
             for k, v in study.best_params.items()
             if (lad := ladders.get(k)) and v in (min(lad), max(lad))]
    for w in warns:
        print(w, flush=True)
    return warns


def export(study, stage):
    rows = []
    for t in study.trials:
        row = {"number": t.number, "state": t.state.name, "value": t.value,
               "cd_final": t.user_attrs.get("cd_final"),
               "train_loss": t.user_attrs.get("train_loss"),
               "cd_evals": " ".join(str(v) for v in t.user_attrs.get("cd_evals", []))}
        row.update({m: t.user_attrs.get(m) for m in AUDIT})
        row.update({f"p_{k}": v for k, v in t.params.items()})
        rows.append(row)
    keys = sorted({k for r in rows for k in r})
    with (OPTDIR / f"{stage}_trials.csv").open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")
    done = [t for t in study.trials if t.value is not None and t.value < BAD]
    if done:
        best = dict(study.best_params)
        best["value"] = study.best_value
        best["edge_warnings"] = edge_check(study)
        (OPTDIR / f"{stage}_best.json").write_text(json.dumps(best, indent=2))
        print(f"BEST {stage}: {best}", flush=True)


def main():
    global ALPHA, PROBE
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=sorted(SPACES))
    ap.add_argument("--n-trials", type=int, default=24, help="TOTAL trial budget")
    ap.add_argument("--rounds", type=int, default=0, help="override rounds (smoke)")
    ap.add_argument("--alpha", type=float, default=None,
                    help="Dirichlet concentration; omit for the single_class split")
    ap.add_argument("--probe-edges", action="store_true",
                    help="bracket the stage's edge-flagged params one rung outward")
    args = ap.parse_args()
    ALPHA = args.alpha
    PROBE = args.probe_edges
    if args.stage.startswith("centralized") and ALPHA is not None:
        sys.exit("centralized holds all data - split-independent, tune once without --alpha")
    OPTDIR.mkdir(parents=True, exist_ok=True)

    grid_stages = {s for s in SPACES if s.endswith("_ext")} | {"gamma"}
    if args.stage in grid_stages:
        sampler, pruner = GridSampler(SPACES[args.stage]), NopPruner()
    elif args.stage == "smoke":
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=1, n_warmup_steps=1, interval_steps=1)
    else:
        sampler = TPESampler(seed=42, n_startup_trials=8)
        # single Hyperband bracket, halving at round 10: earlier rankings are unreliable
        pruner = HyperbandPruner(min_resource=10, max_resource=20, reduction_factor=3)

    tag = "b32" if ALPHA is None else f"dir{ALPHA:g}"
    name = f"{args.stage}_probe_{tag}" if PROBE else f"{args.stage}_{tag}"
    if PROBE:
        pruner = NopPruner()  # probes must run to completion
    study = optuna.create_study(
        study_name=name, direction="minimize", load_if_exists=True,
        storage=RDBStorage(f"sqlite:///{OPTDIR / 'tune.db'}",
                           engine_kwargs={"connect_args": {"timeout": 60}}),
        sampler=sampler, pruner=pruner)

    budget = args.n_trials
    if PROBE:
        cfgs = probe_configs(args.stage, tag)
        if not cfgs:
            print(f"{name}: no edge warnings to probe", flush=True)
            return
        if not study.trials:
            for c in cfgs:
                study.enqueue_trial(c)
        budget = len(cfgs)
    finished = [t for t in study.trials
                if t.state.name in ("COMPLETE", "PRUNED", "FAIL")]
    remaining = max(0, budget - len(finished))
    print(f"{name}: {len(finished)} trials recorded, running {remaining} more", flush=True)
    if remaining:
        if (not PROBE and ALPHA is not None and not study.trials
                and args.stage in ("fedmuon", "scaffold", "fedavg")):
            study.enqueue_trial(reference_params(args.stage))  # seed = reference optimum
        study.optimize(lambda t: objective(t, args.stage, args.rounds),
                       n_trials=remaining)
    export(study, name.replace("_b32", "") if not PROBE and ALPHA is None else name)


if __name__ == "__main__":
    main()

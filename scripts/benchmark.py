#!/usr/bin/env python
"""Benchmark runner: methods x splits x seeds x local-epochs, driven by configs/.

One process per GPU; runs its work list sequentially and skips anything already done
(summary.csv present), so it is safely resumable and two instances can split the work:

  python scripts/benchmark.py --suite headline --gpu 1
  python scripts/benchmark.py --suite splits --setting proxy --gpu 2
  python scripts/benchmark.py --suite localsteps --setting proxy --gpu 1
  python scripts/benchmark.py --suite gamma --setting full --gpu 2
  python scripts/benchmark.py --methods fedmuon scaffold --splits single_class --seeds 42 --gpu 0
"""
import argparse
import itertools
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
PY = str(Path.home() / "anaconda3" / "envs" / "cfd_transformer" / "bin" / "python")
TRAIN = str(ROOT / "src" / "train.py")
CFG = ROOT / "configs"


def load(path):
    return yaml.safe_load(path.read_text())


def wait_for_memory(gpu, need_mib, poll=300):
    """Block until the card has room. Full runs need ~60 GiB and the GPUs are
    shared, so starting on a filled card OOMs -- sometimes hours in."""
    while True:
        q = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
        free = int(q.stdout.split()[0])
        if free >= need_mib:
            return
        print(f"    waiting for memory: gpu {gpu} has {free // 1024} GiB free, "
              f"need {need_mib // 1024}", flush=True)
        time.sleep(poll)


def build_cmd(base, method_cfg, split_cfg, setting, seed, local_epochs, gamma, outdir,
              rounds=None):
    args = {}
    args.update({k.replace("_", "-"): v for k, v in base["protocol"].items()})
    args.update({k.replace("_", "-"): v for k, v in base[setting].items()})
    # split first, then method: method-specific keys must win (e.g. centralized
    # replaces the split with its trivial one-client partition)
    args.update(split_cfg["args"])
    args.update(method_cfg["args"])
    # hyperparameters are tuned PER DISTRIBUTION: a method yaml's per_split block
    # overrides its reference args for that split (filled from the Dirichlet sweep)
    args.update(method_cfg.get("per_split", {}).get(split_cfg["split"], {}))
    args["seed"] = seed
    args["local-epochs"] = local_epochs
    if rounds is not None:  # localsteps: E x rounds = const (equal gradient work)
        args["rounds"] = rounds
    if gamma is not None:
        args["mars-gamma"] = gamma
    args["output-dir"] = str(outdir)
    cmd = [PY, TRAIN]
    for k, v in args.items():
        if v is True:          # store_true flags (e.g. skip-round0-eval)
            cmd += [f"--{k}"]
        elif v is not False:
            cmd += [f"--{k}", str(v)]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=["headline", "splits", "localsteps", "gamma"])
    ap.add_argument("--setting", choices=["proxy", "full"], default="full")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["single_class"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--local-epochs", nargs="+", type=int, default=[1])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--min-free-gb", type=int,
                    help="wait for this much free GPU memory before each run "
                         "(default: 45 full, 18 proxy)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = load(CFG / "base.yaml")
    seeds = a.seeds or base["benchmark"]["seeds"]
    methods, splits, epochs, gammas = a.methods, a.splits, a.local_epochs, [None]

    if a.suite == "headline":       # main table: all methods, natural split, full task
        methods = ["fedavg", "scaffold", "fedmuon", "fedmars", "centralized"]
    elif a.suite == "splits":       # heterogeneity sweep
        methods = methods or ["fedavg", "scaffold", "fedmuon", "fedmars"]
        splits = ["single_class", "dirichlet_0.5", "dirichlet_10", "dirichlet_1000"]
    elif a.suite == "localsteps":   # communication ablation: rounds-to-target at fixed E
        methods = methods or ["fedavg", "scaffold", "fedmuon", "fedmars"]
        epochs = base["benchmark"]["local_epochs_ablation"]
    elif a.suite == "gamma":        # does MARS help on top of tuned FedMuon?
        methods = ["fedmars"]
        gammas = base["benchmark"]["gamma_ablation"]
    if not methods:
        sys.exit("give --suite or --methods")

    budget = base["benchmark"]["localsteps_budget_epochs"]
    todo, skipped = [], 0
    for m, sp, e, s, g in itertools.product(methods, splits, epochs, seeds, gammas):
        rounds = budget // e if a.suite == "localsteps" else None
        # the budget is part of the tag: without it the localsteps E=1 arm (80 rounds)
        # would collide with the standard 20-round run of the same name
        tag = (f"{m}_E{e}_s{s}" + (f"_R{rounds}" if rounds is not None else "")
               + (f"_g{g:g}" if g is not None else ""))
        outdir = ROOT / "results" / "runs" / a.setting / sp / tag
        if (outdir / "summary.csv").exists():
            skipped += 1
            continue
        todo.append((m, sp, e, s, g, rounds, outdir))

    need_mib = (a.min_free_gb or (45 if a.setting == "full" else 18)) * 1024
    failed = []
    print(f"{len(todo)} runs to do, {skipped} already done", flush=True)
    for i, (m, sp, e, s, g, rounds, outdir) in enumerate(todo, 1):
        cmd = build_cmd(base, load(CFG / "method" / f"{m}.yaml"),
                        load(CFG / "split" / f"{sp}.yaml"), a.setting, s, e, g, outdir,
                        rounds=rounds)
        print(f"[{i}/{len(todo)}] {outdir.relative_to(ROOT)}", flush=True)
        if a.dry_run:
            print("   ", " ".join(cmd), flush=True)
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        wait_for_memory(a.gpu, need_mib)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"    exit {r.returncode}", flush=True)
        if r.returncode:
            failed.append(str(outdir.relative_to(ROOT)))

    if failed:  # nonzero exit so the queue script retries (finished runs are skipped)
        sys.exit(f"{len(failed)} run(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

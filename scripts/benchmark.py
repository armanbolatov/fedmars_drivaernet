#!/usr/bin/env python
"""Benchmark runner: methods x splits x seeds x local-epochs, driven by configs/.

One process per GPU; runs its work list sequentially and skips anything already done
(summary.csv present), so it is safely resumable and two instances can split the work:

  python scripts/benchmark.py --suite headline --gpu 1
  python scripts/benchmark.py --suite splits --gpu 2
  python scripts/benchmark.py --suite gamma --gpu 2
  python scripts/benchmark.py --methods fedmuon scaffold --splits single_class --seeds 42 --gpu 0
"""
import argparse
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

from tune_full import SINGLE_LR

ROOT = Path(__file__).resolve().parent.parent
PY = str(Path.home() / "anaconda3" / "envs" / "cfd_transformer" / "bin" / "python")
TRAIN = str(ROOT / "src" / "train.py")
CFG = ROOT / "configs"
LOCKS = ROOT / "log" / "gpu_slots"


def load(path):
    return yaml.safe_load(path.read_text())


def free_mib(gpu):
    q = subprocess.run(["nvidia-smi", f"--id={gpu}", "--query-gpu=memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    return int(q.stdout.split()[0])


def acquire_slot(gpu, slots, need_mib, poll=120):
    """Hold one of the card's file-lock slots for the run; closing releases it."""
    LOCKS.mkdir(parents=True, exist_ok=True)
    while True:
        for k in range(slots):
            fh = (LOCKS / f"gpu{gpu}.slot{k}").open("w")
            try:
                fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                fh.close()
                continue                       # another runner holds this slot
            if free_mib(gpu) >= need_mib:
                return fh
            fh.close()                         # slot free but the card is not
            break
        print(f"    waiting: gpu {gpu} full ({slots} slot(s), "
              f"{free_mib(gpu) // 1024} GiB free, need {need_mib // 1024})", flush=True)
        time.sleep(poll)


def tuned_lr(method, split, local_epochs):
    """Selected LR: the re-bracket at tuned E if present, else the per-split ladder."""
    from tune_full import LR_KEY, REFERENCE_SPLIT, SPLIT_INDEPENDENT
    if method not in LR_KEY:
        return None                       # a generated one-off ladder config
    if method in SPLIT_INDEPENDENT:
        split = REFERENCE_SPLIT
    stems = ["tune_full"] if local_epochs == 1 else ["tune_rebracket", "tune_full"]
    for stem in stems:
        path = ROOT / "results" / f"{stem}_{method}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row["split"] == split:
                return LR_KEY[method], float(row["best"])
    return None


def build_cmd(base, method_cfg, split_cfg, setting, seed, local_epochs, gamma, outdir,
              method=None, clients_per_round=None):
    args = {}
    args.update({k.replace("_", "-"): v for k, v in base["protocol"].items()})
    args.update({k.replace("_", "-"): v for k, v in base[setting].items()})
    # split first, then method: method keys win (centralized carries its own split)
    args.update(split_cfg["args"])
    args.update(method_cfg["args"])
    # per-split overrides from the method yaml
    args.update(method_cfg.get("per_split", {}).get(split_cfg["split"], {}))
    # the tuning of record lives in results/tune_*.json, not in the method yaml
    if method and (sel := tuned_lr(method, split_cfg["split"], local_epochs)):
        key, lr = sel
        args[key] = lr
        if method in SINGLE_LR:
            args["scalar-lr"] = lr
    args["seed"] = seed
    args["local-epochs"] = local_epochs
    if gamma is not None:
        args["mars-gamma"] = gamma
    if clients_per_round is not None:
        args["clients-per-round"] = clients_per_round
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
    ap.add_argument("--suite", choices=["headline", "splits", "gamma", "participation"])
    ap.add_argument("--setting", choices=["proxy", "full", "full30"], default="full")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["single_class"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--local-epochs", nargs="+", type=int, default=[1])
    ap.add_argument("--gpu", required=True)
    ap.add_argument("--min-free-gb", type=int,
                    help="wait for this much free GPU memory before each run "
                         "(default: 35 full-resolution, 18 proxy)")
    ap.add_argument("--slots", type=int, default=1,
                    help="how many of our runs may share one GPU (default 1)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    base = load(CFG / "base.yaml")
    seeds = a.seeds or base["benchmark"]["seeds"]
    methods, splits, epochs = a.methods, a.splits, a.local_epochs
    gammas, samplings = [None], [None]

    federated = ["fedavg", "scaffold", "fedlion", "fedmuon", "fedmars", "fedolion"]
    if a.suite == "headline":       # main table: natural split, each method at its
        methods = federated + ["centralized", "centralized_muon"]   # tuned local epochs
    elif a.suite == "splits":       # heterogeneity sweep at matched local compute (E=1)
        methods = methods or federated
        splits = ["single_class", "dirichlet_0.5", "dirichlet_10", "dirichlet_1000"]
        epochs = [1]
    elif a.suite == "gamma":        # does MARS help on top of tuned FedMuon?
        methods = ["fedmars"]
        gammas = base["benchmark"]["gamma_ablation"]
    elif a.suite == "participation":  # clients sampled per round, each method at its
        methods = methods or federated  # tuned local epochs; K=6 is the headline row
        samplings = base["benchmark"]["clients_per_round"]
    if not methods:
        sys.exit("give --suite or --methods")

    def tuned_epochs(m):
        """The local-epoch count selected by the ladder on the reference split."""
        path = ROOT / "results" / f"tune_epochs_{m}.json"
        if not path.exists():
            sys.exit(f"missing {path.name}: run tune_full.py --epochs first")
        return int(json.loads(path.read_text())[0]["best"])

    todo, skipped = [], 0
    for m, sp, e, s, g, k in itertools.product(methods, splits, epochs, seeds, gammas,
                                               samplings):
        if a.suite in ("headline", "participation"):
            e = 1 if m.startswith("centralized") else tuned_epochs(m)
        tag = (f"{m}_E{e}" + (f"_K{k}" if k is not None else "") + f"_s{s}"
               + (f"_g{g:g}" if g is not None else ""))
        outdir = ROOT / "results" / "runs" / a.setting / sp / tag
        if (outdir / "summary.csv").exists():
            skipped += 1
            continue
        todo.append((m, sp, e, s, g, k, outdir))

    # a run peaks at ~31 GiB; per-card concurrency is set by --slots
    need_mib = (a.min_free_gb or {"full": 35, "full30": 35}.get(a.setting, 18)) * 1024
    failed = []
    print(f"{len(todo)} runs to do, {skipped} already done", flush=True)
    for i, (m, sp, e, s, g, k, outdir) in enumerate(todo, 1):
        cmd = build_cmd(base, load(CFG / "method" / f"{m}.yaml"),
                        load(CFG / "split" / f"{sp}.yaml"), a.setting, s, e, g, outdir,
                        method=m, clients_per_round=k)
        print(f"[{i}/{len(todo)}] {outdir.relative_to(ROOT)}", flush=True)
        if a.dry_run:
            print("   ", " ".join(cmd), flush=True)
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        newest = max((f.stat().st_mtime for f in outdir.iterdir() if f.name != ".lock"),
                     default=0)
        if time.time() - newest < 1800:
            print("    in progress (fresh output), skipping", flush=True)
            failed.append(str(outdir.relative_to(ROOT)))
            continue
        claim = (outdir / ".lock").open("w")   # one writer per row, across runners
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("    in progress elsewhere, skipping", flush=True)
            claim.close()
            failed.append(str(outdir.relative_to(ROOT)))
            continue
        slot = acquire_slot(a.gpu, a.slots, need_mib)
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            slot.close()               # releases the card slot
            claim.close()              # and the row
        print(f"    exit {r.returncode}", flush=True)
        if r.returncode:
            failed.append(str(outdir.relative_to(ROOT)))

    if failed:  # nonzero exit so the queue script retries (finished runs are skipped)
        sys.exit(f"{len(failed)} run(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

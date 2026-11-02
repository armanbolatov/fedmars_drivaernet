#!/usr/bin/env python
"""Benchmark runner: one process per GPU over methods x splits x seeds x epochs.

Examples:
  python scripts/benchmark.py --suite headline --gpu 0
  python scripts/benchmark.py --suite splits --gpu 0
  python scripts/benchmark.py --suite participation --gpu 0
  python scripts/benchmark.py --suite epochs --setting full30 --gpu 0
  python scripts/benchmark.py --methods fedmuon --splits dirichlet_10 --local-epochs 4 --gpu 0

FedMARS hyperparameter grid example:
  python scripts/benchmark.py --methods fedmars --splits single_class \\
      --local-epochs 2 4 --muon-lrs 0.001 0.002 0.003 0.005 \\
      --scalar-lrs 0.00003 0.0001 --muon-wds 0.01 0.05 0.1 \\
      --fed-betas 0.8 0.9 0.95 --setting proxy --model omniarch \\
      --run-folder omniarch_fedmars_tuning --gpu 0 --dry-run

Finished rows (summary.csv) are skipped, a row another runner is writing is skipped,
and a row with a checkpoint resumes, so runners on several cards can share one work
list. Card slots are file locks under log/gpu_slots (--slots caps our runs per card).
scripts/run_bench.sh wraps this in a retry loop for a shared machine.
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


def tuned_lr(method, split, local_epochs, suffix=""):
    """Selected LR: the re-bracket at tuned E if present, else the per-split ladder."""
    from tune_full import LR_KEY, REFERENCE_SPLIT, SPLIT_INDEPENDENT
    if method not in LR_KEY:
        return None                       # a generated one-off ladder config
    if method in SPLIT_INDEPENDENT:
        split = REFERENCE_SPLIT
    stems = ["tune_full"] if local_epochs == 1 else ["tune_rebracket", "tune_full"]
    for stem in stems:
        path = ROOT / "results" / f"{stem}_{method}{suffix}.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text()):
            if row["split"] == split:
                return LR_KEY[method], float(row["best"])
    return None


def build_cmd(base, method_cfg, split_cfg, setting, seed, local_epochs, gamma, outdir,
              method=None, clients_per_round=None, model=None, overrides=None):
    args = {}
    args.update({k.replace("_", "-"): v for k, v in base["protocol"].items()})
    args.update({k.replace("_", "-"): v for k, v in base[setting].items()})
    # split first, then method: method keys win (centralized carries its own split)
    args.update(split_cfg["args"])
    args.update(method_cfg["args"])
    # per-split overrides from the method yaml
    args.update(method_cfg.get("per_split", {}).get(split_cfg["split"], {}))
    # The tuning record supplies the default LR, unless an explicit grid override
    # below replaces it.
    suffix = f"-{model}" if model else ""
    if method and (sel := tuned_lr(method, split_cfg["split"], local_epochs, suffix)):
        key, lr = sel
        args[key] = lr
        if method in SINGLE_LR:
            args["scalar-lr"] = lr

    # Explicit benchmark-grid values take precedence over YAML and tuning records.
    if overrides:
        args.update({
            key.replace("_", "-"): value
            for key, value in overrides.items()
            if value is not None
        })

    if model:
        args["model"] = model
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


def _tag_value(value):
    """Filesystem-friendly, reasonably readable representation of a float."""
    return format(value, ".8g").replace("-", "m").replace(".", "p").replace("+", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite",
                    choices=["headline", "splits", "gamma", "participation", "epochs"])
    ap.add_argument("--setting", choices=["proxy", "full", "full30"], default="full")
    ap.add_argument("--model", default="transolver",
                    help="configs/model/<name>.yaml; another model keeps its runs and "
                         "tuning records in their own namespace")
    ap.add_argument("--run-folder", default=None,
                    help="custom folder under results/runs; defaults to "
                         "<setting>[-<model>], e.g. omniarch_fedmars_tuning")
    ap.add_argument("--methods", nargs="+")
    ap.add_argument("--splits", nargs="+", default=["single_class"])
    ap.add_argument("--seeds", nargs="+", type=int)
    ap.add_argument("--local-epochs", nargs="+", type=int)
    ap.add_argument("--muon-lrs", nargs="+", type=float,
                    help="FedMARS Muon learning-rate grid")
    ap.add_argument("--scalar-lrs", nargs="+", type=float,
                    help="FedMARS scalar-parameter learning-rate grid")
    ap.add_argument("--muon-wds", nargs="+", type=float,
                    help="FedMARS Muon weight-decay grid")
    ap.add_argument("--fed-betas", nargs="+", type=float,
                    help="FedMARS beta grid")
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
    methods, splits, epochs = a.methods, a.splits, a.local_epochs or [1]
    gammas, samplings = [None], [None]
    model = None if a.model == "transolver" else a.model
    suffix = f"-{model}" if model else ""

    grid_requested = any([
        a.muon_lrs is not None,
        a.scalar_lrs is not None,
        a.muon_wds is not None,
        a.fed_betas is not None,
    ])
    if grid_requested and methods is not None and any(m != "fedmars" for m in methods):
        sys.exit("FedMARS hyperparameter grids currently support --methods fedmars only")

    muon_lrs = a.muon_lrs if a.muon_lrs is not None else [None]
    scalar_lrs = a.scalar_lrs if a.scalar_lrs is not None else [None]
    muon_wds = a.muon_wds if a.muon_wds is not None else [None]
    fed_betas = a.fed_betas if a.fed_betas is not None else [None]

    federated = ["fedavg", "scaffold", "fedlion", "fedmuon", "fedmars", "fedolion"]
    if a.suite == "headline":       # main table: natural split, each method at its
        methods = methods or federated + ["centralized", "centralized_muon"]  # tuned E
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
    elif a.suite == "epochs":       # local-epoch ladder on the Dirichlet splits
        methods = methods or federated
        splits = ["dirichlet_0.5", "dirichlet_10", "dirichlet_1000"]
        epochs = a.local_epochs or [4, 16]

    if not methods:
        sys.exit("give --suite or --methods")
    if grid_requested and any(m != "fedmars" for m in methods):
        sys.exit("FedMARS hyperparameter grids currently support fedmars only")

    def tuned_epochs(m):
        """The local-epoch count selected by the ladder on the reference split."""
        path = ROOT / "results" / f"tune_epochs_{m}{suffix}.json"
        if not path.exists():
            sys.exit(f"missing {path.name}: run tune_full.py --epochs first")
        return int(json.loads(path.read_text())[0]["best"])

    # Preserve the historical directory layout unless --run-folder is supplied.
    run_folder = a.run_folder or f"{a.setting}{suffix}"

    todo, skipped = [], 0
    grid = itertools.product(
        methods, splits, epochs, seeds, gammas, samplings,
        muon_lrs, scalar_lrs, muon_wds, fed_betas
    )

    for m, sp, e, s, g, k, mlr, slr, wd, beta in grid:
        if a.suite in ("headline", "participation"):
            e = 1 if m.startswith("centralized") else tuned_epochs(m)

        overrides = {
            "muon_lr": mlr,
            "scalar_lr": slr,
            "muon_wd": wd,
            "fed_beta": beta,
        }

        hp_parts = []
        if mlr is not None:
            hp_parts.append(f"mlr{_tag_value(mlr)}")
        if slr is not None:
            hp_parts.append(f"slr{_tag_value(slr)}")
        if wd is not None:
            hp_parts.append(f"wd{_tag_value(wd)}")
        if beta is not None:
            hp_parts.append(f"beta{_tag_value(beta)}")
        hp_tag = "_" + "_".join(hp_parts) if hp_parts else ""

        tag = (f"{m}_E{e}" + (f"_K{k}" if k is not None else "") + f"_s{s}"
               + (f"_g{g:g}" if g is not None else "") + hp_tag)
        outdir = ROOT / "results" / "runs" / run_folder / sp / tag

        if (outdir / "summary.csv").exists():
            skipped += 1
            continue
        todo.append((m, sp, e, s, g, k, outdir, overrides))

    todo.sort(key=lambda t: -t[2] * (t[5] or 6))   # longest rows first
    # a run peaks at ~31 GiB; per-card concurrency is set by --slots
    need_mib = (a.min_free_gb or {"full": 35, "full30": 35}.get(a.setting, 18)) * 1024
    failed = []
    print(f"{len(todo)} runs to do, {skipped} already done", flush=True)

    for i, (m, sp, e, s, g, k, outdir, overrides) in enumerate(todo, 1):
        cmd = build_cmd(base, load(CFG / "method" / f"{m}.yaml"),
                        load(CFG / "split" / f"{sp}.yaml"), a.setting, s, e, g, outdir,
                        method=m, clients_per_round=k, model=model, overrides=overrides)
        print(f"[{i}/{len(todo)}] {outdir.relative_to(ROOT)}", flush=True)
        if a.dry_run:
            print("   ", " ".join(cmd), flush=True)
            continue
        outdir.mkdir(parents=True, exist_ok=True)
        newest = max((f.stat().st_mtime for f in outdir.iterdir() if f.name != ".lock"),
                     default=0)
        if time.time() - newest < 7200:        # longer than any round: a run whose
            print("    in progress (fresh output), skipping", flush=True)
            failed.append(str(outdir.relative_to(ROOT)))
            continue
        # card first, then row: a runner waiting on a busy card must not hold a row
        slot = acquire_slot(a.gpu, a.slots, need_mib)
        claim = (outdir / ".lock").open("w")
        try:
            fcntl.flock(claim, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            print("    in progress elsewhere, skipping", flush=True)
            claim.close()
            slot.close()
            failed.append(str(outdir.relative_to(ROOT)))
            continue
        if (outdir / "summary.csv").exists():
            claim.close()
            slot.close()
            continue
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=a.gpu,
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        try:
            r = subprocess.run(cmd, cwd=str(ROOT), env=env,
                               pass_fds=(slot.fileno(), claim.fileno()),
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finally:
            slot.close()
            claim.close()
        print(f"    exit {r.returncode}", flush=True)
        if r.returncode:
            failed.append(str(outdir.relative_to(ROOT)))

    if failed:
        sys.exit(f"{len(failed)} run(s) failed: {', '.join(failed)}")


if __name__ == "__main__":
    main()

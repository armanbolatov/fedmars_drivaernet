#!/usr/bin/env python
"""Figures from results/runs (seaborn: mean across seeds, +-1 std band/bars).

Each figure is produced only if its data exists:

  convergence_full   c_d MAE vs round, headline methods (full task)
  trainloss_full     training field-MSE vs round (per-round where logged)
  combined_full      the two above side by side
  gamma              c_d MAE vs MARS gamma (full task, paired seeds)
  heterogeneity      c_d MAE vs split per method (proxy alpha sweep)
  localsteps         c_d MAE vs local epochs E at equal gradient work (proxy)
  drift_rounds       client drift / aggregate step vs round (full task)
  drift_localsteps   the same ratio vs local epochs E (proxy) -- why MARS needs E>1

Every coefficient figure is emitted for lift as well, with a `_lift` suffix
(convergence_lift_full, gamma_lift, heterogeneity_lift, localsteps_lift).

Usage: python scripts/plot.py            # writes results/figures/*.pdf and .png
"""
import csv
import re
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="paper", font_scale=1.15)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"
N_TAIL = 5

LABELS = {"fedavg": "FedAvg", "scaffold": "SCAFFOLD-Adam", "fedmuon": "FedMuon",
          "fedmars": "FedMARS-M", "centralized": "Centralized AdamW",
          "centralized_muon": "Centralized Muon", "centralized_mars": "Centralized MARS-M"}
PALETTE = {LABELS[m]: c for m, c in {
    "fedavg": "tab:gray", "scaffold": "tab:orange", "fedmuon": "tab:blue",
    "fedmars": "tab:red", "centralized": "tab:green",
    "centralized_muon": "tab:purple", "centralized_mars": "tab:brown"}.items()}
DASHED = {LABELS[m] for m in LABELS if m.startswith("centralized")}

# (history column, axis label, file suffix). Lift is integrated from the SAME
# model outputs as drag -- surface integration along z instead of x -- so every
# error figure is produced for both. Expect the lift curves to sit higher and
# noisier: c_l is a small residual of cancelling integrals (see README).
COEFFS = (
    ("c_d_mae", r"$c_d$ MAE", ""),
    ("c_l_mae", r"$c_l$ MAE", "_lift"),
)


# ---------------------------------------------------------------- data access

def read_history(run_dir, metric="c_d_mae"):
    """[(round, value)] for body=all rows of one run."""
    h = run_dir / "history.csv"
    if not h.exists():
        return []
    return [(int(r["round"]), float(r[metric])) for r in csv.DictReader(h.open())
            if r.get("body") == "all" and int(r["round"]) > 0 and r.get(metric)]


def read_stat(run_dir, col):
    """[(round, value)] from round_stats.csv, which every method logs per round."""
    f = run_dir / "round_stats.csv"
    if not f.exists():
        return []
    return [(int(r["round"]), float(r[col])) for r in csv.DictReader(f.open())
            if r.get(col) not in (None, "", "nan")]


def read_train_loss(run_dir):
    """Per-round train loss when logged (round_stats/train_loss.csv), else the
    eval-round samples from history.csv."""
    for name in ("round_stats.csv", "train_loss.csv"):
        f = run_dir / name
        if f.exists():
            return [(int(r["round"]), float(r["train_loss"]))
                    for r in csv.DictReader(f.open())]
    return read_history(run_dir, "train_loss")


def tail_score(run_dir, metric="c_d_mae"):
    """Mean MAE over the last N_TAIL evaluated rounds (the paper convention)."""
    pts = read_history(run_dir, metric)
    if not pts:
        return None
    last = sorted({r for r, _ in pts})[-N_TAIL:]
    return statistics.mean(v for r, v in pts if r in last)


def runs(setting, split, pattern):
    """(seed, run_dir) for completed runs whose tag matches the regex."""
    base = RUNS / setting / split
    if not base.is_dir():
        return []
    out = []
    for d in base.iterdir():
        m = re.fullmatch(pattern, d.name)
        if m and (d / "summary.csv").exists():
            out.append((int(m.group("seed")), d))
    return sorted(out)


def curve_frame(reader):
    """Tidy frame (method, seed, round, value) over all headline methods."""
    rows = []
    for m, label in LABELS.items():
        for seed, d in runs("full", "single_class", rf"{m}_E1_s(?P<seed>\d+)"):
            rows += [{"method": label, "seed": seed, "round": r, "value": v}
                     for r, v in reader(d)]
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- figures

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    print(f"wrote {name}")


def style_lines(ax):
    """Dash the centralized (ceiling) lines; keep the legend in sync."""
    handles, labels = ax.get_legend_handles_labels()
    for artist, label in zip(ax.get_lines(), labels):
        if label in DASHED:
            artist.set_linestyle("--")
    for handle, label in zip(handles, labels):
        if label in DASHED:
            handle.set_linestyle("--")
    ax.legend(handles, labels, frameon=False, fontsize=8)


def draw_convergence(ax, frame, ylabel):
    sns.lineplot(frame, x="round", y="value", hue="method", palette=PALETTE,
                 errorbar="sd", ax=ax, lw=1.6)
    ax.set(xlabel="communication round", ylabel=ylabel, yscale="log")
    style_lines(ax)


def fig_convergence():
    for metric, label, suffix in COEFFS:
        frame = curve_frame(lambda d, m=metric: read_history(d, m))
        if frame.empty or frame["method"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        draw_convergence(ax, frame, label)
        save(fig, f"convergence{suffix}_full")


def fig_trainloss():
    frame = curve_frame(read_train_loss)
    if frame.empty or frame["method"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    draw_convergence(ax, frame, "training loss (field MSE)")
    save(fig, "trainloss_full")


def fig_combined():
    loss, cd = curve_frame(read_train_loss), curve_frame(read_history)
    if loss.empty or cd.empty or cd["method"].nunique() < 2:
        return
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(10, 3.8))
    draw_convergence(ax_l, loss, "training loss (field MSE)")
    draw_convergence(ax_r, cd, r"$c_d$ MAE")
    ax_r.get_legend().remove()
    save(fig, "combined_full")


def fig_gamma():
    for metric, label, suffix in COEFFS:
        rows = [{"gamma": f"{g:g}", "seed": s, "value": v}
                for g in (0.0, 0.01, 0.04)
                for s, d in runs("full", "single_class",
                                 rf"fedmars_E1_s(?P<seed>\d+)_g{re.escape(f'{g:g}')}")
                if (v := tail_score(d, metric)) is not None]
        frame = pd.DataFrame(rows)
        if frame.empty or frame["gamma"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(4, 3.2))
        sns.pointplot(frame, x="gamma", y="value", errorbar="sd", capsize=0.15,
                      color=PALETTE["FedMARS-M"], ax=ax)
        ax.set(xlabel=r"MARS $\gamma$", ylabel=label)
        save(fig, f"gamma{suffix}")


def dodge_for(frame):
    """seaborn divides by (n_hue_levels - 1), so dodging a single-method frame
    raises ZeroDivisionError -- which happens whenever a suite is partly run."""
    return 0.25 if frame["method"].nunique() > 1 else False


def sweep_frame(setting, splits, methods, tag, metric="c_d_mae"):
    rows = [{"x": label, "method": LABELS[m], "seed": s, "value": v}
            for label, sub in splits for m in methods
            for s, d in runs(setting, sub, tag(m))
            if (v := tail_score(d, metric)) is not None]
    return pd.DataFrame(rows)


def fig_heterogeneity():
    splits = [("single-\nclass", "single_class"), (r"$\alpha$=0.5", "dirichlet_0.5"),
              (r"$\alpha$=10", "dirichlet_10"), (r"$\alpha$=1000", "dirichlet_1000")]
    for metric, label, suffix in COEFFS:
        frame = sweep_frame("proxy", splits,
                            ("fedavg", "scaffold", "fedmuon", "fedmars"),
                            lambda m: rf"{m}_E1_s(?P<seed>\d+)", metric)
        if frame.empty or frame["x"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.pointplot(frame, x="x", y="value", hue="method", palette=PALETTE,
                      errorbar="sd", capsize=0.1, dodge=dodge_for(frame), ax=ax)
        ax.set(xlabel="client heterogeneity (high to low)", ylabel=label,
               yscale="log")
        ax.legend(frameon=False, fontsize=8)
        save(fig, f"heterogeneity{suffix}")


def fig_localsteps():
    for metric, label, suffix in COEFFS:
        rows = [{"x": str(e), "method": LABELS[m], "seed": s, "value": v}
                for m in ("fedavg", "scaffold", "fedmuon", "fedmars")
                for e in (1, 2, 4, 8)
                for s, d in runs("proxy", "single_class",
                                 rf"{m}_E{e}_s(?P<seed>\d+)_R\d+")
                if (v := tail_score(d, metric)) is not None]
        frame = pd.DataFrame(rows)
        if frame.empty or frame["x"].nunique() < 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.pointplot(frame, x="x", y="value", hue="method", palette=PALETTE,
                      errorbar="sd", capsize=0.1, dodge=dodge_for(frame), ax=ax)
        ax.set(xlabel="local epochs E (rounds = 80/E: equal gradient work)",
               ylabel=label, yscale="log")
        ax.legend(frameon=False, fontsize=8)
        save(fig, f"localsteps{suffix}")


def fig_drift_rounds():
    """How far clients pull apart relative to the step actually taken. This is the
    quantity MARS variance reduction corrects, so it says whether there is anything
    for gamma to do at E=1."""
    rows = [{"method": LABELS[m], "seed": s, "round": r, "value": v}
            for m in ("fedavg", "scaffold", "fedmuon", "fedmars")
            for s, d in runs("full", "single_class", rf"{m}_E1_s(?P<seed>\d+)")
            for r, v in read_stat(d, "drift_ratio")]
    frame = pd.DataFrame(rows)
    if frame.empty:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.lineplot(frame, x="round", y="value", hue="method", palette=PALETTE,
                 errorbar="sd", ax=ax, lw=1.6)
    ax.set(xlabel="communication round", ylabel="client drift / aggregate update",
           yscale="log")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "drift_rounds")


def fig_drift_localsteps():
    rows = [{"x": str(e), "method": LABELS[m], "seed": s,
             "value": statistics.mean(vals)}
            for m in ("fedavg", "scaffold", "fedmuon", "fedmars")
            for e in (1, 2, 4, 8)
            for s, d in runs("proxy", "single_class", rf"{m}_E{e}_s(?P<seed>\d+)_R\d+")
            if (vals := [v for _, v in read_stat(d, "drift_ratio")])]
    frame = pd.DataFrame(rows)
    if frame.empty or frame["x"].nunique() < 2:
        return
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.pointplot(frame, x="x", y="value", hue="method", palette=PALETTE,
                  errorbar="sd", capsize=0.1, dodge=dodge_for(frame), ax=ax)
    ax.set(xlabel="local epochs E", ylabel="client drift / aggregate update",
           yscale="log")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "drift_localsteps")


if __name__ == "__main__":
    fig_convergence()
    fig_trainloss()
    fig_combined()
    fig_gamma()
    fig_heterogeneity()
    fig_localsteps()
    fig_drift_rounds()
    fig_drift_localsteps()

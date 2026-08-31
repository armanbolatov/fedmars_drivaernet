#!/usr/bin/env python
"""Figures from results/. Each writes a PDF, a PNG and the CSV behind it.

  python scripts/plot.py

Drawn for whatever exists, so this is safe to run mid-campaign:
  heterogeneity  error against client heterogeneity, at one local epoch
  local_epochs   error against the local-epoch budget
  convergence    error against communication round (needs benchmark runs)
  ladders        every learning rate walked (appendix; selection is on c_d)

Style set once globally; colours are semantic and fixed per series (CUD palette).
"""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

RC = {
    "axes.facecolor": "#EAEAF2", "figure.facecolor": "white",
    "axes.edgecolor": "white", "axes.linewidth": 0.0,
    "grid.color": "white", "grid.linewidth": 1.0, "axes.grid": True,
    "axes.axisbelow": True,
    "xtick.bottom": False, "ytick.left": False,
    "xtick.color": "#4A4A4A", "ytick.color": "#4A4A4A",
    "axes.labelcolor": "#2A2A2A", "text.color": "#2A2A2A",
    "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.solid_capstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}
try:
    import seaborn as sns
    sns.set_theme(style="darkgrid", context="notebook", rc=RC)
except ImportError:                       # same look without the dependency
    plt.rcParams.update(RC)

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

# label, colour, marker, linestyle. FedOLion = FedMuon variant (blue, dashed);
# ceilings = gray references; FedAvg = the vermilion baseline.
SERIES = {
    "fedmuon":          ("FedMuon",           "#0173b2", "o", "-"),
    "fedmars":          ("FedMARS-M",         "#029e73", "s", "-"),
    "fedolion":         ("FedOLion",          "#0173b2", "D", "--"),
    "scaffold":         ("SCAFFOLD-Adam",     "#56b4e9", "^", "-"),
    "fedlion":          ("FedLion",           "#cc78bc", "P", "-"),
    "fedavg":           ("FedAvg",            "#d55e00", "v", "-"),
    "centralized":      ("Centralized AdamW", "#949494", "X", (0, (5, 3))),
    "centralized_muon": ("Centralized Muon",  "#949494", "*", (0, (1, 1.5))),
}
ORDER = list(SERIES)
SPLITS = [("single_class", "single-class"), ("dirichlet_0.5", r"$\alpha$=0.5"),
          ("dirichlet_10", r"$\alpha$=10"), ("dirichlet_1000", r"$\alpha$=1000")]
SPLIT_COLORS = [matplotlib.cm.viridis(t) for t in np.linspace(0.05, 0.8, 4)]
METRICS = [("c_d_mae", "drag MAE  $c_d$"), ("c_l_mae", "lift MAE  $c_l$")]
EPOCHS = (1, 2, 4, 8, 16)
CEILINGS = ("centralized", "centralized_muon")   # hold all data: split-independent


def log_y(ax):
    """Plain numbers instead of powers of ten, with labelled 2/3/5 minor ticks."""
    ax.set_yscale("log")
    ax.minorticks_on()
    ax.grid(which="minor", color="white", lw=0.5, alpha=0.6)
    ax.tick_params(which="minor", length=0)
    fmt = FuncFormatter(lambda v, _: f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_locator(LogLocator(subs=(2, 3, 5)))
    ax.yaxis.set_minor_formatter(fmt)


def save(fig, name, rows, header):
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=200)
    plt.close(fig)
    with (TABLES / f"{name}.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    print(f"  {name}: {len(rows)} rows")


def selections(stem):
    """{method: {split: record}} from the ladder result files that exist."""
    out = {}
    for m in ORDER:
        path = ROOT / "results" / f"{stem}_{m}.json"
        if path.exists():
            out[m] = {r["split"]: r for r in json.loads(path.read_text())}
    return out


def scores(run, lo=20, hi=30):
    """Mean of each metric over the evaluations in rounds [lo, hi]."""
    if not (run / "summary.csv").exists():
        return {}
    rows = [r for r in csv.DictReader((run / "history.csv").open())
            if r.get("body") == "all" and lo <= int(r["round"]) <= hi]
    out = {}
    for key, _ in METRICS:
        vals = [float(r[key]) for r in rows if r.get(key) not in (None, "", "nan")]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out


def selected_run(sel, method, split, epochs):
    """The tuning run at this method's selected learning rate for the split."""
    rec = sel.get(method, {}).get(split)
    if not rec:
        return None
    lr = float(rec["best"])
    return RUNS / "full30" / split / f"{method}_lr{lr:g}_{split}_E{epochs}_s42"


def tuned_epochs(method):
    path = ROOT / "results" / f"tune_epochs_{method}.json"
    return int(json.loads(path.read_text())[0]["best"]) if path.exists() else None


def legend_handle(m):
    label, colour, marker, ls = SERIES[m]
    return Line2D([], [], color=colour, marker=marker, ms=5, ls=ls, label=label)


def two_panel(name, xlabel, series, xticks=None, log2_x=False,
              flat_ceilings=False, sparse_markers=False):
    """Drag left, lift right, one legend below; no titles (captions live in the
    paper). `series` = {method: [(x, {metric: value}), ...]}."""
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.2), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    rows = []
    for ax, (key, ylabel) in zip(axes, METRICS):
        for m, pts in series.items():
            label, colour, marker, ls = SERIES[m]
            xy = [(x, s[key]) for x, s in pts if key in s]
            if not xy:
                continue
            if flat_ceilings and m in CEILINGS:
                ax.axhline(xy[0][1], color=colour, ls=ls, lw=1.5, zorder=2)
                continue
            xs, ys = [p[0] for p in xy], [p[1] for p in xy]
            if sparse_markers:            # dense curves: line + a few dots
                ax.plot(xs, ys, color=colour, ls=ls, zorder=3)
                step = max(1, len(xs) // 6)
                ax.scatter(xs[::step], ys[::step], s=22, color=colour,
                           marker=marker, zorder=4)
            else:                         # 4-5 points: every point is information
                ax.plot(xs, ys, color=colour, ls=ls, marker=marker, ms=5, zorder=3)
        log_y(ax)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log2_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_minor_formatter(NullFormatter())
        if xticks:
            ax.set_xticks([t[0] for t in xticks])
            ax.set_xticklabels([t[1] for t in xticks])
    for m, pts in series.items():
        rows += [[m, x] + [s.get(k, "") for k, _ in METRICS] for x, s in pts]
    handles = [legend_handle(m) for m in ORDER if m in series]
    fig.legend(handles=handles, loc="outside lower center",
               ncols=len(handles) if len(handles) <= 6 else 4)
    save(fig, name, rows, ["method", "x"] + [k for k, _ in METRICS])


def fig_heterogeneity():
    """Every method at E=1, per-split LR: benchmark rows once the sweep is
    complete (last-5-round mean), else the 30-round tuning runs."""
    bench = {m: {k: RUNS / "full" / ("single_class" if m in CEILINGS else k)
                 / f"{m}_E1_s42" for k, _ in SPLITS} for m in ORDER}
    complete = all((r / "summary.csv").exists()
                   for m in ORDER for r in bench[m].values())
    sel = selections("tune_full")
    series = {}
    for m in ORDER:
        pts = []
        for i, (key, _) in enumerate(SPLITS):
            if complete:
                s = scores(bench[m][key], lo=96, hi=100)
            else:
                run = selected_run(sel, m, "single_class" if m in CEILINGS else key, 1)
                s = scores(run) if run else {}
            if s:
                pts.append((i, s))
        if pts:
            series[m] = pts
    two_panel("heterogeneity",
              "client distribution  (most heterogeneous on the left)", series,
              xticks=list(enumerate(lbl for _, lbl in SPLITS)), flat_ceilings=True)


def fig_local_epochs():
    sel = selections("tune_full")
    series = {}
    for m in ORDER:
        if m in CEILINGS:
            continue
        pts = []
        for e in EPOCHS:
            run = selected_run(sel, m, "single_class", e)
            if run and (s := scores(run)):
                pts.append((e, s))
        if len(pts) >= 2:
            series[m] = pts
    two_panel("local_epochs",
              "local epochs per round", series,
              xticks=[(e, str(e)) for e in EPOCHS], log2_x=True)


def fig_convergence():
    """Benchmark runs on the reference split: each method at its tuned local-epoch
    budget (the ceilings at E=1), so exactly one run per method."""
    base = RUNS / "full" / "single_class"
    if not base.is_dir():
        return
    series = {}
    for m in ORDER:
        e = 1 if m in CEILINGS else tuned_epochs(m)
        run = base / f"{m}_E{e}_s42"
        if e is None or not (run / "history.csv").exists():
            continue
        pts = []
        for r in csv.DictReader((run / "history.csv").open()):
            if r.get("body") != "all":
                continue
            vals = {k: float(r[k]) for k, _ in METRICS
                    if r.get(k) not in (None, "", "nan")}
            if vals:
                pts.append((int(r["round"]), vals))
        if pts:
            series[m] = sorted(pts)
    two_panel("convergence",
              "communication round", series, sparse_markers=True)


def fig_ladders():
    """Appendix, 2x4: every rung walked, one panel per method, the four
    distributions on the ordered viridis ramp. Selection is on c_d."""
    sel = selections("tune_full")
    methods = [m for m in ORDER if m in sel]
    if not methods:
        return
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.8), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.06, wspace=0.03, hspace=0.05)
    rows = []
    for i, (ax, m) in enumerate(zip(axes.flat, methods)):
        finite = [float(s) for rec in sel[m].values()
                  for s in rec["evaluated"].values() if s != "diverged"]
        # a rung four times worse than the panel's best says only "it failed"
        cap = min(finite) * 4
        for i, (key, _) in enumerate(SPLITS):
            rec = sel[m].get(key)
            if not rec:
                continue
            xs, ys, div = [], [], []
            for lr, s in sorted(rec["evaluated"].items(), key=lambda kv: float(kv[0])):
                if s == "diverged" or float(s) > cap:
                    div.append(float(lr))
                else:
                    xs.append(float(lr))
                    ys.append(float(s))
                rows.append([m, key, lr, s])
            ax.plot(xs, ys, color=SPLIT_COLORS[i], marker="o", ms=4, zorder=3)
            ax.plot(div, [cap] * len(div), ls="none", marker="^", ms=6,
                    mfc="white", mec=SPLIT_COLORS[i], mew=1.2, zorder=3)
            if rec["evaluated"].get(rec["best"]) not in (None, "diverged"):
                ax.plot([float(rec["best"])], [float(rec["evaluated"][rec["best"]])],
                        ls="none", marker="o", ms=11, mfc="none", mec="#2A2A2A",
                        mew=1.3, zorder=4)
        ax.set_xscale("log")
        log_y(ax)
        ax.set_ylim(top=cap * 1.3)
        ax.set_title(SERIES[m][0], loc="left", fontsize=10)
        if i % 4 == 0:                    # y-label once per row, not eight times
            ax.set_ylabel(METRICS[0][1], fontsize=9)
        if i >= 4:                        # x-label on the bottom row only
            ax.set_xlabel("learning rate", fontsize=9)
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=SPLIT_COLORS[i], marker="o", ms=4, label=lbl)
               for i, (_, lbl) in enumerate(SPLITS)]
    handles.append(Line2D([], [], ls="none", marker="^", ms=6, mfc="white",
                          mec="#4A4A4A", mew=1.2, label="diverged"))
    handles.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="none",
                          mec="#2A2A2A", mew=1.3, label="selected"))
    fig.legend(handles=handles, loc="outside lower center", ncols=6)
    save(fig, "ladders", rows, ["method", "split", "learning_rate", "c_d_mae"])


if __name__ == "__main__":
    print("figures ->", OUT)
    for fn in (fig_heterogeneity, fig_local_epochs, fig_convergence, fig_ladders):
        fn()

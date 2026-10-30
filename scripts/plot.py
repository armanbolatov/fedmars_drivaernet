#!/usr/bin/env python
"""Figures from results/. Each writes a PDF, a PNG and the CSV behind it.

  python scripts/plot.py

Drawn for whatever exists, so this is safe to run mid-campaign:
  convergence       loss and error against communication round, headline rows
  heterogeneity     100-round rows on the four client partitions at one local epoch
                    (drawn once every row exists)
  heterogeneity30   the same at 30 rounds: the stage-2 tuning scores
  local_epochs      error against local epochs per round on every partition, a 2x4
                    grid: the stage-3 ladder on single-class, the methods run there
                    on the Dirichlet partitions
  participation_*   loss and error against rounds and against client uploads
  ladders           every learning rate walked (appendix; selection is on c_d)

Style set once globally; colours are semantic and fixed per series (CUD palette).
Every figure has the same width and font sizes, so the paper, which scales each to the
text width, prints them all at one text size; errors are in units of 1e-3, as in the
tables.
"""
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import (FixedLocator, FuncFormatter, LogLocator, MaxNLocator,
                               NullFormatter, NullLocator)

from tune_full import SCORE_ROUNDS

INK, MUTED, RULE = "#2A2A2A", "#4A4A4A", "#9A9A9A"
RC = {    # seaborn's darkgrid style and notebook context, written out: no dependency
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12, "axes.labelsize": 12, "axes.titlesize": 12,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.facecolor": "#EAEAF2", "figure.facecolor": "white",
    "axes.edgecolor": "white", "axes.linewidth": 0.0,
    "grid.color": "white", "grid.linewidth": 1.0, "axes.grid": True,
    "axes.axisbelow": True, "axes.titlelocation": "left",
    "xtick.bottom": False, "ytick.left": False,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.labelcolor": INK, "text.color": INK,
    "legend.frameon": False,
    "lines.linewidth": 1.5, "lines.markersize": 5, "lines.solid_capstyle": "round",
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
}
plt.rcParams.update(RC)
WIDTH = 10.0              # inches, every figure

ROOT = Path(__file__).resolve().parent.parent
RUNS = ROOT / "results" / "runs"
OUT = ROOT / "results" / "figures"
TABLES = ROOT / "results" / "tables"

# label, colour, marker, linestyle. FedOLion = FedMuon variant (blue, dashed);
# ceilings = gray references; FedAvg in vermilion. Every method anneals its
# learning rate: the non-spectral methods are their cosine-schedule configs.
SERIES = {
    "fedmuon":            ("FedMuon",           "#0173b2", "o", "-"),
    "fedmars":            ("FedMARS-M",         "#029e73", "s", "-"),
    "fedolion":           ("FedOLion",          "#0173b2", "D", "--"),
    "scaffold_cosine":    ("SCAFFOLD-Adam",     "#56b4e9", "^", "-"),
    "fedlion_cosine2":    ("FedLion",           "#cc78bc", "P", "-"),
    "fedavg_cosine":      ("FedAvg",            "#d55e00", "v", "-"),
    "centralized_cosine": ("Centralised AdamW", "#949494", "X", (0, (5, 3))),
    "centralized_muon":   ("Centralised Muon",  "#949494", "*", (0, (1, 1.5))),
}
ORDER = list(SERIES)
CEILINGS = ("centralized_cosine", "centralized_muon")  # hold all data: split-independent
FEDERATED = [m for m in ORDER if m not in CEILINGS]
SPLITS = [("single_class", "single-class"), ("dirichlet_0.5", r"$\alpha$=0.5"),
          ("dirichlet_10", r"$\alpha$=10"), ("dirichlet_1000", r"$\alpha$=1000")]
SPLIT_COLORS = [matplotlib.cm.viridis(t) for t in np.linspace(0.05, 0.8, 4)]
METRICS = [("c_d_mae", r"$c_d$ error ($10^{-3}$)"), ("c_l_mae", r"$c_l$ error ($10^{-3}$)")]
SCALE = {"c_d_mae": 1e3, "c_l_mae": 1e3, "train_loss": 1.0}   # drawn as in the tables
EPOCHS = (1, 2, 4, 8, 16)


def shown(key, values):
    """Values of a metric in the units the figures use; the CSVs keep raw values."""
    return [v * SCALE[key] for v in values]


def log_ticks(axis, most=6):
    """One set of equally sized, plainly written ticks on a log axis. Under 0.7 decades,
    evenly spaced round numbers (10, 15, 20); otherwise the coarsest of 1, 1-3, 1-2-5,
    1-2-3-5 x 10^k with three to `most` ticks that leaves no more than a fifth of the
    axis unticked at either end (the closest to that if none does). Call once the
    limits are final."""
    lo, hi = sorted(axis.get_view_interval())
    span = math.log10(hi / lo)
    if span < 0.7:
        ticks = [t for t in MaxNLocator(5, steps=[1, 2, 5, 10]).tick_values(lo, hi)
                 if lo <= t <= hi]
    else:
        decades = range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)) + 1)
        options = []
        for subs in ((1,), (1, 3), (1, 2, 5), (1, 2, 3, 5)):
            t = [m * 10.0 ** k for k in decades for m in subs if lo <= m * 10.0 ** k <= hi]
            if 3 <= len(t) <= most:
                options.append((max(math.log10(t[0] / lo), math.log10(hi / t[-1])) / span, t))
        good = [t for gap, t in options if gap <= 0.2]
        ticks = good[0] if good else min(options)[1] if options else [lo, hi]
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}" if v >= 1000 else f"{v:.3g}"))
    axis.set_minor_locator(LogLocator(subs=range(2, 10)))   # faint, unlabelled grid
    axis.set_minor_formatter(NullFormatter())
    axis.axes.grid(which="minor", axis=axis.axis_name, color="white", lw=0.5, alpha=0.6)


def lr_ticks(axis):
    """Learning-rate axis: ticks on the 1e-x / 3e-x grid of the ladders, as 3e-4; on a
    long ladder the powers of ten only."""
    lo, hi = sorted(axis.get_view_interval())
    ticks = [m * 10.0 ** k for k in range(math.floor(math.log10(lo)), math.ceil(math.log10(hi)))
             for m in (1, 3) if lo <= m * 10.0 ** k <= hi]
    if len(ticks) > 4:
        ticks = [t for t in ticks if round(t / 10.0 ** math.floor(math.log10(t) + 1e-9)) == 1]
    def label(v, _):
        k = math.floor(math.log10(v) + 1e-9)
        return f"{v / 10.0 ** k:.0f}e{k}"
    axis.set_major_locator(FixedLocator(ticks))
    axis.set_minor_locator(NullLocator())
    axis.set_major_formatter(FuncFormatter(label))
    axis.set_minor_formatter(NullFormatter())


def save(fig, name, rows, header):
    OUT.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        if ax.get_yscale() == "log":
            log_ticks(ax.yaxis)
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


def history_rows(run):
    """The whole-body evaluation rows of a run, or [] if it has none."""
    if not (run / "history.csv").exists():
        return []
    return [r for r in csv.DictReader((run / "history.csv").open())
            if r.get("body") == "all"]


def scores(run, rounds=SCORE_ROUNDS):
    """Mean of each metric over the given evaluated rounds (default: the tuning score)."""
    if not (run / "summary.csv").exists():
        return {}
    rows = [r for r in history_rows(run) if int(r["round"]) in rounds]
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
    return Line2D([], [], color=colour, marker=marker, ls=ls, label=label)


def legend(fig, handles, ncols=4):
    fig.legend(handles=handles, loc="outside lower center", ncols=min(ncols, len(handles)))


def two_panel(name, xlabel, series, xticks=None, xcol="x", xnames=None,
              flat_ceilings=False, log2_x=False):
    """Drag left, lift right, one legend below; no titles (captions live in the
    paper). `series` = {method: [(x, {metric: value}), ...]}; `xnames` maps x to
    the table's value."""
    if not series:
        return
    fig, axes = plt.subplots(1, 2, figsize=(WIDTH, 3.4), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    rows = []
    for ax, (key, ylabel) in zip(axes, METRICS):
        for m, pts in series.items():
            label, colour, marker, ls = SERIES[m]
            xy = [(x, s[key]) for x, s in pts if key in s]
            if not xy:
                continue
            ys = shown(key, [p[1] for p in xy])
            if flat_ceilings and m in CEILINGS:
                ax.axhline(ys[0], color=colour, ls=ls, zorder=2)
                continue
            ax.plot([p[0] for p in xy], ys, color=colour, ls=ls, marker=marker, zorder=3)
        ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if log2_x:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_minor_locator(NullLocator())
        if xticks:
            ax.set_xticks([t[0] for t in xticks])
            ax.set_xticklabels([t[1] for t in xticks])
            ax.grid(axis="x", visible=False)    # categories or a few rungs: no x grid
    for m, pts in series.items():
        rows += [[m, xnames[x] if xnames else x] + [s.get(k, "") for k, _ in METRICS]
                 for x, s in pts]
    legend(fig, [legend_handle(m) for m in ORDER if m in series])
    save(fig, name, rows, ["method", xcol] + [k for k, _ in METRICS])


def fig_heterogeneity():
    """Every method at E=1, per-split LR, 100 rounds (last-5-round mean). Drawn only
    once every row exists; the 30-round version is heterogeneity30."""
    bench = {m: {k: RUNS / "full" / ("single_class" if m in CEILINGS else k)
                 / f"{m}_E1_s42" for k, _ in SPLITS} for m in ORDER}
    if not all((r / "summary.csv").exists() for m in ORDER for r in bench[m].values()):
        return
    series = {}
    for m in ORDER:
        pts = [(i, s) for i, (key, _) in enumerate(SPLITS)
               if (s := scores(bench[m][key], rounds=range(96, 101)))]
        if pts:
            series[m] = pts
    two_panel("heterogeneity", "client distribution", series,
              xticks=list(enumerate(lbl for _, lbl in SPLITS)), xcol="split",
              xnames=[key for key, _ in SPLITS], flat_ceilings=True)


def epochs_run(sel, m, split, e):
    """The 30-round run for (method, split, E): the tuning ladder where it exists
    (E = 1 everywhere, every E on single-class), else the `epochs` suite row."""
    run = selected_run(sel, m, split, e)
    return run if run and run.exists() else RUNS / "full30" / split / f"{m}_E{e}_s42"


def fig_local_epochs():
    """Error against local epochs per round on every partition, a 2x4 grid with drag
    on top and lift below; every point is a 30-round run scored like a rung, at the
    method's stage-2 rate for that partition. The single-class column is the stage-3
    ladder of every method; the Dirichlet columns hold the methods run there."""
    sel = selections("tune_full")
    data = {split: {} for split, _ in SPLITS}
    rows = []
    for split in data:
        for m in FEDERATED:
            pts = [(e, s) for e in EPOCHS if (s := scores(epochs_run(sel, m, split, e)))]
            rows += [[m, split, e] + [s.get(k, "") for k, _ in METRICS] for e, s in pts]
            if len(pts) >= 2:
                data[split][m] = pts
    if not any(data.values()):
        return
    fig, axes = plt.subplots(2, 4, figsize=(WIDTH, 5.6), sharex=True, sharey="row",
                             constrained_layout=True)   # one scale per metric, so the
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.06, wspace=0.03, hspace=0.05)
    drawn = set()                                      # partitions compare directly
    for j, (split, label) in enumerate(SPLITS):
        axes[0, j].set_title(label)
        for m, pts in data[split].items():
            drawn.add(m)
            _, colour, marker, ls = SERIES[m]
            for i, (key, _) in enumerate(METRICS):
                xy = [(e, s[key]) for e, s in pts if key in s]
                axes[i, j].plot([x for x, _ in xy], shown(key, [y for _, y in xy]),
                                color=colour, ls=ls, marker=marker, zorder=3)
        if not data[split]:
            for ax in axes[:, j]:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", color=MUTED)
    for (i, j), ax in np.ndenumerate(axes):
        ax.set_yscale("log")
        ax.set_xscale("log", base=2)
        ax.set_xticks(EPOCHS)
        ax.set_xticklabels([str(e) for e in EPOCHS])
        ax.xaxis.set_minor_locator(NullLocator())
        ax.grid(axis="x", visible=False)
        if j == 0:
            ax.set_ylabel(METRICS[i][1])
        else:
            ax.tick_params(labelleft=False)
        if i == 1:
            ax.set_xlabel("local epochs per round")
    legend(fig, [legend_handle(m) for m in FEDERATED if m in drawn], ncols=6)
    save(fig, "local_epochs", rows,
         ["method", "split", "local_epochs"] + [k for k, _ in METRICS])


def fig_heterogeneity30():
    """The heterogeneity comparison after 30 rounds: every federated method at one
    local epoch at its per-split stage-2 rate, scored like a rung."""
    sel = selections("tune_full")
    series = {}
    for m in FEDERATED:
        pts = [(i, s) for i, (key, _) in enumerate(SPLITS)
               if (run := selected_run(sel, m, key, 1)) and (s := scores(run))]
        if pts:
            series[m] = pts
    two_panel("heterogeneity30", "client distribution", series,
              xticks=list(enumerate(lbl for _, lbl in SPLITS)), xcol="split",
              xnames=[key for key, _ in SPLITS])


def curve(run, key="c_d_mae"):
    """(round, value) per evaluated round of a finished run, or [] if unfinished;
    key "train_loss" reads the per-round training loss instead of a test metric."""
    if not (run / "summary.csv").exists():
        return []
    if key == "train_loss":   # a diverged run's loss is NaN from some round on
        return sorted((int(r["round"]), float(r["train_loss"]))
                      for r in csv.DictReader((run / "round_stats.csv").open())
                      if math.isfinite(float(r["train_loss"])))
    return sorted((int(r["round"]), float(r[key])) for r in history_rows(run)
                  if r.get(key) not in (None, "", "nan"))


PANELS = [("train_loss", "training loss")] + METRICS   # per-round figures: loss, drag, lift


def marked_line(ax, xs, ys, colour, ls, marker, **kw):
    """A dense curve: the line, plus its marker at a handful of points so the series
    can be told apart without colour."""
    ax.plot(xs, ys, color=colour, ls=ls, zorder=3, **kw)
    step = max(1, len(xs) // 6)
    ax.plot(xs[::step], ys[::step], ls="none", color=colour, marker=marker, zorder=4,
            **{k: v for k, v in kw.items() if k != "lw"})


def fig_convergence():
    """Benchmark rows on the reference split, each method at its tuned local-epoch
    budget (the ceilings at E=1): training loss every round, drag and lift at the
    evaluated rounds."""
    base = RUNS / "full" / "single_class"
    runs = {}
    for m in ORDER:
        e = 1 if m in CEILINGS else tuned_epochs(m)
        if e is not None and (base / f"{m}_E{e}_s42" / "summary.csv").exists():
            runs[m] = base / f"{m}_E{e}_s42"
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 3.2), sharex=True,
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    rows = []
    for m, run in runs.items():
        _, colour, marker, ls = SERIES[m]
        panels = [curve(run, key) for key, _ in PANELS]
        for ax, (key, _), pts in zip(axes, PANELS, panels):
            marked_line(ax, [r for r, _ in pts], shown(key, [v for _, v in pts]),
                        colour, ls, marker)
        cd, cl = dict(panels[1]), dict(panels[2])
        rows += [[m, r, v, cd.get(r, ""), cl.get(r, "")] for r, v in panels[0]]
    for ax, (_, ylabel) in zip(axes, PANELS):
        ax.set_yscale("log")
        ax.set_xticks([0, 25, 50, 75, 100])
        ax.set_xlabel("communication round")
        ax.set_ylabel(ylabel)
    legend(fig, [legend_handle(m) for m in ORDER if m in runs])
    save(fig, "convergence", rows, ["method", "round"] + [k for k, _ in PANELS])


def fig_participation():
    """Clients per round K on the reference split, each method at its tuned local
    epochs (K=6 is the headline row): one block per metric (training loss, drag,
    lift), each a 2x3 grid of the methods on one y scale, so the methods compare
    directly and every panel is as wide as in the convergence figure; K on the
    ordered viridis ramp, the centralized ceilings in gray. Drawn twice --
    against communication rounds, and against cumulative client uploads (K per
    round; a centralized model uploads nothing: a horizontal line)."""
    base = RUNS / "full" / "single_class"
    ks = (1, 2, 3, 6)
    colours = [matplotlib.cm.viridis(t) for t in np.linspace(0.05, 0.8, len(ks))]
    methods = FEDERATED
    runs = {(m, k): base / (f"{m}_E{tuned_epochs(m)}_s42" if k == 6
                            else f"{m}_E{tuned_epochs(m)}_K{k}_s42")
            for m in methods for k in ks}
    curves = {(m, k, key): curve(run, key)
              for (m, k), run in runs.items() for key, _ in PANELS}
    ceilings = {(c, key): curve(base / f"{c}_E1_s42", key)
                for c in CEILINGS for key, _ in PANELS}
    if not any(curves.values()):
        return
    markers = ("o", "s", "^", "D")
    for name, xlabel, xof in (("participation_rounds", "communication round",
                               lambda k, r: r),
                              ("participation_comm", "client uploads (cumulative)",
                               lambda k, r: k * r)):
        # one block per metric, each a 2x3 grid of the methods on one y scale (the
        # spectral methods above the non-spectral), so the methods compare directly
        ncol = 3
        per_block = -(-len(methods) // ncol)
        fig, axes = plt.subplots(len(PANELS) * per_block, ncol,
                                 figsize=(WIDTH, 2.0 * len(PANELS) * per_block),
                                 sharex=True, constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.06, h_pad=0.06, wspace=0.04, hspace=0.03)
        panel = lambda i, j: axes[i * per_block + j // ncol, j % ncol]
        rows = []
        for i, (key, ylabel) in enumerate(PANELS):
            block = [panel(i, j) for j in range(len(methods))]
            for ax in block[1:]:
                ax.sharey(block[0])
            for r in range(per_block):
                panel(i, r * ncol).set_ylabel(ylabel)
            drawn = []
            for j, m in enumerate(methods):
                ax = block[j]
                ax.set_title(SERIES[m][0])
                if not any(curves[(m, k, "c_d_mae")] for k in ks):   # nothing finished yet
                    ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                            va="center", color=MUTED)
                    continue
                for k, colour, marker in zip(ks, colours, markers):
                    pts = curves[(m, k, key)]
                    if not pts:
                        continue
                    ys = shown(key, [v for _, v in pts])
                    marked_line(ax, [xof(k, r) for r, _ in pts], ys, colour, "-", marker, ms=4)
                    drawn += ys
                for c in CEILINGS:
                    pts = ceilings[(c, key)]
                    if not pts:
                        continue
                    _, colour, _, ls = SERIES[c]
                    ys = shown(key, [v for _, v in pts])
                    if name == "participation_rounds":
                        ax.plot([r for r, _ in pts], ys, color=colour, ls=ls, zorder=2)
                    else:                 # no uploads: its final level is the reference
                        ax.axhline(sum(ys[-5:]) / 5, color=colour, ls=ls, zorder=2)
                    drawn += ys
            for ax in block:
                ax.set_yscale("log")
                if name == "participation_comm":
                    ax.set_xscale("log")
            # the first rounds and a diverging run leave the top of the panel rather
            # than stretch the whole block's scale
            if drawn:
                block[0].set_ylim(min(drawn) / 1.08, np.quantile(drawn, 0.99) * 1.15)
        for ax in axes.flat:
            if ax.get_subplotspec().colspan.start > 0:
                ax.tick_params(labelleft=False)
        for ax in axes[-1, :]:
            ax.set_xlabel(xlabel)
        for m in methods:
            for k in ks:
                by_key = {key: dict(curves[(m, k, key)]) for key, _ in PANELS}
                rows += [[m, k, r, xof(k, r)] + [by_key[key].get(r, "") for key, _ in PANELS]
                         for r in sorted(by_key["train_loss"])]
        if name == "participation_comm":
            log_ticks(axes[-1, 0].xaxis)
        else:
            axes[-1, 0].set_xticks([0, 25, 50, 75, 100])
        handles = [Line2D([], [], color=c, marker=mk, label=f"K = {k}")
                   for k, c, mk in zip(ks, colours, markers)]
        handles += [Line2D([], [], color=SERIES[c][1], ls=SERIES[c][3],
                           label=SERIES[c][0]) for c in CEILINGS if ceilings[(c, "c_d_mae")]]
        legend(fig, handles, ncols=len(handles))
        save(fig, name, rows, ["method", "clients_per_round", "round", "x"]
             + [k for k, _ in PANELS])


def fig_ladders():
    """Appendix, 3x3: every rung walked, one panel per method, the four
    distributions on the ordered viridis ramp. Selection is on c_d."""
    sel = selections("tune_full")
    methods = [m for m in ORDER if m in sel]
    if not methods:
        return
    fig, axes = plt.subplots(3, 3, figsize=(WIDTH, 8.4), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.06, wspace=0.03, hspace=0.05)
    rows = []
    scale = SCALE["c_d_mae"]
    for i, (ax, m) in enumerate(zip(axes.flat, methods)):
        finite = [float(s) * scale for rec in sel[m].values()
                  for s in rec["evaluated"].values() if s != "diverged"]
        # a rung four times worse than the panel's best says only "it failed"
        cap = min(finite) * 4
        for j, (key, _) in enumerate(SPLITS):
            rec = sel[m].get(key)
            if not rec:
                continue
            xs, ys, div = [], [], []
            for lr, s in sorted(rec["evaluated"].items(), key=lambda kv: float(kv[0])):
                if s == "diverged" or float(s) * scale > cap:
                    div.append(float(lr))
                else:
                    xs.append(float(lr))
                    ys.append(float(s) * scale)
                rows.append([m, key, lr, s])
            ax.plot(xs, ys, color=SPLIT_COLORS[j], marker="o", ms=4, zorder=3)
            ax.plot(div, [cap] * len(div), ls="none", marker="^", ms=6,
                    mfc="white", mec=SPLIT_COLORS[j], mew=1.2, zorder=3)
            if rec["evaluated"].get(rec["best"]) not in (None, "diverged"):
                ax.plot([float(rec["best"])], [float(rec["evaluated"][rec["best"]]) * scale],
                        ls="none", marker="o", ms=11, mfc="none", mec=INK, mew=1.3,
                        zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(min(finite) / 1.15, cap * 1.3)   # room for the "selected" ring
        lo, hi = ax.get_xlim()
        ax.set_xlim(lo / 1.3, hi * 1.3)
        lr_ticks(ax.xaxis)
        ax.set_title(SERIES[m][0])
        if i % 3 == 0:                    # y-label once per row, not eight times
            ax.set_ylabel(METRICS[0][1])
        if i >= len(methods) - 3:         # x-label on each column's lowest panel
            ax.set_xlabel("learning rate")
    for ax in axes.flat[len(methods):]:
        ax.set_visible(False)
    handles = [Line2D([], [], color=SPLIT_COLORS[i], marker="o", ms=4, label=lbl)
               for i, (_, lbl) in enumerate(SPLITS)]
    handles.append(Line2D([], [], ls="none", marker="^", ms=6, mfc="white",
                          mec=MUTED, mew=1.2, label="diverged"))
    handles.append(Line2D([], [], ls="none", marker="o", ms=10, mfc="none",
                          mec=INK, mew=1.3, label="selected"))
    legend(fig, handles, ncols=6)
    save(fig, "ladders", rows, ["method", "split", "learning_rate", "c_d_mae"])


if __name__ == "__main__":
    print("figures ->", OUT)
    for fn in (fig_heterogeneity, fig_heterogeneity30, fig_local_epochs,
               fig_convergence, fig_ladders, fig_participation):
        fn()

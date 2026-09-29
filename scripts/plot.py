#!/usr/bin/env python
"""Figures from results/. Each writes a PDF, a PNG and the CSV behind it.

  python scripts/plot.py

Drawn for whatever exists, so this is safe to run mid-campaign:
  heterogeneity  error against client heterogeneity, at one local epoch
  participation_* error against rounds and against client uploads, per K
  local_epochs   error against local samples per round, per client distribution
  convergence    error against communication round (needs benchmark runs)
  ladders        every learning rate walked (appendix; selection is on c_d)
  centralized    training curves of the two centralized references

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

from tune_full import SCORE_ROUNDS

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
SAMPLES_PER_EPOCH = 5703 / 6      # mean client size: 5703 training cars over 6 clients
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
    return Line2D([], [], color=colour, marker=marker, ms=5, ls=ls, label=label)


def two_panel(name, xlabel, series, xticks=None, flat_ceilings=False,
              sparse_markers=False):
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
                s = scores(bench[m][key], rounds=range(96, 101))
            else:
                run = selected_run(sel, m, "single_class" if m in CEILINGS else key, 1)
                s = scores(run) if run else {}
            if s:
                pts.append((i, s))
        if pts:
            series[m] = pts
    two_panel("heterogeneity",
              "client distribution", series,
              xticks=list(enumerate(lbl for _, lbl in SPLITS)), flat_ceilings=True)


def epochs_run(sel, m, split, e):
    """The 30-round run for (method, split, E): the tuning ladder where it exists
    (E = 1 everywhere, every E on single-class), else the `epochs` suite row."""
    run = selected_run(sel, m, split, e)
    return run if run and run.exists() else RUNS / "full30" / split / f"{m}_E{e}_s42"


def fig_local_epochs():
    """2x4: drag (top) and lift (bottom) against local work per round, one column
    per client distribution. Every point is a 30-round run scored like a rung."""
    sel = selections("tune_full")
    fig, axes = plt.subplots(2, 4, figsize=(12.4, 5.6), sharex=True,
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.05, h_pad=0.06, wspace=0.03, hspace=0.05)
    rows, shown = [], set()
    for j, (split, label) in enumerate(SPLITS):
        axes[0, j].set_title(label, loc="left", fontsize=10)
        drawn = False
        for m in (m for m in ORDER if m not in CEILINGS):
            pts = [(e, s) for e in EPOCHS if (s := scores(epochs_run(sel, m, split, e)))]
            if len(pts) < 2:
                continue
            drawn = True
            shown.add(m)
            _, colour, marker, ls = SERIES[m]
            for i, (key, _) in enumerate(METRICS):
                xy = [(e, s[key]) for e, s in pts if key in s]
                axes[i, j].plot([x for x, _ in xy], [y for _, y in xy], color=colour,
                                ls=ls, marker=marker, ms=5, zorder=3)
            rows += [[m, split, e] + [s.get(k, "") for k, _ in METRICS] for e, s in pts]
        if not drawn:
            for ax in axes[:, j]:
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", color="0.55")
    if not shown:
        plt.close(fig)
        return
    for (i, j), ax in np.ndenumerate(axes):
        if ax.lines:
            log_y(ax)
        else:
            ax.set_yticks([])
        ax.set_xscale("log", base=2)
        ax.set_xticks(EPOCHS)
        ax.set_xticklabels([f"{e * SAMPLES_PER_EPOCH / 1000:.1f}k" for e in EPOCHS])
        ax.xaxis.set_minor_formatter(NullFormatter())
        if j == 0:
            ax.set_ylabel(METRICS[i][1], fontsize=9)
        if i == 1:
            ax.set_xlabel("local samples per client per round", fontsize=9)
    fig.legend(handles=[legend_handle(m) for m in ORDER if m in shown],
               loc="outside lower center", ncols=6)
    save(fig, "local_epochs", rows,
         ["method", "split", "local_epochs"] + [k for k, _ in METRICS])


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
        if e is None:
            continue
        pts = []
        for r in history_rows(run):
            vals = {k: float(r[k]) for k, _ in METRICS
                    if r.get(k) not in (None, "", "nan")}
            if vals:
                pts.append((int(r["round"]), vals))
        if pts:
            series[m] = sorted(pts)
    two_panel("convergence",
              "communication round", series, sparse_markers=True)


def curve(run, key="c_d_mae"):
    """(round, value) per evaluated round of a benchmark run, or [] if unfinished."""
    if not (run / "summary.csv").exists():
        return []
    return sorted((int(r["round"]), float(r[key])) for r in history_rows(run)
                  if r.get(key) not in (None, "", "nan"))


def fig_participation():
    """Clients per round K on the reference split, each method at its tuned local
    epochs (K=6 is the headline row): one panel per method, K on the ordered viridis
    ramp, the centralized ceilings in gray. Drawn twice -- against communication
    rounds, and against cumulative client uploads (K per round; a centralized model
    uploads nothing, so it is a horizontal line there)."""
    base = RUNS / "full" / "single_class"
    ks = (1, 2, 3, 6)
    colours = [matplotlib.cm.viridis(t) for t in np.linspace(0.05, 0.8, len(ks))]
    methods = [m for m in ORDER if m not in CEILINGS]
    data = {m: {k: curve(base / (f"{m}_E{tuned_epochs(m)}_s42" if k == 6
                                 else f"{m}_E{tuned_epochs(m)}_K{k}_s42"))
                for k in ks} for m in methods}
    ceilings = {m: curve(base / f"{m}_E1_s42") for m in CEILINGS}
    if not any(pts for d in data.values() for pts in d.values()):
        return
    for name, xlabel, xof in (("participation_rounds", "communication round",
                               lambda k, r: r),
                              ("participation_comm", "client uploads (cumulative)",
                               lambda k, r: k * r)):
        fig, axes = plt.subplots(2, 3, figsize=(12.4, 6.0), constrained_layout=True)
        fig.get_layout_engine().set(w_pad=0.05, h_pad=0.06, wspace=0.03, hspace=0.05)
        rows = []
        for i, (ax, m) in enumerate(zip(axes.flat, methods)):
            ax.set_title(SERIES[m][0], loc="left", fontsize=10)
            if not any(data[m].values()):     # nothing finished for this method yet
                ax.set_xticks([])
                ax.set_yticks([])
                ax.text(0.5, 0.5, "pending", transform=ax.transAxes, ha="center",
                        va="center", color="0.55")
                continue
            for k, colour in zip(ks, colours):
                pts = data[m][k]
                if not pts:
                    continue
                xs = [xof(k, r) for r, _ in pts]
                ax.plot(xs, [v for _, v in pts], color=colour, zorder=3)
                step = max(1, len(xs) // 6)
                ax.scatter(xs[::step], [v for _, v in pts][::step], s=18, color=colour,
                           zorder=4)
                rows += [[m, k, r, xof(k, r), v] for r, v in pts]
            for c, pts in ceilings.items():
                if not pts:
                    continue
                label, colour, _, ls = SERIES[c]
                if name == "participation_rounds":
                    ax.plot([r for r, _ in pts], [v for _, v in pts], color=colour,
                            ls=ls, zorder=2)
                else:                     # no uploads: its final level is the reference
                    ax.axhline(sum(v for _, v in pts[-5:]) / 5, color=colour, ls=ls,
                               zorder=2)
            log_y(ax)
            if name == "participation_comm":
                ax.set_xscale("log")
            if i % 3 == 0:
                ax.set_ylabel(METRICS[0][1], fontsize=9)
            if i >= 3:
                ax.set_xlabel(xlabel, fontsize=9)
        handles = [Line2D([], [], color=c, marker="o", ms=4,
                          label=f"K = {k}" + (" (full)" if k == 6 else ""))
                   for k, c in zip(ks, colours)]
        handles += [Line2D([], [], color=SERIES[c][1], ls=SERIES[c][3],
                           label=SERIES[c][0]) for c in CEILINGS if ceilings[c]]
        fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
        save(fig, name, rows, ["method", "clients_per_round", "round", "x", "c_d_mae"])


def fig_centralized():
    """Training curves of the two centralized references at their selected rates on
    the 30-round setting: train loss every round, drag and lift at the evaluated
    rounds."""
    sel = selections("tune_full")
    colours = {"centralized_muon": SERIES["fedmuon"][1], "centralized": SERIES["fedavg"][1]}
    runs = [(m, run) for m in CEILINGS
            if (run := selected_run(sel, m, "single_class", 1)) and (run / "summary.csv").exists()]
    if not runs:
        return
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 3.4), constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    rows, handles = [], []
    for m, run in runs:
        style = dict(color=colours[m], zorder=3)
        stats = list(csv.DictReader((run / "round_stats.csv").open()))
        axes[0].plot([int(r["round"]) for r in stats],
                     [float(r["train_loss"]) for r in stats], **style)
        for ax, key in zip(axes[1:], ("c_d_mae", "c_l_mae")):
            pts = curve(run, key)
            ax.plot([r for r, _ in pts], [v for _, v in pts], marker="o", ms=4, **style)
        rows += [[m, int(r["round"]), r["train_loss"]] for r in stats]
        handles.append(Line2D([], [], label=SERIES[m][0], **style))
    for ax, ylabel in zip(axes, ["train loss", "test " + METRICS[0][1],
                                 "test " + METRICS[1][1]]):
        log_y(ax)
        ax.set_xlabel("round")
        ax.set_ylabel(ylabel)
    fig.legend(handles=handles, loc="outside lower center", ncols=len(handles))
    save(fig, "centralized", rows, ["method", "round", "train_loss"])


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
        for j, (key, _) in enumerate(SPLITS):
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
            ax.plot(xs, ys, color=SPLIT_COLORS[j], marker="o", ms=4, zorder=3)
            ax.plot(div, [cap] * len(div), ls="none", marker="^", ms=6,
                    mfc="white", mec=SPLIT_COLORS[j], mew=1.2, zorder=3)
            if rec["evaluated"].get(rec["best"]) not in (None, "diverged"):
                ax.plot([float(rec["best"])], [float(rec["evaluated"][rec["best"]])],
                        ls="none", marker="o", ms=11, mfc="none", mec="#2A2A2A",
                        mew=1.3, zorder=4)
        ax.set_xscale("log")
        log_y(ax)
        ax.set_ylim(min(finite) / 1.15, cap * 1.3)   # room for the "selected" ring
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
    for fn in (fig_heterogeneity, fig_local_epochs, fig_convergence, fig_ladders,
               fig_participation, fig_centralized):
        fn()

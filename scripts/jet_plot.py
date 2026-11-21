import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter, NullFormatter

# ----------------------------------------------------------------------------
# CONFIG - edit these
# ----------------------------------------------------------------------------
# For the experiment shown in your screenshot:
ROOT = Path("results/runs/tuned_fedmars")

# The script discovers every seed present in each run folder automatically.
# Example folder names:
# fedmars_E1_s42_mlr0p005_slr0p0001_wd0p01_beta0p9
# fedmars_E1_s123_mlr0p005_slr0p0001_wd0p01_beta0p9

METRIC = "c_d_mae"            # e.g. c_d_mae, c_l_mae, c_p_rmse
Y_LABEL = None                # None -> use METRIC

# Display name -> run-folder prefix in "{prefix}_E{E}_s{seed}..."
# Add other methods here when their runs are present under ROOT.
METHODS = {
    "FedMARS-M": "fedmars",
    # "FedMuon": "fedmuon",
    # "FedOLion": "fedolion",
    # "SCAFFOLD-Adam": "scaffold",
    # "FedLion": "fedlion",
    # "FedAvg": "fedavg",
}

EPOCHS = [1, 2, 4, 8]          # epochs in your current experiment

# Legend label -> sub-folder under ROOT
# Add the other split folders here when those experiments are available.
SERIES = {
    "single-class": "single_class",
    "α=0.5": "dirichlet_0.5",
    "α=10": "dirichlet_10",
    "α=1000": "dirichlet_1000",
}

N_COLS = 3
Y_LOG = True
X_LOG = True
HIGHLIGHT_BEST = True
LOWER_IS_BETTER = not METRIC.endswith("r2")
OUT_FILE = Path("jet_plots") / f"{METRIC}_vs_E.png"
CSV_FILE = Path("jet_plots") / f"{METRIC}_vs_E_statistics.csv"

COLORS = ["#440154", "#3b528b", "#21918c", "#7ad151"]


def seed_from_folder(folder_name):
    """Extract the seed from a run folder name such as ..._E1_s42_mlr..."""
    match = re.search(r"(?:^|_)s(\d+)(?:_|$)", folder_name)
    return int(match.group(1)) if match else None


def load_run_metric(run_dir):
    """Read the final metric value from one run, preserving the original logic."""
    path = run_dir / "summary.csv"
    try:
        if not path.exists():
            return None

        df = pd.read_csv(path)
        if METRIC not in df.columns or df.empty:
            warnings.warn(f"{path}: column '{METRIC}' missing/empty")
            return None

        if "round" in df.columns:
            df = df.sort_values("round")

        # Same evaluation choice as the supplied script: use the final row.
        return float(df[METRIC].iloc[-1])

    except Exception as exc:
        warnings.warn(f"Could not read {path}: {exc}")
        return None


def load_metric_stats(series_dir, method_dir, epoch):
    """
    Find all matching runs for this method/split/epoch.

    One seed: mean=min=max, so the plotted result is unchanged.
    Multiple seeds: calculate mean, minimum and maximum across seed-level
    final metric values.
    """
    parent = ROOT / series_dir
    if not parent.is_dir():
        return None

    pattern = f"{method_dir}_E{epoch}_s*"
    run_dirs = sorted(p for p in parent.glob(pattern) if p.is_dir())

    # Keep only folders with a recognizable seed. If duplicate runs exist for
    # one seed, use the most recently modified folder and warn.
    by_seed = {}
    for run_dir in run_dirs:
        seed = seed_from_folder(run_dir.name)
        if seed is None:
            warnings.warn(f"Cannot identify seed in folder: {run_dir}")
            continue
        if seed in by_seed:
            previous = by_seed[seed]
            warnings.warn(
                f"More than one run found for seed {seed}: "
                f"{previous.name} and {run_dir.name}. "
                "Using the most recently modified folder."
            )
            if run_dir.stat().st_mtime > previous.stat().st_mtime:
                by_seed[seed] = run_dir
        else:
            by_seed[seed] = run_dir

    values = []
    diverged_seeds = []
    for seed, run_dir in sorted(by_seed.items()):
        value = load_run_metric(run_dir)
        if value is None:
            continue
        if np.isfinite(value):
            values.append((seed, value))
        else:
            diverged_seeds.append(seed)

    if not values and not diverged_seeds:
        return None

    finite_values = [value for _, value in values]
    if finite_values:
        return {
            "mean": float(np.mean(finite_values)),
            "min": float(np.min(finite_values)),
            "max": float(np.max(finite_values)),
            "n_seeds": len(finite_values),
            "seeds": [seed for seed, _ in values],
            "diverged_seeds": diverged_seeds,
        }

    # All discovered runs diverged (or produced non-finite final values).
    return {
        "mean": None,
        "min": None,
        "max": None,
        "n_seeds": 0,
        "seeds": [],
        "diverged_seeds": diverged_seeds,
    }


def style_axis(ax, title):
    ax.set_facecolor("#eaeaf2")
    ax.grid(True, which="both", color="white", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.tick_params(length=0, labelsize=9)


def main():
    n_methods = len(METHODS)
    n_rows = int(np.ceil(n_methods / N_COLS))
    fig, axes = plt.subplots(
        n_rows, N_COLS,
        figsize=(4.2 * N_COLS, 3.2 * n_rows),
        squeeze=False
    )
    axes_flat = axes.ravel()
    csv_rows = []

    for ax, (display_name, method_dir) in zip(axes_flat, METHODS.items()):
        style_axis(ax, display_name)
        any_data = False

        for (series_label, series_dir), color in zip(SERIES.items(), COLORS):
            xs, means, mins, maxs = [], [], [], []
            diverged_xs = []
            partial_diverged_xs = []

            for epoch in EPOCHS:
                stats = load_metric_stats(series_dir, method_dir, epoch)
                if stats is None:
                    continue

                if stats["mean"] is None:
                    diverged_xs.append(epoch)
                    continue

                xs.append(epoch)
                means.append(stats["mean"])
                mins.append(stats["min"])
                maxs.append(stats["max"])

                if stats["diverged_seeds"]:
                    partial_diverged_xs.append(epoch)

                csv_rows.append({
                    "method": display_name,
                    "method_folder": method_dir,
                    "series": series_label,
                    "series_folder": series_dir,
                    "local_epochs": epoch,
                    "n_finite_seeds": stats["n_seeds"],
                    "seeds_used": ";".join(map(str, stats["seeds"])),
                    "diverged_seeds": ";".join(map(str, stats["diverged_seeds"])),
                    "mean": stats["mean"],
                    "minimum": stats["min"],
                    "maximum": stats["max"],
                })

            if xs:
                any_data = True
                ax.plot(
                    xs, means, "-o", color=color, ms=4, lw=1.5,
                    label=series_label
                )

                # Show min-to-max variability only when at least one point
                # has multiple finite seeds. A single seed remains an
                # unchanged line/point with no visible band.
                if any(hi > lo for lo, hi in zip(mins, maxs)):
                    ax.fill_between(
                        xs, mins, maxs, color=color, alpha=0.18,
                        linewidth=0, zorder=1
                    )

                if HIGHLIGHT_BEST:
                    best_i = int(
                        np.argmin(means) if LOWER_IS_BETTER
                        else np.argmax(means)
                    )
                    ax.plot(
                        xs[best_i], means[best_i], "o", ms=13,
                        mfc="none", mec="black", mew=1.5
                    )

            if diverged_xs:
                any_data = True
                ax.plot(
                    diverged_xs, [0.93] * len(diverged_xs), "^",
                    mfc="none", mec=color, ms=7,
                    transform=ax.get_xaxis_transform(), clip_on=False
                )

            # A triangle at the top also flags epochs where some seeds
            # diverged but at least one seed produced a finite result.
            if partial_diverged_xs:
                ax.plot(
                    partial_diverged_xs, [0.86] * len(partial_diverged_xs), "v",
                    mfc="none", mec=color, ms=6,
                    transform=ax.get_xaxis_transform(), clip_on=False
                )

        if not any_data:
            ax.text(
                0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, color="gray"
            )

        if Y_LOG and any_data:
            try:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(ScalarFormatter())
                ax.yaxis.set_minor_formatter(ScalarFormatter())
            except Exception:
                pass

        if X_LOG:
            ax.set_xscale("log", base=2)
            ax.xaxis.set_minor_formatter(NullFormatter())

        ax.set_xticks(EPOCHS)
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(min(EPOCHS) * 0.8, max(EPOCHS) * 1.25)

    for ax in axes_flat[n_methods:]:
        ax.axis("off")

    y_label = Y_LABEL or METRIC
    for row in range(n_rows):
        axes[row, 0].set_ylabel(y_label, fontsize=9)

    for col in range(N_COLS):
        for row in range(n_rows):
            if axes[row, col].axison and (
                row == n_rows - 1 or not axes[row + 1, col].axison
            ):
                axes[row, col].set_xlabel("local epochs E", fontsize=9)

    handles = [
        Line2D([0], [0], color=color, marker="o", ms=5, lw=1.5, label=label)
        for (label, _), color in zip(SERIES.items(), COLORS)
    ]
    handles.append(
        Line2D([0], [0], marker="^", mfc="none", mec="k", ls="",
               label="all seeds diverged / non-finite")
    )
    handles.append(
        Line2D([0], [0], marker="v", mfc="none", mec="k", ls="",
               label="some seeds diverged")
    )
    if HIGHLIGHT_BEST:
        handles.append(
            Line2D([0], [0], marker="o", mfc="none", mec="k",
                   mew=1.5, ms=12, ls="", label="best mean")
        )

    # Add an explanatory min-max band key when multiple seeds are present.
    if any(row["n_finite_seeds"] > 1 for row in csv_rows):
        handles.append(
            Line2D([0], [0], color="gray", lw=7, alpha=0.25,
                   label="minimum–maximum across seeds")
        )

    fig.legend(
        handles=handles, loc="lower center",
        ncol=min(len(handles), 5), frameon=False, fontsize=9
    )
    fig.tight_layout(rect=(0, 0.12, 1, 1))

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    CSV_FILE.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_FILE, dpi=200)
    pd.DataFrame(csv_rows).to_csv(CSV_FILE, index=False)

    print(f"Saved plot: {OUT_FILE}")
    print(f"Saved statistics: {CSV_FILE}")
    plt.show()


if __name__ == "__main__":
    main()

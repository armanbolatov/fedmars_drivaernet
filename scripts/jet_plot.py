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
ROOT = Path("results/runs/full30-omniarch")
SEED = 42

N_ROUNDS = 30   # <-- add this near the other CONFIG constants (after SEED, say)

METRIC = "c_d_mae"            # any column of summary.csv (c_l_mae, c_p_rmse, ...)
Y_LABEL = None                # None -> auto label from METRIC

# display name -> folder prefix {M} in "{M}_E{E}_s42"
# (FOLDER NAMES ARE GUESSES except fedavg - fix them to match your disk)
METHODS = {
    "FedMuon":           "fedmuon",
    "FedMARS-M":         "fedmars",
    "FedOLion":          "fedolion",
    "SCAFFOLD-Adam":     "scaffold",
    "FedLion":           "fedlion",
    "FedAvg":            "fedavg",
}

EPOCHS = [1, 2, 4, 8, 16]    # all possible E values

# legend label -> sub-folder under ROOT
# (ONLY single_class IS KNOWN - fix the alpha folder names)
SERIES = {
    "single-class": "single_class",
    "α=0.5":        "alpha0.5",
    "α=10":         "alpha10",
    "α=1000":       "alpha1000",
}

N_COLS = 3                    # 8 methods -> 2 rows x 4 cols, as in the photo
Y_LOG = True
X_LOG = True
HIGHLIGHT_BEST = True         # circle best point per series ("selected")
LOWER_IS_BETTER = not METRIC.endswith("r2")
OUT_FILE = f"jet_plots/{METRIC}_vs_E.png"

# ----------------------------------------------------------------------------
COLORS = ["#440154", "#3b528b", "#21918c", "#7ad151"]  # viridis-like


def load_metric(series_dir: str, method_dir: str, E: int):
    """Return metric value (float) or None if it can't be loaded."""
    run_dir = ROOT / series_dir / f"{method_dir}_E{E}_s{SEED}"
    path = run_dir / "summary.csv"
    try:
        if not path.exists():
            return None

        # --- NEW: skip folders that still have a .lock file (run incomplete) ---
        if (run_dir / ".lock").exists():
            warnings.warn(f"{run_dir}: .lock file present -> skipping")
            return None

        df = pd.read_csv(path)
        if METRIC not in df.columns or df.empty:
            warnings.warn(f"{path}: column '{METRIC}' missing/empty")
            return None
        if "round" in df.columns:          # take the last round
            df = df.sort_values("round")

        # --- NEW: skip folders that don't have the full N_ROUNDS rounds ---
        if "round" in df.columns and len(df) < N_ROUNDS:
            warnings.warn(
                f"{path}: only {len(df)} rounds (< {N_ROUNDS}) -> skipping"
            )
            return None

        return float(df[METRIC].iloc[-1])
    except Exception as e:                 # never crash on a bad file
        warnings.warn(f"Could not read {path}: {e}")
        return None


def style_axis(ax, title):
    ax.set_facecolor("#eaeaf2")
    ax.grid(True, which="both", color="white", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, loc="left", fontsize=10, fontweight="bold")
    ax.tick_params(length=0, labelsize=9)


def main():
    n_methods = len(METHODS)
    n_rows = int(np.ceil(n_methods / N_COLS))
    fig, axes = plt.subplots(n_rows, N_COLS, figsize=(4.2 * N_COLS, 3.2 * n_rows),
                             squeeze=False)
    axes_flat = axes.ravel()

    for ax, (disp_name, method_dir) in zip(axes_flat, METHODS.items()):
        style_axis(ax, disp_name)
        any_data = False

        for (label, series_dir), color in zip(SERIES.items(), COLORS):
            xs, ys, div_xs = [], [], []
            for E in EPOCHS:
                v = load_metric(series_dir, method_dir, E)
                if v is None:
                    continue                      # missing -> just skip
                if np.isfinite(v):
                    xs.append(E)
                    ys.append(v)
                else:
                    div_xs.append(E)              # nan/inf -> diverged

            if xs:
                any_data = True
                ax.plot(xs, ys, "-o", color=color, ms=4, lw=1.5, label=label)
                if HIGHLIGHT_BEST:
                    i = int(np.argmin(ys) if LOWER_IS_BETTER else np.argmax(ys))
                    ax.plot(xs[i], ys[i], "o", ms=13, mfc="none",
                            mec="black", mew=1.5)
            if div_xs:
                any_data = True
                # x in data coords, y in axes coords (near the top)
                ax.plot(div_xs, [0.93] * len(div_xs), "^", mfc="none", mec=color,
                        ms=7, transform=ax.get_xaxis_transform(), clip_on=False)

        if not any_data:
            ax.text(0.5, 0.5, "no data", ha="center", va="center",
                    transform=ax.transAxes, color="gray")

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

    # hide unused axes
    for ax in axes_flat[n_methods:]:
        ax.axis("off")

    # axis labels like the original: y only on first column, x on bottom row
    ylab = Y_LABEL or METRIC
    for r in range(n_rows):
        axes[r, 0].set_ylabel(ylab, fontsize=9)
    for c in range(N_COLS):
        for r in range(n_rows):
            if axes[r, c].axison and (r == n_rows - 1 or not axes[r + 1, c].axison):
                axes[r, c].set_xlabel("local epochs E", fontsize=9)

    # shared legend
    handles = [Line2D([0], [0], color=c, marker="o", ms=5, lw=1.5, label=l)
               for (l, _), c in zip(SERIES.items(), COLORS)]
    handles.append(Line2D([0], [0], marker="^", mfc="none", mec="k", ls="", label="diverged"))
    if HIGHLIGHT_BEST:
        handles.append(Line2D([0], [0], marker="o", mfc="none", mec="k", mew=1.5,
                              ms=12, ls="", label="selected"))
    fig.legend(handles=handles, loc="lower center", ncol=len(handles),
               frameon=False, fontsize=11)

    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(OUT_FILE, dpi=200)
    print(f"Saved {OUT_FILE}")
    plt.show()


if __name__ == "__main__":
    main()

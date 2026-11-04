import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results/runs/fedmars_search/single_class")
SEED = 42
METRIC = "c_d_mae"
LOWER_IS_BETTER = not METRIC.endswith("r2")
OUT_DIR = Path("jet_plots/fedmars_search")

# Example: fedmars_E2_s42_mlr0p001_slr3em05_wd0p01_beta0p9
RUN_RE = re.compile(
    r"^fedmars_E(?P<epochs>\\d+)_s(?P<seed>\\d+)"
    r"_mlr(?P<mlr>[0-9p]+)_slr(?P<slr>[0-9pem]+)"
    r"_wd(?P<wd>[0-9p]+)_beta(?P<beta>[0-9p]+)$"
)


def decode_number(s):
    return float(s.replace("p", ".").replace("em", "e-").replace("ep", "e+"))


def load_runs():
    rows = []
    if not ROOT.exists():
        raise FileNotFoundError(f"Run directory not found: {ROOT.resolve()}")

    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        p = match.groupdict()
        if int(p["seed"]) != SEED:
            continue

        summary = run_dir / "summary.csv"
        if not summary.exists():
            warnings.warn(f"Missing {summary}")
            continue
        try:
            df = pd.read_csv(summary)
            if df.empty or METRIC not in df.columns:
                warnings.warn(f"{summary}: empty or missing {METRIC}")
                continue
            if "round" in df.columns:
                df = df.sort_values("round")
            value = pd.to_numeric(df[METRIC], errors="coerce").iloc[-1]
            rows.append({
                "run": run_dir.name,
                "epochs": int(p["epochs"]),
                "muon_lr": decode_number(p["mlr"]),
                "scalar_lr": decode_number(p["slr"]),
                "weight_decay": decode_number(p["wd"]),
                "beta": decode_number(p["beta"]),
                "metric": float(value) if pd.notna(value) else np.nan,
            })
        except Exception as exc:
            warnings.warn(f"Could not read {summary}: {exc}")

    if not rows:
        raise RuntimeError(f"No matching runs found in {ROOT.resolve()}")
    return pd.DataFrame(rows)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_runs()
    valid = data[np.isfinite(data["metric"])].copy()
    if valid.empty:
        raise RuntimeError(f"No finite values found for {METRIC}")

    ascending = LOWER_IS_BETTER
    ranking = data.sort_values("metric", ascending=ascending, na_position="last")
    csv_path = OUT_DIR / f"{METRIC}_all_runs.csv"
    ranking.to_csv(csv_path, index=False)

    best = ranking.dropna(subset=["metric"]).iloc[0]
    epochs_values = sorted(valid["epochs"].unique())
    wd_values = sorted(valid["weight_decay"].unique())

    fig, axes = plt.subplots(
        len(epochs_values), len(wd_values),
        figsize=(5.2 * len(wd_values), 4.0 * len(epochs_values)),
        squeeze=False, constrained_layout=True,
    )
    vmin, vmax = valid["metric"].min(), valid["metric"].max()
    image = None

    for i, epochs in enumerate(epochs_values):
        for j, wd in enumerate(wd_values):
            ax = axes[i, j]
            subset = valid[
                (valid["epochs"] == epochs)
                & np.isclose(valid["weight_decay"], wd)
            ]
            pivot = subset.pivot_table(
                index="scalar_lr", columns="muon_lr",
                values="metric", aggfunc="mean"
            ).sort_index().sort_index(axis=1)

            image = ax.imshow(
                pivot.values, aspect="auto", origin="lower",
                cmap="viridis_r" if LOWER_IS_BETTER else "viridis",
                vmin=vmin, vmax=vmax,
            )
            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{x:g}" for x in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([f"{y:g}" for y in pivot.index])
            ax.set_xlabel("Muon learning rate")
            ax.set_ylabel("Scalar learning rate")
            ax.set_title(f"E={epochs}, weight decay={wd:g}")

            midpoint = (vmin + vmax) / 2
            for ri, scalar_lr in enumerate(pivot.index):
                for ci, muon_lr in enumerate(pivot.columns):
                    value = pivot.loc[scalar_lr, muon_lr]
                    if pd.isna(value):
                        label = "—"
                        text_color = "black"
                    else:
                        label = f"{value:.4g}"
                        text_color = (
                            "white" if
                            ((LOWER_IS_BETTER and value < midpoint) or
                             (not LOWER_IS_BETTER and value > midpoint))
                            else "black"
                        )
                    ax.text(ci, ri, label, ha="center", va="center",
                            color=text_color, fontsize=9)

                    if (
                        int(epochs) == int(best["epochs"])
                        and np.isclose(wd, best["weight_decay"])
                        and np.isclose(muon_lr, best["muon_lr"])
                        and np.isclose(scalar_lr, best["scalar_lr"])
                    ):
                        ax.add_patch(plt.Rectangle(
                            (ci - 0.48, ri - 0.48), 0.96, 0.96,
                            fill=False, edgecolor="red", linewidth=2.5,
                        ))

    fig.suptitle(
        f"FedMARS hyperparameter search: {METRIC} "
        f"(seed={SEED}, beta={sorted(valid['beta'].unique())})",
        fontsize=14, fontweight="bold",
    )
    cbar = fig.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    cbar.set_label(METRIC)

    plot_path = OUT_DIR / f"{METRIC}_heatmaps.png"
    fig.savefig(plot_path, dpi=220, bbox_inches="tight")
    print(f"Saved plot: {plot_path.resolve()}")

    print("\\nBest run:")
    print(best[["run", "epochs", "muon_lr", "scalar_lr",
                "weight_decay", "beta", "metric"]].to_string())
    print("\\nTop 10 runs:")
    print(ranking[["epochs", "muon_lr", "scalar_lr", "weight_decay",
                   "beta", "metric", "run"]].head(10).to_string(index=False))
    print(f"\\nFound {len(data)} runs, {len(valid)} with finite metrics.")
    print(f"Saved ranking: {csv_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()

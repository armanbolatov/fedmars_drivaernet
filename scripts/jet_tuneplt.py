import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path("results/runs/fedmars_noise/single_class")
METRIC = "c_d_mae"
LOWER_IS_BETTER = not METRIC.endswith("r2")
OUT_DIR = Path("jet_plots/fedmars_noise")

# Example: fedmars_E2_s42_mlr0p001_slr3em05_wd0p01_beta0p9
RUN_RE = re.compile(
    r"^fedmars_E(?P<epochs>\d+)_s(?P<seed>\d+)"
    r"_mlr(?P<mlr>[0-9p]+)_slr(?P<slr>[0-9pem]+)"
    r"_wd(?P<wd>[0-9p]+)_beta(?P<beta>[0-9p]+)$"
)


def decode_number(s):
    return float(s.replace("p", ".").replace("em", "e-").replace("ep", "e+"))


def load_history_runs():
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

        history_path = run_dir / "history.csv"
        if not history_path.exists():
            warnings.warn(f"Missing {history_path}")
            continue
        try:
            df = pd.read_csv(history_path)
            if df.empty or METRIC not in df.columns:
                warnings.warn(f"{history_path}: empty or missing {METRIC}")
                continue
            
            # Filter for overall body evaluation if 'body' column exists
            if "body" in df.columns:
                df = df[df["body"] == "all"]
            if df.empty:
                continue

            df["epochs"] = int(p["epochs"])
            df["seed"] = int(p["seed"])
            df["muon_lr"] = decode_number(p["mlr"])
            df["scalar_lr"] = decode_number(p["slr"])
            df["weight_decay"] = decode_number(p["wd"])
            df["beta"] = decode_number(p["beta"])
            df["run"] = run_dir.name

            rows.append(df)
        except Exception as exc:
            warnings.warn(f"Could not read {history_path}: {exc}")

    if not rows:
        raise RuntimeError(f"No matching history runs found in {ROOT.resolve()}")
    return pd.concat(rows, ignore_index=True)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = load_history_runs()
    
    data[METRIC] = pd.to_numeric(data[METRIC], errors="coerce")
    data = data.dropna(subset=[METRIC, "round"])

    if data.empty:
        raise RuntimeError(f"No valid data found for metric {METRIC}")

    # Find the best hyperparameter configuration per epoch setup 
    # based on the final round performance averaged across seeds.
    hp_cols = ["epochs", "muon_lr", "scalar_lr", "weight_decay", "beta"]
    
    max_rounds = data.groupby("run")["round"].max().reset_index()
    final_df = pd.merge(data, max_rounds, on=["run", "round"])
    
    hp_performance = final_df.groupby(hp_cols, as_index=False)[METRIC].mean()
    ascending = LOWER_IS_BETTER
    hp_performance = hp_performance.sort_values(METRIC, ascending=ascending)

    best_configs = hp_performance.drop_duplicates(subset=["epochs"]).copy()
    print("Best hyperparameter configuration(s) selected for plotting:")
    print(best_configs.to_string(index=False))

    # Filter history data to only include the best hyperparameter configuration for each epoch setup
    filtered_data = pd.merge(data, best_configs[hp_cols], on=hp_cols)

    # Group by epochs and round to compute mean, min, and max across seeds
    agg_df = filtered_data.groupby(["epochs", "round"])[METRIC].agg(
        mean="mean",
        min="min",
        max="max"
    ).reset_index()

    # --- Construct the requested CSV format ---
    # We want rows to be E2_mean, E2_min, E2_max, E4_mean, E4_min, E4_max (or whatever epochs exist)
    # and columns to be the iteration/round numbers.
    pivot_rows = []
    for epochs in sorted(agg_df["epochs"].unique()):
        sub = agg_df[agg_df["epochs"] == epochs].sort_values("round")
        for stat in ["mean", "min", "max"]:
            row_data = {"line_type": f"E{epochs}_{stat}"}
            for _, r in sub.iterrows():
                row_data[str(int(r["round"]))] = r[stat]
            pivot_rows.append(row_data)

    csv_output_df = pd.DataFrame(pivot_rows)
    csv_path = OUT_DIR / f"{METRIC}_learning_curves_summary_matrix.csv"
    csv_output_df.to_csv(csv_path, index=False)
    print(f"\nSaved summary matrix CSV: {csv_path.resolve()}")

    # --- Plotting learning curves with dispersion bands ---
    plt.figure(figsize=(9, 6), constrained_layout=True)
    epochs_values = sorted(agg_df["epochs"].unique())
    colors = plt.cm.tab10(np.linspace(0, 1, len(epochs_values)))

    for idx, epochs in enumerate(epochs_values):
        sub = agg_df[agg_df["epochs"] == epochs].sort_values("round")
        rounds = sub["round"]
        mean_val = sub["mean"]
        min_val = sub["min"]
        max_val = sub["max"]
        color = colors[idx]

        plt.plot(rounds, mean_val, label=f"Epochs = {epochs} (Mean)", color=color, linewidth=2.2)
        plt.plot(rounds, min_val, linestyle="--", color=color, alpha=0.5, linewidth=1)
        plt.plot(rounds, max_val, linestyle="--", color=color, alpha=0.5, linewidth=1)
        plt.fill_between(rounds, min_val, max_val, color=color, alpha=0.15)

    plt.xlabel("Training Round", fontsize=12)
    plt.ylabel(f"{METRIC} (Lower is Better)" if LOWER_IS_BETTER else f"{METRIC} (Higher is Better)", fontsize=12)
    plt.title(f"FedMARS Learning Curves: {METRIC} across Seeds (Min-Max Dispersion)", fontsize=14, fontweight="bold")
    plt.legend(loc="best", frameon=True)
    plt.grid(True, linestyle=":", alpha=0.6)

    plot_path = OUT_DIR / f"{METRIC}_learning_curves_dispersion.png"
    plt.savefig(plot_path, dpi=220, bbox_inches="tight")
    print(f"Saved learning curve plot: {plot_path.resolve()}")
    plt.show()


if __name__ == "__main__":
    main()

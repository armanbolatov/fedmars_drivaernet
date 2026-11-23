import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Results produced by the FedLion command in the experiment.
ROOT = Path("results/runs/fedlion_search/single_class")
OUT_DIR = Path("jet_plots/fedlion_search_tables")

# Expected run name:
# fedlion_E4_s123_llr0p0001_lb10p8_lb20p95_lwd0p005
RUN_RE = re.compile(
    r"^fedlion_E(?P<epochs>\d+)_s(?P<seed>\d+)"
    r"_llr(?P<lion_lr>[0-9pem]+)"
    r"_lb1(?P<beta1>[0-9pem]+)"
    r"_lb2(?P<beta2>[0-9pem]+)"
    r"_lwd(?P<weight_decay>[0-9pem]+)$"
)

# The script will use the first matching loss column it finds.
LOSS_CANDIDATES = [
    "loss", "val_loss", "eval_loss", "total_loss",
    "train_loss", "test_loss", "objective",
]
METRICS = ["loss", "c_d_mae"]


def decode_number(value):
    return float(value.replace("m", "-").replace("p", "."))


def find_loss_column(columns):
    lower_to_original = {str(c).lower(): c for c in columns}
    for candidate in LOSS_CANDIDATES:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    return None


def load_final_runs():
    if not ROOT.exists():
        raise FileNotFoundError(f"Run directory not found: {ROOT.resolve()}")

    rows = []
    found_dirs = 0

    for run_dir in sorted(ROOT.iterdir()):
        if not run_dir.is_dir():
            continue

        match = RUN_RE.match(run_dir.name)
        if not match:
            continue
        found_dirs += 1

        history_path = run_dir / "history.csv"
        if not history_path.exists():
            warnings.warn(f"Skipping {run_dir.name}: no history.csv")
            continue

        try:
            df = pd.read_csv(history_path)
        except Exception as exc:
            warnings.warn(f"Skipping {history_path}: {exc}")
            continue

        if df.empty or "round" not in df.columns:
            warnings.warn(f"Skipping {history_path}: empty or no round column")
            continue

        # Keep overall-body rows when this column is present.
        if "body" in df.columns:
            all_rows = df[df["body"].astype(str).str.lower() == "all"]
            if not all_rows.empty:
                df = all_rows

        df["round"] = pd.to_numeric(df["round"], errors="coerce")
        df = df.dropna(subset=["round"])
        if df.empty:
            continue

        # Use the last available round for this run. If multiple rows exist at
        # that round (e.g. per-body rows), use the mean of numeric metrics.
        final_round = df["round"].max()
        final_rows = df[df["round"] == final_round]

        params = match.groupdict()
        row = {
            "epochs": int(params["epochs"]),
            "seed": int(params["seed"]),
            "lion_lr": decode_number(params["lion_lr"]),
            "beta1": decode_number(params["beta1"]),
            "beta2": decode_number(params["beta2"]),
            "weight_decay": decode_number(params["weight_decay"]),
            "round": final_round,
            "run": run_dir.name,
        }

        loss_col = find_loss_column(final_rows.columns)
        if loss_col is not None:
            row["loss"] = pd.to_numeric(
                final_rows[loss_col], errors="coerce"
            ).mean()
        else:
            row["loss"] = np.nan
            warnings.warn(
                f"{run_dir.name}: no recognized loss column; "
                f"looked for {LOSS_CANDIDATES}"
            )

        if "c_d_mae" in final_rows.columns:
            row["c_d_mae"] = pd.to_numeric(
                final_rows["c_d_mae"], errors="coerce"
            ).mean()
        else:
            row["c_d_mae"] = np.nan
            warnings.warn(f"{run_dir.name}: no c_d_mae column")

        rows.append(row)

    if found_dirs == 0:
        raise RuntimeError(
            f"No run directories matched the expected FedLion naming pattern "
            f"under {ROOT.resolve()}"
        )
    if not rows:
        raise RuntimeError(
            f"Found {found_dirs} matching run directories, but none had usable "
            "history.csv files."
        )

    return pd.DataFrame(rows)


def aggregate_seeds(final_runs):
    # Aggregate only runs/metrics that actually exist. Missing runs are left
    # missing rather than treated as zero.
    keys = ["epochs", "lion_lr", "beta1", "beta2", "weight_decay"]
    pieces = []

    for metric in METRICS:
        available = final_runs.dropna(subset=[metric])
        if available.empty:
            continue
        grouped = (
            available.groupby(keys)[metric]
            .agg(mean="mean", min="min", max="max", n_seeds="count")
            .reset_index()
        )
        grouped["metric"] = metric
        pieces.append(grouped)

    if not pieces:
        raise RuntimeError("No usable loss or c_d_mae values found in history.csv files.")

    return pd.concat(pieces, ignore_index=True)


def make_heatmap(ax, table, metric, title):
    values = table.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)

    # Both loss and c_d_mae are lower-is-better: darkest cell is best.
    image = ax.imshow(masked, aspect="auto", cmap="YlOrRd_r")

    ax.set_xticks(np.arange(len(table.columns)))
    ax.set_xticklabels([f"{x:g}" for x in table.columns])
    ax.set_yticks(np.arange(len(table.index)))
    ax.set_yticklabels([f"{x:g}" for x in table.index])
    ax.set_xlabel("Weight decay")
    ax.set_ylabel("Lion learning rate")
    ax.set_title(title)

    finite = values[np.isfinite(values)]
    best = np.nanmin(values) if finite.size else np.nan

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            if np.isfinite(value):
                is_best = np.isclose(value, best, rtol=1e-9, atol=1e-12)
                ax.text(
                    j, i, f"{value:.5g}" + ("\\nBEST" if is_best else ""),
                    ha="center", va="center",
                    color="black", fontsize=9,
                    fontweight="bold" if is_best else "normal",
                )
                if is_best:
                    ax.add_patch(
                        plt.Rectangle(
                            (j - 0.48, i - 0.48), 0.96, 0.96,
                            fill=False, edgecolor="blue", linewidth=2.5
                        )
                    )
            else:
                ax.text(j, i, "—", ha="center", va="center", color="gray")

    return image


def plot_tables(aggregated):
    for metric in METRICS:
        metric_data = aggregated[aggregated["metric"] == metric]
        if metric_data.empty:
            print(f"Skipping {metric}: no values found.")
            continue

        for epochs in sorted(metric_data["epochs"].unique()):
            beta_values = sorted(metric_data["beta1"].unique())
            ncols = max(1, len(beta_values))
            fig, axes = plt.subplots(
                1, ncols, figsize=(6.2 * ncols, 5.5), squeeze=False,
                constrained_layout=True
            )
            axes = axes[0]

            for ax, beta1 in zip(axes, beta_values):
                subset = metric_data[
                    (metric_data["epochs"] == epochs)
                    & (metric_data["beta1"] == beta1)
                ]
                table = subset.pivot(
                    index="lion_lr", columns="weight_decay", values="mean"
                ).sort_index().sort_index(axis=1)

                image = make_heatmap(
                    ax, table, metric,
                    f"Epochs={epochs}, beta1={beta1:g}, beta2 fixed/configured"
                )
                fig.colorbar(image, ax=ax, shrink=0.82, label=f"Mean final {metric}")

            fig.suptitle(
                f"FedLion final {metric} — local epochs {epochs}\\n"
                "Cell values are means across available seeds; BEST marks minimum",
                fontsize=13
            )
            path = OUT_DIR / f"{metric}_heatmap_E{epochs}.png"
            fig.savefig(path, dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved {path.resolve()}")


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    final_runs = load_final_runs()
    aggregated = aggregate_seeds(final_runs)

    # Save run-level values and seed-aggregated table, including all setups.
    run_csv = OUT_DIR / "final_values_per_run_seed.csv"
    final_runs.to_csv(run_csv, index=False)
    print(f"Saved {run_csv.resolve()}")

    summary_csv = OUT_DIR / "final_values_mean_min_max_across_seeds.csv"
    aggregated.to_csv(summary_csv, index=False)
    print(f"Saved {summary_csv.resolve()}")

    print("\\nAvailable final results:")
    print(aggregated.to_string(index=False))

    plot_tables(aggregated)


if __name__ == "__main__":
    main()

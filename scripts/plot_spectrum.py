"""Figure for the gradient-spectrum measurement (scripts/grad_spectrum.py): per model,
the distribution over the 42 weight matrices of the gradient's stable rank, the gain of
an orthogonalised step and the spectral size of a sign step. Same style as plot.py.

  python scripts/plot_spectrum.py
"""
import csv

from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt

from plot import INK, OUT, RULE, SERIES, TABLES, WIDTH, log_ticks

# model of the table -> method of plot.SERIES (None: the untrained network)
METHOD = {"init": None, "fedavg_cosine_E16_s42": "fedavg_cosine",
          "scaffold_cosine_E16_s42": "scaffold_cosine",
          "fedlion_cosine2_E16_s42": "fedlion_cosine2",
          "centralized_cosine_E1_s42": "centralized_cosine", "fedmuon_E4_s42": "fedmuon",
          "fedmars_E2_s42": "fedmars", "fedolion_E8_s42": "fedolion",
          "centralized_muon_E1_s42": "centralized_muon"}
PANELS = [("stable_rank", "stable rank", True), ("gain", "gain $g$", False),
          ("sign_size", "size of a sign step", True)]


def main():
    rows = list(csv.DictReader((TABLES / "grad_spectrum.csv").open()))
    models = [m for m in METHOD if any(r["model"] == m for r in rows)]
    labels = [SERIES[METHOD[m]][0] if METHOD[m] else "Initialisation" for m in models]
    colours = [SERIES[METHOD[m]][1] if METHOD[m] else RULE for m in models]
    fig, axes = plt.subplots(1, 3, figsize=(WIDTH, 0.9 + 0.45 * len(models)), sharey=True,
                             constrained_layout=True)
    fig.get_layout_engine().set(w_pad=0.06, wspace=0.04)
    for ax, (key, xlabel, log) in zip(axes, PANELS):
        data = [[float(r[key]) for r in rows if r["model"] == m] for m in models]
        kw = dict(whis=(5, 95), widths=0.6, patch_artist=True, showfliers=True)
        try:
            bp = ax.boxplot(data, orientation="horizontal", **kw)
        except TypeError:                 # matplotlib before 3.10
            bp = ax.boxplot(data, vert=False, **kw)
        for i, colour in enumerate(colours):
            bp["boxes"][i].set(facecolor=to_rgba(colour, 0.35), edgecolor=colour, lw=1.2)
            bp["medians"][i].set(color=colour, lw=2.0)
            for part in ("whiskers", "caps"):
                for line in bp[part][2 * i:2 * i + 2]:
                    line.set(color=colour, lw=1.2)
            bp["fliers"][i].set(marker="o", ms=3, mfc="none", mec=colour, mew=0.8)
        if log:
            ax.set_xscale("log")
            log_ticks(ax.xaxis)
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(range(1, len(models) + 1), labels, color=INK)
    axes[0].invert_yaxis()                # the untrained network on top
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"grad_spectrum.{ext}", dpi=200)
    print("wrote", OUT / "grad_spectrum.pdf")


if __name__ == "__main__":
    main()

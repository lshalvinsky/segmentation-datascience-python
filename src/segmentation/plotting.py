"""Matplotlib helpers with one consistent, colour-blind-aware style."""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# Categorical slots in fixed order: colour follows the segment, never its rank
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
SEQ = "#2a78d6"
SEQ_CMAP = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])
DIV_CMAP = LinearSegmentedColormap.from_list("div_blue_red", ["#184f95", "#86b6ef", "#f0efec", "#f0a3a2", "#b52f2f"])


def set_style():
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "figure.dpi": 110,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_2,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 10,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "axes.axisbelow": True,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "lines.linewidth": 2,
        "legend.frameon": False,
    })


def segment_colors(segment_names):
    return {name: CATEGORICAL[i] for i, name in enumerate(segment_names)}


def plot_k_selection(scores, stability, bic, chosen_k):
    """Four single-measure panels (no dual axes): elbow, silhouette, stability, GMM BIC."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.5))
    panels = [
        (scores["inertia"], "K-Means inertia (elbow)", "look for the bend"),
        (scores["silhouette"], "Silhouette score", "higher is better"),
        (stability, "Bootstrap stability (mean ARI)", "higher is better"),
        (bic, "Gaussian mixture BIC", "lower is better"),
    ]
    for ax, (s, title, note) in zip(axes.flat, panels):
        ax.plot(s.index, s.values, color=SEQ, marker="o", markersize=6)
        ax.plot(chosen_k, s.loc[chosen_k], marker="o", markersize=11, color=SEQ,
                markeredgecolor=INK, markeredgewidth=1.5, linestyle="none")
        ax.annotate(f"k = {chosen_k}", (chosen_k, s.loc[chosen_k]), xytext=(8, 8),
                    textcoords="offset points", color=INK, fontweight="bold")
        ax.set_title(title)
        ax.set_xlabel(f"number of clusters (k) - {note}")
        ax.set_xticks(list(s.index))
        ax.grid(axis="x", visible=False)
    fig.tight_layout()
    return fig


def plot_barh(values, colors=None, title="", fmt="{:.0%}", xlabel=""):
    """Horizontal bars with a direct value label on each bar."""
    values = values.iloc[::-1]
    colors = colors or {}
    fig, ax = plt.subplots(figsize=(8, 0.5 * len(values) + 1.2))
    bar_colors = [colors.get(i, SEQ) for i in values.index]
    ax.barh(values.index, values.values, color=bar_colors, height=0.6, edgecolor=SURFACE, linewidth=2)
    for y, v in enumerate(values.values):
        ax.annotate(fmt.format(v), (v, y), xytext=(5, 0), textcoords="offset points",
                    va="center", color=INK_2)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, values.max() * 1.15)
    if "%" in fmt:
        ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    fig.tight_layout()
    return fig


def plot_segment_map(coords, labels, colors, axis_labels=("PC1", "PC2")):
    """Small multiples: each segment highlighted in its colour over the full base in grey.

    Faceting keeps every segment readable without relying on telling 5 hues apart
    in one overlapping scatter.
    """
    names = list(colors)
    fig, axes = plt.subplots(1, len(names), figsize=(3.2 * len(names), 3.4), sharex=True, sharey=True)
    for ax, name in zip(axes, names):
        mask = labels == name
        ax.scatter(coords[~mask, 0], coords[~mask, 1], s=4, color=GRID, linewidths=0)
        ax.scatter(coords[mask, 0], coords[mask, 1], s=6, color=colors[name], alpha=0.7, linewidths=0)
        ax.set_title(f"{name}\n{mask.mean():.0%} of customers", fontsize=10)
        ax.set_xlabel(axis_labels[0])
    axes[0].set_ylabel(axis_labels[1])
    fig.tight_layout()
    return fig


def plot_heatmap(matrix, title="", cmap=DIV_CMAP, center=0.0, fmt="{:+.1f}", vlim=None, cbar_label=""):
    """Annotated heatmap; diverging around ``center`` by default."""
    data = matrix.values.astype(float)
    if vlim is None:
        vlim = np.nanmax(np.abs(data - center))
    vmin, vmax = (center - vlim, center + vlim) if cmap is DIV_CMAP else (np.nanmin(data), np.nanmax(data))
    fig, ax = plt.subplots(figsize=(0.75 * data.shape[1] + 3, 0.55 * data.shape[0] + 1.6))
    im = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(data.shape[1]), matrix.columns, rotation=40, ha="right")
    ax.set_yticks(range(data.shape[0]), matrix.index)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    span = vmax - vmin
    for (i, j), v in np.ndenumerate(data):
        strong = abs(v - center) / (vlim or 1) > 0.6 if cmap is DIV_CMAP else (v - vmin) / span > 0.6
        ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=8.5,
                color="white" if strong else INK)
    cb = fig.colorbar(im, ax=ax, shrink=0.8)
    cb.outline.set_visible(False)
    cb.set_label(cbar_label, color=INK_2)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_roc(results):
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for i, (name, r) in enumerate(results.items()):
        ax.plot(r["fpr"], r["tpr"], color=CATEGORICAL[i], label=f"{name} (AUC {r['auc']:.3f})")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Random")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Campaign response - ROC curve")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_gains(gains):
    x = np.r_[0, gains["cum_pct_customers"].values]
    y = np.r_[0, gains["cum_pct_responders"].values]
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(x, y, color=CATEGORICAL[1], marker="o", markersize=6, label="Model-ranked targeting")
    ax.plot([0, 1], [0, 1], color=MUTED, linestyle="--", linewidth=1.2, label="Random targeting")
    top3 = gains.iloc[2]
    ax.annotate(f"Top 30% of customers\ncapture {top3['cum_pct_responders']:.0%} of responders",
                (top3["cum_pct_customers"], top3["cum_pct_responders"]), xytext=(20, -40),
                textcoords="offset points", color=INK, arrowprops={"arrowstyle": "-", "color": MUTED})
    ax.xaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0))
    ax.set_xlabel("Customers targeted (ranked by predicted response)")
    ax.set_ylabel("Responders captured")
    ax.set_title("Cumulative gains")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig

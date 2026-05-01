"""
Comprehensive visualisation suite for bacterial growth analysis.

All functions return matplotlib Figure objects so they can be saved, shown,
or embedded in notebooks. Display is suppressed by default unless show=True.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe in scripts and notebooks
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

__all__ = [
    "plot_growth_curve",
    "plot_growth_curve_comparison",
    "plot_growth_boundary",
    "plot_confusion_matrix",
    "plot_feature_importance",
    "plot_roc_curves",
    "plot_calibration_curve",
    "plot_kinetics_distributions",
    "plot_conformal_coverage",
    "save_figure",
]

_SPECIES_COLORS = {
    "E. coli": "#2196F3",
    "S. aureus": "#F44336",
    "P. aeruginosa": "#4CAF50",
    "B. subtilis": "#FF9800",
}
_DEFAULT_CMAP = "viridis"


# ---------------------------------------------------------------------------
# Growth curves
# ---------------------------------------------------------------------------

def plot_growth_curve(
    t: np.ndarray,
    log_n_obs: np.ndarray,
    kinetics=None,
    species: str = "",
    show: bool = False,
) -> plt.Figure:
    """Plot observed growth data with optional model overlay."""
    fig, ax = plt.subplots(figsize=(7, 4))

    color = _SPECIES_COLORS.get(species, "#555555")
    ax.scatter(t, log_n_obs, s=30, color=color, alpha=0.7, zorder=3, label="Observed")

    if kinetics is not None:
        from .kinetics import predict_growth_curve
        t_smooth = np.linspace(t[0], t[-1], 300)
        y_smooth = predict_growth_curve(kinetics, t_smooth)
        ax.plot(t_smooth, y_smooth, color=color, linewidth=2,
                label=f"Model: {kinetics.model_name} (RMSE={kinetics.rmse:.3f})")

        # Mark lag and µmax
        ax.axvline(kinetics.lag_time, color="grey", linestyle="--", linewidth=1,
                   label=f"Lag = {kinetics.lag_time:.1f} h")
        slope = kinetics.mu_max
        t_infl = kinetics.lag_time + (kinetics.log_nmax - kinetics.log_n0) / (2 * slope + 1e-9)
        y_infl = kinetics.log_nmax / 2 + kinetics.log_n0 / 2
        ax.annotate(
            f"µmax = {slope:.3f} /h",
            xy=(t_infl, y_infl), xytext=(t_infl + 1, y_infl - 0.5),
            arrowprops=dict(arrowstyle="->", color="black"),
            fontsize=9,
        )

    ax.set_xlabel("Time (h)", fontsize=11)
    ax.set_ylabel("log CFU / mL", fontsize=11)
    ax.set_title(f"Growth Curve — {species or 'Unknown'}", fontsize=12, fontweight="bold")
    ax.legend(framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_growth_curve_comparison(
    curves_df: pd.DataFrame,
    n_per_species: int = 3,
    show: bool = False,
) -> plt.Figure:
    """Plot growth curves for multiple species side-by-side."""
    species_list = curves_df["species"].unique()
    ncols = len(species_list)
    fig, axes = plt.subplots(1, ncols, figsize=(5 * ncols, 4), sharey=True)
    if ncols == 1:
        axes = [axes]

    for ax, sp in zip(axes, species_list):
        ids = curves_df[curves_df["species"] == sp]["curve_id"].unique()[:n_per_species]
        color = _SPECIES_COLORS.get(sp, "#555555")
        for cid in ids:
            sub = curves_df[curves_df["curve_id"] == cid].sort_values("time")
            ax.plot(sub["time"], sub["log_n"], alpha=0.75, color=color, linewidth=1.5)
        ax.set_title(sp, fontsize=11, fontweight="bold")
        ax.set_xlabel("Time (h)")
        ax.grid(alpha=0.3)

    axes[0].set_ylabel("log CFU / mL", fontsize=11)
    fig.suptitle("Simulated Growth Curves by Species", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Cardinal Parameter Model — growth boundary
# ---------------------------------------------------------------------------

def plot_growth_boundary(
    T_grid: np.ndarray,
    pH_grid: np.ndarray,
    mu_grid: np.ndarray,
    species: str = "",
    show: bool = False,
) -> plt.Figure:
    """Contour plot of µmax over (T, pH) space from the Cardinal Parameter Model."""
    fig, ax = plt.subplots(figsize=(7, 5))
    cf = ax.contourf(T_grid, pH_grid, mu_grid, levels=20, cmap=_DEFAULT_CMAP)
    cs = ax.contour(T_grid, pH_grid, mu_grid, levels=[0.01], colors="red",
                    linestyles="--", linewidths=1.5)
    ax.clabel(cs, fmt="No growth", fontsize=9)
    cbar = fig.colorbar(cf, ax=ax, label="µmax (log CFU / mL / h)")
    ax.set_xlabel("Temperature (°C)", fontsize=11)
    ax.set_ylabel("pH", fontsize=11)
    ax.set_title(f"Growth Boundary — {species}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Classification diagnostics
# ---------------------------------------------------------------------------

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: list[str],
    normalise: bool = True,
    show: bool = False,
) -> plt.Figure:
    """Heatmap confusion matrix with optional row normalisation."""
    if normalise:
        cm_plot = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)
        fmt = ".2f"
        title = "Normalised Confusion Matrix"
    else:
        cm_plot = cm
        fmt = "d"
        title = "Confusion Matrix"

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_plot, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names)
    ax.set_xlabel("Predicted", fontsize=11)
    ax.set_ylabel("True", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    thresh = cm_plot.max() / 2.0
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            val = cm_plot[i, j]
            text = format(val, fmt)
            ax.text(j, i, text, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=10)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: list[str],
    top_n: int = 20,
    title: str = "Feature Importance",
    show: bool = False,
) -> plt.Figure:
    """Horizontal bar chart of top-N feature importances."""
    indices = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colors = cm.viridis(np.linspace(0.3, 0.9, len(indices)))
    ax.barh(range(len(indices)), importances[indices], color=colors)
    ax.set_yticks(range(len(indices)))
    ax.set_yticklabels([feature_names[i] for i in indices], fontsize=9)
    ax.set_xlabel("Importance", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_roc_curves(
    y_true_bin: np.ndarray,
    y_score: np.ndarray,
    class_names: list[str],
    show: bool = False,
) -> plt.Figure:
    """One-vs-Rest ROC curves with AUC for multi-class classification."""
    from sklearn.metrics import roc_curve, auc

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = list(_SPECIES_COLORS.values()) + ["purple", "brown", "pink"]

    for i, (name, color) in enumerate(zip(class_names, colors)):
        if y_true_bin.ndim == 1:
            y_bin = (y_true_bin == i).astype(int)
        else:
            y_bin = y_true_bin[:, i]
        fpr, tpr, _ = roc_curve(y_bin, y_score[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f"{name} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
    ax.set_xlabel("False Positive Rate", fontsize=11)
    ax.set_ylabel("True Positive Rate", fontsize=11)
    ax.set_title("One-vs-Rest ROC Curves", fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_calibration_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    show: bool = False,
) -> plt.Figure:
    """Reliability diagram to assess probability calibration."""
    from sklearn.calibration import calibration_curve

    fig, ax = plt.subplots(figsize=(6, 5))
    frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    ax.plot(mean_pred, frac_pos, "s-", label="Calibration curve", color="#2196F3")
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability", fontsize=11)
    ax.set_ylabel("Fraction of positives", fontsize=11)
    ax.set_title("Calibration (Reliability Diagram)", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_kinetics_distributions(
    kinetics_df: pd.DataFrame,
    parameter: str = "mu_max",
    show: bool = False,
) -> plt.Figure:
    """Violin plot of a kinetic parameter distribution per species."""
    species_list = sorted(kinetics_df["species"].unique())
    data = [kinetics_df.loc[kinetics_df["species"] == sp, parameter].dropna().values
            for sp in species_list]
    colors = [_SPECIES_COLORS.get(sp, "#777") for sp in species_list]

    fig, ax = plt.subplots(figsize=(8, 5))
    parts = ax.violinplot(data, positions=range(len(species_list)),
                          showmeans=True, showmedians=False)
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.75)

    ax.set_xticks(range(len(species_list)))
    ax.set_xticklabels(species_list, rotation=15, ha="right")
    ax.set_ylabel(parameter.replace("_", " ").title(), fontsize=11)
    ax.set_title(f"Distribution of {parameter} by Species", fontsize=12, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


def plot_conformal_coverage(
    prediction_sets: list[list[str]],
    y_true: np.ndarray,
    target_coverage: float = 0.9,
    show: bool = False,
) -> plt.Figure:
    """Bar chart of empirical coverage per class and overall for conformal sets."""
    classes = sorted(set(y_true))
    overall_covered = [y in pset for y, pset in zip(y_true, prediction_sets)]
    overall_coverage = float(np.mean(overall_covered))

    per_class = {}
    for cls in classes:
        mask = y_true == cls
        covered = [y in ps for y, ps in
                   zip(y_true[mask], [ps for y, ps in zip(y_true, prediction_sets) if y == cls])]
        per_class[cls] = float(np.mean(covered))

    fig, ax = plt.subplots(figsize=(7, 4))
    all_labels = list(per_class.keys()) + ["Overall"]
    all_vals = list(per_class.values()) + [overall_coverage]
    colors = [_SPECIES_COLORS.get(lb, "#777") if lb != "Overall" else "#333" for lb in all_labels]
    bars = ax.bar(all_labels, all_vals, color=colors, alpha=0.8)
    ax.axhline(target_coverage, color="red", linestyle="--", linewidth=1.5,
               label=f"Target coverage ({target_coverage:.0%})")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Empirical Coverage", fontsize=11)
    ax.set_title("Conformal Prediction Coverage", fontsize=12, fontweight="bold")
    ax.legend()
    for bar, val in zip(bars, all_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    if show:
        plt.show()
    return fig


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, path: str | Path, dpi: int = 150) -> None:
    """Save figure to disk with sensible defaults."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved to {path}")

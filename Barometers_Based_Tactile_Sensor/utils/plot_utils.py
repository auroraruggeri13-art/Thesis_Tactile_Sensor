"""
Plotting utilities for tactile sensor force prediction models.

Functions
---------
plot_pred_vs_actual         -- Scatter plot of predicted vs. actual (all targets).
plot_error_distributions    -- Histogram + box-plot of per-target residuals.
plot_training_history_nn    -- Loss / R² curves for a PyTorch MLP.
plot_training_history_keras -- Loss / MAE curves for a Keras History object.
plot_data_distribution      -- Histogram of target variable distributions.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score


def plot_pred_vs_actual(
    y_true,
    y_pred,
    target_cols,
    title_suffix="",
    save_path=None,
    alpha=0.5,
    s=20,
    scatter_color=None,
    show_units=False,
    figsize_factor=(4, 4),
    title_fontsize=10,
    xlabel_fontsize=8,
    ylabel_fontsize=8,
):
    """
    Scatter plot of predicted vs. actual values for every target column.

    Parameters
    ----------
    y_true, y_pred : np.ndarray, shape (N, n_targets)
    target_cols : list of str
    title_suffix : str, optional
        Added to the figure suptitle; if empty, no suptitle is set.
    save_path : str or Path, optional
        If given, the figure is saved here (dpi=300, tight bbox).
    alpha, s : float
        Scatter transparency and marker size.
    scatter_color : color spec or None
        Explicit scatter colour; None uses the matplotlib default cycle.
    show_units : bool
        If True, append "mm" (for x, y) or "N" (for forces) to subplot titles.
    figsize_factor : (w, h)
        Width and height *per target subplot*.
    title_fontsize, xlabel_fontsize, ylabel_fontsize : int
        Font sizes for subplot title and axis labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(
        1, n_targets, figsize=(figsize_factor[0] * n_targets, figsize_factor[1])
    )
    if n_targets == 1:
        axes = [axes]

    scatter_kwargs = dict(alpha=alpha, s=s)
    if scatter_color is not None:
        scatter_kwargs["color"] = scatter_color

    for i, col in enumerate(target_cols):
        ax = axes[i]
        true_vals = y_true[:, i]
        pred_vals = y_pred[:, i]

        ax.scatter(true_vals, pred_vals, **scatter_kwargs)

        min_val = min(true_vals.min(), pred_vals.min())
        max_val = max(true_vals.max(), pred_vals.max())
        lims = [
            min_val - (max_val - min_val) * 0.05,
            max_val + (max_val - min_val) * 0.05,
        ]
        ax.plot(lims, lims, "r--", alpha=0.75, linewidth=1.5)

        mae = mean_absolute_error(true_vals, pred_vals)
        r2 = r2_score(true_vals, pred_vals)

        if show_units:
            unit = "mm" if col in ["x", "y"] else "N"
            ax.set_title(
                f"{col}\nMAE: {mae:.2f} {unit} | R\u00b2: {r2:.3f}",
                fontsize=title_fontsize,
            )
        else:
            ax.set_title(
                f"{col}\nMAE: {mae:.2f} | R\u00b2: {r2:.3f}",
                fontsize=title_fontsize,
            )

        ax.set_xlabel("Actual", fontsize=xlabel_fontsize)
        ax.set_ylabel("Predicted", fontsize=ylabel_fontsize)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    if title_suffix:
        plt.suptitle(f"Predicted vs Actual ({title_suffix})", fontsize=12)
    plt.tight_layout()

    if save_path is not None:
        parent = os.path.dirname(os.path.abspath(save_path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=300)
        print(f"Saved predictions plot to: {save_path}")

    return fig


def plot_error_distributions(y_test, y_pred, target_cols, title_suffix=""):
    """
    Plot error histograms (top row) and box plots (bottom row) for each target.

    Parameters
    ----------
    y_test, y_pred : np.ndarray, shape (N, n_targets)
    target_cols : list of str
    title_suffix : str, optional

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_targets = len(target_cols)
    fig, axes = plt.subplots(2, n_targets, figsize=(4 * n_targets, 8))
    if n_targets == 1:
        axes = axes.reshape(-1, 1)

    for i, col in enumerate(target_cols):
        errors = y_test[:, i] - y_pred[:, i]

        # -- Top row: histogram --
        ax_hist = axes[0, i]
        ax_hist.hist(errors, bins=50, alpha=0.7, color="orange", edgecolor="black")
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        mae = np.mean(np.abs(errors))
        ax_hist.axvline(
            mean_err, color="red", linestyle="--", linewidth=2,
            label=f"Mean: {mean_err:.3f}"
        )
        ax_hist.axvline(0, color="green", linestyle="-", linewidth=1.5, alpha=0.7, label="Zero")
        unit = "mm" if col in ["x", "y"] else "N"
        ax_hist.set_xlabel(f"Error ({unit})", fontsize=10)
        ax_hist.set_ylabel("Frequency", fontsize=10)
        ax_hist.set_title(
            f"{col} Error Distribution\nStd: {std_err:.3f} | MAE: {mae:.3f}",
            fontsize=11,
        )
        ax_hist.legend(fontsize=9)
        ax_hist.grid(True, alpha=0.3)

        # -- Bottom row: box plot --
        ax_box = axes[1, i]
        ax_box.boxplot(
            [errors],
            vert=True,
            widths=0.6,
            patch_artist=True,
            boxprops=dict(facecolor="lightblue", alpha=0.7),
            medianprops=dict(color="red", linewidth=2),
            whiskerprops=dict(linewidth=1.5),
            capprops=dict(linewidth=1.5),
        )
        ax_box.axhline(0, color="green", linestyle="-", linewidth=1.5, alpha=0.7)
        p5 = np.percentile(errors, 5)
        p95 = np.percentile(errors, 95)
        ax_box.set_ylabel(f"Error ({unit})", fontsize=10)
        ax_box.set_title(
            f"{col} Error Box Plot\n5th: {p5:.3f} | 95th: {p95:.3f}", fontsize=11
        )
        ax_box.grid(True, alpha=0.3, axis="y")
        ax_box.set_xticklabels([col])

    if title_suffix:
        plt.suptitle(
            f"Error Distribution Analysis ({title_suffix})", fontsize=13, y=0.995
        )
    plt.tight_layout()
    return fig


def plot_training_history_nn(model):
    """
    Plot loss and R² history for a PyTorch RegressionModelNN instance.

    Parameters
    ----------
    model : RegressionModelNN
        Must expose `.training_losses` (list) and `.validation_metrics` (list of dicts
        with keys 'loss' and 'r2').

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    ax1.plot(model.training_losses, label="Training Loss", alpha=0.8)
    val_losses = [m["loss"] for m in model.validation_metrics]
    ax1.plot(val_losses, label="Validation Loss", alpha=0.8)
    ax1.set_title("Model Training History - Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    val_r2 = [m["r2"] for m in model.validation_metrics]
    ax2.plot(val_r2, label="Validation R\u00b2", color="green", alpha=0.8)
    ax2.set_title("Model Training History - R\u00b2 Score")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("R\u00b2 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_training_history_keras(history, target_cols=None):
    """
    Plot training and validation loss/MAE from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History
    target_cols : list, optional  (unused; kept for call-site compatibility)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    ax.plot(history.history["loss"], label="Training Loss", linewidth=2)
    ax.plot(history.history["val_loss"], label="Validation Loss", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MSE Loss", fontsize=11)
    ax.set_title("Model Loss (MSE)", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(history.history["mae"], label="Training MAE", linewidth=2)
    ax.plot(history.history["val_mae"], label="Validation MAE", linewidth=2)
    ax.set_xlabel("Epoch", fontsize=11)
    ax.set_ylabel("MAE", fontsize=11)
    ax.set_title("Model MAE", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.suptitle("LSTM Training History", fontsize=13, y=1.02)
    plt.tight_layout()
    return fig


def plot_data_distribution(y_data, target_names, title_prefix=""):
    """
    Histograms showing the distribution of each target variable.

    Parameters
    ----------
    y_data : np.ndarray, shape (N, n_targets)
    target_names : list of str
    title_prefix : str, optional
        Prefix for the figure suptitle (e.g. "Training" or "Test").

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    n_targets = y_data.shape[1]
    fig, axes = plt.subplots(1, n_targets, figsize=(4 * n_targets, 4))
    if n_targets == 1:
        axes = [axes]

    for i in range(n_targets):
        ax = axes[i]
        data = y_data[:, i]

        ax.hist(data, bins=50, alpha=0.7, color="steelblue", edgecolor="black")

        mean_val = np.mean(data)
        std_val = np.std(data)
        median_val = np.median(data)

        ax.axvline(
            mean_val, color="red", linestyle="--", linewidth=2,
            label=f"Mean: {mean_val:.2f}"
        )
        ax.axvline(
            median_val, color="green", linestyle="--", linewidth=2,
            label=f"Median: {median_val:.2f}"
        )

        if target_names[i] in ["x", "y"]:
            unit = "mm"
        elif target_names[i] in ["tx", "ty", "tz"]:
            unit = "N\u00b7m"
        else:
            unit = "N"

        ax.set_title(f"{target_names[i]}\nStd: {std_val:.2f} {unit}", fontsize=10)
        ax.set_xlabel(f"{target_names[i]} ({unit})", fontsize=8)
        ax.set_ylabel("Frequency", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"{title_prefix} Data Distribution", fontsize=12, fontweight="bold")
    plt.tight_layout()
    return fig

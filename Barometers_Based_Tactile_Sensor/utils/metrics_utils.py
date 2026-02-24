"""
Evaluation metrics utilities for tactile sensor force prediction models.

Functions
---------
calculate_grouped_rmse  -- Print/return grouped RMSE for contact location,
                           force vector, and optional torque vector.
evaluate_constrained_region -- Evaluate metrics inside a rectangular x-y region.
"""

import numpy as np
from sklearn.metrics import mean_absolute_error, r2_score


def calculate_grouped_rmse(y_true, y_pred, target_names, title_suffix="", return_metrics=False):
    """
    Calculate and print grouped RMSE for contact location, force vector, and torque.

    Parameters
    ----------
    y_true : np.ndarray, shape (N, n_targets)
    y_pred : np.ndarray, shape (N, n_targets)
    target_names : list of str
        Names matching the columns of y_true / y_pred.
    title_suffix : str, optional
        Extra label appended to the printed header.
    return_metrics : bool, optional
        If True, return a dict with the computed RMSE values.

    Returns
    -------
    dict or None
        Keys (present only when the relevant targets exist):
        'contact_location_rmse', 'contact_euclidean_rmse',
        'force_vector_rmse', 'force_euclidean_rmse',
        'torque_vector_rmse', 'torque_euclidean_rmse'.
    """
    print("\n" + "=" * 70 + f"\nGROUPED RMSE METRICS {title_suffix}\n" + "=" * 70)
    metrics = {}

    # ---- Contact location (x, y) ----
    contact_indices = [i for i, name in enumerate(target_names) if name in ["x", "y"]]
    if len(contact_indices) >= 2:
        t_true = y_true[:, contact_indices]
        t_pred = y_pred[:, contact_indices]
        err = t_true - t_pred
        rmse_comp = np.sqrt(np.mean(err ** 2))
        dist = np.sqrt(np.sum(err ** 2, axis=1))
        rmse_euc = np.sqrt(np.mean(dist ** 2))
        print(f"\nContact Location (x, y):")
        print(f"  - Component-wise RMSE: {rmse_comp:.4f} mm")
        print(f"  - Euclidean RMSE:      {rmse_euc:.4f} mm")
        print(f"  - Mean error distance: {np.mean(dist):.4f} mm")
        metrics["contact_location_rmse"] = rmse_comp
        metrics["contact_euclidean_rmse"] = rmse_euc

    # ---- Force vector (fx, fy, fz) ----
    force_indices = [i for i, name in enumerate(target_names) if name in ["fx", "fy", "fz"]]
    if force_indices:
        t_true = y_true[:, force_indices]
        t_pred = y_pred[:, force_indices]
        err = t_true - t_pred
        rmse_comp = np.sqrt(np.mean(err ** 2))
        mag = np.sqrt(np.sum(err ** 2, axis=1))
        rmse_euc = np.sqrt(np.mean(mag ** 2))
        force_names = [target_names[i] for i in force_indices]
        print(f"\n{len(force_indices)}-DOF Force Vector ({', '.join(force_names)}):")
        print(f"  - Component-wise RMSE: {rmse_comp:.4f} N")
        print(f"  - Euclidean RMSE:      {rmse_euc:.4f} N")
        print(f"  - Mean error magnitude: {np.mean(mag):.4f} N")
        metrics["force_vector_rmse"] = rmse_comp
        metrics["force_euclidean_rmse"] = rmse_euc

    # ---- Torque vector (tx, ty, tz) ----
    torque_indices = [i for i, name in enumerate(target_names) if name in ["tx", "ty", "tz"]]
    if torque_indices:
        t_true = y_true[:, torque_indices]
        t_pred = y_pred[:, torque_indices]
        err = t_true - t_pred
        rmse_comp = np.sqrt(np.mean(err ** 2))
        mag = np.sqrt(np.sum(err ** 2, axis=1))
        rmse_euc = np.sqrt(np.mean(mag ** 2))
        torque_names = [target_names[i] for i in torque_indices]
        print(f"\n{len(torque_indices)}-DOF Torque Vector ({', '.join(torque_names)}):")
        print(f"  - Component-wise RMSE: {rmse_comp * 1000:.4f} N\u00b7mm")
        print(f"  - Euclidean RMSE:      {rmse_euc * 1000:.4f} N\u00b7mm")
        print(f"  - Mean error magnitude: {np.mean(mag) * 1000:.4f} N\u00b7mm")
        metrics["torque_vector_rmse"] = rmse_comp
        metrics["torque_euclidean_rmse"] = rmse_euc

    if return_metrics:
        return metrics


def evaluate_constrained_region(y_test, y_pred, target_cols, x_range=10, y_range=8):
    """
    Evaluate predictions inside the rectangular region |x| <= x_range, |y| <= y_range.

    Parameters
    ----------
    y_test, y_pred : np.ndarray, shape (N, n_targets)
    target_cols : list of str
    x_range, y_range : float
        Half-width of the evaluation rectangle in mm.

    Returns
    -------
    mask : np.ndarray or None
        Boolean mask selecting samples inside the region, or None if empty.
    """
    try:
        x_idx = target_cols.index("x")
        y_idx = target_cols.index("y")
    except ValueError:
        print("Warning: 'x' or 'y' not found in target columns. Skipping constrained region analysis.")
        return None

    mask = (np.abs(y_test[:, x_idx]) <= x_range) & (np.abs(y_test[:, y_idx]) <= y_range)
    n_samples = np.sum(mask)

    if n_samples == 0:
        print(f"\nNo samples in constrained region (x: \u00b1{x_range}, y: \u00b1{y_range})")
        return None

    print("\n" + "=" * 70)
    print(f"CONSTRAINED REGION ANALYSIS (Rectangle: {2 * x_range} x {2 * y_range} mm)")
    print(f"  x: \u00b1{x_range} mm, y: \u00b1{y_range} mm")
    print("=" * 70)
    print(f"Samples in region: {n_samples}/{len(y_test)} ({100 * n_samples / len(y_test):.1f}%)\n")

    y_test_c = y_test[mask]
    y_pred_c = y_pred[mask]

    print("Individual Target Metrics (Constrained Region):")
    print("-" * 70)
    for i, col in enumerate(target_cols):
        mae = mean_absolute_error(y_test_c[:, i], y_pred_c[:, i])
        r2 = r2_score(y_test_c[:, i], y_pred_c[:, i])
        rmse = np.sqrt(np.mean((y_test_c[:, i] - y_pred_c[:, i]) ** 2))
        corr = np.corrcoef(y_test_c[:, i], y_pred_c[:, i])[0, 1]
        unit = "mm" if col in ["x", "y"] else "N"
        print(
            f"{col:>3} | MAE: {mae:6.3f} {unit} | RMSE: {rmse:6.3f} {unit} "
            f"| R\u00b2: {r2:6.3f} | Corr: {corr:6.3f}"
        )

    calculate_grouped_rmse(y_test_c, y_pred_c, target_cols, title_suffix="(Constrained Region)")
    return mask

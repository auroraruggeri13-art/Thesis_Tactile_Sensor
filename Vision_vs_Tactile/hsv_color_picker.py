"""
HSV Color Picker for Gripper Segmentation
==========================================
Opens two interactive windows:
  Tab 1 – "Include":  click up to 4 pixels that SHOULD be segmented.
  Tab 2 – "Exclude":  click up to 4 pixels that must NOT be included.

After closing both windows the script prints the computed HSV bounds
and patches FinRayProcessor so that process_all() uses them automatically.

Usage
-----
  from hsv_color_picker import pick_hsv_bounds, patch_processor

  green_lower, green_upper, exclude_lower, exclude_upper = pick_hsv_bounds(sample_image)
  patch_processor(processor, green_lower, green_upper, exclude_lower, exclude_upper)
  processor.process_all()
"""

import cv2
import numpy as np
import matplotlib
matplotlib.use("TkAgg")          # interactive backend (works on Windows / Linux / macOS)
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.widgets import Button
from dataclasses import dataclass, field
from typing import List, Optional, Tuple


# ── colour palette ──────────────────────────────────────────────────────────
_DARK   = "#0f0f14"
_PANEL  = "#1a1a24"
_ACCENT = "#00e5ff"
_WARN   = "#ff4f5e"
_TEXT   = "#e8e8f0"
_DIM    = "#5a5a7a"


@dataclass
class _PickState:
    points:   List[Tuple[int, int]] = field(default_factory=list)
    hsv_vals: List[np.ndarray]      = field(default_factory=list)
    max_pts:  int = 4


# ── helpers ──────────────────────────────────────────────────────────────────

def _bgr_to_display_rgb(bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _sample_hsv_patch(hsv_img: np.ndarray, x: int, y: int, r: int = 5) -> np.ndarray:
    """Mean HSV in a small patch around (x, y) — robust to single-pixel noise."""
    h, w = hsv_img.shape[:2]
    y0, y1 = max(0, y - r), min(h, y + r + 1)
    x0, x1 = max(0, x - r), min(w, x + r + 1)
    return hsv_img[y0:y1, x0:x1].reshape(-1, 3).mean(axis=0)


def _hsv_bounds_from_samples(
    samples: List[np.ndarray],
    margin_h: int = 15,
    margin_sv: int = 40,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Expand the bounding box of sampled HSV values by margin_h (hue) and
    margin_sv (sat / val), clamping to valid ranges.
    """
    arr = np.array(samples, dtype=np.float32)
    lo = arr.min(axis=0) - np.array([margin_h, margin_sv, margin_sv])
    hi = arr.max(axis=0) + np.array([margin_h, margin_sv, margin_sv])
    lo = np.clip(lo, [0, 0, 0],   [179, 255, 255])
    hi = np.clip(hi, [0, 0, 0],   [179, 255, 255])
    return lo.astype(np.uint8), hi.astype(np.uint8)


# ── single picker window ─────────────────────────────────────────────────────

def _run_picker(
    bgr_img:   np.ndarray,
    hsv_img:   np.ndarray,
    title:     str,
    dot_color: str,
    state:     _PickState,
) -> None:
    """
    Opens one matplotlib figure.  User clicks on the image; each click adds
    a point (up to state.max_pts).  Close the window or press "Done" to finish.
    """
    rgb = _bgr_to_display_rgb(bgr_img)

    fig = plt.figure(figsize=(12, 7), facecolor=_DARK)
    fig.canvas.manager.set_window_title(title)

    # ── layout: main image + right info panel ──
    ax_img  = fig.add_axes([0.01, 0.12, 0.64, 0.84])
    ax_info = fig.add_axes([0.67, 0.12, 0.31, 0.84])
    ax_btn  = fig.add_axes([0.40, 0.02, 0.20, 0.07])

    for ax in (ax_img, ax_info, ax_btn):
        ax.set_facecolor(_PANEL)
        for spine in ax.spines.values():
            spine.set_edgecolor(_DIM)

    ax_img.imshow(rgb)
    ax_img.set_title(title, color=_ACCENT, fontsize=13,
                     fontfamily="monospace", pad=8)
    ax_img.set_xlabel("Click up to 4 pixels  •  close or press Done when finished",
                      color=_DIM, fontsize=9)
    ax_img.tick_params(colors=_DIM, labelsize=7)

    ax_info.axis("off")
    ax_info.set_title("Sampled HSV", color=_ACCENT,
                      fontsize=11, fontfamily="monospace", pad=6)

    btn = Button(ax_btn, "✓  Done", color=_DIM, hovercolor=_ACCENT)
    btn.label.set_color(_TEXT)
    btn.label.set_fontfamily("monospace")

    scatter_artists = []
    info_texts      = []

    def _refresh_info():
        for t in info_texts:
            t.remove()
        info_texts.clear()
        if not state.hsv_vals:
            t = ax_info.text(
                0.5, 0.5, "No points yet",
                transform=ax_info.transAxes, ha="center", va="center",
                color=_DIM, fontsize=10, fontfamily="monospace",
            )
            info_texts.append(t)
            fig.canvas.draw_idle()
            return

        for idx, (hsv, (px, py)) in enumerate(
            zip(state.hsv_vals, state.points), start=1
        ):
            h, s, v = hsv
            # small colour swatch via a rectangle annotation
            swatch_color = np.array([[[h, s, v]]], dtype=np.uint8)
            swatch_rgb   = cv2.cvtColor(swatch_color, cv2.COLOR_HSV2RGB)[0, 0] / 255
            ypos = 0.90 - (idx - 1) * 0.22
            ax_info.add_patch(mpatches.FancyBboxPatch(
                (0.05, ypos - 0.06), 0.12, 0.12,
                boxstyle="round,pad=0.01",
                facecolor=swatch_rgb, edgecolor=_DIM, linewidth=1,
                transform=ax_info.transAxes,
            ))
            line = (
                f"#{idx}  ({px:4d},{py:4d})\n"
                f"    H={h:5.1f}  S={s:5.1f}  V={v:5.1f}"
            )
            t = ax_info.text(
                0.25, ypos,
                line,
                transform=ax_info.transAxes, ha="left", va="center",
                color=_TEXT, fontsize=9, fontfamily="monospace",
            )
            info_texts.append(t)

        # show computed range if ≥1 sample
        if state.hsv_vals:
            lo, hi = _hsv_bounds_from_samples(state.hsv_vals)
            rng_txt = (
                f"\nComputed range\n"
                f"  Lo: H{lo[0]:3d} S{lo[1]:3d} V{lo[2]:3d}\n"
                f"  Hi: H{hi[0]:3d} S{hi[1]:3d} V{hi[2]:3d}"
            )
            t = ax_info.text(
                0.05, 0.10, rng_txt,
                transform=ax_info.transAxes, ha="left", va="bottom",
                color=_ACCENT, fontsize=8.5, fontfamily="monospace",
            )
            info_texts.append(t)

        fig.canvas.draw_idle()

    def _on_click(event):
        if event.inaxes is not ax_img:
            return
        if len(state.points) >= state.max_pts:
            print(f"  [picker] Max {state.max_pts} points reached. Remove one or press Done.")
            return
        x, y = int(event.xdata), int(event.ydata)
        h_img, w_img = hsv_img.shape[:2]
        x = np.clip(x, 0, w_img - 1)
        y = np.clip(y, 0, h_img - 1)

        hsv_val = _sample_hsv_patch(hsv_img, x, y)
        state.points.append((x, y))
        state.hsv_vals.append(hsv_val)

        sc = ax_img.scatter(x, y, s=120, c=dot_color,
                            marker="+", linewidths=2.5, zorder=5)
        ax_img.annotate(
            f" #{len(state.points)}",
            (x, y), color=_TEXT, fontsize=8, fontfamily="monospace",
            xytext=(6, 6), textcoords="offset points",
        )
        scatter_artists.append(sc)
        print(f"  [picker] Point #{len(state.points)}: ({x},{y})  "
              f"HSV=({hsv_val[0]:.1f}, {hsv_val[1]:.1f}, {hsv_val[2]:.1f})")
        _refresh_info()

    def _on_done(_):
        plt.close(fig)

    btn.on_clicked(_on_done)
    fig.canvas.mpl_connect("button_press_event", _on_click)

    _refresh_info()
    plt.show(block=True)


# ── public API ────────────────────────────────────────────────────────────────

def pick_hsv_bounds(
    bgr_img:      np.ndarray,
    margin_h:     int = 15,
    margin_sv:    int = 40,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Open two interactive picker windows.

    Parameters
    ----------
    bgr_img   : BGR image (numpy array)
    margin_h  : hue margin to expand around sampled hue values
    margin_sv : sat/val margin to expand around sampled values

    Returns
    -------
    green_lower   : np.uint8 array [H, S, V] — lower include bound
    green_upper   : np.uint8 array [H, S, V] — upper include bound
    exclude_lower : np.uint8 array or None   — lower exclude bound
    exclude_upper : np.uint8 array or None   — upper exclude bound
    """
    # Precompute HSV once
    filtered = cv2.bilateralFilter(bgr_img, d=9, sigmaColor=75, sigmaSpace=75)
    hsv_img  = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

    # ── Tab 1: include ────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(" STEP 1 / 2  —  INCLUDE picker")
    print(" Click up to 4 pixels that SHOULD be segmented.")
    print(" Close the window or press ✓ Done when finished.")
    print("═"*60)
    inc_state = _PickState(max_pts=4)
    _run_picker(bgr_img, hsv_img,
                title="TAB 1 — Include pixels (to segment)",
                dot_color=_ACCENT,
                state=inc_state)

    if not inc_state.hsv_vals:
        raise ValueError("No include points selected — cannot build HSV range.")

    green_lower, green_upper = _hsv_bounds_from_samples(
        inc_state.hsv_vals, margin_h=margin_h, margin_sv=margin_sv
    )
    print(f"\n  ✔ Include  lower = {green_lower}")
    print(f"  ✔ Include  upper = {green_upper}")

    # ── Tab 2: exclude ────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    print(" STEP 2 / 2  —  EXCLUDE picker")
    print(" Click up to 4 pixels that must NOT be segmented.")
    print(" (Leave empty and press Done if nothing to exclude.)")
    print("═"*60)
    exc_state = _PickState(max_pts=4)
    _run_picker(bgr_img, hsv_img,
                title="TAB 2 — Exclude pixels (to reject)",
                dot_color=_WARN,
                state=exc_state)

    exclude_lower = exclude_upper = None
    if exc_state.hsv_vals:
        exclude_lower, exclude_upper = _hsv_bounds_from_samples(
            exc_state.hsv_vals, margin_h=margin_h, margin_sv=margin_sv
        )
        print(f"\n  ✔ Exclude lower = {exclude_lower}")
        print(f"  ✔ Exclude upper = {exclude_upper}")
    else:
        print("\n  (no exclude region defined)")

    return green_lower, green_upper, exclude_lower, exclude_upper


# ── processor patch ──────────────────────────────────────────────────────────

def patch_processor(processor, purple_lower, purple_upper,
                    exclude_lower=None, exclude_upper=None):
    """
    Monkey-patch a FinRayProcessor instance to use the interactively
    selected HSV bounds for every subsequent call to segment_gripper().
    """
    processor.purple_lower   = purple_lower
    processor.purple_upper   = purple_upper
    processor.exclude_lower = exclude_lower
    processor.exclude_upper = exclude_upper

    import types

    def _segment_with_hsv_patched(self, image):
        """Simple HSV segmentation using interactively picked bounds."""
        image_area = image.shape[0] * image.shape[1]

        filtered = cv2.bilateralFilter(image, d=9, sigmaColor=75, sigmaSpace=75)
        hsv = cv2.cvtColor(filtered, cv2.COLOR_BGR2HSV)

        mask = cv2.inRange(hsv, self.purple_lower, self.purple_upper)

        # Close: fills small noise gaps within each rib (~10-20 px) without
        # merging separate arms or overriding larger structural rib gaps
        close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel, iterations=2)
        # Open: removes isolated speckles smaller than the kernel
        open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, open_kernel, iterations=1)

        if self.exclude_lower is not None and self.exclude_upper is not None:
            exc = cv2.inRange(hsv, self.exclude_lower, self.exclude_upper)
            exc = cv2.dilate(exc, np.ones((5, 5), np.uint8), iterations=1)
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(exc))

        # Drop blobs smaller than 0.3 % of the image
        num_labels, lbl_map, stats, _ = cv2.connectedComponentsWithStats(
            mask, connectivity=8)
        result = np.zeros_like(mask)
        for lbl in range(1, num_labels):
            if stats[lbl, cv2.CC_STAT_AREA] >= image_area * 0.003:
                result[lbl_map == lbl] = 255
        return result

    # Bind the patched method to the instance
    processor._segment_with_hsv = types.MethodType(_segment_with_hsv_patched, processor)

    print("\n  [patch_processor] FinRayProcessor updated with picked HSV bounds.")
    print(f"    Include: {purple_lower} → {purple_upper}")
    if exclude_lower is not None:
        print(f"    Exclude: {exclude_lower} → {exclude_upper}")

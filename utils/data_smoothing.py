import numpy as np
import matplotlib.pyplot as plt

import os
from pathlib import Path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.resolve()


directory_demos = os.path.join(ROOT_DIR, f"demonstrations/pick_right/")
demo = 'recorded_9'
save_directory = os.path.join(ROOT_DIR, f"demonstrations/pick_right_processed/")

def smooth_and_crop_data(demo, change_threshold_ratio=0.05, margin = 1, show_plots=False, verbose=False):
    # -----------------------------
    # 2. Helper functions
    # -----------------------------
    def find_active_segment(signal,
                            min_active_dims=3,
                            change_threshold_ratio=0.05,
                            margin=5):
        """
        Find [start_idx, end_idx] where the signal is 'active', based on how many
        dimensions are changing.

        signal: array of shape (T, D)
        min_active_dims: how many dimensions must be 'active' to consider that
                        the signal has started/stopped changing (x = 3 here).
        change_threshold_ratio: per-dimension threshold as a fraction of that
                                dimension's max absolute change.
        margin: extra samples kept before first change and after last change.
        """
        T, D = signal.shape

        # Differences over time: shape (T - 1, D)
        diff = np.diff(signal, axis=0)
        abs_diff = np.abs(diff)

        # Per-dimension max absolute change
        max_abs_diff = abs_diff.max(axis=0)  # shape (D,)

        # Dimensions that never change (max_abs_diff == 0) will never be "active"
        thresholds = change_threshold_ratio * max_abs_diff
        # Avoid NaNs when max_abs_diff == 0
        thresholds[max_abs_diff == 0] = np.inf

        # Active in each dimension (T - 1, D)
        active_dims = abs_diff > thresholds

        # Count how many dimensions are active at each time step (T - 1,)
        active_count = active_dims.sum(axis=1)

        # A time step is active if at least min_active_dims are moving
        active = active_count >= min_active_dims

        # If nothing is active, keep the whole signal
        if not np.any(active):
            return 0, T - 1

        # First and last active steps (indices in [0, T-2])
        first_step = np.argmax(active)
        last_step = len(active) - 1 - np.argmax(active[::-1])

        # Map diff indices to sample indices:
        # change at diff[k] occurs between samples k and k+1
        start_idx = max(first_step - margin, 0)
        end_idx = min(last_step + 1 + margin, T - 1)

        return start_idx, end_idx

    # def smooth_signal(signal, window_size=11):
    def smooth_signal(signal, window_size=41):
        """
        Moving-average smoothing along time without introducing artificial
        jumps at the edges.

        signal: array of shape (T, D)
        window_size: length of moving window (odd is recommended).
        """
        if window_size < 2:
            return signal.copy()

        T, D = signal.shape
        pad = window_size // 2
        kernel = np.ones(window_size) / window_size

        smoothed = np.empty_like(signal)

        # Pad in time with edge values, then do 'valid' convolution
        for d in range(D):
            # shape becomes (T + 2*pad,)
            padded = np.pad(signal[:, d], pad_width=pad, mode='edge')
            # 'valid' will now give back exactly T samples
            smoothed[:, d] = np.convolve(padded, kernel, mode='valid')

        return smoothed



    # -----------------------------
    # 3. Crop the signal
    # -----------------------------
    start_idx, end_idx = find_active_segment(
        demo,
        min_active_dims=3,        # x = 3 dimensions must be active
        change_threshold_ratio=change_threshold_ratio,
        margin=margin
    )
    if verbose:
        print("Cropping from index", start_idx, "to", end_idx)

    cropped = demo[start_idx:end_idx + 1]
    smoothed = smooth_signal(cropped, window_size=11)

    if not show_plots:
        return smoothed
    else:
        # -----------------------------
        # 4. Visualizations
        # -----------------------------
        T_full = demo.shape[0]
        T_cropped = cropped.shape[0]
        time_full = np.arange(T_full)
        time_cropped = np.arange(T_cropped)

        # --- 4.1 Original signal with crop points ---
        n_dims = demo.shape[1]
        n_cols = 4
        n_rows = int(np.ceil(n_dims / n_cols))
        fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True)
        axes1 = axes1.ravel()
        for dim in range(demo.shape[1]):
            ax = axes1[dim]
            ax.plot(time_full, demo[:, dim])
            ax.axvline(start_idx, linestyle='--')  # crop start
            ax.axvline(end_idx, linestyle='--')    # crop end
            ax.set_title(f'Dim {dim}', fontsize=8)
        fig1.suptitle('Original signal with crop start/end', fontsize=14)
        plt.tight_layout()
        plt.show()

        # --- 4.2 Cropped vs smoothed ---
        n_cols = 4
        n_rows = int(np.ceil(n_dims / n_cols))
        fig2, axes2 = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4), sharex=True)
        axes2 = axes2.ravel()
        for dim in range(demo.shape[1]):
            ax = axes2[dim]
            ax.plot(time_cropped, cropped[:, dim], label='cropped', alpha=0.8)
            ax.plot(time_cropped, smoothed[:, dim], label='smoothed', alpha=0.8)
            ax.set_title(f'Dim {dim}', fontsize=8)
            if dim == 0:
                ax.legend(fontsize=8)
        fig2.suptitle('Cropped vs smoothed signal', fontsize=14)
        plt.tight_layout()
        plt.show()

        return smoothed

    # # -----------------------------
    # # 5. Save processed data
    # # -----------------------------
    # np.savez_compressed(
    #     save_directory + demo,
    #     data=smoothed_q,
    #     crop_start=start_idx,
    #     crop_end=end_idx
    # )
"""
Preprocess and glue separate left-arm / right-arm RBY1 letter demonstrations
into a paired multi-demo dataset in the standard pipeline .npz format.

Raw demo format (each .npz file)
─────────────────────────────────
  key: 'data'  shape: (T, 24)
  recorded_l_*.npz  →  left  arm moved  (cols 15-21 are active)
  recorded_r_*.npz  →  right arm moved  (cols  8-14 are active)

Pairing strategy
────────────────
  Files are sorted by name and paired by index:
    (recorded_l_1.npz, recorded_r_1.npz), (l_2, r_2), …
  so N_DEMOS = min(#left_files, #right_files).

  For each pair:
    1. Trim only the leading stationary frames (trailing kept for stop-learning).
    2. Smooth with a moving-average filter.
    3. Resample the shorter arm to the longer one's length (within the pair).
    4. Append N_HOLD_STEPS copies of the final frame.

  After all pairs are processed the demos are right-padded to the same
  length (the longest demo's T) by repeating their own final frame, so the
  array is rectangular.

Output .npz keys
────────────────
  q_data  : (N, T, 14)   joint positions  [q_left | q_right]
  x_data  : (N, T, nb_x) task-space       [x_left | x_right]
  dq_data : (N, T, 14)   joint velocities
  dt, nb_q, nb_x, nb_q_left, nb_q_right, nb_x_left, nb_x_right

Run from the repository root:
    python examples/rby1/make_rby1_letters_dataset.py
"""

import os
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import interp1d
import torch

# ── repo root on sys.path ─────────────────────────────────────────────────────
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR    = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytorch_kinematics as pk  # noqa: E402

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here
# ─────────────────────────────────────────────────────────────────────────────

RAW_DIR     = ROOT_DIR / "demonstrations" / "rby1_letters_raw"
OUTPUT_DIR  = ROOT_DIR / "demonstrations" / "rby1_letters"
DATASET_NAME = "left-rby1-letters_right-rby1-letters_ndofs-14"

RECORDING_HZ = 200.0   # Hz — used to compute dq and stored in output

# ── Trimming ─────────────────────────────────────────────────────────────────
# A dimension is "active" when its per-step change exceeds this fraction of
# its own maximum change across the whole trajectory.
TRIM_CHANGE_THRESHOLD = 0.05
# At least this many joints must move simultaneously to mark the start.
TRIM_MIN_ACTIVE_DIMS  = 3
# Extra timesteps kept before the first detected motion.
TRIM_MARGIN           = 10   # 50 ms at 200 Hz

# ── Smoothing ─────────────────────────────────────────────────────────────────
SMOOTH_WINDOW = 71   # must be odd; 0 or 1 = no smoothing

# ── Hold at end ───────────────────────────────────────────────────────────────
# Append this many copies of the final frame so the network learns to stop.
# Rule of thumb: should be at least as large as nb_steps * stride in training
# so that the sampler can draw windows that lie entirely in the hold region.
N_HOLD_STEPS = 150

# ── Task-space format ─────────────────────────────────────────────────────────
# True  → x = [x, y]         per arm  (nb_x_per_arm = 2)
# False → x = [x,y,z, R(9)]  per arm  (nb_x_per_arm = 12)
IS_XY_TASKSPACE = True

# ─────────────────────────────────────────────────────────────────────────────
# Column indices in the raw 24-column data
# ─────────────────────────────────────────────────────────────────────────────
COL_RIGHT_ARM = slice(8, 15)    # right_arm_0 .. right_arm_6
COL_LEFT_ARM  = slice(15, 22)   # left_arm_0  .. left_arm_6

# ─────────────────────────────────────────────────────────────────────────────
# FK chains
# ─────────────────────────────────────────────────────────────────────────────
_urdf_data  = open(ROOT_DIR / "urdf" / "rby1a" / "model.urdf").read()
chain_right = pk.build_serial_chain_from_urdf(
    _urdf_data, end_link_name="ee_right", root_link_name="link_torso_5"
).to(device=device)
chain_left  = pk.build_serial_chain_from_urdf(
    _urdf_data, end_link_name="ee_left",  root_link_name="link_torso_5"
).to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing helpers (mirrors make_rby1_dataset.py)
# ─────────────────────────────────────────────────────────────────────────────

def _find_start(signal: np.ndarray,
                change_threshold_ratio: float,
                min_active_dims: int,
                margin: int) -> int:
    """
    Return the index of the first 'active' timestep, minus margin.

    Only the leading dead region is trimmed; the trailing end is left intact.
    """
    diff  = np.abs(np.diff(signal, axis=0))          # (T-1, D)
    maxd  = diff.max(axis=0)                          # (D,)
    thr   = change_threshold_ratio * maxd
    thr[maxd == 0] = np.inf

    active = (diff > thr).sum(axis=1) >= min_active_dims  # (T-1,)
    if not np.any(active):
        return 0
    first = int(np.argmax(active))
    return max(first - margin, 0)


def _moving_average(signal: np.ndarray, window: int) -> np.ndarray:
    """Edge-padded moving-average smooth along the time axis."""
    if window < 2:
        return signal.copy()
    T, D  = signal.shape
    pad   = window // 2
    kern  = np.ones(window) / window
    out   = np.empty_like(signal)
    for d in range(D):
        padded    = np.pad(signal[:, d], pad_width=pad, mode='edge')
        out[:, d] = np.convolve(padded, kern, mode='valid')
    return out


def _resample(signal: np.ndarray, target_len: int) -> np.ndarray:
    """
    Resample a (T, D) array to (target_len, D) using linear interpolation.
    """
    T, D = signal.shape
    if T == target_len:
        return signal.copy()
    t_old = np.linspace(0.0, 1.0, T)
    t_new = np.linspace(0.0, 1.0, target_len)
    f     = interp1d(t_old, signal, axis=0, kind='linear')
    return f(t_new)


def compute_task_space(q_arm: np.ndarray, chain) -> np.ndarray:
    """
    FK for one arm.

    IS_XY_TASKSPACE=True  → (T, 2)  Y–Z position only.
        The robot writes letters in the Y–Z plane of the link_torso_5 frame:
        X barely changes (~5 mm) while Y and Z each span ~100 mm.
    IS_XY_TASKSPACE=False → (T, 12) [pos(3) + rot_matrix_row_major(9)]
    """
    qt = torch.tensor(q_arm, dtype=torch.float32, device=device)
    with torch.no_grad():
        m = chain.forward_kinematics(qt).get_matrix()   # (T, 4, 4)
    pos = m[:, :3, 3].cpu().numpy()                     # (T, 3)
    if IS_XY_TASKSPACE:
        return pos[:, 1:3]                              # (T, 2)  — y, z
    rot = m[:, :3, :3].cpu().numpy().reshape(-1, 9)     # (T, 9)
    return np.concatenate([pos, rot], axis=1)           # (T, 12)


# ─────────────────────────────────────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────────────────────────────────────

def load_and_preprocess_arm(path: Path, arm: str) -> tuple[np.ndarray, int]:
    """
    Load a single arm recording, trim the leading dead region, and smooth.

    Returns
    -------
    q_clean : (T_clean, 7)  smoothed joint positions of the active arm
    T_clean : int
    """
    raw   = np.load(path)["data"]                      # (T, 24)
    col   = COL_LEFT_ARM if arm == "left" else COL_RIGHT_ARM
    q     = raw[:, col]                                # (T, 7)

    start = _find_start(q,
                        change_threshold_ratio=TRIM_CHANGE_THRESHOLD,
                        min_active_dims=TRIM_MIN_ACTIVE_DIMS,
                        margin=TRIM_MARGIN)
    q_trim  = q[start:]                                # trim only the front
    q_clean = _moving_average(q_trim, SMOOTH_WINDOW)

    return q_clean, q_clean.shape[0]


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    left_files  = sorted(RAW_DIR.glob("recorded_l_*.npz"))
    right_files = sorted(RAW_DIR.glob("recorded_r_*.npz"))

    if not left_files:
        raise FileNotFoundError(f"No recorded_l_*.npz files found in {RAW_DIR}")
    if not right_files:
        raise FileNotFoundError(f"No recorded_r_*.npz files found in {RAW_DIR}")

    n_pairs = min(len(left_files), len(right_files))
    print(f"Found {len(left_files)} left-arm and {len(right_files)} right-arm recordings.")
    print(f"Pairing by index → {n_pairs} demo(s).")

    # ── Step 1: load, trim and smooth each recording ──────────────────────────
    left_demos  = []   # list of (q_clean, T_active, path)
    right_demos = []

    print("\nLeft arm recordings:")
    for p in left_files[:n_pairs]:
        q, T = load_and_preprocess_arm(p, arm="left")
        left_demos.append((q, T, p))
        print(f"  {p.name}: T_active = {T}  ({T / RECORDING_HZ:.2f} s)")

    print("\nRight arm recordings:")
    for p in right_files[:n_pairs]:
        q, T = load_and_preprocess_arm(p, arm="right")
        right_demos.append((q, T, p))
        print(f"  {p.name}: T_active = {T}  ({T / RECORDING_HZ:.2f} s)")

    # ── Step 2: process each same-index pair ─────────────────────────────────
    # For each pair: resample to common length, append hold frames, glue arms.
    print("\nPair synchronisation:")
    print(f"  {'pair':<6} {'T_left':>8} {'T_right':>8} {'diff_steps':>12} {'diff_ms':>10}  {'T_after_resample':>18}")
    print("  " + "-" * 70)

    q_demos  = []   # list of (T_i, 14) arrays
    x_demos  = []   # list of (T_i, nb_x) arrays
    dq_demos = []   # list of (T_i, 14) arrays
    nb_x_per_arm = None

    for k, ((q_l_raw, Tl, path_l), (q_r_raw, Tr, path_r)) in \
            enumerate(zip(left_demos, right_demos), start=1):

        diff     = abs(Tl - Tr)
        T_target = max(Tl, Tr)
        print(f"  {k:<6} {Tl:>8}   {Tr:>8}   {diff:>10}   {diff/RECORDING_HZ*1e3:>8.0f} ms"
              f"   → {T_target} steps ({T_target/RECORDING_HZ:.2f} s)")

        # Resample the shorter arm to match the longer
        q_left  = _resample(q_l_raw, T_target)   # (T_target, 7)
        q_right = _resample(q_r_raw, T_target)   # (T_target, 7)

        # Append hold-at-end frames
        if N_HOLD_STEPS > 0:
            q_left  = np.concatenate([q_left,  np.tile(q_left[-1:],  (N_HOLD_STEPS, 1))], axis=0)
            q_right = np.concatenate([q_right, np.tile(q_right[-1:], (N_HOLD_STEPS, 1))], axis=0)

        # Glue arms
        q = np.concatenate([q_left, q_right], axis=1)   # (T, 14)

        # FK → task-space
        x_left_arr  = compute_task_space(q_left,  chain_left)
        x_right_arr = compute_task_space(q_right, chain_right)
        x = np.concatenate([x_left_arr, x_right_arr], axis=1)

        if nb_x_per_arm is None:
            nb_x_per_arm = x_left_arr.shape[1]

        # Joint velocities
        dq = np.zeros_like(q)
        dq[:-1] = (q[1:] - q[:-1]) * RECORDING_HZ

        q_demos.append(q)
        x_demos.append(x)
        dq_demos.append(dq)

    # ── Step 3: pad all demos to the same length ──────────────────────────────
    # Different pairs may have different T (because left/right active lengths
    # vary).  Right-pad shorter demos by repeating their final frame.
    T_max = max(q.shape[0] for q in q_demos)
    print(f"\nPadding all demos to T_max = {T_max} steps  ({T_max/RECORDING_HZ:.2f} s)")

    def _pad_to(arr, T_max):
        T = arr.shape[0]
        if T == T_max:
            return arr
        pad = np.tile(arr[-1:], (T_max - T,) + (1,) * (arr.ndim - 1))
        return np.concatenate([arr, pad], axis=0)

    q_arr  = np.stack([_pad_to(q,  T_max) for q  in q_demos],  axis=0)   # (N, T, 14)
    x_arr  = np.stack([_pad_to(x,  T_max) for x  in x_demos],  axis=0).astype(np.float32)
    dq_arr = np.stack([_pad_to(dq, T_max) for dq in dq_demos], axis=0)   # (N, T, 14)

    nb_x     = nb_x_per_arm * 2
    nb_demos = q_arr.shape[0]

    print(f"\nDataset summary:")
    print(f"  nb_demos : {nb_demos}")
    print(f"  T        : {T_max}  timesteps  ({T_max/RECORDING_HZ:.2f} s)")
    print(f"  nb_q     : 14  (7 left + 7 right)")
    print(f"  nb_x     : {nb_x}  ({nb_x_per_arm} per arm)")
    print(f"  dt       : {1.0/RECORDING_HZ:.4f} s  ({RECORDING_HZ:.0f} Hz)")

    # ── Step 4: save ──────────────────────────────────────────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{DATASET_NAME}.npz"

    np.savez(
        output_path,
        q_data   = q_arr,
        x_data   = x_arr,
        dq_data  = dq_arr,
        dt       = np.float64(1.0 / RECORDING_HZ),
        nb_q     = np.int64(14),
        nb_x     = np.int64(nb_x),
        nb_q_left  = np.int64(7),
        nb_q_right = np.int64(7),
        nb_x_left  = np.int64(nb_x_per_arm),
        nb_x_right = np.int64(nb_x_per_arm),
    )

    print(f"\nSaved → {output_path}")
    print(f"  q_data  shape: {q_arr.shape}")
    print(f"  x_data  shape: {x_arr.shape}")
    print(f"  dq_data shape: {dq_arr.shape}")


if __name__ == "__main__":
    main()

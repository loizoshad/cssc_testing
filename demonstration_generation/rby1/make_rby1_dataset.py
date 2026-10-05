"""
Convert real RBY1 robot demonstrations into the pipeline .npz format.

Raw demo format (each .npz file)
─────────────────────────────────
  key: 'data'  shape: (T, 24)  dtype: float64

  The 24 columns are the joint positions of all controllable joints, ordered as
  the RBY1 SDK reports them:

    cols  0-1  : right_wheel, left_wheel          (wheels, ~stationary)
    cols  2-7  : torso_0 .. torso_5               (torso, ~stationary)
    cols  8-14 : right_arm_0 .. right_arm_6       (7-DoF right arm)
    cols 15-21 : left_arm_0  .. left_arm_6        (7-DoF left arm)
    cols 22-23 : head_0, head_1                   (head, ~stationary)

Output .npz format (consumed by DataVisualizer.construct_q_x_dq_real_data)
───────────────────────────────────────────────────────────────────────────
  q_data : (nb_demos, T_min, 14)    joint positions  [q_left | q_right]
  x_data : (nb_demos, T_min, nb_x)  task-space       [x_left | x_right]
  dq_data: (nb_demos, T_min, 14)    joint velocities (finite differences)

  nb_q = 14  (7 left + 7 right)

  Task-space per arm is determined by TASK_SPACE:
    'xyz_theta' → [x, y, z, yaw]           nb_x = 8  (4 per arm, default for pan)
    'full'      → [pos(3) + rot_mat(9)]    nb_x = 24 (12 per arm)

  The left arm comes first in both q and x to match the pipeline naming
  convention  left-..._right-...

Run from the repository root:
    python examples/rby1/make_rby1_dataset.py
"""

import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps"  if torch.backends.mps.is_available() else
    "cpu"
)
warnings.filterwarnings("ignore")

# ── repo root on sys.path ─────────────────────────────────────────────────────
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR    = CURRENT_DIR.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytorch_kinematics as pk  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here
# ─────────────────────────────────────────────────────────────────────────────
# Task-space representation stored in x_data.
# Must match the task_space used in create_rby1_robot() for training/augmentation.
#   'xyz_theta' → [x, y, z, yaw]           4 values per arm (pan experiment default)
#   'full'      → [pos(3) + rot_mat(9)]    12 values per arm
TASK_SPACE = 'xyz_theta'

clean_data_flag = True

# Demo sources: list of (folder_path, glob_pattern) tuples.
SOURCES = [
    (ROOT_DIR / "demonstrations" / "rby1_pan_v1_raw", "recorded_3.npz"),
    (ROOT_DIR / "demonstrations" / "rby1_pan_v1_raw", "recorded_4.npz"),
]

# Output
OUTPUT_DIR   = ROOT_DIR / "demonstrations" / "rby1_pan_v1"
DATASET_NAME = "left-rby1-all_right-rby1-all_ndofs-14"

# Recording frequency of the RBY1 robot (Hz).  Used for stored dq = Δq/dt.
RECORDING_HZ = 200.0

# ─────────────────────────────────────────────────────────────────────────────
# CLEANING — trim stationary start/end and smooth
# ─────────────────────────────────────────────────────────────────────────────
TRIM_CHANGE_THRESHOLD = 0.05   # fraction of max-change per dim to classify as active
TRIM_MIN_ACTIVE_DIMS  = 3      # min dims active simultaneously to be "in motion"
TRIM_MARGIN           = 10     # extra steps kept around the active region (50 ms @200 Hz)
SMOOTH_WINDOW         = 71     # moving-average kernel length (must be odd; 0/1 = no smooth)
N_HOLD_STEPS          = 40     # extra copies of the last pose appended at the end

# ─────────────────────────────────────────────────────────────────────────────
# ORIENTATION NUDGE — optional step to tilt EE approach axis toward downward
#
# During pan grasping the gripper should face straight down, but demonstrations
# are often slightly off.  When NUDGE_EE_DOWN=True a null-space Jacobian step
# is applied at every timestep to partially correct the orientation while
# keeping the EE position unchanged.
#
# Tuning guide:
#   NUDGE_APPROACH_COL  — which column of the EE rotation matrix R is the
#                         "approach" (gripper-opening) direction.  Typically 2
#                         (z-axis); print the EE R matrix on a known pose to
#                         check.
#   NUDGE_TARGET_DIR    — desired approach direction in the link_torso_5 frame.
#                         [0, 0, -1] = straight down along the torso -z axis.
#   NUDGE_ALPHA         — max fraction of the current orientation error to
#                         correct.  0 = no change, 1 = full correction.
#                         Start with 0.2–0.4; too large causes large joint jumps.
#   NUDGE_RAMP          — if True, linearly ramp correction from 0 at trajectory
#                         start to NUDGE_ALPHA at the end.  Keeps the approach
#                         phase natural and corrects mainly the grasp pose.
#   NUDGE_LAMBDA        — damping coefficient for the pseudoinverse.  Larger =
#                         more stable but less responsive (try 0.01–0.1).
# ─────────────────────────────────────────────────────────────────────────────
NUDGE_EE_DOWN      = False
NUDGE_APPROACH_COL = 2
NUDGE_TARGET_DIR   = np.array([0., 0., -1.])
NUDGE_ALPHA        = 0.3
NUDGE_RAMP         = True
NUDGE_LAMBDA       = 0.05

# ─────────────────────────────────────────────────────────────────────────────
# Column indices in the raw 24-column data
# ─────────────────────────────────────────────────────────────────────────────
COL_RIGHT_ARM = slice(8, 15)   # right_arm_0 .. right_arm_6
COL_LEFT_ARM  = slice(15, 22)  # left_arm_0  .. left_arm_6

# ─────────────────────────────────────────────────────────────────────────────
# Build FK chains  (root = link_torso_5, end = ee_left / ee_right)
# ─────────────────────────────────────────────────────────────────────────────
_urdf_data  = open(ROOT_DIR / "urdf" / "rby1a" / "model.urdf").read()
chain_right = pk.build_serial_chain_from_urdf(
    _urdf_data, end_link_name="ee_right", root_link_name="link_torso_5"
).to(device=device)
chain_left  = pk.build_serial_chain_from_urdf(
    _urdf_data, end_link_name="ee_left",  root_link_name="link_torso_5"
).to(device=device)


# ─────────────────────────────────────────────────────────────────────────────
# Cleaning helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_active_crop(signal: np.ndarray,
                      change_threshold_ratio: float,
                      min_active_dims: int,
                      margin: int):
    """Return (start_idx, end_idx) bracketing the active region of `signal`."""
    T, D   = signal.shape
    diff   = np.abs(np.diff(signal, axis=0))
    maxd   = diff.max(axis=0)
    thr    = change_threshold_ratio * maxd
    thr[maxd == 0] = np.inf
    active = (diff > thr).sum(axis=1) >= min_active_dims
    if not np.any(active):
        return 0, T - 1
    first = int(np.argmax(active))
    last  = int(len(active) - 1 - np.argmax(active[::-1]))
    return max(first - margin, 0), min(last + 1 + margin, T - 1)


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


def clean_demo(q: np.ndarray) -> np.ndarray:
    """Trim stationary leading/trailing frames and smooth joint positions."""
    start, end = _find_active_crop(
        q,
        change_threshold_ratio=TRIM_CHANGE_THRESHOLD,
        min_active_dims=TRIM_MIN_ACTIVE_DIMS,
        margin=TRIM_MARGIN,
    )
    return _moving_average(q[start : end + 1], SMOOTH_WINDOW)


# ─────────────────────────────────────────────────────────────────────────────
# Task-space FK helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fk_matrix(q_arm: np.ndarray, chain) -> np.ndarray:
    """q_arm: (T, 7) → homogeneous transforms (T, 4, 4)."""
    q_t = torch.tensor(q_arm, dtype=torch.float32, device=device)
    with torch.no_grad():
        m = chain.forward_kinematics(q_t).get_matrix()   # (T, 4, 4)
    return m.cpu().numpy()


def compute_task_space(q_arm: np.ndarray, chain) -> np.ndarray:
    """Full task-space: [pos(3) + rot_matrix_row_major(9)] — shape (T, 12)."""
    m   = _fk_matrix(q_arm, chain)
    pos = m[:, :3, 3]                          # (T, 3)
    rot = m[:, :3, :3].reshape(-1, 9)          # (T, 9)
    return np.concatenate([pos, rot], axis=1)  # (T, 12)


def compute_task_space_xyz_theta(q_arm: np.ndarray, chain) -> np.ndarray:
    """xyz_theta task-space: [x, y, z, yaw] — shape (T, 4).

    yaw = atan2(R[1,0], R[0,0]) — rotation about the torso z-axis,
    consistent with the 'xyz_theta' task_space in RBY1SingleArm.
    """
    m   = _fk_matrix(q_arm, chain)
    pos = m[:, :3, 3]                                           # (T, 3)
    yaw = np.arctan2(m[:, 1, 0], m[:, 0, 0])[:, None]          # (T, 1)
    return np.concatenate([pos, yaw], axis=1)                   # (T, 4)


def compute_x(q_arm: np.ndarray, chain) -> np.ndarray:
    """Compute task-space according to the global TASK_SPACE setting."""
    if TASK_SPACE == 'xyz_theta':
        return compute_task_space_xyz_theta(q_arm, chain)
    else:  # 'full'
        return compute_task_space(q_arm, chain)


# ─────────────────────────────────────────────────────────────────────────────
# Orientation nudge helper
# ─────────────────────────────────────────────────────────────────────────────

def nudge_ee_toward_down(q_arm: np.ndarray, chain) -> np.ndarray:
    """Nudge joint trajectory so the EE approach axis points more toward
    NUDGE_TARGET_DIR, while keeping EE position (nearly) unchanged.

    Algorithm (per timestep t):
      1. FK  →  rotation matrix R
      2. approach vector  v = R[:, NUDGE_APPROACH_COL]
      3. orientation error  e = v × v_target   (zero when aligned)
      4. null-space projection of J_pos  →  N = I − J_pos⁺ J_pos
      5. least-squares correction in null space:
             dq = N Jₐᵀ (Jₐ N Jₐᵀ + λ²I)⁻¹ (alpha_t · e)
      6. q[t] += dq

    The null-space projection ensures the EE position barely changes
    (to first order), so only the orientation is corrected.
    """
    T, n_dof = q_arm.shape
    q_out    = q_arm.copy()
    v_target = NUDGE_TARGET_DIR / np.linalg.norm(NUDGE_TARGET_DIR)

    for t in range(T):
        alpha_t = NUDGE_ALPHA * (t / max(T - 1, 1)) if NUDGE_RAMP else NUDGE_ALPHA
        if alpha_t < 1e-6:
            continue

        q_t = torch.tensor(q_out[t:t+1], dtype=torch.float32, device=device)

        with torch.no_grad():
            m = chain.forward_kinematics(q_t).get_matrix()   # (1, 4, 4)
        R = m[0, :3, :3].cpu().numpy()                        # (3, 3)

        v_curr  = R[:, NUDGE_APPROACH_COL]           # (3,)
        e       = np.cross(v_curr, v_target)          # (3,) — ∝ sin(θ)
        if np.linalg.norm(e) < 1e-4:
            continue  # already aligned — nothing to do

        e_scaled = alpha_t * e                        # (3,) scaled error

        # Geometric Jacobian: rows 0-2 linear, rows 3-5 angular
        J     = chain.jacobian(q_t)[0].cpu().numpy()  # (6, n_dof)
        J_pos = J[:3, :]                               # (3, n_dof)
        J_ang = J[3:, :]                               # (3, n_dof)

        # Null-space projector of the position Jacobian (damped)
        W_pos      = J_pos @ J_pos.T + NUDGE_LAMBDA**2 * np.eye(3)
        J_pos_pinv = J_pos.T @ np.linalg.inv(W_pos)            # (n_dof, 3)
        N          = np.eye(n_dof) - J_pos_pinv @ J_pos        # (n_dof, n_dof)

        # Damped pseudoinverse of (J_ang @ N) projected into null space
        JaN  = J_ang @ N                                        # (3, n_dof)
        W_a  = JaN @ JaN.T + NUDGE_LAMBDA**2 * np.eye(3)
        dq   = N @ J_ang.T @ np.linalg.solve(W_a, e_scaled)    # (n_dof,)

        q_out[t] = q_out[t] + dq

    return q_out


# ─────────────────────────────────────────────────────────────────────────────
# Load, process, and store demonstrations
# ─────────────────────────────────────────────────────────────────────────────
print("Loading demonstrations...")

_NB_X_PER_ARM = {'xyz_theta': 4, 'full': 12}
nb_x_per_arm  = _NB_X_PER_ARM.get(TASK_SPACE, 12)

q_demos  = []   # list of (T_i, 14) arrays  [q_left | q_right]
x_demos  = []   # list of (T_i, nb_x) arrays [x_left | x_right]
dq_demos = []   # list of (T_i, 14) arrays

for folder, pattern in SOURCES:
    files = sorted(folder.glob(pattern))
    if not files:
        print(f"  WARNING: no files matched {folder}/{pattern}")
    for fpath in files:
        raw     = np.load(fpath)["data"]          # (T, 24)
        q_right = raw[:, COL_RIGHT_ARM]           # (T, 7)
        q_left  = raw[:, COL_LEFT_ARM]            # (T, 7)
        T_raw   = q_right.shape[0]

        # ── Clean: trim stationary initial/final states and smooth ────────
        if clean_data_flag:
            q_both = np.concatenate([q_left, q_right], axis=1)   # (T, 14)
            q_both = clean_demo(q_both)
            q_left  = q_both[:, :7]
            q_right = q_both[:, 7:]

        # ── Optional orientation nudge ────────────────────────────────────
        if NUDGE_EE_DOWN:
            q_left  = nudge_ee_toward_down(q_left,  chain_left)
            q_right = nudge_ee_toward_down(q_right, chain_right)

        T_clean = q_left.shape[0]

        # ── Task-space FK (computed after all joint corrections) ──────────
        x_left  = compute_x(q_left,  chain_left)
        x_right = compute_x(q_right, chain_right)

        # ── Concatenate: left first, then right ──────────────────────────
        q  = np.concatenate([q_left,  q_right],  axis=1)   # (T_clean, 14)
        x  = np.concatenate([x_left,  x_right],  axis=1)   # (T_clean, nb_x)

        # ── Hold the last pose for N_HOLD_STEPS extra frames ─────────────
        if N_HOLD_STEPS > 0:
            q = np.concatenate([q, np.tile(q[-1:],  (N_HOLD_STEPS, 1))], axis=0)
            x = np.concatenate([x, np.tile(x[-1:],  (N_HOLD_STEPS, 1))], axis=0)

        # ── Joint velocities via finite differences (zero-padded at end) ─
        dq        = np.zeros_like(q)
        dq[:-1]   = (q[1:] - q[:-1]) * RECORDING_HZ

        q_demos.append(q)
        x_demos.append(x)
        dq_demos.append(dq)
        print(f"  {fpath.parent.name}/{fpath.name}  "
              f"T_raw={T_raw} → T_clean={T_clean}"
              + (" [nudged]" if NUDGE_EE_DOWN else ""))

print(f"\nLoaded {len(q_demos)} demonstrations.")

# ─────────────────────────────────────────────────────────────────────────────
# Trim all demos to the shortest trajectory length
# ─────────────────────────────────────────────────────────────────────────────
lengths = [q.shape[0] for q in q_demos]
T_min   = min(lengths)
T_max   = max(lengths)
print(f"Trajectory lengths: min={T_min}, max={T_max}")
if T_min < T_max:
    print(f"Trimming all demos to T={T_min} timesteps.")

q_arr  = np.stack([q[:T_min]  for q  in q_demos])    # (N, T_min, 14)
x_arr  = np.stack([x[:T_min]  for x  in x_demos])    # (N, T_min, nb_x)
dq_arr = np.stack([dq[:T_min] for dq in dq_demos])   # (N, T_min, 14)

nb_demos, T, nb_q = q_arr.shape
nb_x = x_arr.shape[2]

print(f"\nDataset summary:")
print(f"  nb_demos : {nb_demos}")
print(f"  T        : {T}  timesteps")
print(f"  nb_q     : {nb_q}  (7 left + 7 right arm joints)")
print(f"  nb_x     : {nb_x}  ({nb_x_per_arm} per arm, task_space='{TASK_SPACE}')")
print(f"  dt       : {1.0 / RECORDING_HZ:.4f} s  ({RECORDING_HZ:.0f} Hz)")
if NUDGE_EE_DOWN:
    print(f"  nudge    : ON  alpha={NUDGE_ALPHA}  ramp={NUDGE_RAMP}  "
          f"col={NUDGE_APPROACH_COL}  target={NUDGE_TARGET_DIR.tolist()}")

# ─────────────────────────────────────────────────────────────────────────────
# Save
# ─────────────────────────────────────────────────────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
output_path = OUTPUT_DIR / f"{DATASET_NAME}.npz"

np.savez(
    output_path,
    q_data=q_arr,
    x_data=x_arr,
    dq_data=dq_arr,
    dt=np.float64(1.0 / RECORDING_HZ),
    nb_q=np.int64(nb_q),
    nb_x=np.int64(nb_x),
    nb_q_left=np.int64(7),
    nb_q_right=np.int64(7),
    nb_x_left=np.int64(nb_x_per_arm),
    nb_x_right=np.int64(nb_x_per_arm),
)

print(f"\nSaved → {output_path}")
print(f"  q_data  shape: {q_arr.shape}")
print(f"  x_data  shape: {x_arr.shape}")
print(f"  dq_data shape: {dq_arr.shape}")

import glob
import os
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

CURRENT_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR = CURRENT_DIR.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.utils import create_rby1_robot

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = ROOT_DIR / 'results_rby1' / 'rby1_quantitative_results'
PLOTS_SPATIAL_DIR = RESULTS_DIR / 'plots_spatial'

LOG_DIRS = {
    'pan':     [ROOT_DIR / 'results_rby1' / 'real_robot_test_logs' / 'pan'],
    'letters': [ROOT_DIR / 'results_rby1' / 'real_robot_test_logs' / 'letters'],
}

DEMO_TASK_SPACE = {
    'rby1_pan_v1':  'xyz_theta',
    'rby1_letters': 'yz',
}

MATCH_DIST_TOLERANCE = 1e-2  # nominal-demo match sanity threshold (summed endpoint L2, radians)
N_RESAMPLE = 200

_PERM = np.array([1., -1., -1., 1., -1., 1., -1.])
_PURE_C2_FALLBACK_COND = np.array([0.0, 1.0, 1.0])
_PURE_C2_THRESHOLD = 0.15


_demo_cache = {}
_robot_cache = {}


def collect_realrobot_logs(dirs) -> list:
    """All REALROBOT_*.npz files in `dirs` (SIM_*.npz simulated rollouts are excluded)."""
    seen = {}
    for d in dirs:
        for f in sorted(glob.glob(str(Path(d) / 'REALROBOT_*.npz'))):
            seen[Path(f).name] = f
    return sorted(seen.values())


# ─────────────────────────────────────────────────────────────────────────────
# Nominal trajectory / dataset lookup
# ─────────────────────────────────────────────────────────────────────────────

def load_augmented_demos(demo_folder: str, task_name: str, group: str) -> dict:
    key = (demo_folder, task_name, group)
    if key not in _demo_cache:
        path = ROOT_DIR / 'demonstrations' / demo_folder / f'{task_name}_augmented_config_{group}.npz'
        if not path.exists():
            raise FileNotFoundError(f'Augmented dataset not found: {path}')
        _demo_cache[key] = np.load(path, allow_pickle=True)['demonstrations'].item()
    return _demo_cache[key]


def _is_pure_c2_fallback(cond_used: np.ndarray) -> bool:
    return (
        abs(cond_used[0]) < _PURE_C2_THRESHOLD and
        abs(cond_used[1] - 1.0) < _PURE_C2_THRESHOLD and
        cond_used[2] < 0
    )


def find_matching_nominal_traj(demos: dict, q_init_l, q_init_r, q_tgt_l, q_tgt_r, cond_used: np.ndarray):
    """Returns (best_idx, best_dist, second_best_dist, nom_q_left, nom_q_right, is_pure_c2).

    nom_q_left/nom_q_right are the full nominal joint trajectories to compare against —
    normally just the matched demo's own columns, but for the pure-C2 fallback case
    (see _PERM above) they are the arm-swapped, sign-flipped columns of the matched
    *plain* demo, matching what the real controller actually tracked.
    """
    is_pure_c2 = _is_pure_c2_fallback(cond_used)
    target_cond = _PURE_C2_FALLBACK_COND if is_pure_c2 else cond_used

    dists = []
    for traj in demos['train_in']:
        traj = np.asarray(traj)
        cond = traj[0, -3:]
        if np.linalg.norm(cond - target_cond) > 0.5:  # cheap pre-filter, well above any real threshold
            dists.append(np.inf)
            continue
        q_left, q_right = traj[:, :7], traj[:, 7:14]
        if is_pure_c2:
            ql0, qr0 = q_right[0] * _PERM, q_left[0] * _PERM
            ql1, qr1 = q_right[-1] * _PERM, q_left[-1] * _PERM
        else:
            ql0, qr0 = q_left[0], q_right[0]
            ql1, qr1 = q_left[-1], q_right[-1]
        d = (np.linalg.norm(ql0 - q_init_l) + np.linalg.norm(qr0 - q_init_r) +
             np.linalg.norm(ql1 - q_tgt_l) + np.linalg.norm(qr1 - q_tgt_r))
        dists.append(d)
    dists = np.asarray(dists)
    order = np.argsort(dists)
    best_idx = int(order[0])
    best_dist = float(dists[best_idx])
    second_best_dist = float(dists[order[1]]) if len(order) > 1 else float('nan')

    best_traj = np.asarray(demos['train_in'][best_idx])
    if is_pure_c2:
        nom_q_left = best_traj[:, 7:14] * _PERM
        nom_q_right = best_traj[:, :7] * _PERM
    else:
        nom_q_left = best_traj[:, :7]
        nom_q_right = best_traj[:, 7:14]

    return best_idx, best_dist, second_best_dist, nom_q_left, nom_q_right, is_pure_c2


# ─────────────────────────────────────────────────────────────────────────────
# Alignment — pluggable registry so DTW or other methods can be dropped in later
# without touching any call sites.
# ─────────────────────────────────────────────────────────────────────────────

def _align_normalized_time(traj_a: np.ndarray, traj_b: np.ndarray, n_samples: int = N_RESAMPLE):
    """Resample both trajectories to n_samples via linear interpolation over their own
    normalized [0, 1] time fraction (sample-index based). Assumes roughly constant
    sample rate within each trajectory and a monotonic start->target motion.

    Caveat: for a real-robot run that was cut short before reaching the target
    (done=False), this stretches the partial motion to fill [0, 1], which can distort
    the comparison — see the `done` column in the output CSV to filter those out.
    """
    def resample(traj, n):
        T = traj.shape[0]
        if T == 1:
            return np.repeat(traj, n, axis=0)
        x_old = np.linspace(0.0, 1.0, T)
        x_new = np.linspace(0.0, 1.0, n)
        out = np.empty((n, traj.shape[1]))
        for d in range(traj.shape[1]):
            out[:, d] = np.interp(x_new, x_old, traj[:, d])
        return out
    return resample(traj_a, n_samples), resample(traj_b, n_samples)


ALIGNMENT_METHODS = {
    'normalized_time': _align_normalized_time,
    # 'dtw': _align_dtw,  # plug in here if needed later
}


def align_trajectories(traj_a: np.ndarray, traj_b: np.ndarray,
                        method: str = 'normalized_time', n_samples: int = N_RESAMPLE):
    if method not in ALIGNMENT_METHODS:
        raise ValueError(f'Unknown alignment method: {method!r}. Options: {list(ALIGNMENT_METHODS)}')
    return ALIGNMENT_METHODS[method](traj_a, traj_b, n_samples)


# ─────────────────────────────────────────────────────────────────────────────
# Forward kinematics / metrics
# ─────────────────────────────────────────────────────────────────────────────

def get_robot(task_space: str):
    if task_space not in _robot_cache:
        _robot_cache[task_space] = create_rby1_robot(task_space=task_space, ee_joint=False, is_taskspace=False)
    return _robot_cache[task_space]


def fk_task_space(q_left: np.ndarray, q_right: np.ndarray, task_space: str) -> np.ndarray:
    robot = get_robot(task_space)
    ql = torch.tensor(np.asarray(q_left), dtype=torch.float32)
    qr = torch.tensor(np.asarray(q_right), dtype=torch.float32)
    xl = robot.fk_func_left_torch(ql).detach().cpu().numpy()
    xr = robot.fk_func_right_torch(qr).detach().cpu().numpy()
    return np.concatenate([xl, xr], axis=1)  # (T, 2 * nb_x_per_arm)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def per_dim_rmse(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((a - b) ** 2, axis=0))


# ─────────────────────────────────────────────────────────────────────────────
# Per-log processing
# ─────────────────────────────────────────────────────────────────────────────

_RUN_RE = re.compile(r'__run_(\d+)')


def process_log(log_path: str, task_type: str, alignment_method: str = 'normalized_time') -> dict:
    d = np.load(log_path, allow_pickle=True)

    demo_folder = str(d['meta_demo_folder'])
    task_name   = str(d['meta_task_name'])
    group       = str(d['meta_group'])
    model_id    = str(d['meta_mlp_model_id'])
    cond_used   = d['meta_conditioning_val_used']
    done        = bool(d['ts_done'][-1])

    task_space = DEMO_TASK_SPACE.get(demo_folder)
    if task_space is None:
        raise ValueError(f'No task_space mapping for demo_folder={demo_folder!r} (log: {log_path})')

    ckpt_path = ROOT_DIR / 'networks_rby1' / 'models' / f'{model_id}.pt'
    ckpt_exists = ckpt_path.exists()

    demos = load_augmented_demos(demo_folder, task_name, group)
    best_idx, best_dist, second_best_dist, nom_q_left, nom_q_right, is_pure_c2 = find_matching_nominal_traj(
        demos,
        d['meta_q_init_left_arm'], d['meta_q_init_right_arm'],
        d['meta_q_target_left_arm'], d['meta_q_target_right_arm'],
        cond_used,
    )
    match_ok = best_dist < MATCH_DIST_TOLERANCE
    n_samples_nominal = nom_q_left.shape[0]
    real_q_left, real_q_right = d['ts_q_left_arm'], d['ts_q_right_arm']

    # Task space
    x_nom = fk_task_space(nom_q_left, nom_q_right, task_space)
    x_real = fk_task_space(real_q_left, real_q_right, task_space)
    x_nom_r, x_real_r = align_trajectories(x_nom, x_real, method=alignment_method)
    task_space_rmse = rmse(x_nom_r, x_real_r)
    task_space_rmse_per_dim = per_dim_rmse(x_nom_r, x_real_r)
    final_task_space_error = float(np.linalg.norm(x_real_r[-1] - x_nom_r[-1]))

    # Joint space (both arms concatenated)
    q_nom = np.concatenate([nom_q_left, nom_q_right], axis=1)
    q_real = np.concatenate([real_q_left, real_q_right], axis=1)
    q_nom_r, q_real_r = align_trajectories(q_nom, q_real, method=alignment_method)
    joint_space_rmse = rmse(q_nom_r, q_real_r)

    run_match = _RUN_RE.search(Path(log_path).name)

    record = dict(
        log_file=Path(log_path).name,
        task_type=task_type,
        run=int(run_match.group(1)) if run_match else None,
        demo_folder=demo_folder,
        group=group,
        model_id=model_id,
        checkpoint_exists=ckpt_exists,
        theta_deg=float(np.rad2deg(cond_used[0])),
        scale=float(cond_used[1]),
        reflection=float(cond_used[2]),
        nominal_demo_idx=best_idx,
        is_pure_c2_fallback=is_pure_c2,
        match_dist=best_dist,
        match_dist_second_best=second_best_dist,
        match_ok=match_ok,
        done=done,
        n_samples_real=int(d['ts_timestamp'].shape[0]),
        n_samples_nominal=int(n_samples_nominal),
        alignment_method=alignment_method,
        task_space=task_space,
        task_space_rmse=task_space_rmse,
        joint_space_rmse=joint_space_rmse,
        final_task_space_error=final_task_space_error,
        created_at=str(d['meta_created_at']),
        git_commit_control_repo=str(d['meta_git_commit']),
    )
    # Kept only for plotting — stripped before writing the CSV.
    record['_task_space_rmse_per_dim'] = task_space_rmse_per_dim.tolist()
    record['_x_nom'] = x_nom_r
    record['_x_real'] = x_real_r
    return record


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _plot_spatial_arm(ax, x_nom_arm: np.ndarray, x_real_arm: np.ndarray, task_space: str,
                       invert_x: bool = False, arrow_subsample: int = 15):
    """
    Combined position (+ orientation, where available) trace for one arm — the actual
    2-D path traced by the end effector, not component-vs-time. For 'yz' this is the
    letter shape itself; for 'xyz_theta' it's the x-y path with yaw shown as arrows.
    """
    if task_space == 'yz':
        y_n, z_n = x_nom_arm[:, 0], x_nom_arm[:, 1]
        y_r, z_r = x_real_arm[:, 0], x_real_arm[:, 1]
        ax.plot(y_n, z_n, color='#444444', linewidth=2.0, label='nominal')
        ax.plot(y_r, z_r, color='#C0392B', linewidth=1.6, linestyle='--', label='real robot')
        for y, z, color in [(y_n, z_n, '#444444'), (y_r, z_r, '#C0392B')]:
            ax.plot(y[0], z[0], 'o', color=color, markersize=6)
            ax.plot(y[-1], z[-1], 'x', color=color, markersize=8, markeredgewidth=2)
        ax.set_xlabel("y [m]  (← robot's left)")
        ax.set_ylabel('z [m]')
        if invert_x:
            ax.invert_xaxis()

    elif task_space == 'xyz_theta':
        # Top-down (x, y) position trace with yaw drawn as orientation arrows.
        x_n, y_n, yaw_n = x_nom_arm[:, 0], x_nom_arm[:, 1], x_nom_arm[:, 3]
        x_r, y_r, yaw_r = x_real_arm[:, 0], x_real_arm[:, 1], x_real_arm[:, 3]
        ax.plot(x_n, y_n, color='#444444', linewidth=2.0, label='nominal')
        ax.plot(x_r, y_r, color='#C0392B', linewidth=1.6, linestyle='--', label='real robot')
        for x, y, color in [(x_n, y_n, '#444444'), (x_r, y_r, '#C0392B')]:
            ax.plot(x[0], y[0], 'o', color=color, markersize=6)
            ax.plot(x[-1], y[-1], 'x', color=color, markersize=8, markeredgewidth=2)

        span = max(x_n.max() - x_n.min(), y_n.max() - y_n.min(), 1e-3)
        arrow_len = 0.08 * span
        for x, y, yaw, color in [(x_n, y_n, yaw_n, '#444444'), (x_r, y_r, yaw_r, '#C0392B')]:
            idx = np.arange(0, len(x), arrow_subsample)
            ax.quiver(x[idx], y[idx], np.cos(yaw[idx]) * arrow_len, np.sin(yaw[idx]) * arrow_len,
                      color=color, alpha=0.55, angles='xy', scale_units='xy', scale=1.0, width=0.005)
        ax.set_xlabel('x [m]')
        ax.set_ylabel('y [m]')

    else:
        raise ValueError(f'No spatial-plot layout defined for task_space={task_space!r}')

    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, linewidth=0.4)


def plot_log_spatial(record: dict, out_dir: Path):
    """Combined position(+orientation) trace per arm — e.g. the drawn letter shape
    itself for the letters task, rather than component-vs-time curves."""
    x_nom, x_real = record['_x_nom'], record['_x_real']
    task_space = record['task_space']
    n_x_per_arm = x_nom.shape[1] // 2

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    for arm_i, arm_name in enumerate(['left', 'right']):
        cols = slice(arm_i * n_x_per_arm, (arm_i + 1) * n_x_per_arm)
        _plot_spatial_arm(axes[arm_i], x_nom[:, cols], x_real[:, cols], task_space,
                          invert_x=(task_space == 'yz'))
        axes[arm_i].set_title(f'{arm_name} arm', fontsize=9)
    axes[0].legend(fontsize=8, loc='best')

    status = 'done' if record['done'] else 'INCOMPLETE (done=False)'
    fig.suptitle(
        f"{record['log_file']}  |  θ={record['theta_deg']:.1f}°, s={record['scale']:.2f}, "
        f"r={record['reflection']:.0f}  |  task-space RMSE={record['task_space_rmse']:.4f}  |  {status}",
        fontsize=9,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    out_path = out_dir / f"{record['task_type']}__{Path(record['log_file']).stem}.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def _clear_pngs(out_dir: Path):
    """Remove plots of a previous run, so the folder only holds plots of the current logs."""
    for f in out_dir.glob('*.png'):
        f.unlink()


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_SPATIAL_DIR.mkdir(parents=True, exist_ok=True)
    _clear_pngs(PLOTS_SPATIAL_DIR)

    records = []
    for task_type, log_dirs in LOG_DIRS.items():
        log_paths = collect_realrobot_logs(log_dirs)
        print(f'\n=== {task_type}: {len(log_paths)} logs in {[str(p) for p in log_dirs]} ===')
        for log_path in log_paths:
            print(f'  processing {Path(log_path).name} ...')
            try:
                record = process_log(log_path, task_type)
            except Exception as e:
                print(f'    [ERROR] {e}')
                continue
            if not record['match_ok']:
                print(f"    [WARNING] weak nominal-demo match (dist={record['match_dist']:.4f}) "
                      f"— verify manually.")
            if not record['checkpoint_exists']:
                print(f"    [WARNING] checkpoint not found on disk: {record['model_id']}.pt")
            plot_log_spatial(record, PLOTS_SPATIAL_DIR)
            records.append(record)

    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in records])
    df = df.sort_values(['task_type', 'theta_deg', 'scale', 'reflection'], kind='stable')
    csv_path = RESULTS_DIR / 'rby1_quantitative_results.csv'
    df.to_csv(csv_path, index=False)
    print(f'\nSaved {len(df)} results (one row per log) -> {csv_path}')
    print(f'Saved per-log spatial plots -> {PLOTS_SPATIAL_DIR}')

    # Mean ± std over the completed runs (done=True); incomplete runs are listed in the CSV but not counted.
    print('\n' + '=' * 70 + '\nSUMMARY\n' + '=' * 70)
    for task_type in df['task_type'].unique():
        all_sub = df[df.task_type == task_type]
        sub = all_sub[all_sub.done]
        print(f'\n{task_type}  (n_logs={len(all_sub)}, of which done=True: {len(sub)}):')
        if len(sub) == 0:
            print('  no completed runs')
            continue
        print(f'  task-space  RMSE: mean={sub.task_space_rmse.mean():.4f}  std={sub.task_space_rmse.std():.4f}')
        print(f'  joint-space RMSE: mean={sub.joint_space_rmse.mean():.4f}  std={sub.joint_space_rmse.std():.4f}')


if __name__ == '__main__':
    main()

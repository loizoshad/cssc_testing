import os
from math import ceil
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import numpy as np
import random
import torch
import shutil

from utils.groups import *
from utils.utils import *
from utils.networks_pytorch import CustomMLP

cols = shutil.get_terminal_size().columns
np.set_printoptions(precision=4, suppress=True, linewidth=cols)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = Path(CURRENT_DIR).parent.resolve()
MODELS_DIR  = ROOT_DIR / 'networks' / 'models'
device      = torch.device('cuda' if torch.cuda.is_available()
                           else 'mps' if torch.backends.mps.is_available()
                           else 'cpu')

# ──────────────────────────────────────────────────────────────────────────────
# Configuration  ← edit this section
# ──────────────────────────────────────────────────────────────────────────────
is_taskspace = False
robot = create_two_arm_robot(nb_dofs_left=2, nb_dofs_right=2, nb_x_left=2, nb_x_right=2, ee_joint=False, is_taskspace=is_taskspace)
# robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2, ee_joint=False)

# margin=1e-10
# margin=0.05

margin=0.8

# margin=2.0
# margin=5.0
# margin=100.0
decouple_arms = True
num_iterations = 20000
nb_steps = 5

demo_folder = 'planar_robot'
demo_type_left = 'LASA'; demo_name_left = 'CShape'
demo_type_right = 'LASA'; demo_name_right = 'NShape'    
# Load the dual arm data with name following the convention: demos_name = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{nb_dofs}.npz'
task = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{robot.nb_dofs}'
task = task + '_taskspace' if is_taskspace else task

model_id = f'MLP_BASELINE_{task}_epochs={str(num_iterations)}_nsteps={str(nb_steps)}_' + f'norm_bounds={margin}_'

if decouple_arms:
    model_id += 'decoupled'

BASE_MODEL_ID = (model_id)


# Checkpoint iterations to load.  None → auto-discover all saved checkpoints.
# CHECKPOINT_ITERS = [200, 6000, 12000]
# CHECKPOINT_ITERS = [200, 1000, 2000, 
#                     3000, 4000, 5000, 
#                     6000, 7000, 8000, 
#                     10000, 12000, 14000, 
#                     16000, 18000]
# CHECKPOINT_ITERS = [200, 1000, 5000, 
#                     10000, 15000]


CHECKPOINT_ITERS = [200, 5000, 
                    10000, 15000]


# # Pick the first 10 checkpoints for a quick test ruqq0)][:84-1] # 12 columns and 7 rows


# CHECKPOINT_ITERS = [200]

# Always append the final model (saved without an iter suffix)
INCLUDE_FINAL = True

# Demonstrations used for evaluation
DEMO_CONFIG = dict(
    demo_labels=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
    num_of_demos=[7, 0, 0, 0, 0],
)

N_COLS      = 3      # columns in the trajectory checkpoint grid
# N_COLS      = 12      # columns in the trajectory checkpoint grid
# N_GRID      = 25     # grid resolution for vector fields
N_GRID      = 30     # grid resolution for vector fields
# N_GRID      = 50     # grid resolution for vector fields
VF_FRACTIONS = [0.25, 0.50, 0.75]  # fixed-point fractions along reference demo
SAVE_PLOTS  = True
# ──────────────────────────────────────────────────────────────────────────────


# ── Helpers ───────────────────────────────────────────────────────────────────

def _discover_checkpoints(base_id: str) -> list[int]:
    prefix = base_id + 'iter'
    iters  = []
    for f in MODELS_DIR.glob(f'{prefix}*.pt'):
        try:
            iters.append(int(f.stem.replace(prefix, '')))
        except ValueError:
            pass
    return sorted(iters)


def _to_numpy(x):
    """Tensor or ndarray → float32 ndarray."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy().astype(np.float32)
    return np.asarray(x, dtype=np.float32)


def _denorm_grid(grid_norm_2d: np.ndarray, q_min, q_max) -> np.ndarray:
    """Normalised [-1, 1] grid → physical space using per-arm bounds."""
    q_min = _to_numpy(q_min)
    q_max = _to_numpy(q_max)
    return (grid_norm_2d / 2 + 0.5) * (q_max - q_min) + q_min


# ── Trajectory evaluation ─────────────────────────────────────────────────────

def get_trajectories(network, robot, horizon: int, eval_data: list, is_taskspace: bool = True):
    """
    Returns
    -------
    gt_trajs   : list of (T, 2*n_x) numpy arrays — ground truth in task space
    pred_trajs : list of (horizon, 2*n_x) numpy arrays — network rollout in task space

    In taskspace mode the state columns already are task positions.
    In config-space mode, FK is applied to convert joint angles → task positions.
    """
    Q_min, Q_max = network.demonstrations['Q_min'], network.demonstrations['Q_max']
    X_min, X_max = network.demonstrations['X_min'], network.demonstrations['X_max']
    nb_dofs = robot.nb_dofs

    state_denorm = [denormalize_state(t[:, :nb_dofs], Q_min, Q_max) for t in eval_data]
    goal_denorm  = [denormalize_state(t[:, nb_dofs:], X_min, X_max) for t in eval_data]
    data_denorm  = [torch.hstack([s, g]) for s, g in zip(state_denorm, goal_denorm)]

    ic = torch.stack([t[0] for t in data_denorm])

    with torch.no_grad():
        pred_qs, _ = network.forward_denorm_multistep(
            horizon=horizon, inp_batch=ic, return_traj=True, use_grad=False
        )

    if is_taskspace:
        gt_trajs   = [s[:, :nb_dofs].cpu().numpy() for s in state_denorm]
        pred_trajs = [pred_qs[i, :, :nb_dofs].cpu().numpy() for i in range(pred_qs.shape[0])]
    else:
        n_l, n_r = robot.nb_dofs_left, robot.nb_dofs_right

        def _q_to_task(q_traj: torch.Tensor) -> np.ndarray:
            """(T, nb_dofs) physical joint angles → (T, n_x_left+n_x_right) task positions."""
            with torch.no_grad():
                x_l = robot.fk_func_left_torch(q_traj[:, :n_l])       # (T, 2)
                x_r = robot.fk_func_right_torch(q_traj[:, n_l:n_l+n_r])  # (T, 2)
            return torch.cat([x_l, x_r], dim=-1).cpu().numpy()

        gt_trajs   = [_q_to_task(s[:, :nb_dofs]) for s in state_denorm]
        pred_trajs = [_q_to_task(pred_qs[i, :, :nb_dofs]) for i in range(pred_qs.shape[0])]

    return gt_trajs, pred_trajs


# ── Vector-field query ────────────────────────────────────────────────────────

def query_vf(model, arm: str, fixed_pt: np.ndarray,
             x_goal_C: np.ndarray, x_goal_N: np.ndarray,
             grid_norm: np.ndarray,
             n_q: int,
             dq_min_arm: np.ndarray, dq_max_arm: np.ndarray,
             q_min_arm:  np.ndarray, q_max_arm:  np.ndarray,
             grid_phys:  np.ndarray,
             is_taskspace: bool = True,
             arm_fk_torch=None, arm_jacob_torch=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Query unit-normalised velocity arrows for `arm` ('C' or 'N') over
    `grid_norm` while holding the other arm fixed at `fixed_pt`.

    Taskspace mode  : grid_phys are task-space positions; returns (grid_phys, v_unit).
    Config-space mode: grid_phys are physical joint angles; FK maps them to task-space
                       positions for plotting, and J @ dq maps joint velocity to
                       task-space velocity.  Requires arm_fk_torch and arm_jacob_torch.

    Returns (plot_positions, v_unit) both shape (n_pts, 2) in task space.
    """
    n_pts = len(grid_norm)
    fp    = np.tile(fixed_pt,  (n_pts, 1)).astype(np.float32)
    xg_C  = np.tile(x_goal_C,  (n_pts, 1)).astype(np.float32)
    xg_N  = np.tile(x_goal_N,  (n_pts, 1)).astype(np.float32)

    if arm == 'N':
        # sweep N's state, C is fixed
        inp      = np.hstack([fp, grid_norm, xg_C, xg_N])
        vel_cols = slice(n_q, 2 * n_q)
    else:  # 'C'
        # sweep C's state, N is fixed
        inp      = np.hstack([grid_norm, fp, xg_C, xg_N])
        vel_cols = slice(0, n_q)

    with torch.no_grad():
        t_inp  = torch.tensor(inp, device=device)
        v_norm = model(t_inp)[:, vel_cols].cpu().numpy()

    # Denormalise: network output → physical velocity (joint or task space)
    v_dq   = (v_norm / 2 + 0.5) * (dq_max_arm - dq_min_arm) + dq_min_arm
    v_phys = v_dq * 0.5 * (q_max_arm - q_min_arm)

    if is_taskspace:
        v_unit = v_phys / (np.linalg.norm(v_phys, axis=1, keepdims=True) + 1e-8)
        return grid_phys, v_unit

    # Config-space: grid_phys holds physical joint angles.
    # Map to task-space positions (for plotting) and task-space velocities (via Jacobian).
    q_tensor = torch.tensor(grid_phys, dtype=torch.float32, device=device)
    with torch.no_grad():
        grid_task = arm_fk_torch(q_tensor).cpu().numpy()   # (n_pts, 2)
        J         = arm_jacob_torch(q_tensor).cpu().numpy() # (n_pts, 2, n_q)

    v_task = np.einsum('bij,bj->bi', J, v_phys)            # (n_pts, 2)
    v_unit = v_task / (np.linalg.norm(v_task, axis=1, keepdims=True) + 1e-8)
    return grid_task, v_unit


# ── Visualisation: trajectory grid ───────────────────────────────────────────

def _draw_traj_panel(ax, gt_trajs, pred_trajs, n_q: int, title: str):
    """One checkpoint panel: both arms, GT=green, pred=red; solid=left, dashed=right."""
    for gt in gt_trajs:
        ax.plot(gt[:, 0],   gt[:, 1],   color='green', lw=2.5, alpha=0.8)
        ax.plot(gt[:, n_q], gt[:, n_q+1], color='green', lw=2.5, alpha=0.8, ls='--')
    for pred in pred_trajs:
        ax.plot(pred[:, 0],   pred[:, 1],   color='red', lw=1.5, alpha=0.75)
        ax.plot(pred[:, n_q], pred[:, n_q+1], color='red', lw=1.5, alpha=0.75, ls='--')
    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x', fontsize=8); ax.set_ylabel('y', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)


def plot_checkpoint_grid(panels, n_q: int, n_cols: int = 3,
                         suptitle: str = 'Checkpoint evaluation',
                         save_path=None):
    n      = len(panels)
    n_rows = ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows), squeeze=False)
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    for idx, p in enumerate(panels):
        _draw_traj_panel(axes[idx // n_cols][idx % n_cols],
                         p['gt_trajs'], p['pred_trajs'], n_q, p['label'])

    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    fig.legend(handles=[
        mlines.Line2D([], [], color='green', lw=2.5,          label='GT – left arm'),
        mlines.Line2D([], [], color='green', lw=2.5, ls='--', label='GT – right arm'),
        mlines.Line2D([], [], color='red',   lw=1.5,          label='Pred – left arm'),
        mlines.Line2D([], [], color='red',   lw=1.5, ls='--', label='Pred – right arm'),
    ], loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.04),
       fontsize=9, framealpha=0.9)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] Saved {save_path}')
    return fig


# ── Visualisation: vector-field grid ─────────────────────────────────────────

def plot_vf_grid(panels, arm: str, gt_trajs: list, n_q: int,
                 fractions: list, demo_name_arm: str, demo_name_other: str,
                 suptitle: str = '', save_path=None):
    """
    rows = fractions (len(fractions))
    cols = checkpoints (len(panels))

    Each cell: unit-normalised quiver for `arm`, overlaid with the physical
    ground-truth trajectories of that arm and a star at the attractor.

    panels : list of dicts with key 'vf_{arm}' → list[(grid_phys, v_unit)]
             one entry per fraction, in the same order as `fractions`.
    """
    n_ckpts = len(panels)
    n_fracs = len(fractions)

    fig, axs = plt.subplots(n_fracs, n_ckpts,
                            figsize=(3.8 * n_ckpts, 3.8 * n_fracs),
                            squeeze=False)
    fig.suptitle(suptitle, fontsize=11, y=1.01)

    arm_col  = slice(n_q, 2*n_q) if arm == 'N' else slice(0, n_q)
    color    = 'tomato'    if arm == 'N' else 'steelblue'
    attractor = gt_trajs[0][-1, arm_col]   # last point of first demo = goal

    for row, frac in enumerate(fractions):
        for col, panel in enumerate(panels):
            ax = axs[row, col]
            grid_phys, v_unit = panel[f'vf_{arm}'][row]

            # Vector field
            ax.quiver(grid_phys[:, 0], grid_phys[:, 1],
                      v_unit[:, 0],    v_unit[:, 1],
                      color='darkorange', alpha=0.9, scale=35, width=0.003)

            # Ground-truth demos
            for gt in gt_trajs:
                arm_traj = gt[:, arm_col]
                ax.plot(arm_traj[:, 0], arm_traj[:, 1],
                        color=color, lw=1.2, alpha=0.5)

            # Attractor / goal
            ax.scatter(attractor[0], attractor[1],
                       color=color, marker='*', s=120, zorder=5)

            # Labels
            col_title = panel['label'] if row == 0 else ''
            row_label = f'{demo_name_other} fixed at {int(frac*100)}%'
            ax.set_title(col_title, fontsize=9)
            ax.set_xlabel(row_label, fontsize=7)
            ax.grid(True, alpha=0.3)
            ax.set_aspect('equal', adjustable='datalim')

    # Row labels on left edge
    for row, frac in enumerate(fractions):
        axs[row, 0].set_ylabel(f'{demo_name_other} fixed\nat {int(frac*100)}%',
                               fontsize=8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] Saved {save_path}')
    return fig


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    # ── Data ──────────────────────────────────────────────────────────────────
    data_vis = DataVisualizer(task=task, isotropic_normalization=False, margin=margin)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(demo_folder='planar_robot', load_augmented_data=False)
    demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, train_demo_types=['original'], num_train_demos_per_type=[7])

    # ── Network skeleton (weights loaded per-checkpoint below) ────────────────
    network = CustomMLP(model_id=BASE_MODEL_ID, load_model_flag=False, save_model_flag=False, demonstrations=demonstrations_normalized, robot=robot)
    data_vis.set_network(network)

    horizon = min(t.shape[0] for t in network.demonstrations['train_in'])

    demonstrations_eval = choose_demos_for_evaluation(network.demonstrations, config=DEMO_CONFIG)
    print(f'[INFO] Evaluating on {len(demonstrations_eval)} demonstrations, horizon={horizon}')

    # ── VF setup (computed once — does not depend on network weights) ─────────
    n_q = network.n_q   # per-arm state dims (= 2)
    n_x = network.n_x   # per-arm goal dims  (= 2)

    Q_min_np  = _to_numpy(network.demonstrations['Q_min'])
    Q_max_np  = _to_numpy(network.demonstrations['Q_max'])
    Dq_min_np = _to_numpy(network.demonstrations['Dq_min'])
    Dq_max_np = _to_numpy(network.demonstrations['Dq_max'])

    # Per-arm bounds slices
    Q_min_C  = Q_min_np[:n_q];      Q_max_C  = Q_max_np[:n_q]
    Q_min_N  = Q_min_np[n_q:2*n_q]; Q_max_N  = Q_max_np[n_q:2*n_q]
    Dq_min_C = Dq_min_np[:n_q];     Dq_max_C = Dq_max_np[:n_q]
    Dq_min_N = Dq_min_np[n_q:2*n_q]; Dq_max_N = Dq_max_np[n_q:2*n_q]

    # Reference demo (normalised) used to define fixed-point fractions
    ref_demo      = demonstrations_eval[0]                    # (T, 2*n_q + 2*n_x)
    ref_C_norm    = ref_demo[:, :n_q].cpu().numpy()          # normalised q_C
    ref_N_norm    = ref_demo[:, n_q:2*n_q].cpu().numpy()     # normalised q_N
    x_goal_C_norm = _to_numpy(ref_demo[0, 2*n_q:2*n_q+n_x]) # goal C (constant)
    x_goal_N_norm = _to_numpy(ref_demo[0, 2*n_q+n_x:])      # goal N (constant)

    fixed_C_pts = [ref_C_norm[int(f * (len(ref_C_norm) - 1))] for f in VF_FRACTIONS]
    fixed_N_pts = [ref_N_norm[int(f * (len(ref_N_norm) - 1))] for f in VF_FRACTIONS]

    # Normalised query grid (same for both arms)
    gx, gy    = np.meshgrid(np.linspace(-1.4, 1.4, N_GRID),
                             np.linspace(-1.4, 1.4, N_GRID))
    grid_norm = np.stack([gx.ravel(), gy.ravel()], axis=1).astype(np.float32)

    grid_phys_C = _denorm_grid(grid_norm, Q_min_C, Q_max_C)
    grid_phys_N = _denorm_grid(grid_norm, Q_min_N, Q_max_N)

    # ── Resolve checkpoints ───────────────────────────────────────────────────
    if CHECKPOINT_ITERS is None:
        iters_to_load = _discover_checkpoints(BASE_MODEL_ID)
        print(f'[INFO] Auto-discovered {len(iters_to_load)} checkpoints')
    else:
        iters_to_load = sorted(CHECKPOINT_ITERS)

    checkpoints: list[tuple[str, str]] = []
    for it in iters_to_load:
        ckpt_id   = f'{BASE_MODEL_ID}iter{it}'
        ckpt_path = MODELS_DIR / (ckpt_id + '.pt')
        if ckpt_path.exists():
            checkpoints.append((f'iter {it}', ckpt_id))
        else:
            print(f'[WARN] Not found, skipping: {ckpt_path.name}')

    if INCLUDE_FINAL:
        final_path = MODELS_DIR / (BASE_MODEL_ID + '.pt')
        if final_path.exists():
            checkpoints.append(('Final', BASE_MODEL_ID))
        else:
            print(f'[WARN] Final model not found: {final_path.name}')

    if not checkpoints:
        raise FileNotFoundError('No checkpoint files found. Check BASE_MODEL_ID.')

    print(f'[INFO] Loading {len(checkpoints)} checkpoints: '
          + ', '.join(lbl for lbl, _ in checkpoints))

    # ── Evaluate each checkpoint ───────────────────────────────────────────────
    panels: list[dict] = []

    for label, model_id in checkpoints:
        network.load_model(model_id=model_id)
        network.model.eval()

        # Trajectories
        gt_trajs, pred_trajs = get_trajectories(
            network, robot, horizon, demonstrations_eval, is_taskspace=is_taskspace
        )

        # Vector fields: 3 entries per arm, one per fraction
        vf_N, vf_C = [], []
        for fi, frac in enumerate(VF_FRACTIONS):
            vf_N.append(query_vf(
                model          = network.model,
                arm            = 'N',
                fixed_pt       = fixed_C_pts[fi],
                x_goal_C       = x_goal_C_norm,
                x_goal_N       = x_goal_N_norm,
                grid_norm      = grid_norm,
                n_q            = n_q,
                dq_min_arm     = Dq_min_N, dq_max_arm = Dq_max_N,
                q_min_arm      = Q_min_N,  q_max_arm  = Q_max_N,
                grid_phys      = grid_phys_N,
                is_taskspace   = is_taskspace,
                arm_fk_torch   = robot.fk_func_right_torch,
                arm_jacob_torch= robot.jacob0_right_torch,
            ))
            vf_C.append(query_vf(
                model          = network.model,
                arm            = 'C',
                fixed_pt       = fixed_N_pts[fi],
                x_goal_C       = x_goal_C_norm,
                x_goal_N       = x_goal_N_norm,
                grid_norm      = grid_norm,
                n_q            = n_q,
                dq_min_arm     = Dq_min_C, dq_max_arm = Dq_max_C,
                q_min_arm      = Q_min_C,  q_max_arm  = Q_max_C,
                grid_phys      = grid_phys_C,
                is_taskspace   = is_taskspace,
                arm_fk_torch   = robot.fk_func_left_torch,
                arm_jacob_torch= robot.jacob0_left_torch,
            ))

        panels.append(dict(label=label, gt_trajs=gt_trajs,
                           pred_trajs=pred_trajs, vf_N=vf_N, vf_C=vf_C))
        print(f'  evaluated {label}')

    # Ground-truth trajectories are identical across checkpoints; use first panel
    gt_trajs_ref = panels[0]['gt_trajs']

    # ── Figure 1: trajectory checkpoint grid ──────────────────────────────────
    save_traj = (ROOT_DIR / 'evaluation' / 'plots' / f'{BASE_MODEL_ID}_checkpoint_traj.png'
                 if SAVE_PLOTS else None)
    plot_checkpoint_grid(
        panels, n_q=n_q, n_cols=N_COLS,
        suptitle=f'Trajectory evaluation — {task}',
        save_path=save_traj,
    )

    # ── Figure 2: N-arm vector-field grid ─────────────────────────────────────
    #   rows = fractions, cols = checkpoints
    #   "N velocity field with C fixed at X% of a C demo"
    save_vf_N = (ROOT_DIR / 'evaluation' / 'plots' / f'{BASE_MODEL_ID}_vf_N.png'
                 if SAVE_PLOTS else None)
    plot_vf_grid(
        panels         = panels,
        arm            = 'N',
        gt_trajs       = gt_trajs_ref,
        n_q            = n_q,
        fractions      = VF_FRACTIONS,
        demo_name_arm  = demo_name_right,   # NShape
        demo_name_other= demo_name_left,    # CShape
        suptitle       = (f'{demo_name_right} velocity field\n'
                          f'rows: {demo_name_left} state fixed at 25 / 50 / 75% of a demo  |  '
                          f'cols: training checkpoint'),
        save_path      = save_vf_N,
    )

    # ── Figure 3: C-arm vector-field grid ─────────────────────────────────────
    #   rows = fractions, cols = checkpoints
    #   "C velocity field with N fixed at X% of a N demo"
    save_vf_C = (ROOT_DIR / 'evaluation' / 'plots' / f'{BASE_MODEL_ID}_vf_C.png'
                 if SAVE_PLOTS else None)
    plot_vf_grid(
        panels         = panels,
        arm            = 'C',
        gt_trajs       = gt_trajs_ref,
        n_q            = n_q,
        fractions      = VF_FRACTIONS,
        demo_name_arm  = demo_name_left,    # CShape
        demo_name_other= demo_name_right,   # NShape
        suptitle       = (f'{demo_name_left} velocity field\n'
                          f'rows: {demo_name_right} state fixed at 25 / 50 / 75% of a demo  |  '
                          f'cols: training checkpoint'),
        save_path      = save_vf_C,
    )

    plt.show()

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


seed = 42
# seed = 7223598  # 20001
# seed = 63724918 # 20001
# seed = 249831868 # 20001




# seed = 32482183 # 20001


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = Path(CURRENT_DIR).parent.resolve()
MODELS_DIR  = ROOT_DIR / 'networks' / 'models'


# ──────────────────────────────────────────────────────────────────────────────
# Configuration  ← edit this section
# ──────────────────────────────────────────────────────────────────────────────
is_taskspace = False
# robot = create_two_arm_robot(nb_dofs_left=2, nb_dofs_right=2, nb_x_left=2, nb_x_right=2, ee_joint=False, is_taskspace=is_taskspace)
robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2, ee_joint=False)

demo_folder = 'planar_robot'

demo_type_left = 'LASA'; demo_name_left = 'CShape'
demo_type_right = 'LASA'; demo_name_right = 'NShape'

# demo_type_left = 'LASA'; demo_name_left = 'PShape'
# demo_type_right = 'LASA'; demo_name_right = 'CShape'
    
# demo_type_left = 'LASA'; demo_name_left = 'PShape'
# demo_type_right = 'LASA'; demo_name_right = 'SShape'

# demo_type_left = 'LASA'; demo_name_left = 'CShape'
# demo_type_right = 'LASA'; demo_name_right = 'SShape'

# Load the dual arm data with name following the convention: demos_name = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{nb_dofs}.npz'
task = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{robot.nb_dofs}'
task = task + '_taskspace' if is_taskspace else task
############################################################################################################
# Network settings
############################################################################################################    
margin=1e-10
# margin=0.05
# margin=0.1


# num_iterations = 20000
# num_iterations = 200
# num_iterations = 20000

# num_iterations = 4000

# num_iterations = 10000
num_iterations = 50000

# num_iterations = 20000
# num_iterations = 20001
# num_iterations = 100000


# num_iterations = 8000

# num_iterations = 20001
# num_iterations = 7001


# nb_steps = 5
# stride = 1

## Failed for 4 DoF on the 20k sample at least
# nb_steps = 10
# stride = 5

# nb_steps = 20
# stride = 5


# nb_steps = 50
# stride = 5

# nb_steps = 10
# stride = 10

nb_steps = 15
stride = 10


batch_size = 250

train = False
load_model_flag = True
save_model_flag = False
save_plots = False
save_loss = False

# decouple_arms = True
decouple_arms = True
isotropic_normalization = False


# conditioning=None#'goal'
# normalize_conditioning = False
# conditioning='goal'
# normalize_conditioning = True

conditioning='symmetry'
normalize_conditioning = False

task_space_loss = False

## Fully trained model id
model_id = f'MLP_BASELINE_{task}_epochs={str(num_iterations)}_nsteps={str(nb_steps)}_stride={stride}' + f'norm_bounds={margin}_'

if decouple_arms:
    model_id += 'decoupled'
else:
    model_id += 'coupled'

# Add the condioning type to the model
model_id += f'_conditioning={conditioning}' if conditioning is not None else '_no_conditioning'
model_id += f'_normalize_conditioning={normalize_conditioning}' if normalize_conditioning else ''
model_id += f'_task_space_loss' if task_space_loss else ''
# add the seed
model_id += f'_seed={seed}_'




# model_id = 'MLP_DA_SO2Scaling2Group_left-LASA-CShape_right-LASA-NShape_ndofs-8_epochs=100000_nsteps=15_stride=10norm_bounds=1e-10_decoupled_conditioning=symmetry_seed=42__CustomMLP.pt'

# so2
# model_id = 'MLP_DA_SO2_left-LASA-CShape_right-LASA-NShape_ndofs-8_epochs=20000_nsteps=15_stride=10norm_bounds=1e-10_decoupled_conditioning=symmetry_seed=42_'
# scaling2
# model_id = 'MLP_DA_Scaling2_left-LASA-CShape_right-LASA-NShape_ndofs-8_epochs=20000_nsteps=15_stride=10norm_bounds=1e-10_decoupled_conditioning=symmetry_seed=42_'
# so2+scaling2
# model_id='MLP_DA_SO2Scaling2Group_left-LASA-CShape_right-LASA-NShape_ndofs-8_epochs=100000_nsteps=15_stride=10norm_bounds=1e-10_decoupled_conditioning=symmetry_seed=42_'
# c2+so2+scaling2
model_id='MLP_DA_C2SO2Scaling2Group_left-LASA-CShape_right-LASA-NShape_ndofs-8_epochs=50000_nsteps=15_stride=10norm_bounds=1e-10_decoupled_conditioning=symmetry_seed=42_'

BASE_MODEL_ID = (model_id)






# # Checkpoint iterations to load.  Use None to auto-discover all saved checkpoints.
# N_COLS = 8          # columns in the checkpoint grid
# # CHECKPOINT_ITERS = [200, 1000, 2000, 4000, 6000, 10000, 14000, 18000, 20000]
# # CHECKPOINT_ITERS = [200, 1000, 2000, 
# #                     3000, 4000, 5000, 
# #                     6000, 7000, 8000, 
# #                     10000, 12000, 14000, 
# #                     16000, 18000]

# # plot every 100 the first 3k iterationso
# # CHECKPOINT_ITERS = list(range(0, 3001, 100))
# # CHECKPOINT_ITERS = list(range(0, num_iterations+1, 200))
# # plot_every = 200
# plot_every = 500
# # max_num_of_plots = int(N_COLS * 7 - 1)
# max_num_of_plots = int(N_COLS * 8 - 1)
# CHECKPOINT_ITERS = [i for i in range(plot_every, num_iterations, plot_every)][:max_num_of_plots]
# if CHECKPOINT_ITERS[-1] != num_iterations and len(CHECKPOINT_ITERS):
#     CHECKPOINT_ITERS.append(num_iterations)


# N_COLS = 5
# CHECKPOINT_ITERS = [i for i in range(1000, num_iterations + 1, 1000) if i < num_iterations]

N_COLS = 5
# CHECKPOINT_ITERS = [10000, 20000, 30000, 40000, 100000]

CHECKPOINT_ITERS = [i for i in range(2000, num_iterations + 1, 2000) if i < num_iterations]
# every 5k
# CHECKPOINT_ITERS = [i for i in range(5000, num_iterations + 1, 5000) if i < num_iterations]

# Always include the final model (saved without an iter suffix)?
INCLUDE_FINAL = True

# Demonstrations used for evaluation
DEMO_CONFIG = dict(
    demo_labels=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
    # num_of_demos=[0, 6, 0, 0, 0],
    # num_of_demos=[0, 0, 6, 0, 0],
    # num_of_demos=[0, 0, 0, 8, 0],
    num_of_demos=[0, 3, 3, 3, 3],
)

SAVE_PLOTS = True  # set True to save the figure to results/plots/
# ──────────────────────────────────────────────────────────────────────────────


def _discover_checkpoints(base_id: str) -> list[int]:
    """Return all checkpoint iterations found on disk, sorted."""
    prefix = base_id + 'iter'
    iters = []
    for f in MODELS_DIR.glob(f'{prefix}*.pt'):
        try:
            iters.append(int(f.stem.replace(prefix, '')))
        except ValueError:
            pass
    return sorted(iters)

def get_trajectories(network, robot, horizon: int, eval_data: list, goal_conditioned=False):
    """
    Denormalize eval_data, take initial conditions, and run the network
    forward open-loop for `horizon` steps.

    Returns
    -------
    gt_trajs   : list of (T, nb_dofs) numpy arrays  — ground truth
    pred_trajs : list of (horizon, nb_dofs) numpy arrays  — predicted
    """
    Q_min, Q_max = network.demonstrations['Q_min'], network.demonstrations['Q_max']
    X_min, X_max = network.demonstrations['X_min'], network.demonstrations['X_max']
    nb_dofs = robot.nb_dofs

    state_denorm = [denormalize_state(t[:, :nb_dofs], Q_min, Q_max) for t in eval_data]
    if goal_conditioned:
        goal_denorm  = [denormalize_state(t[:, nb_dofs:], X_min, X_max) for t in eval_data]
    else:
        # just copy it
        goal_denorm  = [t[:, nb_dofs:] for t in eval_data]
    data_denorm  = [torch.hstack([s, g]) for s, g in zip(state_denorm, goal_denorm)]

    ic = torch.stack([t[0] for t in data_denorm])          # (N, state+goal)

    with torch.no_grad():
        pred_qs, _ = network.forward_denorm_multistep(horizon=horizon, inp_batch=ic, return_traj=True, use_grad=False)

    if is_taskspace:    
        gt_trajs   = [s[:, :nb_dofs].cpu().numpy() for s in state_denorm]
        pred_trajs = [pred_qs[i, :, :nb_dofs].cpu().numpy() for i in range(pred_qs.shape[0])]
        return gt_trajs, pred_trajs
    else:
        # We need to convert them from joint space to task space.
        fkine_left = robot.fk_func_left_torch
        fkine_right = robot.fk_func_right_torch

        gt_trajs = []
        pred_trajs = []
        for i in range(len(data_denorm)):
            gt_q = state_denorm[i][:, :nb_dofs]  # (T, nb_dofs)
            pred_q = pred_qs[i, :, :nb_dofs]     # (horizon, nb_dofs)

            gt_left = fkine_left(gt_q[:, :robot.nb_dofs_left])    # (T, 2)
            gt_right = fkine_right(gt_q[:, robot.nb_dofs_left:]) # (T, 2)
            gt_traj = torch.hstack([gt_left, gt_right])          # (T, 4)
            gt_trajs.append(gt_traj.cpu().numpy())

            pred_left = fkine_left(pred_q[:, :robot.nb_dofs_left])    # (horizon, 2)
            pred_right = fkine_right(pred_q[:, robot.nb_dofs_left:]) # (horizon, 2)
            pred_traj = torch.hstack([pred_left, pred_right])       # (horizon, 4)
            pred_trajs.append(pred_traj.cpu().numpy())
        return gt_trajs, pred_trajs


def _draw_panel(ax, gt_trajs, pred_trajs, n_x: int, title: str):
    """
    Draw one checkpoint panel.  Both arms plotted on the same axes.
    Left arm  → solid lines.
    Right arm → dashed lines.
    Ground truth → green (thick).   Predicted → red (thinner).
    """
    for gt in gt_trajs:
        ax.plot(gt[:, 0],    gt[:, 1],    color='green', lw=2.5, alpha=0.8)
        ax.plot(gt[:, n_x],  gt[:, n_x+1], color='green', lw=2.5, alpha=0.8,
                linestyle='--')

    for pred in pred_trajs:
        ax.plot(pred[:, 0],   pred[:, 1],    color='red', lw=1.5, alpha=0.75)
        ax.plot(pred[:, n_x], pred[:, n_x+1], color='red', lw=1.5, alpha=0.75,
                linestyle='--')

    ax.set_title(title, fontsize=10)
    ax.set_xlabel('x', fontsize=8)
    ax.set_ylabel('y', fontsize=8)
    ax.set_aspect('equal', adjustable='datalim')
    ax.grid(True, alpha=0.3)

def plot_checkpoint_grid(panels, n_x: int, n_cols: int = 3,
                         suptitle: str = 'Checkpoint evaluation',
                         save_path=None):
    """
    Parameters
    ----------
    panels   : list of (label, gt_trajs, pred_trajs)
    n_x      : task-space dims per arm (columns 0:n_x = left, n_x:2*n_x = right)
    n_cols   : number of columns in the grid
    """
    n       = len(panels)
    n_rows  = ceil(n / n_cols)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5 * n_cols, 4.5 * n_rows),
                             squeeze=False)
    fig.suptitle(suptitle, fontsize=13, y=1.01)

    for idx, (label, gt_trajs, pred_trajs) in enumerate(panels):
        ax = axes[idx // n_cols][idx % n_cols]
        _draw_panel(ax, gt_trajs, pred_trajs, n_x, label)

    # hide unused axes
    for idx in range(n, n_rows * n_cols):
        axes[idx // n_cols][idx % n_cols].set_visible(False)

    # shared legend
    legend_handles = [
        mlines.Line2D([], [], color='green', lw=2.5,              label='Ground truth – left arm'),
        mlines.Line2D([], [], color='green', lw=2.5, ls='--',     label='Ground truth – right arm'),
        mlines.Line2D([], [], color='red',   lw=1.5,              label='Predicted – left arm'),
        mlines.Line2D([], [], color='red',   lw=1.5, ls='--',     label='Predicted – right arm'),
    ]
    fig.legend(handles=legend_handles, loc='lower center', ncol=4,
               bbox_to_anchor=(0.5, -0.04), fontsize=9, framealpha=0.9)

    plt.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f'[INFO] Figure saved to {save_path}')

    return fig


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    taskspace_dim = 2
    # # ── Robot & task ──────────────────────────────────────────────────────────
    # is_taskspace = True
    # robot = create_two_arm_robot(nb_dofs_left=2, nb_dofs_right=2, nb_x_left=2,   nb_x_right=2, ee_joint=False, is_taskspace=is_taskspace)
    # demo_folder     = 'planar_robot'
    # demo_type_left  = 'LASA'; demo_name_left  = 'CShape'
    # demo_type_right = 'LASA'; demo_name_right = 'NShape'
    # task = (f'left-{demo_type_left}-{demo_name_left}'
    #         f'_right-{demo_type_right}-{demo_name_right}'
    #         f'_ndofs-{robot.nb_dofs}')
    # task = task + '_taskspace' if is_taskspace else task

    # ── Data ──────────────────────────────────────────────────────────────────
    # Load the SAME augmented dataset used during training so that the
    # normalisation bounds (Q_min/Q_max/Dq_min/Dq_max) stored in
    # network.demonstrations match those the model was trained with.
    # Using load_augmented_data=False (original demos only) gives different,
    # narrower bounds and causes the model to receive out-of-distribution inputs.
    
    # G = SO2()
    # G = Scaling2()
    # G = SO2Scaling2Group()
    G = C2SO2Scaling2Group()
    data_vis = DataVisualizer(task=task, isotropic_normalization=False, margin=margin,
                              conditioning=conditioning, normalize_conditioning=normalize_conditioning,
                              group=G)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(demo_folder=demo_folder, load_augmented_data=True)
    demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(
        demonstrations, demonstrations_normalized,
        # train_demo_types=['original', 'SO2'],
        # train_demo_types=['original', 'Scaling2'],
        # train_demo_types=['original', 'SO2Scaling2Group'],
        train_demo_types=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
        num_train_demos_per_type=[1, 1, 1, 1, 3],
    )

    # ── Network (weights loaded per-checkpoint below, not here) ───────────────
    network = CustomMLP(model_id=BASE_MODEL_ID, load_model_flag=False, save_model_flag=False, demonstrations=demonstrations_normalized, robot=robot)
    data_vis.set_network(network)

    horizon = min(t.shape[0] for t in network.demonstrations['train_in'])

    # ── Choose evaluation demonstrations ──────────────────────────────────────
    demonstrations_eval = choose_demos_for_evaluation(network.demonstrations, config=DEMO_CONFIG)
    print(f'[INFO] Evaluating on {len(demonstrations_eval)} demonstrations, ' f'horizon={horizon}')

    # ── Resolve which checkpoints to load ─────────────────────────────────────
    if CHECKPOINT_ITERS is None:
        iters_to_load = _discover_checkpoints(BASE_MODEL_ID)
        print(f'[INFO] Auto-discovered {len(iters_to_load)} checkpoints')
    else:
        iters_to_load = sorted(CHECKPOINT_ITERS)

    # Build ordered list of (label, model_id) pairs
    checkpoints: list[tuple[str, str]] = []
    for it in iters_to_load:
        ckpt_id = f'{BASE_MODEL_ID}iter{it}'
        ckpt_path = MODELS_DIR / (ckpt_id + '.pt')
        if ckpt_path.exists():
            checkpoints.append((f'iter {it}', ckpt_id))
        else:
            print(f'[WARN] Checkpoint not found, skipping: {ckpt_path.name}')

    if INCLUDE_FINAL:
        final_path = MODELS_DIR / (BASE_MODEL_ID + '.pt')
        if final_path.exists():
            checkpoints.append(('Final', BASE_MODEL_ID))
        else:
            print(f'[WARN] Final model not found: {final_path.name}')

    if not checkpoints:
        raise FileNotFoundError(f'No checkpoint files found. Check BASE_MODEL_ID and MODELS_DIR.\n Model name used: {BASE_MODEL_ID}')

    print(f'[INFO] Loading {len(checkpoints)} checkpoints: ' + ', '.join(label for label, _ in checkpoints))

    # ── Evaluate each checkpoint ───────────────────────────────────────────────
    panels: list[tuple[str, list, list]] = []

    for label, model_id in checkpoints:
        network.load_model(model_id=model_id)
        network.model.eval()

        gt_trajs, pred_trajs = get_trajectories(network, robot, horizon, demonstrations_eval, goal_conditioned=False)
        panels.append((label, gt_trajs, pred_trajs))
        print(f'  evaluated {label}')

    # ── Plot ──────────────────────────────────────────────────────────────────
    save_path = None
    if SAVE_PLOTS:
        save_path = ROOT_DIR / 'evaluation' / 'plots' / f'{BASE_MODEL_ID}_checkpoint_eval.png'

    fig = plot_checkpoint_grid(panels, n_x=taskspace_dim, n_cols=N_COLS, suptitle=f'Checkpoint evaluation — {task}', save_path=save_path)
    plt.show()

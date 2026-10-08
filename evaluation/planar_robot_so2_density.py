import argparse
import csv
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from utils.utils import DataVisualizer, create_two_arm_robot
from utils.networks_pytorch import CustomMLP, NETWORKS_DIR
from evaluation.planar_robot_rmse_table import check_inputs

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = NETWORKS_DIR / 'models'                       # networks/models/: where training saves checkpoints
DATASET_FOLDER = 'planar_robot/planar_robot_lasa_so2_1deg'    # under demonstrations/

TASK = 'left-LASA-CShape_right-LASA-NShape_ndofs-8'
DATASET = f'{TASK}_augmented_config_SO2.npz'      # SO2 augmentation on a 1-degree grid (test set of this evaluation)

# augmentation step [deg] -> (epochs in the checkpoint name, checkpoint iteration)
STEPS = {
    5.0: (311000, 211000), 10.0: (162000, 142000), 15.0: (112000, 112000), 30.0: (62000, 18000),
    45.0: (46000, 4000), 60.0: (38000, 2000), 75.0: (29000, 20000), 90.0: (29000, 4000),
}
N_TRIM = 60     # drop the first N steps of every test trajectory (warm-up)


def checkpoint_name(step):
    epochs, iteration = STEPS[step]
    return (f'MLP_DA_SO2_{TASK}_epochs={epochs}_nsteps=10_stride=15_augmstep={step}_'
            f'decoupled_conditioning=symmetry_seed=42_iter{iteration}')


def csv_name(step):
    return f'rmse_SO2_{step}deg_iter{STEPS[step][1]}.csv'


def load_dataset():
    """(test trajectories grouped by SO2 angle in degrees, normalized demonstrations for the network bounds)"""
    data_vis = DataVisualizer(task=TASK, conditioning='symmetry', normalize_conditioning=False)
    demos, demos_norm = data_vis.construct_demonstrations(demo_folder=DATASET_FOLDER,
                                                          load_augmented_data=True, predefined_dataset=DATASET)
    by_angle = {}
    for traj, demo_type in zip(demos['test_in'], demos['test_in_demo_type']):
        if demo_type != 'original' and 'SO2' not in demo_type:
            continue
        traj = traj[N_TRIM:]
        by_angle.setdefault(round(float(np.rad2deg(np.mean(traj[:, -3]))), 1), []).append(traj)
    return dict(sorted(by_angle.items())), demos_norm


def load_policy(step, demos_norm, robot):
    net = CustomMLP(model_id=checkpoint_name(step), load_model_flag=False, save_model_flag=False,
                    demonstrations=demos_norm, robot=robot)
    weights = CHECKPOINT_DIR / f'{checkpoint_name(step)}.pt'
    net.model.load_state_dict(torch.load(weights, map_location=device))
    net.model.eval()
    return net


def to_task_space(q, robot):
    """(T, nb_dofs) joint trajectory -> (T, 4) end-effector positions [x_left, y_left, x_right, y_right]."""
    q = torch.as_tensor(q, dtype=torch.float32).to(device)
    x_left = robot.fk_func_left_torch(q[:, :robot.nb_dofs_left])
    x_right = robot.fk_func_right_torch(q[:, robot.nb_dofs_left:robot.nb_dofs])
    return torch.cat([x_left, x_right], dim=-1).cpu().numpy()


def rmse_per_trajectory(net, trajs, robot):
    horizon = min(t.shape[0] for t in trajs) - 1
    initial_states = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in trajs]).to(device)
    with torch.no_grad():
        pred_q, _ = net.forward_denorm_multistep(horizon=horizon, inp_batch=initial_states, return_traj=True, use_grad=False)
    rmses = []
    for pred, gt in zip(pred_q[:, :, :robot.nb_dofs], trajs):
        x_pred, x_gt = to_task_space(pred, robot), to_task_space(gt[:, :robot.nb_dofs], robot)
        T = min(len(x_pred), len(x_gt))
        rmses.append(float(np.sqrt(np.mean((x_pred[:T] - x_gt[:T]) ** 2))))
    return rmses


def required_files(steps=None):
    """Checkpoints and dataset this evaluation needs (not stored in git)."""
    return ([CHECKPOINT_DIR / f'{checkpoint_name(step)}.pt' for step in (steps or STEPS)]
            + [ROOT_DIR / 'demonstrations' / DATASET_FOLDER / DATASET])


def compute(steps=None, verbose=True):
    """results[step][angle_deg] = [RMSE of each test trajectory at that angle]"""
    robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2, ee_joint=False)
    if verbose:
        print(f'loading {DATASET_FOLDER}/{DATASET} ...', flush=True)
    by_angle, demos_norm = load_dataset()
    steps = steps or list(STEPS)
    # one bar step per (step size, test angle) evaluation; the postfix names the running step
    bar = tqdm(total=len(steps) * len(by_angle), desc='SO2 density', disable=not verbose)
    results = {}
    for step in steps:
        net = load_policy(step, demos_norm, robot)
        results[step] = {}
        for angle, trajs in by_angle.items():
            bar.set_postfix_str(f'step {step:g} deg, angle {angle:g} deg')
            results[step][angle] = rmse_per_trajectory(net, trajs, robot)
            bar.update()
        if verbose:
            bar.write(f'SO2 {step:g} deg: {len(by_angle)} angles, mean RMSE {np.mean([np.mean(v) for v in results[step].values()]):.4f}')
    bar.close()
    return results


def write_csvs(results, out_dir):
    """One CSV per step size, same layout and names as the paper's cache files."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for step, by_angle in results.items():
        with open(out_dir / csv_name(step), 'w', newline='') as f:
            w = csv.writer(f)
            w.writerow(['angle_deg', 'rmse'])
            for angle in sorted(by_angle):
                for rmse in by_angle[angle]:
                    w.writerow([f'{angle:.1f}', f'{rmse:.8f}'])


def read_csvs(in_dir, steps=None):
    results = {}
    for step in (steps or STEPS):
        by_angle = {}
        with open(Path(in_dir) / csv_name(step), newline='') as f:
            for row in csv.DictReader(f):
                by_angle.setdefault(float(row['angle_deg']), []).append(float(row['rmse']))
        results[step] = by_angle
    return results


# Styles of the paper figure (one per step size, in the order of STEPS).
COLORS = ['#2196F3', '#E63946', '#FF9800', '#E91E63', '#2CA02C', '#8B4513', '#00BCD4', '#9C27B0']
PAPER_STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '#B2A5A5', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'font.family': 'STIXGeneral', 'mathtext.fontset': 'stix', 'text.usetex': False,
}


def plot(results, path):
    """Mean RMSE over the test trajectories at each angle, one line per step size."""
    import matplotlib.pyplot as plt
    with plt.rc_context(PAPER_STYLE):
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for i, (step, by_angle) in enumerate(results.items()):
            angles = sorted(by_angle)
            ax.plot(angles, [np.mean(by_angle[a]) for a in angles], color=COLORS[i % len(COLORS)],
                    linewidth=2.2, label=f'SO2 {step:g}°', zorder=3)
        ax.set_xlim(-185, 185)
        ticks = np.arange(-180, 181, 45)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f'{t:.0f}°' for t in ticks])
        ax.set_xlabel(r'$\theta$', fontsize=21)
        ax.set_ylabel('RMSE', fontsize=21)
        ax.tick_params(labelsize=21)
        fig.tight_layout()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=300, bbox_inches='tight')
        plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(ROOT_DIR / 'results' / 'planar_paper' / 'so2_density'), help='output folder (default: %(default)s)')
    p.add_argument('--steps', type=float, nargs='+', help=f'subset of step sizes in degrees (default: all of {list(STEPS)})')
    p.add_argument('--plot-only', metavar='CSV_DIR', help='skip evaluation; plot the CSVs in this folder')
    args = p.parse_args()
    steps = args.steps or list(STEPS)
    if args.plot_only:
        results = read_csvs(args.plot_only, steps)
    else:
        check_inputs(required_files(steps))
        results = compute(steps)
        write_csvs(results, args.out)
    plot(results, Path(args.out) / 'rmse_by_so2_angle.pdf')
    print(f'saved -> {args.out}')


if __name__ == '__main__':
    main()

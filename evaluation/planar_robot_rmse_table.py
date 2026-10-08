import argparse
import csv
import sys
from pathlib import Path

from tqdm import tqdm
import numpy as np
import torch

from utils.utils import DataVisualizer, create_two_arm_robot
from utils.networks_pytorch import CustomMLP, NETWORKS_DIR

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

ROOT_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = NETWORKS_DIR / 'models'                # networks/models/: where training saves checkpoints
DATASET_FOLDER = 'planar_robot/planar_robot_lasa'      # under demonstrations/: raw + augmented datasets

TASK = 'left-LASA-CShape_right-LASA-NShape_ndofs-8'
SUFFIX = 'nsteps=10_stride=15_augmstep=15.0_decoupled_conditioning=symmetry_seed=42'

# policy name -> (checkpoint stem, training dataset; None = the raw, non-augmented demonstrations)
POLICIES = {
    'Baseline':      (f'MLP_BASELINE_{TASK}_epochs=20002_{SUFFIX}_iter400', None),
    'SO2':           (f'MLP_DA_SO2_{TASK}_epochs=200000_{SUFFIX}_iter67000', f'{TASK}_augmented_config_SO2.npz'),
    'SO2Scaling2':   (f'MLP_DA_SO2Scaling2Group_{TASK}_epochs=200000_{SUFFIX}_iter179000', f'{TASK}_augmented_config_SO2Scaling2Group.npz'),
    'C2SO2Scaling2': (f'MLP_DA_C2SO2Scaling2Group_{TASK}_epochs=400000_{SUFFIX}_iter287000', f'{TASK}_augmented_config_C2SO2Scaling2Group.npz'),
}
EVAL_DATASET = f'{TASK}_augmented_config_C2SO2Scaling2Group.npz'
CATEGORIES = ['original', 'SO2', 'SO2Scaling2Group', 'C2SO2Scaling2Group']

N_TRIM = 60                     # drop the first N steps of every test trajectory (warm-up)
ANGLE_TOL = np.deg2rad(2.0)     # |angle| below this counts as "not rotated"
SCALE_TOL = 0.05                # |scale - 1| below this counts as "not scaled"


def load_policy(checkpoint, dataset, robot):
    """Build the MLP with the normalization bounds of its training dataset and load its weights."""
    data_vis = DataVisualizer(task=TASK, conditioning='symmetry', normalize_conditioning=False)
    _, demos_norm = data_vis.construct_demonstrations(demo_folder=DATASET_FOLDER,
                                                      load_augmented_data=dataset is not None,
                                                      predefined_dataset=dataset)
    net = CustomMLP(model_id=checkpoint, load_model_flag=False, save_model_flag=False,
                    demonstrations=demos_norm, robot=robot)
    weights = CHECKPOINT_DIR / f'{checkpoint}.pt'
    net.model.load_state_dict(torch.load(weights, map_location=device))
    net.model.eval()
    return net


def build_test_pool():
    """All trajectories (train + test) of the evaluation dataset, trimmed and split into nested categories."""
    data_vis = DataVisualizer(task=TASK, conditioning='symmetry', normalize_conditioning=False)
    demos, _ = data_vis.construct_demonstrations(demo_folder=DATASET_FOLDER,
                                                 load_augmented_data=True, predefined_dataset=EVAL_DATASET)
    pool = {category: [] for category in CATEGORIES}
    for traj in demos['train_in'] + demos['test_in']:
        traj = traj[N_TRIM:]
        angle, scale, refl = np.mean(traj[:, -3]), np.mean(traj[:, -2]), np.mean(traj[:, -1])
        pool['C2SO2Scaling2Group'].append(traj)
        if refl > 0:
            pool['SO2Scaling2Group'].append(traj)
            if abs(scale - 1.0) < SCALE_TOL:
                pool['SO2'].append(traj)
                if abs(angle) < ANGLE_TOL:
                    pool['original'].append(traj)
    return pool


def to_task_space(q, robot):
    """(T, nb_dofs) joint trajectory -> (T, 4) end-effector positions [x_left, y_left, x_right, y_right]."""
    q = torch.as_tensor(q, dtype=torch.float32).to(device)
    x_left = robot.fk_func_left_torch(q[:, :robot.nb_dofs_left])
    x_right = robot.fk_func_right_torch(q[:, robot.nb_dofs_left:robot.nb_dofs])
    return torch.cat([x_left, x_right], dim=-1).cpu().numpy()


def evaluate(net, trajs, robot):
    """Roll out the policy from each trajectory's first state; RMSE (and MSE spread) in task space."""
    horizon = min(t.shape[0] for t in trajs) - 1
    initial_states = torch.stack([torch.tensor(t[0], dtype=torch.float32) for t in trajs]).to(device)
    with torch.no_grad():
        pred_q, _ = net.forward_denorm_multistep(horizon=horizon, inp_batch=initial_states, return_traj=True, use_grad=False)
    mses = []
    for pred, gt in zip(pred_q[:, :, :robot.nb_dofs], trajs):
        x_pred, x_gt = to_task_space(pred, robot), to_task_space(gt[:, :robot.nb_dofs], robot)
        T = min(len(x_pred), len(x_gt))
        mses.append(float(np.mean((x_gt[:T] - x_pred[:T]) ** 2)))
    mses = np.array(mses)
    return {'rmse': float(np.sqrt(np.mean(mses))), 'std': float(np.std(mses)), 'n': len(mses)}


def required_files():
    """Checkpoints and datasets this evaluation needs (not stored in git)."""
    data = ROOT_DIR / 'demonstrations' / DATASET_FOLDER
    return ([CHECKPOINT_DIR / f'{checkpoint}.pt' for checkpoint, _ in POLICIES.values()]
            + [data / f'{TASK}.npz'] + [data / dataset for _, dataset in POLICIES.values() if dataset])


def check_inputs(files):
    """Exit with a clear message if datasets/checkpoints (not stored in git) have not been downloaded."""
    missing = [f for f in files if not f.exists()]
    if missing:
        sys.exit('Missing inputs (see the README for how to download the datasets and checkpoints):\n  '
                 + '\n  '.join(str(f.relative_to(ROOT_DIR)) for f in missing))


def compute_table(verbose=True):
    """results[policy][category] = {'rmse', 'std', 'n'}"""
    robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2, ee_joint=False)
    # one bar step per dataset load and per (policy, category) evaluation; the postfix names the running step
    bar = tqdm(total=1 + len(POLICIES) * (1 + len(CATEGORIES)), desc='RMSE table', disable=not verbose)
    bar.set_postfix_str('loading test pool')
    pool = build_test_pool()
    bar.update()
    if verbose:
        bar.write('test pool: ' + ', '.join(f'{c}={len(t)}' for c, t in pool.items()))
    results = {}
    for name, (checkpoint, dataset) in POLICIES.items():
        bar.set_postfix_str(f'loading {name}')
        net = load_policy(checkpoint, dataset, robot)
        bar.update()
        results[name] = {}
        for category in CATEGORIES:
            bar.set_postfix_str(f'{name} on {category}')
            results[name][category] = evaluate(net, pool[category], robot)
            bar.update()
        if verbose:
            bar.write(f'{name:14s} ' + '  '.join(f'{c}: {r["rmse"]:.3f}±{r["std"]:.3f}' for c, r in results[name].items()))
    bar.close()
    return results


def write_csv(results, path):
    """Same layout as the paper's cross_eval_rmse.csv."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['network'] + CATEGORIES + [f'{c}_std' for c in CATEGORIES] + [f'{c}_n' for c in CATEGORIES])
        for name, row in results.items():
            w.writerow([name] + [f'{row[c]["rmse"]:.6f}' for c in CATEGORIES]
                       + [f'{row[c]["std"]:.6f}' for c in CATEGORIES] + [str(row[c]['n']) for c in CATEGORIES])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(ROOT_DIR / 'results' / 'planar_paper'), help='output folder (default: %(default)s)')
    args = p.parse_args()
    check_inputs(required_files())
    results = compute_table()
    out = Path(args.out) / 'cross_eval_rmse.csv'
    write_csv(results, out)
    print(f'saved -> {out}')


if __name__ == '__main__':
    main()

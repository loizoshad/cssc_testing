import os
import copy
from pathlib import Path
import numpy as np
import random
import torch
import shutil # Just making things prettier when printing in the console
import matplotlib.pyplot as plt

from utils.groups import *
from utils.utils import *
from utils.networks_pytorch import CustomMLP

cols = shutil.get_terminal_size().columns
np.set_printoptions(precision = 8, suppress = True, linewidth=cols)
seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.resolve()

def split_train_test(demonstrations, demonstrations_normalized,
                     so2_train_step=None,
                     scaling_train_values=None,
                     original_train_indices=None,
                     max_original_demos=None,
                     max_demos_per_cell=None,
                     axis_match_tolerance=0.05):
    """
    Split augmented demonstrations into train / test sets.

    For the non-'original' demo types, trajectories are first partitioned by the
    sign of their reflection conditioning variable (index -1):
      - positive reflection group  (refl ≈ +1, i.e. non-reflected)
      - negative reflection group  (refl ≈ -1, i.e. reflected)

    Within each reflection group, a 2D grid is built over (SO2, Scaling2)
    conditioning values.  Training cells are the intersection of:
      - SO2 rows    selected by matching targets spaced every `so2_train_step`
                    radians across the full SO2 range present in the data
      - Scaling2 columns selected by matching the explicit list `scaling_train_values`

    The same grid selection logic is applied independently to each reflection
    group, and the train_local sets from both groups are combined.

    For each axis the closest grid value within `axis_match_tolerance` is used;
    values with no sufficiently close match are silently skipped.

    Parameters
    ----------
    so2_train_step : float (radians)
        Spacing between training SO2 targets, e.g. np.deg2rad(20).
    scaling_train_values : list of float
        Explicit scaling values to include as training columns,
        e.g. [0.1, 0.45, 0.80].
    original_train_indices : list of int or None
        Explicit 0-based positions (in the original group) to keep in training.
        Takes priority over max_original_demos.  None means use max_original_demos
        (or all originals if that is also None).
    max_original_demos : int or None
        How many original demos to keep in training (randomly sampled).
        Ignored when original_train_indices is provided.  None = keep all.
    max_demos_per_cell : int or None
        Maximum number of demos taken from each training (SO2, Scaling2) cell,
        applied independently within each reflection group.
        Excess demos in a training cell are sent to test instead.
        None = keep all demos in each training cell.
    axis_match_tolerance : float
        Maximum distance between a target value and the nearest grid value for
        the grid value to be considered a match.
    """
    demonstrations = copy.deepcopy(demonstrations)
    demonstrations_normalized = copy.deepcopy(demonstrations_normalized)

    new_demonstrations = {key: [] for key in ['train_in', 'train_out', 'test_in', 'test_out',
                                                'train_in_demo_type', 'train_out_demo_type',
                                                'test_in_demo_type', 'test_out_demo_type']}
    new_demonstrations_norm = {key: [] for key in ['train_in', 'train_out', 'test_in', 'test_out',
                                                    'train_in_demo_type', 'train_out_demo_type',
                                                    'test_in_demo_type', 'test_out_demo_type']}

    for key in demonstrations:
        if key not in new_demonstrations:
            new_demonstrations[key] = demonstrations[key]
    for key in demonstrations_normalized:
        if key not in new_demonstrations_norm:
            new_demonstrations_norm[key] = demonstrations_normalized[key]

    # Carry over any already-assigned test demos unchanged
    for key in ['test_in', 'test_out', 'test_in_demo_type', 'test_out_demo_type']:
        new_demonstrations[key] = list(demonstrations[key])
        new_demonstrations_norm[key] = list(demonstrations_normalized.get(key, []))

    # Group current train indices by demo type (preserving insertion order)
    type_to_indices = {}
    for i, demo_type in enumerate(demonstrations['train_in_demo_type']):
        type_to_indices.setdefault(demo_type, []).append(i)

    def cond_mean(global_i):
        traj = demonstrations['train_in'][global_i]
        return traj.mean(axis=0) if hasattr(traj, 'mean') else np.mean(traj, axis=0)

    def so2_scaling2_cell(global_i):
        m = cond_mean(global_i)
        return (round(float(m[-3]), 6), round(float(m[-2]), 6))

    def reflection_sign(global_i):
        """Returns +1 if the reflection conditioning variable is non-negative, -1 otherwise."""
        return +1 if float(cond_mean(global_i)[-1]) >= 0 else -1

    def match_targets_to_grid(grid_vals, targets, tol):
        """Return the subset of grid_vals closest to each target, within tol."""
        matched = set()
        for t in targets:
            closest = min(grid_vals, key=lambda v: abs(v - t))
            if abs(closest - t) <= tol:
                matched.add(closest)
        return matched

    def select_train_from_grid(grid, label):
        """
        Given a 2D grid {(so2, scaling2): [local_i, ...]}, apply the SO2/Scaling2
        selection and return the set of local indices assigned to training.
        """
        so2_vals      = sorted({k[0] for k in grid})
        scaling2_vals = sorted({k[1] for k in grid})

        so2_targets = np.arange(so2_vals[0], so2_vals[-1] + so2_train_step / 2, so2_train_step)
        train_so2_vals     = match_targets_to_grid(so2_vals,     so2_targets,          axis_match_tolerance)
        train_scaling2_vals = match_targets_to_grid(scaling2_vals, scaling_train_values, axis_match_tolerance)

        print(f'  [{label}] SO2 train ({len(train_so2_vals)}/{len(so2_vals)}): '
              f'{sorted(round(np.rad2deg(v), 1) for v in train_so2_vals)}°')
        print(f'  [{label}] Scaling2 train ({len(train_scaling2_vals)}/{len(scaling2_vals)}): '
              f'{sorted(round(v, 3) for v in train_scaling2_vals)}')

        train_local = set()
        for (so2_val, scaling2_val), cell_local in grid.items():
            if so2_val in train_so2_vals and scaling2_val in train_scaling2_vals:
                if max_demos_per_cell is not None and len(cell_local) > max_demos_per_cell:
                    kept = random.sample(cell_local, max_demos_per_cell)
                else:
                    kept = cell_local
                train_local.update(kept)
        return train_local

    for demo_type, indices in type_to_indices.items():
        if demo_type == 'original':
            if original_train_indices is not None:
                train_local = set(original_train_indices)
            elif max_original_demos is not None:
                n_keep = min(max_original_demos, len(indices))
                train_local = set(random.sample(range(len(indices)), n_keep))
                print(f'  original: keeping {n_keep}/{len(indices)} demos in train')
            else:
                train_local = set(range(len(indices)))
        else:
            print(f'\n  demo_type={demo_type}')

            # Partition by reflection sign, building one 2D grid per sign
            grid_pos = {}   # refl ≈ +1
            grid_neg = {}   # refl ≈ -1
            for local_i, global_i in enumerate(indices):
                cell = so2_scaling2_cell(global_i)
                if reflection_sign(global_i) >= 0:
                    grid_pos.setdefault(cell, []).append(local_i)
                else:
                    grid_neg.setdefault(cell, []).append(local_i)

            train_local = set()
            if grid_pos:
                train_local |= select_train_from_grid(grid_pos, 'refl=+1')
            if grid_neg:
                train_local |= select_train_from_grid(grid_neg, 'refl=-1')

        for local_i, global_i in enumerate(indices):
            dest = 'train' if local_i in train_local else 'test'
            new_demonstrations[f'{dest}_in'].append(demonstrations['train_in'][global_i])
            new_demonstrations[f'{dest}_out'].append(demonstrations['train_out'][global_i])
            new_demonstrations[f'{dest}_in_demo_type'].append(demo_type)
            new_demonstrations[f'{dest}_out_demo_type'].append(demo_type)
            new_demonstrations_norm[f'{dest}_in'].append(demonstrations_normalized['train_in'][global_i])
            new_demonstrations_norm[f'{dest}_out'].append(demonstrations_normalized['train_out'][global_i])
            new_demonstrations_norm[f'{dest}_in_demo_type'].append(demo_type)
            new_demonstrations_norm[f'{dest}_out_demo_type'].append(demo_type)

    return new_demonstrations, new_demonstrations_norm


def choose_demos_by_conditioning(demonstrations, queries, use_train=None, tolerance=0.15):
    """
    Select demonstrations by matching specific symmetry conditioning variable values.

    Parameters
    ----------
    demonstrations : dict
        The demonstrations dict (e.g., network.demonstrations).
    queries : list of (so2_deg, scaling, reflection, idx) tuples
        so2_deg    – target SO2 angle in **degrees** (converted to radians internally)
        scaling    – target scaling factor
        reflection – target reflection indicator (e.g. 1 or -1)
        idx        – 0-based index into the group of matched trajectories
    use_train : bool or None
        True  → search only train_in
        False → search only test_in
        None  → search both
    tolerance : float
        Maximum L∞ distance (in natural units: radians, scaling, reflection) for a
        valid match. Queries whose nearest grid point exceeds this are skipped with
        a warning.

    Returns
    -------
    list of trajectories
    """
    if use_train is None:
        pool = demonstrations['train_in'] + demonstrations['test_in']
    elif use_train:
        pool = demonstrations['train_in']
    else:
        pool = demonstrations['test_in']

    # Group trajectories by their rounded (SO2, Scaling2, Reflection) key
    groups = {}
    for traj in pool:
        # mean_cond = np.mean(traj, axis=0) if not hasattr(traj, 'numpy') else traj.numpy().mean(axis=0)
        mean_cond = np.mean(traj, axis=0) if not hasattr(traj, 'numpy') else traj.cpu().numpy().mean(axis=0)
        key = tuple(round(float(mean_cond[i]), 6) for i in (-3, -2, -1))
        groups.setdefault(key, []).append(traj)

    result = []
    for so2_deg, scaling, reflection, idx in queries:
        target = np.array([np.deg2rad(float(so2_deg)), float(scaling), float(reflection)])

        # Find the closest group by L∞ distance
        best_key, best_dist = None, float('inf')
        for key in groups:
            dist = float(np.max(np.abs(np.array(key) - target)))
            if dist < best_dist:
                best_dist, best_key = dist, key

        if best_key is None or best_dist > tolerance:
            print(f'[WARN] No match within tolerance={tolerance} for query '
                  f'({so2_deg}°, {scaling}, {reflection}). '
                  f'Closest dist={best_dist:.4f} at key={best_key}')
            continue

        matching = groups[best_key]
        if idx >= len(matching):
            print(f'[WARN] Index {idx} out of range for query '
                  f'({so2_deg}°, {scaling}, {reflection}) — '
                  f'only {len(matching)} trajectories matched.')
            continue

        result.append(matching[idx])

    return result


if __name__ == '__main__':
    ############################################################################################################
    # Initialize pybullet and robot
    ############################################################################################################
    is_taskspace = False
    robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2, ee_joint=False)

    demo_folder = 'planar_robot/planar_robot_lasa'    
    demo_type_left = 'LASA'; demo_name_left = 'PShape'
    demo_type_right = 'LASA'; demo_name_right = 'NShape'

    # Load the dual arm data with name following the convention: demos_name = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{nb_dofs}.npz'
    task = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{robot.nb_dofs}'
    task = task + '_taskspace' if is_taskspace else task
    ############################################################################################################
    # Network settings
    ############################################################################################################    
    # num_iterations = 100000
    num_iterations = 1000

    step = 5.0
    nb_steps = 15
    stride = 10
    batch_size = 250
    
    train = True
    load_model_flag = False
    save_model_flag = True
    save_plots = True
    save_loss = False

    data_augmentation = True
    decouple_arms = True
    isotropic_normalization = False

    conditioning='symmetry'
    normalize_conditioning = False
    task_space_loss = False

    G = C2SO2Scaling2Group()
    rep_in = C2SO2Scaling2DualArmConfigTaskRepIn( robot=robot, G=G, is_vf_constant=False, dt=0.1)
    rep_out = C2SO2Scaling2DualArmConfigTaskRepOut(robot=robot, G=G, is_vf_constant=False, dt=0.1)

    model_id = f'test_augm_density_MLP_DA_{G}_{task}_epochs={str(num_iterations)}_nsteps={str(nb_steps)}_stride={stride}_augmstep={step}_'

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

    predefined_dataset = None
    ############################################################################################################
    # Get data and initialize network
    ############################################################################################################
    data_vis = DataVisualizer(task=task, isotropic_normalization=isotropic_normalization, conditioning=conditioning, normalize_conditioning=normalize_conditioning, group=G)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(demo_folder=demo_folder, load_augmented_data=True, predefined_dataset=predefined_dataset)
    if data_augmentation:
        num_c2so2scaling2 = sum([1 for demo_type in demonstrations['train_in_demo_type'] if 'C2SO2Scaling2Group' in demo_type])
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, 
                                                                                      train_demo_types=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'], 
                                                                                      num_train_demos_per_type=[7, 0, 0, 0, num_c2so2scaling2])

    so2_train_step = np.deg2rad(step) # e.g., one training angle every 20°
    scaling_train_values = list(np.arange(1.0, 0.1-0.01, -np.deg2rad(step))) # e.g., 5 evenly spaced scaling values from 0.1 to 1.0
    demonstrations, demonstrations_normalized = split_train_test(
        demonstrations, demonstrations_normalized,
        so2_train_step=so2_train_step,
        scaling_train_values=scaling_train_values,
        original_train_indices=None,   # or e.g. [0, 2, 4] for explicit picks
        max_original_demos=5,       # or e.g. 5  to keep only 5 of the originals
        max_demos_per_cell=5,       # or e.g. 5  to cap demos per (SO2, Scaling2) cell
        axis_match_tolerance=np.deg2rad(step) / 2,   # half the augmentation step
    )

    network = CustomMLP(model_id=model_id, load_model_flag=load_model_flag, save_model_flag=save_model_flag, 
                        demonstrations=demonstrations_normalized, 
                        robot=robot)

    model = network.model
    data_vis.set_network(network)
    ############################################################################################################
    # Train or load network
    ############################################################################################################
    if train:
        network.train_network(robot, num_iterations=num_iterations, steps=nb_steps, stride=stride, verbose=True, batch_size=batch_size, verbose_every=100, decouple_arms=decouple_arms, task_space_loss=task_space_loss)
    if network.save_model_flag:
        network.save_model(model_id=network.model_id)

    # ###########################################################################################################
    # Evaluate network's performance - Full trajectory simulation
    # ###########################################################################################################
    horizon = min([traj.shape[0] for traj in network.demonstrations['train_in']])

    #### Choose randomly
    demo_config_train = dict(
        demo_labels=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
        # num_of_demos = [7, 0, 0, 0, 0]
        num_of_demos = [1, 0, 0, 0, 5]
    )
    demo_config_test = dict(
        demo_labels=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
        # num_of_demos = [7, 0, 0, 0, 0]
        num_of_demos = [1, 0, 0, 0, 5]
    )    

    demonstrations_eval_test = choose_demos_for_evaluation(network.demonstrations, config=demo_config_test, use_train = False)
    demonstrations_eval_train = choose_demos_for_evaluation(network.demonstrations, config=demo_config_train, use_train = True)

    evaluate_full_trajectory(network, data_vis, robot, horizon, demonstrations_eval_test, demonstrations_eval_train, title_prefix='Full Evaluation', save_plots=save_plots, save_trajs=False, goal_conditioned=False)
        
    plt.show()


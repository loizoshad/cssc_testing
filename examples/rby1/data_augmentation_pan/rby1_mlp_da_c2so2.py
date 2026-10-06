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

    For the non-'original' demo type a 2D grid is built over (SO2, Scaling2)
    conditioning values.  Training cells are the intersection of:
      - SO2 rows   selected by matching targets spaced every `so2_train_step` radians
                   across the full SO2 range present in the data
      - Scaling2 columns selected by matching the explicit list `scaling_train_values`

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
        Maximum number of demos taken from each training (SO2, Scaling2) cell.
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

    def so2_scaling2_cell(global_i):
        traj = demonstrations['train_in'][global_i]
        mean = traj.mean(axis=0) if hasattr(traj, 'mean') else np.mean(traj, axis=0)
        return (round(float(mean[-3]), 6), round(float(mean[-2]), 6))

    def match_targets_to_grid(grid_vals, targets, tol):
        """Return the subset of grid_vals closest to each target, within tol."""
        matched = set()
        for t in targets:
            closest = min(grid_vals, key=lambda v: abs(v - t))
            if abs(closest - t) <= tol:
                matched.add(closest)
        return matched

    for demo_type, indices in type_to_indices.items():
        if demo_type == 'original':
            if original_train_indices is not None:
                # Explicit index list takes priority
                train_local = set(original_train_indices)
            elif max_original_demos is not None:
                n_keep = min(max_original_demos, len(indices))
                train_local = set(random.sample(range(len(indices)), n_keep))
                print(f'  original: keeping {n_keep}/{len(indices)} demos in train')
            else:
                train_local = set(range(len(indices)))
        else:
            # Build 2D grid: (so2_val, scaling2_val) → [local_i, ...]
            grid = {}
            for local_i, global_i in enumerate(indices):
                cell = so2_scaling2_cell(global_i)
                grid.setdefault(cell, []).append(local_i)

            so2_vals      = sorted({k[0] for k in grid})
            scaling2_vals = sorted({k[1] for k in grid})

            # --- SO2 axis: targets spaced every so2_train_step across the data range ---
            so2_targets = np.arange(so2_vals[0], so2_vals[-1] + so2_train_step / 2, so2_train_step)
            train_so2_vals = match_targets_to_grid(so2_vals, so2_targets, axis_match_tolerance)

            # --- Scaling2 axis: explicitly specified values ---
            train_scaling2_vals = match_targets_to_grid(scaling2_vals, scaling_train_values, axis_match_tolerance)

            print(f'  SO2 train values ({len(train_so2_vals)}/{len(so2_vals)}):      '
                  f'{sorted(np.round(np.rad2deg(v), 0) for v in train_so2_vals)}°')
            # In rad
            print(f'  SO2 train values ({len(train_so2_vals)}/{len(so2_vals)}):      '
                  f'{sorted(np.round(v, 3) for v in train_so2_vals)} rad')
            print(f'  Scaling2 train values ({len(train_scaling2_vals)}/{len(scaling2_vals)}): '
                  f'{sorted(np.round(v, 3) for v in train_scaling2_vals)}')

            # Intersection: a cell trains only if BOTH its SO2 and Scaling2 are selected.
            # If max_demos_per_cell is set, randomly keep that many; extras go to test.
            train_local = set()
            for (so2_val, scaling2_val), cell_local in grid.items():
                if so2_val in train_so2_vals and scaling2_val in train_scaling2_vals:
                    if max_demos_per_cell is not None and len(cell_local) > max_demos_per_cell:
                        kept = random.sample(cell_local, max_demos_per_cell)
                    else:
                        kept = cell_local
                    train_local.update(kept)

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


if __name__ == '__main__':
    ############################################################################################################
    # Initialize pybullet and robot
    ############################################################################################################
    is_taskspace = False
    robot = create_rby1_robot(task_space='xy', ee_joint=False, is_taskspace=is_taskspace)
    demo_folder = 'rby1'
    task = 'left-rby1-all_right-rby1-all_ndofs-14'
    task = task + '_taskspace' if is_taskspace else task
    ############################################################################################################
    # Network settings
    ############################################################################################################    
    # num_iterations = 100000
    num_iterations = 10000

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

    conditioning='symmetry'
    normalize_conditioning = False

    dt = np.deg2rad(1.0)
    G = C2SO2RBY1()
    rep_in = C2SO2RBY1RepIn(robot=robot, G=G, is_vf_constant=False, dt=dt)
    rep_out = C2SO2RBY1RepOut(robot=robot, G=G, is_vf_constant=False, dt=dt)

    model_id = f'MLP_DA_{G}_{task}_epochs={str(num_iterations)}_nsteps={str(nb_steps)}_stride={stride}_augmstep={step}_'


    if decouple_arms:
        model_id += 'decoupled'
    else:
        model_id += 'coupled'

    # Add the condioning type to the model
    model_id += f'_conditioning={conditioning}' if conditioning is not None else '_no_conditioning'
    model_id += f'_normalize_conditioning={normalize_conditioning}' if normalize_conditioning else ''
    # add the seed
    model_id += f'_seed={seed}_'

    predefined_dataset = None
    ############################################################################################################
    # Get data and initialize network
    ############################################################################################################
    data_vis = DataVisualizer(task=task, conditioning=conditioning, normalize_conditioning=normalize_conditioning, group=G)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(demo_folder=demo_folder, load_augmented_data=True, predefined_dataset=predefined_dataset)
    if data_augmentation:
        num_original = sum([1 for demo_type in demonstrations['train_in_demo_type'] if 'original' in demo_type])
        num_c2so2 = sum([1 for demo_type in demonstrations['train_in_demo_type'] if 'C2SO2RBY1' in demo_type])
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, 
                                                                                      train_demo_types=['original', 'C2SO2RBY1'], 
                                                                                      num_train_demos_per_type=[num_original, num_c2so2])

    network = CustomMLP(model_id=model_id, load_model_flag=load_model_flag, save_model_flag=save_model_flag, demonstrations=demonstrations_normalized, robot=robot)

    model = network.model
    data_vis.set_network(network)
    ############################################################################################################
    # Train or load network
    ############################################################################################################
    if train:
        network.train_network(robot, num_iterations=num_iterations, steps=nb_steps, stride=stride, verbose=True, batch_size=batch_size, verbose_every=100, decouple_arms=decouple_arms)
    if network.save_model_flag:
        network.save_model(model_id=network.model_id)

    data_vis.plot_loss_history(network, save_plots=False)

    # ###########################################################################################################
    # Evaluate network's performance - Full trajectory simulation
    # ###########################################################################################################
    horizon = min([traj.shape[0] for traj in network.demonstrations['train_in']])

    #### Choose randomly
    demo_config_train = dict(demo_labels=['original', 'C2SO2RBY1'], num_of_demos = [2, 8])

    demonstrations_eval_train = choose_demos_for_evaluation(network.demonstrations, config=demo_config_train, use_train = True)
    demonstrations_eval_test = demonstrations_eval_train # NOTE: We don't care about test trajectories at the moment. Just trying to get proof of concept.
    demonstrations_eval_test = []

    evaluate_full_trajectory(network, data_vis, robot, horizon, demonstrations_eval_test, demonstrations_eval_train, title_prefix='Full Evaluation', save_plots=save_plots, save_trajs=False, goal_conditioned=False)
        
    plt.show()








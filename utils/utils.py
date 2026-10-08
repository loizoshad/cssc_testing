import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from matplotlib.patches import Circle
import random
import time
import torch
from typing import List
import os
from pathlib import Path
import copy
import time

from torch.func import jacrev, vmap

from utils.data_smoothing import smooth_and_crop_data
from utils.robots import TwoArmsPlanarManipulator, TwoArmsRBY1
from utils.vector_fields import *
# from utils.visualization_tools import *
from utils.visualization_tools import plot_equivariance_sampling_bounds
from robots.planar_robot.visualization import plot_planar_robot
from robots.planar_robot.kinematics import *

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Circle
import torch


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.resolve()

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def split_train_test(demonstrations, demonstrations_normalized,
                     so2_train_step=None,
                     scaling_train_values=None,
                     original_train_indices=None,
                     max_original_demos=None,
                     max_demos_per_cell=None,
                     so2_match_tolerance=0.05,
                     scaling_match_tolerance=0.05):
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

    For each axis the closest grid value within the respective tolerance is used;
    values with no sufficiently close match are silently skipped.

    Parameters
    ----------
    so2_train_step : float (radians)
        Spacing between training SO2 targets, e.g. np.deg2rad(20).
        None → keep ALL augmented rotation angles.
    scaling_train_values : list of float or None
        Explicit scaling conditioning values to include as training columns,
        e.g. [0.7, 0.85, 1.0].  None → keep ALL augmented scale values.
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
    so2_match_tolerance : float (radians)
        Maximum distance between an SO2 target and the nearest grid value.
        Typically set to half the augmentation step size (e.g. deg2rad(5)/2 for
        a 5° augmentation grid).  Default 0.05 rad ≈ 2.9°.
    scaling_match_tolerance : float (dimensionless)
        Maximum distance between a scaling target and the nearest grid value.
        Should be set relative to the augmentation step size on the scale axis
        (e.g. 0.05 for a grid spaced at 0.1 units).  Default 0.05.
        Kept separate from so2_match_tolerance because the two axes use
        different units (radians vs. dimensionless ratio).
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

        None values for so2_train_step / scaling_train_values mean "keep all".
        The two axes use independent tolerances (so2_match_tolerance for the
        rotation axis in radians, scaling_match_tolerance for the scale axis).
        """
        so2_vals      = sorted({k[0] for k in grid})
        scaling2_vals = sorted({k[1] for k in grid})

        if so2_train_step is None:
            train_so2_vals = set(so2_vals)
        else:
            so2_targets    = np.arange(so2_vals[0], so2_vals[-1] + so2_train_step / 2, so2_train_step)
            train_so2_vals = match_targets_to_grid(so2_vals, so2_targets, so2_match_tolerance)

        train_scaling2_vals = (set(scaling2_vals) if scaling_train_values is None
                               else match_targets_to_grid(scaling2_vals, scaling_train_values, scaling_match_tolerance))

        print(f'  [{label}] SO2 train ({len(train_so2_vals)}/{len(so2_vals)}): '
              f'{"all" if so2_train_step is None else sorted(round(np.rad2deg(v), 1) for v in train_so2_vals)}°')
        print(f'  [{label}] Scaling2 train ({len(train_scaling2_vals)}/{len(scaling2_vals)}): '
              f'{"all" if scaling_train_values is None else sorted(round(v, 4) for v in train_scaling2_vals)}')

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


######################

def choose_demos_for_evaluation(demonstrations: dict, config: dict,
                                 max_demos_per_label: int = 1,
                                 indices_per_label: dict = None, use_train = None):
    """
    Choose demonstrations for evaluation based on the provided configuration.
    If indices_per_label is provided, those specific indices are used instead
    of uniform sampling for the corresponding labels.
    """
    # Build per-label pool
    demo_dict = {label: [] for label in config['demo_labels']}
    if use_train is None:
        for label in config['demo_labels']:
            for i, _ in enumerate(demonstrations['train_in']):
                if label == demonstrations['train_in_demo_type'][i]:
                    demo_dict[label].append(demonstrations['train_in'][i])
            for i, _ in enumerate(demonstrations['test_in']):
                if label == demonstrations['test_in_demo_type'][i]:
                    demo_dict[label].append(demonstrations['test_in'][i])
    elif use_train:
        for label in config['demo_labels']:
            for i, _ in enumerate(demonstrations['train_in']):
                if label == demonstrations['train_in_demo_type'][i]:
                    demo_dict[label].append(demonstrations['train_in'][i])
    elif use_train == False:
        # use only test demos
        for label in config['demo_labels']:
            for i, _ in enumerate(demonstrations['test_in']):
                if label == demonstrations['test_in_demo_type'][i]:
                    demo_dict[label].append(demonstrations['test_in'][i])

    demonstrations_eval = []
    for label, num in zip(config['demo_labels'], config['num_of_demos']):
        pool = demo_dict[label]

        # ── Specific indices requested ────────────────────────────────────────
        if indices_per_label is not None and label in indices_per_label:
            requested = indices_per_label[label]
            for idx in requested:
                if idx < len(pool):
                    demonstrations_eval.append(pool[idx])
                else:
                    print(f'  [WARN] Index {idx} out of range for label '
                          f'"{label}" (pool size={len(pool)}) — skipped.')
            continue

        # ── Default: uniform sampling ─────────────────────────────────────────
        if num == -1:
            demonstrations_eval.extend(pool[:max_demos_per_label])
        elif num > 0 and pool:
            indices = np.linspace(0, len(pool) - 1,
                                  num=min(num, len(pool)), dtype=int)
            for idx in indices:
                demonstrations_eval.append(pool[idx])

    return demonstrations_eval


def choose_demos_for_evaluation_original(demonstrations: dict, config: dict,
                                 max_demos_per_label: int = 1,
                                 indices_per_label: dict = None):
    """
    Choose demonstrations for evaluation based on the provided configuration.
    If indices_per_label is provided, those specific indices are used instead
    of uniform sampling for the corresponding labels.
    """
    # Build per-label pool
    demo_dict = {label: [] for label in config['demo_labels']}
    for label in config['demo_labels']:
        for i, _ in enumerate(demonstrations['train_in']):
            if label == demonstrations['train_in_demo_type'][i]:
                demo_dict[label].append(demonstrations['train_in'][i])
        for i, _ in enumerate(demonstrations['test_in']):
            if label == demonstrations['test_in_demo_type'][i]:
                demo_dict[label].append(demonstrations['test_in'][i])

    demonstrations_eval = []
    for label, num in zip(config['demo_labels'], config['num_of_demos']):
        pool = demo_dict[label]

        # ── Specific indices requested ────────────────────────────────────────
        if indices_per_label is not None and label in indices_per_label:
            requested = indices_per_label[label]
            for idx in requested:
                if idx < len(pool):
                    demonstrations_eval.append(pool[idx])
                else:
                    print(f'  [WARN] Index {idx} out of range for label '
                          f'"{label}" (pool size={len(pool)}) — skipped.')
            continue

        # ── Default: uniform sampling ─────────────────────────────────────────
        if num == -1:
            demonstrations_eval.extend(pool[:max_demos_per_label])
        elif num > 0 and pool:
            indices = np.linspace(0, len(pool) - 1,
                                  num=min(num, len(pool)), dtype=int)
            for idx in indices:
                demonstrations_eval.append(pool[idx])

    return demonstrations_eval


def _to_tensor_bounds(state, x_min, x_max):
    """Convert numpy bounds to torch tensors matching state's dtype and device."""
    if isinstance(state, torch.Tensor):
        if isinstance(x_min, np.ndarray):
            x_min = torch.tensor(x_min, dtype=state.dtype, device=state.device)
        if isinstance(x_max, np.ndarray):
            x_max = torch.tensor(x_max, dtype=state.dtype, device=state.device)
    return x_min, x_max

def normalize_state(state, x_min, x_max):
    """Normalize state to [-1, 1] using per-variable bounds."""
    x_min, x_max = _to_tensor_bounds(state, x_min, x_max)
    state = (((state - x_min) / (x_max - x_min)) - 0.5) * 2
    return state

def denormalize_state(state, x_min, x_max):
    """Denormalize state from [-1, 1] using per-variable bounds."""
    x_min, x_max = _to_tensor_bounds(state, x_min, x_max)
    state = ((state / 2) + 0.5) * (x_max - x_min) + x_min
    return state

def denormalize_state_derivative(state_dot, x_min, x_max):
    x_min, x_max = _to_tensor_bounds(state_dot, x_min, x_max)
    scale = 0.5 * (x_max - x_min)
    return state_dot * scale


class RobotHandler:
    '''
    This is just a convenience class to bring all the properties of the robot on the same level of abstraction and make it easier to pass around the robot as a single object.
    '''
    def __init__(self, robot):
        self.robot = robot
        self.robot_left = robot.robot_left_arm
        self.robot_right = robot.robot_right_arm

        # Set task space string for easy access
        self.task_space = 'xy' # this is the default.

        # update if the robot has the task_space attribute
        if hasattr(robot.robot_left_arm, 'task_space'):
            self.task_space = robot.robot_left_arm.task_space
            print(f'[INFO] RobotHandler: detected task space "{self.task_space}" from robot attribute.')

        # Forward kinematic functions
        self.fk_func_left = self.robot_left.fkine
        self.fk_func_right = self.robot_right.fkine
        # Jacobians
        self.jacob0_left = self.robot_left.jacob0
        self.jacob0_right = self.robot_right.jacob0

        # Torch Forward kinematic functions
        self.fk_func_left_torch = self.robot_left.fkine_torch
        self.fk_func_right_torch = self.robot_right.fkine_torch
        # Torch Jacobians
        self.jacob0_left_torch = self.robot_left.jacob0_torch
        self.jacob0_right_torch = self.robot_right.jacob0_torch

        # Degree of freedom in joint space
        self.nb_dofs_left = self.robot_left.nb_dofs if not self.robot_left.ee_joint else self.robot_left.nb_dofs + 1
        self.nb_dofs_right = self.robot_right.nb_dofs if not self.robot_right.ee_joint else self.robot_right.nb_dofs + 1
        self.nb_dofs = self.nb_dofs_left + self.nb_dofs_right        
        
        # Dimension of the task space
        self.nb_x_left = self.robot_left.nb_x
        self.nb_x_right = self.robot_right.nb_x
        self.nb_x = self.nb_x_left + self.nb_x_right

        # Auxiliary properties
        self.arm_length = self.robot_left.arm_length # Assuming all joints for both arms have the same length for now, but we can easily change this if needed.
        self.ee_joint = self.robot_left.ee_joint # Assuming both arms either have or don't have the ee_joint, but we can easily change this if needed.

        # Check if we are working only in task space
        if self.robot_left.is_taskspace != self.robot_right.is_taskspace:
            raise ValueError('The left and right arms must both be either in task space or configuration space. Mixed spaces are not supported yet.')
        self.is_taskspace = self.robot_left.is_taskspace

class PlanarRobotVisualizer_original:
    def __init__(self, robot, ax=None):
        self.robot = robot
        self.arm_length = robot.arm_length
        self.ee_joint = robot.ee_joint
        if ax is None:
            self.fig, self.ax = plt.subplots()
        else:
            self.ax = ax

    def draw_robot(self, qt):
        plot_planar_robot(self.ax, qt, self.arm_length, facecolor='black')

    def draw_demos(self, demos, color='purple'):
        for demo in demos:
            self.ax.plot(demo[:, 0], demo[:, 1], color=color, alpha=0.8, linewidth=1.5)

    def draw_points(self, xd, xt):
        # self.ax.scatter(*xd, color='seagreen', s=80)
        # self.ax.scatter(*xt, color='navy', s=80)
        r = 0.02
        self.ax.add_patch(Circle(xd, r, color='seagreen', alpha=0.8))
        self.ax.add_patch(Circle(xt, r, color='navy', alpha=0.8))

    def draw_orientation(self, xd, theta):
        # Draw an arrow indicating the orientation at the end-effector position
        arrow_length = 1.0
        arrow_dx = arrow_length * np.cos(theta)
        arrow_dy = arrow_length * np.sin(theta)
        self.ax.arrow(xd[0], xd[1], arrow_dx, arrow_dy, head_width=0.2, head_length=0.2, fc='orange', ec='orange')

    def animate_robot_demos(self, ax=None, demos=None, title='Planar Robot Demos', save_plots=False):
        '''
        This is just like the animate_robot_demo, but it animates multiple demos simulenously on the same plot. It is useful for visualizing the diversity of the demos.
        '''
        if ax is None:
            ax = self.ax
        
        # Check, if data is a torch tensor and convert it to numpy array if needed
        demos_np = []
        for i in range(len(demos)):
            if isinstance(demos[0], torch.Tensor):
                demos_np.append(demos[i].cpu().numpy())
            else:
                demos_np.append(demos[i])

        # Add title
        ax.set_title(title, fontsize=16)

        if save_plots:
            print(f'[WARNING]: Saving the robot animation is not implemented yet.')

        # Extract the left and right arm joint space trajectories from the demos:
        demos_left_q = [demo[:, :self.robot.nb_dofs_left] for demo in demos_np]
        demos_right_q = [demo[:, self.robot.nb_dofs_left:self.robot.nb_dofs_left + self.robot.nb_dofs_right] for demo in demos_np]

        # Compute the task space trajectories for both arms using the forward kinematics functions
        demos_left_x = [np.array([self.robot.fk_func_left(q) for q in demo_left_q]) for demo_left_q in demos_left_q]
        demos_right_x = [np.array([self.robot.fk_func_right(q) for q in demo_right_q]) for demo_right_q in demos_right_q]

        # Animate the robot following the demo trajectories simultaneously
        for t in range(max([demo.shape[0] for demo in demos_np])):
            ax.cla()
            ax.axes.xaxis.set_ticks([])
            ax.axes.yaxis.set_ticks([])
            ax.margins(0.1)
            ax.grid(True)
            # ax.set_xlim(-4, 15)
            # ax.set_ylim(-4, 15)

            for spine in ax.spines.values():
                spine.set_visible(False)

            for demo_left_q, demo_right_q in zip(demos_left_q, demos_right_q):
                if t < demo_left_q.shape[0]:
                    self.draw_robot(demo_left_q[t, :])
                if t < demo_right_q.shape[0]:
                    self.draw_robot(demo_right_q[t, :])
            # self.draw_demos(demos_left_x + demos_right_x)
            self.draw_demos(demos_left_x, color='orange')
            self.draw_demos(demos_right_x, color='purple')
            for demo_left_x, demo_right_x in zip(demos_left_x, demos_right_x):
                if t < demo_left_x.shape[0]:
                    self.draw_points(demo_left_x[t, :2], demo_right_x[t, :2])
                    if self.ee_joint:
                        print(f'[WARNING]: I am not sure if this functionality is accurate at the moment')
                        self.draw_orientation(demo_left_x[t, :2], demo_left_x[t, 2])
                        self.draw_orientation(demo_right_x[t, :2], demo_right_x[t, 2])
            plt.pause(0.01)


def create_two_arm_robot(nb_dofs_left, nb_dofs_right, nb_x_left, nb_x_right, ee_joint = False, is_taskspace = False):
    robot = TwoArmsPlanarManipulator(nb_dofs_left, nb_dofs_right, ee_joint=ee_joint, nb_x_left=nb_x_left, nb_x_right=nb_x_right, is_taskspace=is_taskspace)
    robot_handler = RobotHandler(robot)
    return robot_handler

def create_rby1_robot(task_space: str = 'full', ee_joint=False, is_taskspace=False):
    '''
    task_space: one of 'xy', 'yz', 'xy_theta', 'xyz', 'xyz_theta', 'full'.
                Controls the dimensionality and content of FK output and Jacobian rows.
                    'xy'        → [x, y]                       nb_x = 2
                    'yz'        → [y, z]  (letter-writing plane) nb_x = 2
                    'xy_theta'  → [x, y, yaw]                  nb_x = 3
                    'xyz'       → [x, y, z]                    nb_x = 3
                    'xyz_theta' → [x, y, z, yaw]               nb_x = 4
                    'full'      → [x, y, z, R_flat(9)]         nb_x = 12
                yaw is the z-axis rotation in the link_torso_5 frame: atan2(R[1,0], R[0,0]).
    '''
    robot = TwoArmsRBY1(task_space=task_space, ee_joint=ee_joint, is_taskspace=is_taskspace)
    robot_handler = RobotHandler(robot)
    return robot_handler


def evaluate_full_trajectory(network, data_vis, robot, horizon, test_in_data, train_in_data, title_prefix='', save_plots: bool = False, save_trajs: bool = False, save_loss = False, plot_equivariance_sampling_bounds_flag=False, goal_conditioned=True):
    '''
    This method performs a qualitative evaluation of the network's performance by asking it to generate full trajectories starting from the same initial conditions as both the training and test data.
    Then it visualizes them.
    '''
    #### Step 1: To be consistent with the rest of the code, we first need to denormalize the input data.
    # Training data
    train_in_data_state_denorm = [denormalize_state(traj[:, :robot.nb_dofs], x_min=network.demonstrations["Q_min"], x_max=network.demonstrations["Q_max"]) for traj in train_in_data]
    # just copy it
    train_in_data_goal_denorm = [traj[:, robot.nb_dofs:] for traj in train_in_data]
    train_in_data_denorm = [torch.hstack([train_in_data_state_denorm[i], train_in_data_goal_denorm[i]]) for i in range(len(train_in_data))]

    # Testing data
    test_in_data_state_denorm = [denormalize_state(traj[:, :robot.nb_dofs], x_min=network.demonstrations["Q_min"], x_max=network.demonstrations["Q_max"]) for traj in test_in_data]
    # just copy it
    test_in_data_goal_denorm = [traj[:, robot.nb_dofs:] for traj in test_in_data]
    test_in_data_denorm = [torch.hstack([test_in_data_state_denorm[i], test_in_data_goal_denorm[i]]) for i in range(len(test_in_data))]

    #### Step 2: For the network predictions, we only need the initial conditions, which are the first row in each trajectory. So the input to the network will be a batch of initial conditions: (batch_size, 2*nb_dofs + 2*nb_x)
    # Check if the lists are empty and handle that case
    if len(train_in_data_denorm) == 0:
        print('[WARNING]: No training data available for evaluation.')
        train_in_data_initial_conditions = torch.empty((0, robot.nb_dofs + 3), device=device)
    else:
        train_in_data_initial_conditions = torch.stack([traj[0, :] for traj in train_in_data_denorm], dim=0)
    if len(test_in_data_denorm) == 0:
        print('[WARNING]: No testing data available for evaluation.')
        test_in_data_initial_conditions = torch.empty((0, robot.nb_dofs + 3), device=device)
    else:
        test_in_data_initial_conditions = torch.stack([traj[0, :] for traj in test_in_data_denorm], dim=0)

    #### Step 3: Get the network predictions for the full trajectories starting from the initial conditions of both the training and test data. The output will be two tensors of shape (batch_size, horizon, 2*nb_dofs) for the predicted states and their derivatives.
    pred_traj_state_train, pred_traj_state_dot_train = network.forward_denorm_multistep(horizon=horizon, inp_batch=train_in_data_initial_conditions, return_traj=True, use_grad=False)
    pred_traj_state_test, pred_traj_state_dot_test = network.forward_denorm_multistep(horizon=horizon, inp_batch=test_in_data_initial_conditions, return_traj=True, use_grad=False)

    #### Step 4: Visualize the predicted data along with the true training and testing data.

    ### Step 4.1: Convert everything to lists of numpy arrays for visualization
    pred_traj_state_train = [pred_traj_state_train[i, :, :robot.nb_dofs].cpu().numpy() for i in range(pred_traj_state_train.shape[0])]
    pred_traj_state_dot_train = [pred_traj_state_dot_train[i, :, :robot.nb_dofs].cpu().numpy() for i in range(pred_traj_state_dot_train.shape[0])]
    pred_traj_state_test = [pred_traj_state_test[i, :, :robot.nb_dofs].cpu().numpy() for i in range(pred_traj_state_test.shape[0])]
    pred_traj_state_dot_test = [pred_traj_state_dot_test[i, :, :robot.nb_dofs].cpu().numpy() for i in range(pred_traj_state_dot_test.shape[0])]
    train_in_data_state_denorm = [traj[:, :robot.nb_dofs].cpu().numpy() for traj in train_in_data_denorm]
    test_in_data_state_denorm = [traj[:, :robot.nb_dofs].cpu().numpy() for traj in test_in_data_denorm]

    # Combine the prediction lists into a single list because we will anyways plot them together
    pred_traj_state = pred_traj_state_train + pred_traj_state_test
    pred_traj_state_dot = pred_traj_state_dot_train + pred_traj_state_dot_test

    ### Step 4.2: If the robot's state evolves in the joint space, we need to compute its task space trajectories as well.
    if not robot.is_taskspace:
        true_train_traj_state_X = data_vis.get_X_from_Q_trajectories(robot.fk_func_left, robot.fk_func_right, train_in_data_state_denorm, robot.nb_dofs_left, robot.nb_dofs_right)
        true_test_traj_state_X = data_vis.get_X_from_Q_trajectories(robot.fk_func_left, robot.fk_func_right, test_in_data_state_denorm, robot.nb_dofs_left, robot.nb_dofs_right)
        pred_traj_state_X = data_vis.get_X_from_Q_trajectories(robot.fk_func_left, robot.fk_func_right, pred_traj_state, robot.nb_dofs_left, robot.nb_dofs_right)

        # Visualize
        data_vis.plot_trajectories_colored( traj_q = {'train': train_in_data_state_denorm, 'test': test_in_data_state_denorm, 'predicted': pred_traj_state}, 
                                            traj_x = {'train': true_train_traj_state_X, 'test': true_test_traj_state_X, 'predicted': pred_traj_state_X}, 
                                            traj_dq = {'train': [], 'test': [], 'pred': pred_traj_state_dot}, 
                                            merged_task_space = True, save_plots=save_plots, title_prefix=title_prefix, 
                                            color_scheme = {'train': 'green', 'test': 'blue', 'predicted': 'red',},
                                            line_width = {'train': 3, 'test': 3, 'predicted': 1.5},
                                            plot_equivariance_sampling_bounds_flag=plot_equivariance_sampling_bounds_flag)
    else:
        data_vis.plot_trajectories_colored( traj_q = None, 
                                            traj_x = {'train': train_in_data_state_denorm, 'test': test_in_data_state_denorm, 'predicted': pred_traj_state}, 
                                            traj_dq = {'train': [], 'test': [], 'pred': pred_traj_state_dot}, 
                                            merged_task_space = True, save_plots=save_plots, title_prefix=title_prefix,
                                            color_scheme = {'train': 'green', 'test': 'blue', 'predicted': 'red',},
                                            line_width = {'train': 3, 'test': 3, 'predicted': 1.5},
                                            plot_equivariance_sampling_bounds_flag=plot_equivariance_sampling_bounds_flag)

# ============================================================
# Data utilities
# ============================================================

def _normalize_traj_input(traj):
    """Convert a list of arrays to {'default': list}, or pass dicts through unchanged."""
    if traj is None:
        return None
    if isinstance(traj, dict):
        return traj
    return {'default': traj}

def _get_style(label, color_scheme, line_width):
    color = None if color_scheme is None else color_scheme.get(label)
    lw    = None if line_width   is None else line_width.get(label)
    return color, lw

# ============================================================
# Plot primitives (stateless, reusable)
# ============================================================

def plot_time_series(axs, traj_dict, ylabel_prefix, color_scheme=None, line_width=None):
    axs_flat = np.ravel(axs)
    nb_dims  = next(iter(traj_dict.values()))[0].shape[1]

    for i in range(nb_dims):
        ax = axs_flat[i]
        time_offset = 0

        for label, traj_list in traj_dict.items():
            color, lw = _get_style(label, color_scheme, line_width)
            for traj in traj_list:
                t = np.arange(traj.shape[0]) + time_offset
                ax.plot(t, traj[:, i], color=color, linewidth=lw,
                        label=label if time_offset == 0 else None)
                time_offset += traj.shape[0]
                ax.axvline(time_offset, color='k', linestyle='--', alpha=0.3)

        ax.set_ylabel(f'{ylabel_prefix} {i+1}')
        ax.grid()
        ax.set_facecolor('#f0f0f0')

    for ax in axs_flat:
        ax.set_xlabel('Time step')
    axs_flat[0].legend()

def plot_task_space_xy(ax, traj_dict, color_scheme=None, line_width=None, network=None):
    for label, traj_list in traj_dict.items():
        color, lw = _get_style(label, color_scheme, line_width)
        for traj in traj_list:
            ax.plot(traj[:, 0], traj[:, 1], color=color, linewidth=lw)
            if traj.shape[1] > 2:
                ax.plot(traj[:, 2], traj[:, 3], color=color, linewidth=lw)

    if network is not None:
        plot_equivariance_sampling_bounds(robot=network.robot, network=network, ax=ax)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#f0f0f0')

    unique_labels = list(dict.fromkeys(traj_dict.keys()))
    handles = [plt.Line2D([0], [0], color=_get_style(l, color_scheme, line_width)[0],
                          linewidth=_get_style(l, color_scheme, line_width)[1])
               for l in unique_labels]
    ax.legend(handles, unique_labels)


def plot_task_space_xyz(ax, traj_dict, color_scheme=None, line_width=None):
    """3D task-space plot. Expects cols [x_l,y_l,z_l, x_r,y_r,z_r]."""
    for label, traj_list in traj_dict.items():
        color, lw = _get_style(label, color_scheme, line_width)
        for traj in traj_list:
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], color=color, linewidth=lw)
            if traj.shape[1] >= 6:
                ax.plot(traj[:, 3], traj[:, 4], traj[:, 5], color=color, linewidth=lw)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.grid(True)

    unique_labels = list(dict.fromkeys(traj_dict.keys()))
    handles = [plt.Line2D([0], [0], color=_get_style(l, color_scheme, line_width)[0],
                          linewidth=_get_style(l, color_scheme, line_width)[1])
               for l in unique_labels]
    ax.legend(handles, unique_labels)


def _arrow_len_from_data(traj_dict, pos_cols):
    """Fixed arrow length = 3% of the overall position data range."""
    all_vals = []
    for traj_list in traj_dict.values():
        for traj in traj_list:
            t = traj.cpu().numpy() if isinstance(traj, torch.Tensor) else np.asarray(traj)
            for c in pos_cols:
                if c < t.shape[1]:
                    all_vals.append(t[:, c])
    if not all_vals:
        return 0.03
    vals = np.concatenate(all_vals)
    return max((vals.max() - vals.min()) * 0.03, 1e-4)


def plot_task_space_xy_theta(ax, traj_dict, color_scheme=None, line_width=None, quiver_interval=10):
    """2D task-space plot with orientation quivers. Expects cols [x_l,y_l,θ_l, x_r,y_r,θ_r]."""
    arrow_len = _arrow_len_from_data(traj_dict, [0, 1, 3, 4])
    for label, traj_list in traj_dict.items():
        color, lw = _get_style(label, color_scheme, line_width)
        for traj in traj_list:
            t = traj.cpu().numpy() if isinstance(traj, torch.Tensor) else np.asarray(traj)
            ax.plot(t[:, 0], t[:, 1], color=color, linewidth=lw)
            idx = np.arange(0, len(t), quiver_interval)
            ax.quiver(t[idx, 0], t[idx, 1],
                      arrow_len * np.cos(t[idx, 2]), arrow_len * np.sin(t[idx, 2]),
                      color=color, angles='xy', scale_units='xy', scale=1,
                      width=0.003, headwidth=4, alpha=0.8)
            if t.shape[1] >= 6:
                ax.plot(t[:, 3], t[:, 4], color=color, linewidth=lw)
                ax.quiver(t[idx, 3], t[idx, 4],
                          arrow_len * np.cos(t[idx, 5]), arrow_len * np.sin(t[idx, 5]),
                          color=color, angles='xy', scale_units='xy', scale=1,
                          width=0.003, headwidth=4, alpha=0.8)

    ax.set_xlabel('X'); ax.set_ylabel('Y')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#f0f0f0')

    unique_labels = list(dict.fromkeys(traj_dict.keys()))
    handles = [plt.Line2D([0], [0], color=_get_style(l, color_scheme, line_width)[0],
                          linewidth=_get_style(l, color_scheme, line_width)[1])
               for l in unique_labels]
    ax.legend(handles, unique_labels)


def plot_task_space_xyz_theta(ax, traj_dict, color_scheme=None, line_width=None, quiver_interval=10):
    """3D task-space plot with orientation quivers. Expects cols [x_l,y_l,z_l,θ_l, x_r,y_r,z_r,θ_r]."""
    arrow_len = _arrow_len_from_data(traj_dict, [0, 1, 2, 4, 5, 6])
    for label, traj_list in traj_dict.items():
        color, lw = _get_style(label, color_scheme, line_width)
        for traj in traj_list:
            t = traj.cpu().numpy() if isinstance(traj, torch.Tensor) else np.asarray(traj)
            ax.plot(t[:, 0], t[:, 1], t[:, 2], color=color, linewidth=lw)
            idx = np.arange(0, len(t), quiver_interval)
            ax.quiver(t[idx, 0], t[idx, 1], t[idx, 2],
                      arrow_len * np.cos(t[idx, 3]), arrow_len * np.sin(t[idx, 3]),
                      np.zeros(len(idx)),
                      color=color, length=1.0, normalize=False, alpha=0.8)
            if t.shape[1] >= 8:
                ax.plot(t[:, 4], t[:, 5], t[:, 6], color=color, linewidth=lw)
                ax.quiver(t[idx, 4], t[idx, 5], t[idx, 6],
                          arrow_len * np.cos(t[idx, 7]), arrow_len * np.sin(t[idx, 7]),
                          np.zeros(len(idx)),
                          color=color, length=1.0, normalize=False, alpha=0.8)

    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
    ax.grid(True)

    unique_labels = list(dict.fromkeys(traj_dict.keys()))
    handles = [plt.Line2D([0], [0], color=_get_style(l, color_scheme, line_width)[0],
                          linewidth=_get_style(l, color_scheme, line_width)[1])
               for l in unique_labels]
    ax.legend(handles, unique_labels)


def plot_task_space_yz(ax, traj_dict, color_scheme=None, line_width=None):
    """2-D task-space plot for the Y–Z (letter-writing) plane.

    Data layout: cols [y_l, z_l, y_r, z_r].

    The robot's Y-axis points to its left.  Inverting the horizontal axis makes
    the plot match what a viewer standing alongside the robot sees on the board.
    """
    for label, traj_list in traj_dict.items():
        color, lw = _get_style(label, color_scheme, line_width)
        for traj in traj_list:
            t = traj.cpu().numpy() if isinstance(traj, torch.Tensor) else np.asarray(traj)
            ax.plot(t[:, 0], t[:, 1], color=color, linewidth=lw)
            if t.shape[1] >= 4:
                ax.plot(t[:, 2], t[:, 3], color=color, linewidth=lw)

    ax.invert_xaxis()   # +Y is robot's left; invert so letter reads correctly
    ax.set_xlabel('Y  (← robot\'s left)  [m]')
    ax.set_ylabel('Z  [m]')
    ax.grid(True)
    ax.set_aspect('equal', adjustable='box')
    ax.set_facecolor('#f0f0f0')

    unique_labels = list(dict.fromkeys(traj_dict.keys()))
    handles = [plt.Line2D([0], [0], color=_get_style(l, color_scheme, line_width)[0],
                          linewidth=_get_style(l, color_scheme, line_width)[1])
               for l in unique_labels]
    ax.legend(handles, unique_labels)


def _plot_task_space(fig, traj_dict, task_space, color_scheme=None, line_width=None, network=None):
    """Dispatch to the right task-space plotting function based on task_space string."""
    is_3d = task_space in ('xyz', 'xyz_theta')
    print(f'[INFO] Plotting in task space "{task_space}" using {"3D" if is_3d else "2D"} projection.')
    ax = fig.add_subplot(111, projection='3d') if is_3d else fig.add_subplot(111)
    if task_space == 'xyz':
        plot_task_space_xyz(ax, traj_dict, color_scheme, line_width)
    elif task_space == 'xy_theta':
        plot_task_space_xy_theta(ax, traj_dict, color_scheme, line_width)
    elif task_space == 'xyz_theta':
        plot_task_space_xyz_theta(ax, traj_dict, color_scheme, line_width)
    elif task_space == 'yz':
        plot_task_space_yz(ax, traj_dict, color_scheme, line_width)
    else:  # default: 'xy'
        plot_task_space_xy(ax, traj_dict, color_scheme, line_width, network)
    return ax

# ============================================================
# Layout helpers
# ============================================================

def _create_grid(fig, nb_dims):
    return fig.subplots(nrows=(nb_dims + 3) // 4, ncols=min(nb_dims, 4), sharex=True)

# ============================================================
# Tab UI controller
# ============================================================

class TabController:
    """Manages tab-like button switching within a single matplotlib figure."""

    def __init__(self, fig):
        self.fig = fig
        self.axes = []       # currently displayed content axes
        self.buttons = {}    # key -> Button
        self.actions = {}    # button_ax -> callback
        self.connection_ids = []

        self.fig.canvas.mpl_connect('close_event',
            lambda e: [self.fig.canvas.mpl_disconnect(c) for c in self.connection_ids])

    def register_button(self, key, label, callback, x_pos, w=0.18, h=0.05, y=0.88):
        ax = self.fig.add_axes([x_pos, y, w, h])
        self.buttons[key] = Button(ax, label)
        self.actions[ax] = callback

    def activate(self, active_key):
        for key, btn in self.buttons.items():
            btn.color = '#d0d0d0' if key == active_key else '#f0f0f0'
        self.fig.canvas.draw_idle()

    def connect(self):
        def handler(event):
            if event.button == 1 and event.inaxes in self.actions:
                self.actions[event.inaxes]()
        cid = self.fig.canvas.mpl_connect('button_press_event', handler)
        self.connection_ids.append(cid)

    def clear(self):
        for ax in self.axes:
            ax.remove()
        self.axes.clear()

class DataVisualizer:
    '''
    A class for visualizing trajectories in 3D space.
    As well as saving trajectories.
    '''
    def __init__(self, 
                 network=None, task=None,
                 group=None, 
                 isotropic_normalization = False,
                 margin=1e-10,
                 conditioning=None,
                 normalize_conditioning=False,
                 task_space = 'xy'
                 ):
        self.network = network
        self.task = task
        self.group = group
        self.isotropic_normalization = isotropic_normalization
        self.margin = margin
        self.conditioning = conditioning
        self.normalize_conditioning = normalize_conditioning
        self.task_space = task_space


    def construct_demonstrations(self, demo_folder : str, dt: float = 1.0, load_augmented_data=True, predefined_dataset=None):
        ##### First load the npz file and extract the raw joint space trajectories, the task space trajectories and compute the joint velocities.
        
        #### Step 0: Choose between the group-augmented and non group-augmented demonstrations.
        if not load_augmented_data and predefined_dataset is None:
            q_real, x_real, dq_real = self.construct_q_x_dq_real_data(demo_folder, dt)

            demonstrations_real = self.build_network_real_dataset(q_real, x_real, dq_real)
        else:
            # Load the group-augmented demonstrations dictionary
            if predefined_dataset is None:
                demos_name = self.task + f'_augmented_config_{self.group}.npz'
            else:
                demos_name = predefined_dataset
                        
            data = np.load(ROOT_DIR / 'demonstrations' /demo_folder / demos_name, allow_pickle=True)

            # Group-augmented demonstrations have been built based on the original demonstrations. So we do not need to smoothen them and repeat many of the processes done when working wit original demonstrations.
            demonstrations_real = data['demonstrations'].item() # Convert from 0-d array to dictionary            

        ### Step ?: Normalize the demonstrations.
        demonstrations_norm = self.normalize_demonstrations(demonstrations_real)

        # return demonstrations_real, None
        return demonstrations_real, demonstrations_norm

    def construct_q_x_dq_real_data(self, demo_folder : str, dt: float = 1.0):
        '''
        This method returns the real demosntrations of the robot in the raw format. Meaning that these data re not normalized.

        The original npz files have the following format:
        - q_data: (nb_demos, nb_time_steps, nb_q). Note, this does not concern itself with the fact that the network might output velocities, or need a goal in the task space. This array is simply the raw joint space trajectories as recorded from the real robot.
        - x_data: (nb_demos, nb_time_steps, nb_x). This array contains the task space trajectories corresponding to the joint space trajectories in q_data. It is computed by applying the forward kinematics to the joint space trajectories.
        '''
        demos_name = self.task + '.npz'
        data = np.load(ROOT_DIR / 'demonstrations' /demo_folder / demos_name)

        #### Step 1: Extract the raw joint space trajectories from the robot demonstrations.
        q_real = data['q_data']

        #### Step 2: Compute the joint velocities from the joint positions using finite differences.
        dq_real = (q_real[:, 1:] - q_real[:, :-1]) / dt

        # Add a zero velocity at the end of each trajectory to maintain the same length as q_real. This is a bit hacky and should be fixed later.
        zeros = np.zeros((dq_real.shape[0], 1, dq_real.shape[2]), dtype=dq_real.dtype)
        dq_real = np.concatenate([dq_real, zeros], axis=1)
        
        #### Step 3: Extract the task space trajectories from the robot demonstrations.
        x_real = data['x_data']
        nb_demos = q_real.shape[0]

        #### Step 4: Convert to lists of arrays for consistency with the rest of the code.
        q_real = [q_real[i] for i in range(nb_demos)]
        x_real = [x_real[i] for i in range(nb_demos)]
        dq_real = [dq_real[i] for i in range(nb_demos)]
        
        #### Step 5: Remove extra entries of any trajectories (from their begininning) to ensure that all trajectories have the same length. This is a bit hacky and should be fixed later. TODO
        min_traj_length = min([demo.shape[0] for demo in q_real])
        q_real = [demo[:min_traj_length] for demo in q_real]
        x_real = [demo[:min_traj_length] for demo in x_real]
        dq_real = [demo[:min_traj_length] for demo in dq_real]

        return q_real, x_real, dq_real

    def build_network_real_dataset(self, q_real, x_real, dq_real):
        '''
        This method takes the generic demonstrations and builds the dataset in the format expected by the network. Still, these are NOT normalized.

        The network has the form: f: (q_state, x_goal) -> dq.

        So, the input to the network is a combination of the current joint state and the goal in the task space. The output is the joint velocity.

        The goal x_goal is defined as the last task space state of each trajectory, which corresponds to the final position of the end-effector in the demonstrations.

        In this case we do not have any test data. Test data are only relevant when we have augmented data.
        '''
        demonstrations = {'train_in': [], 'train_out': [], 'test_in': [], 'test_out': [],
                          'nb_q': None, 'nb_x': None, 'nb_q_left': None, 'nb_q_right': None, 'nb_x_left': None, 'nb_x_right': None,
                          'Q_min': None, 'Q_max': None, 'X_min': None, 'X_max': None, 'Dq_min': None, 'Dq_max': None,
                          'inp_min': None, 'inp_max': None, 'out_min': None, 'out_max': None,
                          'train_in_demo_type': [], 'train_out_demo_type': [], 'test_in_demo_type': [], 'test_out_demo_type': [] # Used to define the "type" of trajectory augmentation (e.g., original, SO2, Scaling2, ...)
                          }

        # The output data can be directly taken as the joint velocities computed from the real demonstrations.
        demonstrations['train_out'] = dq_real

        if self.conditioning is None:
            # The input data is simply the current joint state, without any goal information.
            demonstrations['train_in'] = q_real
            for _ in q_real:
                # All these demos are from the original dataset without any augmentation, so we label them as 'original' in the "demo_type" lists.
                demonstrations['train_in_demo_type'].append('original')
                demonstrations['train_out_demo_type'].append('original')

            # Add the number of joint states to the demonstrations dictionary for later use in the network.
            demonstrations['nb_q'] = q_real[0].shape[1]
            demonstrations['nb_q_left'] = q_real[0].shape[1] // 2
            demonstrations['nb_q_right'] = q_real[0].shape[1] // 2
            demonstrations['nb_x'] = 0
            demonstrations['nb_x_left'] = 0
            demonstrations['nb_x_right'] = 0
        elif self.conditioning == 'goal':
            # The input data is a combination of the current joint state and the goal in the task space. The goal is defined as the last task space state of each trajectory.
            for q_traj, x_traj in zip(q_real, x_real):
                x_goal = x_traj[-1, :] # Get the last task space state as the goal
                x_goal_repeated = np.tile(x_goal, (q_traj.shape[0], 1)) # Repeat the goal for each time step in the trajectory
                input_traj = np.hstack([q_traj, x_goal_repeated]) # Concatenate the joint states and the repeated goal to form the input trajectory
                demonstrations['train_in'].append(input_traj)
                # All these demos are from the original dataset without any augmentation, so we label them as 'original' in the "demo_type" lists.
                demonstrations['train_in_demo_type'].append('original')
                demonstrations['train_out_demo_type'].append('original')

            # Add the number of joint states and task space states to the demonstrations dictionary for later use in the network.
            demonstrations['nb_q'] = q_real[0].shape[1]
            demonstrations['nb_x'] = x_real[0].shape[1]
            demonstrations['nb_q_left'] = q_real[0].shape[1] // 2
            demonstrations['nb_q_right'] = q_real[0].shape[1] // 2
            demonstrations['nb_x_left'] = x_real[0].shape[1] // 2
            demonstrations['nb_x_right'] = x_real[0].shape[1] // 2
        elif self.conditioning == 'symmetry':
            '''
            The symmetry conditioning is a bit more complex. We have 3 types of symmetriers. We have three symmetries.
            For each symmetry we have 1 variable:
            - SO2: angle of rotation - reference angle is "zero".
            - Scaling2: scaling factor - reference is 1.
            - Reflection: binary variable: 1 if no reflection is applied, -1 if reflection is applied. Reference is 1 (no reflection).
            '''
            # This is just the original dataset, so it should only have the reference values for the symmetry variables (i.e., zero rotation, scaling factor of 1, and no reflection).
            for q_traj in q_real:
                # Define the symmetry variables for this trajectory. Since this is the original trajectory without any augmentation, we set them to their reference values.
                symmetry_variables = np.array([0.0, 1.0, 1.0]) # [SO2_angle, Scaling_factor, Reflection]
                symmetry_variables_repeated = np.tile(symmetry_variables, (q_traj.shape[0], 1)) # Repeat the symmetry variables for each time step in the trajectory
                input_traj = np.hstack([q_traj, symmetry_variables_repeated]) # Concatenate the joint states, task space states and the repeated symmetry variables to form the input trajectory
                demonstrations['train_in'].append(input_traj)
                # All these demos are from the original dataset without any augmentation, so we label them as 'original' in the "demo_type" lists.
                demonstrations['train_in_demo_type'].append('original')
                demonstrations['train_out_demo_type'].append('original')

                demonstrations['nb_q'] = q_real[0].shape[1]
                demonstrations['nb_q_left'] = q_real[0].shape[1] // 2
                demonstrations['nb_q_right'] = q_real[0].shape[1] // 2
                demonstrations['nb_x'] = 3 # 3 symmetry variables
                demonstrations['nb_x_left'] = 0
                demonstrations['nb_x_right'] = 0
        return demonstrations                

    def normalize_demonstrations(self, demonstrations, dt=1.0):
        #### Step 1: Set the normalization for the position components Q and X if not already set
        margin = self.margin
        
        nb_q = demonstrations['nb_q']

        inp_data_real = np.vstack(demonstrations['train_in'] + demonstrations['test_in'])

        # Get the actual min max bounds of the input data
        inp_min = np.min(inp_data_real, axis=0)
        inp_max = np.max(inp_data_real, axis=0)

        # Add some margin
        self.inp_min = inp_min - margin * (inp_max - inp_min)
        self.inp_max = inp_max + margin * (inp_max - inp_min)

        # Protect against near-constant features: when a feature barely moves
        # (e.g. a stationary arm's joints), inp_max - inp_min ≈ noise ≈ 1e-4.
        # The standard formula amplifies that noise ~10000× to fill [-1, 1],
        # causing the network to learn noise patterns instead of real motion.
        # Fix: ensure every feature has at least MIN_FEATURE_RANGE as its
        # normalisation range, centred on the feature's actual midpoint.
        MIN_FEATURE_RANGE = 1e-3
        inp_range = self.inp_max - self.inp_min
        is_tiny   = inp_range < MIN_FEATURE_RANGE
        if np.any(is_tiny):
            mid            = (self.inp_max + self.inp_min) / 2.0
            self.inp_min   = np.where(is_tiny, mid - MIN_FEATURE_RANGE / 2.0, self.inp_min)
            self.inp_max   = np.where(is_tiny, mid + MIN_FEATURE_RANGE / 2.0, self.inp_max)
            tiny_dims      = np.where(is_tiny)[0].tolist()
            # print(f"[normalize_demonstrations] WARNING: {len(tiny_dims)} near-constant "
            #       f"feature(s) detected (dims {tiny_dims}). "
            #       f"Clamping normalisation range to {MIN_FEATURE_RANGE}.")

        # split into Q and X components
        demonstrations['inp_min'] = self.inp_min
        demonstrations['inp_max'] = self.inp_max
        demonstrations['Q_min'] = self.inp_min[:nb_q]
        demonstrations['Q_max'] = self.inp_max[:nb_q]
        demonstrations['X_min'] = self.inp_min[nb_q:]
        demonstrations['X_max'] = self.inp_max[nb_q:]

        #### Step 2: Normalize the position components Q and X using the set normalization bounds.
        if self.normalize_conditioning:
            q_data_train_norm = [normalize_state(demo[:, :demonstrations['nb_q']], demonstrations['Q_min'], demonstrations['Q_max']) for demo in demonstrations['train_in']]
            x_data_train_norm = [normalize_state(demo[:, demonstrations['nb_q']:], demonstrations['X_min'], demonstrations['X_max']) for demo in demonstrations['train_in']]
            q_data_test_norm = [normalize_state(demo[:, :demonstrations['nb_q']], demonstrations['Q_min'], demonstrations['Q_max']) for demo in demonstrations['test_in']]
            x_data_test_norm = [normalize_state(demo[:, demonstrations['nb_q']:], demonstrations['X_min'], demonstrations['X_max']) for demo in demonstrations['test_in']]
        else:
            q_data_train_norm = [normalize_state(demo[:, :demonstrations['nb_q']], demonstrations['Q_min'], demonstrations['Q_max']) for demo in demonstrations['train_in']]
            x_data_train_norm = [demo[:, demonstrations['nb_q']:] for demo in demonstrations['train_in']] # No normalization
            q_data_test_norm = [normalize_state(demo[:, :demonstrations['nb_q']], demonstrations['Q_min'], demonstrations['Q_max']) for demo in demonstrations['test_in']]
            x_data_test_norm = [demo[:, demonstrations['nb_q']:] for demo in demonstrations['test_in']] # No normalization


        #### Step 3: Obtain the velocities of the normalized position components by applying finite differences to the normalized position trajectories.
        q_data_norm = q_data_train_norm + q_data_test_norm
        dq_data_for_q_norm = [(q[1:] - q[:-1]) / dt for q in q_data_norm]
        # Add a zero velocity at the end of each trajectory to maintain the same length as q_data_norm. This is a bit hacky and should be fixed later.
        dq_data_for_q_norm = [np.vstack([dq, np.zeros((1, dq.shape[1]))]) for dq in dq_data_for_q_norm]

        #### Step 4: Set the normalization bounds for the normalized position derivatives (velocities) if not already set.
        dq_data_norm = np.vstack(dq_data_for_q_norm)

        # Get the actual min max bounds of the velocity data
        dq_min = np.min(dq_data_norm, axis=0)
        dq_max = np.max(dq_data_norm, axis=0)

        #### SYMMETRIC
        # Use symmetric bounds so that zero velocity maps to zero network output
        # and denormalize_state(0, Dq_min, Dq_max) = 0 exactly.
        # Asymmetric min-max bounds (old approach) shift the zero point, causing
        # attractor drift during rollout.
        max_abs_dq = np.maximum(np.abs(dq_min), np.abs(dq_max))
        # Prevent zero velocity range for constant features (would cause 0/0 = NaN)
        max_abs_dq = np.maximum(max_abs_dq, 1e-6)
        self.Dq_min = -max_abs_dq * (1.0 + margin)
        self.Dq_max =  max_abs_dq * (1.0 + margin)

        # Include the normalization bounds in the dictionary
        demonstrations["Dq_min"] = self.Dq_min
        demonstrations["Dq_max"] = self.Dq_max


        # #### ASYMMETRIC
        # # Add some margin
        # self.Dq_min = dq_min - margin * (dq_max - dq_min)
        # self.Dq_max = dq_max + margin * (dq_max - dq_min)

        # # Include the normalization bounds in the dictionary
        # demonstrations["Dq_min"] = self.Dq_min
        # demonstrations["Dq_max"] = self.Dq_max




        #### Step 5: Normalize the position derivatives (velocities) using the set normalization bounds.
        dq_data_train_norm = [normalize_state(dq, demonstrations["Dq_min"], demonstrations["Dq_max"]) for dq in dq_data_for_q_norm[:len(q_data_train_norm)]]
        dq_data_test_norm = [normalize_state(dq, demonstrations["Dq_min"], demonstrations["Dq_max"]) for dq in dq_data_for_q_norm[len(q_data_train_norm):]]

        #### Step 6: Rebuild the demonstrations dictionary in the format expected by the network.
        train_in_norm = [np.hstack([q_data_train_norm[i], x_data_train_norm[i]]) for i in range(len(q_data_train_norm))]
        test_in_norm = [np.hstack([q_data_test_norm[i], x_data_test_norm[i]]) for i in range(len(q_data_test_norm))]
        train_out_norm = dq_data_train_norm
        test_out_norm = dq_data_test_norm

        normalized_demonstrations = {'train_in': train_in_norm, 'test_in': test_in_norm, 'train_out': train_out_norm, 'test_out': test_out_norm}

        # Add all the other fields from the original demonstrations dictionary to the normalized demonstrations dictionary.
        for key in demonstrations.keys():
            if key not in normalized_demonstrations:
                normalized_demonstrations[key] = demonstrations[key]

        return normalized_demonstrations
    

    def set_network(self, network):
        self.network = network

    def rearrange_demonstrations_v1(self, demonstrations, demonstrations_normalized, train_demo_types=['original'], num_train_demos_per_type = [1]):
        # Deep copy everything first to avoid shared references
        demonstrations = copy.deepcopy(demonstrations)
        demonstrations_normalized = copy.deepcopy(demonstrations_normalized)

        new_demonstrations = {key: [] for key in demonstrations.keys() if key in ['train_in', 'train_out', 'test_in', 'test_out', 'train_in_demo_type', 'train_out_demo_type', 'test_in_demo_type', 'test_out_demo_type']}
        new_demonstrations_norm = {key: [] for key in demonstrations_normalized.keys() if key in ['train_in', 'train_out', 'test_in', 'test_out', 'train_in_demo_type', 'train_out_demo_type', 'test_in_demo_type', 'test_out_demo_type']}

        for key in demonstrations.keys():
            if key not in new_demonstrations:
                new_demonstrations[key] = demonstrations[key]
        for key in demonstrations_normalized.keys():
            if key not in new_demonstrations_norm:
                new_demonstrations_norm[key] = demonstrations_normalized[key]

        for data_key, type_key in [('train_in', 'train_in_demo_type'), ('train_out', 'train_out_demo_type')]:
            for demo, demo_type in zip(demonstrations[data_key], demonstrations[type_key]):
                new_demonstrations[data_key].append(demo)
                new_demonstrations[type_key].append(demo_type)
            for demo, demo_type in zip(demonstrations_normalized[data_key], demonstrations_normalized[type_key]):
                new_demonstrations_norm[data_key].append(demo)
                new_demonstrations_norm[type_key].append(demo_type)

        for data_key, type_key in [('test_in', 'test_in_demo_type'), ('test_out', 'test_out_demo_type')]:
            train_key = data_key.replace('test', 'train')
            train_type_key = type_key.replace('test', 'train')
            for demo, demo_type in zip(demonstrations[data_key], demonstrations[type_key]):
                if demo_type in train_demo_types:
                    new_demonstrations[train_key].append(demo)
                    new_demonstrations[train_type_key].append(demo_type)
                else:
                    new_demonstrations[data_key].append(demo)
                    new_demonstrations[type_key].append(demo_type)
            for demo, demo_type in zip(demonstrations_normalized[data_key], demonstrations_normalized[type_key]):
                if demo_type in train_demo_types:
                    new_demonstrations_norm[train_key].append(demo)
                    new_demonstrations_norm[train_type_key].append(demo_type)
                else:
                    new_demonstrations_norm[data_key].append(demo)
                    new_demonstrations_norm[type_key].append(demo_type)

        return new_demonstrations, new_demonstrations_norm    

    def rearrange_demonstrations(self, demonstrations, demonstrations_normalized, train_demo_types=['original'], num_train_demos_per_type=[1], preserve_test_split=False):        
    # def rearrange_demonstrations(self, demonstrations, demonstrations_normalized, train_demo_types=['original'], num_train_demos_per_type=[1]):
        demonstrations = copy.deepcopy(demonstrations)
        demonstrations_normalized = copy.deepcopy(demonstrations_normalized)

        new_demonstrations = {key: [] for key in ['train_in', 'train_out', 'test_in', 'test_out',
                                                'train_in_demo_type', 'train_out_demo_type',
                                                'test_in_demo_type', 'test_out_demo_type']}
        new_demonstrations_norm = {key: [] for key in ['train_in', 'train_out', 'test_in', 'test_out',
                                                        'train_in_demo_type', 'train_out_demo_type',
                                                        'test_in_demo_type', 'test_out_demo_type']}

        # Carry over any keys not related to splits (e.g. metadata, scaler, etc.)
        for key in demonstrations:
            if key not in new_demonstrations:
                new_demonstrations[key] = demonstrations[key]
        for key in demonstrations_normalized:
            if key not in new_demonstrations_norm:
                new_demonstrations_norm[key] = demonstrations_normalized[key]

        # Count how many demos of each type have been added to training so far
        train_type_counts = {t: 0 for t in train_demo_types}

        # # Pool all demos from both train and test splits together
        # all_in  = demonstrations['train_in']  + demonstrations['test_in']
        # all_out = demonstrations['train_out'] + demonstrations['test_out']
        # all_in_types  = demonstrations['train_in_demo_type']  + demonstrations['test_in_demo_type']
        # all_out_types = demonstrations['train_out_demo_type'] + demonstrations['test_out_demo_type']

        # all_in_norm  = demonstrations_normalized['train_in']  + demonstrations_normalized['test_in']
        # all_out_norm = demonstrations_normalized['train_out'] + demonstrations_normalized['test_out']

        if preserve_test_split:
            # test_in was assigned during data augmentation — carry it over unchanged
            # and only redistribute train_in according to the quotas.
            for key in ['test_in', 'test_out', 'test_in_demo_type', 'test_out_demo_type']:
                new_demonstrations[key]      = list(demonstrations[key])
                new_demonstrations_norm[key] = list(demonstrations_normalized.get(key, []))
            all_in        = demonstrations['train_in']
            all_out       = demonstrations['train_out']
            all_in_types  = demonstrations['train_in_demo_type']
            all_out_types = demonstrations['train_out_demo_type']
            all_in_norm   = demonstrations_normalized['train_in']
            all_out_norm  = demonstrations_normalized['train_out']
        else:
            # Pool all demos from both train and test splits together
            all_in        = demonstrations['train_in']  + demonstrations['test_in']
            all_out       = demonstrations['train_out'] + demonstrations['test_out']
            all_in_types  = demonstrations['train_in_demo_type']  + demonstrations['test_in_demo_type']
            all_out_types = demonstrations['train_out_demo_type'] + demonstrations['test_out_demo_type']
            all_in_norm   = demonstrations_normalized['train_in']  + demonstrations_normalized['test_in']
            all_out_norm  = demonstrations_normalized['train_out'] + demonstrations_normalized['test_out']


        # Shuffle jointly to ensure random selection
        indices = list(range(len(all_in)))
        random.shuffle(indices)

        for idx in indices:
            demo_in       = all_in[idx]
            demo_out      = all_out[idx]
            demo_in_norm  = all_in_norm[idx]
            demo_out_norm = all_out_norm[idx]
            demo_type     = all_in_types[idx]  # in/out types should be aligned

            # Sanity check alignment
            assert all_in_types[idx] == all_out_types[idx], \
                f"Mismatched demo types at index {idx}: {all_in_types[idx]} vs {all_out_types[idx]}"

            # Determine the allowed quota for this type
            if demo_type in train_demo_types:
                type_idx = train_demo_types.index(demo_type)
                quota = num_train_demos_per_type[type_idx]
            else:
                quota = 0

            if demo_type in train_demo_types and train_type_counts[demo_type] < quota:
                # Add to training
                new_demonstrations['train_in'].append(demo_in)
                new_demonstrations['train_out'].append(demo_out)
                new_demonstrations['train_in_demo_type'].append(demo_type)
                new_demonstrations['train_out_demo_type'].append(demo_type)
                new_demonstrations_norm['train_in'].append(demo_in_norm)
                new_demonstrations_norm['train_out'].append(demo_out_norm)
                new_demonstrations_norm['train_in_demo_type'].append(demo_type)
                new_demonstrations_norm['train_out_demo_type'].append(demo_type)
                train_type_counts[demo_type] += 1
            else:
                # Add to testing
                new_demonstrations['test_in'].append(demo_in)
                new_demonstrations['test_out'].append(demo_out)
                new_demonstrations['test_in_demo_type'].append(demo_type)
                new_demonstrations['test_out_demo_type'].append(demo_type)
                new_demonstrations_norm['test_in'].append(demo_in_norm)
                new_demonstrations_norm['test_out'].append(demo_out_norm)
                new_demonstrations_norm['test_in_demo_type'].append(demo_type)
                new_demonstrations_norm['test_out_demo_type'].append(demo_type)

        return new_demonstrations, new_demonstrations_norm


    def split_train_test(self, demonstrations, demonstrations_normalized, test_ratio=0.2, original_train_indices=None):
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

        def cond_key(global_i):
            traj = demonstrations['train_in'][global_i]
            mean = traj.mean(axis=0) if hasattr(traj, 'mean') else np.mean(traj, axis=0)
            return (round(float(mean[-3]), 6), round(float(mean[-2]), 6), round(float(mean[-1]), 6))

        for demo_type, indices in type_to_indices.items():
            indices = sorted(indices, key=cond_key)

            if demo_type == 'original':
                if original_train_indices is None:
                    train_local = set(range(len(indices)))
                else:
                    train_local = set(original_train_indices)
            else:
                # Cluster consecutive sorted demos that share the same conditioning values
                clusters = []
                current_cluster = []
                prev_key = None
                for local_i, global_i in enumerate(indices):
                    key = cond_key(global_i)
                    if key != prev_key:
                        if current_cluster:
                            clusters.append(current_cluster)
                        current_cluster = [local_i]
                        prev_key = key
                    else:
                        current_cluster.append(local_i)
                if current_cluster:
                    clusters.append(current_cluster)

                # Evenly select (1-test_ratio) of the clusters for training,
                # spanning the full conditioning range
                n_clusters = len(clusters)
                n_train_clusters = max(1, round(n_clusters * (1 - test_ratio)))
                train_cluster_pos = set(
                    np.round(np.linspace(0, n_clusters - 1, n_train_clusters)).astype(int).tolist()
                )

                # Choose a subset of each cluster randomly
                percentage_to_keep_per_cluster = 0.8  # Keep all demos in selected clusters

                # For each cluster, randomly select a subset of demos to keep in the training set
                for pos, cluster in enumerate(clusters):
                    if pos in train_cluster_pos:
                        n_to_keep = max(1, round(len(cluster) * percentage_to_keep_per_cluster))
                        kept_indices = set(random.sample(cluster, n_to_keep))
                        cluster[:] = [i for i in cluster if i in kept_indices]
                    else:
                        cluster.clear()  # Clear the cluster to exclude all its demos from training
                

                # Selected clusters → all demos to train. Unselected → all demos to test.
                train_local = set()
                for pos, cluster in enumerate(clusters):
                    if pos in train_cluster_pos:
                        train_local.update(cluster)

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





    def plot_trajectories_colored(self, traj_q=None, traj_x=None, traj_dq=None,
                                        merged_task_space: bool = False, title_prefix: str = '', save_plots: bool = False,
                                        color_scheme=None, line_width=None, plot_equivariance_sampling_bounds_flag=False):
        """
        Plot joint positions, task-space positions and joint velocities
        in a single figure with tab-like buttons (single-click, robust).
        """
        traj_q  = _normalize_traj_input(traj_q)   # NOTE: not data normalization — just dict wrapping
        traj_x  = _normalize_traj_input(traj_x)
        traj_dq = _normalize_traj_input(traj_dq)

        fig  = plt.figure(figsize=(16, 8))
        ctrl = TabController(fig)
        network = self.network if plot_equivariance_sampling_bounds_flag else None

        def save_figure(suffix):
            if save_plots:
                # check if the robot is the rby1
                if 'rby1' in self.task:
                    path = ROOT_DIR / 'results_rby1' / 'plots' / f"{self.network.model_id}_{self.task}_{title_prefix.replace(' ', '_')}_{suffix}.png"
                else:
                    path = ROOT_DIR / 'results' / 'plots' / f"{self.network.model_id}{title_prefix.replace(' ', '_')}_{suffix}.png"
                path.parent.mkdir(parents=True, exist_ok=True)
                fig.savefig(path, dpi=300, bbox_inches='tight')

        # Resolve task_space for spatial plotting
        _task_space = None
        if merged_task_space:
            # if self.network is not None and hasattr(self.network, 'robot') and hasattr(self.network.robot, 'task_space'):
            #     _task_space = self.network.robot.task_space
            # else:
            #     _task_space = 'xy'
            _task_space = self.task_space

        def make_callback(key, traj_dict, ylabel_prefix, ts, title, save_suffix):
            def show():
                ctrl.clear()
                ctrl.activate(key)
                fig.suptitle(title, fontsize=16)
                if ts is not None:
                    ax = _plot_task_space(fig, traj_dict, ts, color_scheme, line_width, network)
                    ctrl.axes.append(ax)
                else:
                    nb_dims = next(iter(traj_dict.values()))[0].shape[1]
                    axs = _create_grid(fig, nb_dims)
                    plot_time_series(axs, traj_dict, ylabel_prefix, color_scheme, line_width)
                    ctrl.axes.extend(np.ravel(axs))
                save_figure(save_suffix)
            return show

        x_title = f'{title_prefix}: Task Space' if merged_task_space else f'{title_prefix}: End-Effector Position'
        panels = []
        if traj_x  is not None: panels.append(('x',  'Task space',       traj_x,  'EE',     _task_space, x_title,                             'task_space'))
        if traj_q  is not None: panels.append(('q',  'Joint positions',  traj_q,  'Joint',  None,        f'{title_prefix}: Joint Positions',  'joint_positions'))
        if traj_dq is not None: panels.append(('dq', 'Joint velocities', traj_dq, 'dJoint', None,        f'{title_prefix}: Joint Velocities', 'joint_velocities'))

        x0 = 0.25
        for key, label, traj_dict, ylabel_prefix, ts, title, save_suffix in panels:
            ctrl.register_button(key, label, make_callback(key, traj_dict, ylabel_prefix, ts, title, save_suffix), x0)
            x0 += 0.19

        ctrl.connect()
        ctrl.actions[next(iter(ctrl.actions))]()  # show first available tab

    def plot_loss_history(self, network, save_plots: bool = False):
        # Create a new figure to plot training loss
        fig3, ax3 = plt.subplots(figsize=(8,6))
        ax3.plot(self.network.loss_history, label='Training Loss')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Loss')
        ax3.set_title('Training Loss over Epochs')
        # ax3.set_ylim(0, max(network.loss_history)*1.1)
        # ax3.set_ylim(0, 0.1)
        ax3.legend()
        ax3.grid(True)

        # Save plot
        if save_plots:        
            filename = f"{self.network.model_id}_loss_history.png"
            filename = ROOT_DIR / 'results' / 'loss' / filename
            filename.parent.mkdir(parents=True, exist_ok=True)
            fig3.savefig(filename, dpi=300, bbox_inches='tight')
            # Save loss history as a text file
            loss_filename = f"{self.network.model_id}_loss_history.txt"
            loss_filename = ROOT_DIR / 'results' / 'loss' / loss_filename
            np.savetxt(loss_filename, np.array(self.network.loss_history), fmt='%.6f')

    def plot_1step_predictions(self, network, nb_train_demos, save_plots: bool = False, scale='norm'):
        nb_dofs = self.network.demonstrations['train_out'][0].shape[1] # Use train out data because it only has configuration space dimensions

        # Plot 1-step predictions for training data
        nrows_traj = (nb_dofs + 3) // 4
        nrows_total = nrows_traj * 2
        ncols = min(nb_dofs, 4)
        fig_train, axs_train = plt.subplots(nrows=nrows_total, ncols=ncols, figsize=(16, 4 * nrows_total))
        fig_train.suptitle('1-Step Predictions on Training Data', fontsize=16)
        train_in_data = self.network.demonstrations['train_in']
        train_out_data = [traj.cpu() for traj in self.network.demonstrations['train_out']]
        # Feed the training data through the model to get predictions
        train_out_pred = [self.network.model(traj_in.clone()).detach().cpu().numpy() for traj_in in train_in_data]


        if scale == 'real':
            # Denormalize outputs
            train_out_data_denorm = [denormalize_state(traj, self.network.demonstrations['Dq_min'], self.network.demonstrations['Dq_max']) for traj in train_out_data]
            train_out_pred_denorm = [denormalize_state(traj, self.network.demonstrations['Dq_min'], self.network.demonstrations['Dq_max']) for traj in train_out_pred]
            
            # Denormalize state derivatives (velocities)
            train_out_data = [denormalize_state_derivative(traj, self.network.demonstrations['Q_min'], self.network.demonstrations['Q_max']) for traj in train_out_data_denorm]
            train_out_pred = [denormalize_state_derivative(traj, self.network.demonstrations['Q_min'], self.network.demonstrations['Q_max']) for traj in train_out_pred_denorm]

        cmap_train = plt.cm.get_cmap('tab10', len(train_in_data))
        # Plot the true vs predicted velocities for each joint
        for i in range(nb_dofs):
            # Top half: trajectory axes
            ax_traj = axs_train[i // 4, i % 4] if nb_dofs > 4 else axs_train[i // 4, i % 4]
            # Bottom half: error axes (offset by nrows_traj)
            ax_err  = axs_train[nrows_traj + i // 4, i % 4]

            for j in range(nb_train_demos):
                color = cmap_train(j)
                traj_true = train_out_data[j]
                traj_pred = train_out_pred[j]
                error = traj_true[:, i] - traj_pred[:, i]
                time_steps = np.arange(traj_true.shape[0])

                # --- Trajectory plot ---
                ax_traj.plot(time_steps, traj_true[:, i],
                            label=f'Traj {j+1}' if i == 0 else "",
                            color=color, linewidth=2)
                ax_traj.fill_between(time_steps,
                                    traj_true[:, i],
                                    traj_pred[:, i],
                                    color=color, alpha=0.25)

                # --- Error plot ---
                ax_err.plot(time_steps, error,
                            label=f'Traj {j+1}' if i == 0 else "",
                            color=color, linewidth=1.5)

            # Style trajectory axis
            ax_traj.set_ylabel(f'Joint {i+1} Velocity')
            ax_traj.set_title(f'Joint {i+1}')
            ax_traj.grid()
            ax_traj.set_facecolor('#f0f0f0')
            ax_traj.set_ylim(-torch.max(torch.abs(traj_true)) * 1.1,
                            torch.max(torch.abs(traj_true)) * 1.1)
            ax_traj.tick_params(labelbottom=False)  # Hide x-ticks, shared with error plot below

            # Style error axis
            ax_err.axhline(0, color='k', linewidth=0.8, linestyle='--')
            ax_err.set_xlabel('Time Step')
            ax_err.set_ylabel(f'Error J{i+1}')
            ax_err.grid()
            ax_err.set_facecolor('#f0f0f0')


        # Hide any unused axes
        for idx in range(nb_dofs, nrows_traj * ncols):
            axs_train[idx // ncols, idx % ncols].set_visible(False)
            axs_train[nrows_traj + idx // ncols, idx % ncols].set_visible(False)

        # Shared legend for trajectories
        handles, labels = axs_train[0, 0].get_legend_handles_labels()
        fig_train.legend(handles, labels, loc='lower center',
                        ncol=min(nb_train_demos, 5), bbox_to_anchor=(0.5, 0.0))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # ax.legend()
        # plt.tight_layout(rect=[0, 0.03, 1, 0.95])





        # Save plot
        if save_plots:
            filename = f"{network.model_id}_1step_predictions_train.png"
            filename = ROOT_DIR / 'results' / 'plots' / filename
            filename.parent.mkdir(parents=True, exist_ok=True)
            fig_train.savefig(filename, dpi=300, bbox_inches='tight')

        # Plot 1-step predictions for testing data
        fig_test, axs_test = plt.subplots(nrows=(nb_dofs + 3)//4, ncols=min(nb_dofs,4), figsize=(16, 4 * ((nb_dofs + 3)//4)))
        fig_test.suptitle('1-Step Predictions on Testing Data', fontsize=16)
        test_in_data = network.demonstrations['test_in']   # These are augmented inputs
        # test_out_data = network.demonstrations['test_out'] # These are augmented outputs
        test_out_data = [traj.cpu() for traj in network.demonstrations['test_out']] # These are augmented outputs
        # Feed the testing data through the model to get predictions
        # test_out_pred = [network.model(torch.tensor(traj_in, dtype=torch.float32)).detach() for traj_in in test_in_data]
        test_out_pred = [network.model(traj_in.clone()).detach().cpu().numpy() for traj_in in test_in_data]

        # Plot the true vs predicted velocities for each joint
        for i in range(nb_dofs):
            ax = axs_test[i//4, i%4] if nb_dofs > 4 else axs_test[i]
            for j in range(len(test_in_data)):
                traj_true = test_out_data[j]
                traj_pred = test_out_pred[j]
                time_steps = np.arange(traj_true.shape[0])
                ax.plot(time_steps, traj_true[:, i], label='True' if j == 0 else "", color='blue')
                ax.plot(time_steps, traj_pred[:, i], label='Predicted' if j == 0 else "", linestyle='dashed', color='red')
            ax.set_xlabel('Time Step')
            ax.set_ylabel(f'Joint {i+1} Velocity')
            ax.grid()
            ax.set_facecolor('#f0f0f0')
            # ax.set_ylim(-np.max(np.abs(traj_true))*1.1, np.max(np.abs(traj_true))*1.1)
            ax.set_ylim(-torch.max(torch.abs(traj_true))*1.1, torch.max(torch.abs(traj_true))*1.1)
        ax.legend()
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])

        # Save plot
        if save_plots:
            filename = f"{network.model_id}_1step_predictions_test.png"
            filename = ROOT_DIR / 'results' / 'plots' / filename
            filename.parent.mkdir(parents=True, exist_ok=True)
            fig_test.savefig(filename, dpi=300, bbox_inches='tight')


    def get_X_from_Q_trajectories(self, fk_func_left, fk_func_right, traj_q_list, nb_dofs_left, nb_dofs_right):
        x_trajectories = []
        for traj_q in traj_q_list:
            x_traj = []
            for q in traj_q:
                q_left = q[:nb_dofs_left]
                q_right = q[nb_dofs_left:nb_dofs_left+nb_dofs_right]
                x_left = fk_func_left(np.array(q_left))
                x_right = fk_func_right(np.array(q_right))
                x_traj.append( np.hstack([x_left, x_right]) )
            x_trajectories.append( np.array(x_traj) )
        return x_trajectories
    
    def get_X_from_Q_trajectories_torch(self, fk_func_left, fk_func_right, traj_q_list, nb_dofs_left, nb_dofs_right):
        x_trajectories = []
        for traj_q in traj_q_list:
            x_traj_left = fk_func_left(traj_q[:, :nb_dofs_left])
            x_traj_right = fk_func_right(traj_q[:, nb_dofs_left:nb_dofs_left+nb_dofs_right])
            x_traj = torch.hstack([x_traj_left, x_traj_right])
            x_trajectories.append(x_traj)
        return x_trajectories

    def get_TX_from_TQ_trajectories(self, J_left, J_right, traj_q_list, traj_dq_list, nb_dofs_left, nb_dofs_right):
        tx_trajectories = []
        for traj_q, traj_dq in zip(traj_q_list, traj_dq_list):
            tx_traj = []
            for q, dq in zip(traj_q, traj_dq):
                q_left = q[:nb_dofs_left]
                q_right = q[nb_dofs_left:nb_dofs_left+nb_dofs_right]
                dq_left = dq[:nb_dofs_left]
                dq_right = dq[nb_dofs_left:nb_dofs_left+nb_dofs_right]
                J_l = J_left(np.array(q_left))
                J_r = J_right(np.array(q_right))
                x_dot_left = J_l @ np.array(dq_left)
                x_dot_right = J_r @ np.array(dq_right)
                tx_traj.append( np.hstack([x_dot_left, x_dot_right]) )
            tx_trajectories.append( np.array(tx_traj) )
        return tx_trajectories

    def get_TX_from_TQ_trajectories_torch(self, J_left, J_right, traj_q_list, traj_dq_list, nb_dofs_left, nb_dofs_right):
        tx_trajectories = []
        for traj_q, traj_dq in zip(traj_q_list, traj_dq_list):
            J_l = J_left(traj_q[:, :nb_dofs_left])
            J_r = J_right(traj_q[:, nb_dofs_left:nb_dofs_left+nb_dofs_right])
            x_dot_left = torch.bmm(J_l, traj_dq[:, :nb_dofs_left].unsqueeze(-1)).squeeze(-1)
            x_dot_right = torch.bmm(J_r, traj_dq[:, nb_dofs_left:nb_dofs_left+nb_dofs_right].unsqueeze(-1)).squeeze(-1)
            tx_traj = torch.hstack([x_dot_left, x_dot_right])
            tx_trajectories.append(tx_traj)
        return tx_trajectories


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


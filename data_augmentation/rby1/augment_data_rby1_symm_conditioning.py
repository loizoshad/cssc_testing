import os
from pathlib import Path
import numpy as np
import copy
import shutil
import time

import torch
from utils.groups import *
from utils.utils import *

cols = shutil.get_terminal_size().columns
np.set_printoptions(precision=4, suppress=True, linewidth=cols)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = Path(CURRENT_DIR).parent.parent.resolve()
device      = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f'device: {device}')


### Select group

## pan_v1 dataset (xyz_theta task space): # These carry their own task_space/demo_folder/task so TASK_SPACE above does not need to be changed when switching between pan and letters.
# GROUP_NAME = 'C2Pan'
GROUP_NAME = 'SO2Pan'
# GROUP_NAME = 'C2SO2Pan'

## letters dataset (yz_in_6d task space)
# GROUP_NAME = 'SO2Letters'
# GROUP_NAME = 'Scaling2Letters'
# GROUP_NAME = 'C2Letters'
# GROUP_NAME = 'SO2Scaling2Letters'
# GROUP_NAME = 'C2SO2Scaling2Letters'



# Pan dataset constants (task_space='xyz_theta')
_PAN_TASK   = 'left-rby1-all_right-rby1-all_ndofs-14'
_PAN_FOLDER = 'rby1_pan_v1'
_PAN_TS     = 'xyz_theta'   # [x, y, z, yaw] per arm — planar pan experiment

# ── Letters dataset factory functions (task_space='yz_in_6d') ─────────────────
_LETTERS_TASK   = 'left-rby1-letters_right-rby1-letters_ndofs-14'
_LETTERS_FOLDER = 'rby1_letters'
_LETTERS_TS     = 'yz_in_6d'


# TASK_SPACE = 'xyz_theta'
TASK_SPACE = 'yz'
# Integration step sizes — tune independently for rotation and scaling.
# Smaller dt -> more Euler steps -> more accurate but slower augmentation.
# DT_SO2      = np.deg2rad(0.5)

# DT_SO2      = np.deg2rad(5.0)
# DT_SCALING2 = np.deg2rad(5.0)
DT_SO2      = np.deg2rad(5.0)
DT_SCALING2 = 0.1
IS_TASKSPACE = False
# task_space   = 'xy'   # only used when is_taskspace=False

ROBOT = create_rby1_robot(task_space=TASK_SPACE, ee_joint=False, is_taskspace=IS_TASKSPACE)


def augment_demonstrations_discrete(group, rep_in, rep_out, demonstrations: dict, augment_test: bool = False):
    """
    Alternative version, to augment both train and test data.
    """
    for key in ['train_in', 'train_out', 'test_in', 'test_out']:
        demonstrations[key] = [
            traj.to(device) if isinstance(traj, torch.Tensor)
            else torch.tensor(traj, dtype=torch.float32, device=device)
            for traj in demonstrations[key]
        ]

    if 'train_in_demo_type' not in demonstrations:
        for key in ['train_in', 'train_out', 'test_in', 'test_out']:
            demonstrations[f'{key}_demo_type'] = ['original'] * len(demonstrations[key])

    num_train_traj = len(demonstrations['train_in'])
    nb_q, nb_x     = demonstrations['nb_q'], demonstrations['nb_x']
    augmented      = copy.deepcopy(demonstrations)
    group_str      = f'{group}'

    splits = ['train', 'test'] if augment_test else ['train']
    print(f'Augmenting {splits} with {len(group.discrete_generators)} discrete generator(s) | '
          f'train: {num_train_traj}, test: {len(demonstrations["test_in"])}')

    for gen in group.discrete_generators:
        g = torch.tensor(gen, dtype=torch.float32, device=device)
        new_in, new_out = [], []

        for split in splits:
            for traj_in, traj_out in zip(demonstrations[f'{split}_in'], demonstrations[f'{split}_out']):
                q0 = traj_in[:, :nb_q]
                x0 = traj_in[:, nb_q:nb_q+nb_x]
                v0 = traj_out[:, :nb_q]

                aug_in  = rep_in.act_on( g, q0=q0, x0=x0, full_traj=False)
                aug_out = rep_out.act_on(g, q0=q0, v0=v0, full_traj=False)

                new_in.append(aug_in)
                new_out.append(aug_out)

        # All augmented trajectories go to test regardless of source split
        augmented['test_in'].extend(new_in)
        augmented['test_out'].extend(new_out)
        for key in ['train_in', 'train_out', 'test_in', 'test_out']:
            augmented[f'{key}_demo_type'].extend([group_str] * len(new_in))

        print(f'  Generator applied — {len(new_in)} new test trajs | test total: {len(augmented["test_in"])}')

    return augmented

def build_group_elements(group_str: str, bounds: dict) -> tuple[list, list]:
    """
    Discretize the group action space into a finite set of group elements.
    
    Parameters
    ----------
    group_str : str
        Name of the group.
    bounds : dict
        For SO2:             {'max_angle': float}          (radians)
        For Scaling2:        {'min_scale': float, 'max_scale': float}
        For SO2Scaling2Group: both of the above combined
    Returns
    -------
    g_list   : list of np.ndarray  — the group elements to apply
    g_labels : list of str         — label for each element
    """
    def _rot(a):
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
    def _scale(s):
        return np.eye(2) * s

    g_list, g_labels = [], []

    if group_str == 'SO2RBY1':
        max_angle = bounds['max_angle']
        # Set the angles to be the max_angle and -max_angle only
        angles = np.array([-max_angle, max_angle])
        # angles = np.array([max_angle])
        for a in angles:
            g_list.append(_rot(a))
            g_labels.append('SO2RBY1')
    elif group_str == 'Scaling2RBY1':
        # Use only the minimum and maximum scale factors, skipping s≈1 (identity).
        min_s, max_s = bounds['min_scale'], bounds['max_scale']
        for s in [min_s, max_s]:
            if abs(s - 1.0) < 1e-6:
                continue   # identity element — no augmentation value
            g_list.append(_scale(s))
            g_labels.append('Scaling2RBY1')
    elif group_str == 'C2SO2RBY1':
        max_angle = bounds['max_angle']
        c2_2d = np.array([[-1., 0.], [0., 1.]])
        for a in [-max_angle, max_angle]:
            g_list.append(_rot(a))
            g_labels.append('SO2RBY1')
            g_list.append(c2_2d @ _rot(a))
            g_labels.append('C2SO2RBY1')
        # Pure C2 reflection (zero rotation component)
        g_list.append(c2_2d.copy())
        g_labels.append('C2RBY1')
    elif group_str == 'C2Scaling2RBY1':
        min_s, max_s = bounds['min_scale'], bounds['max_scale']
        c2_2d = np.array([[-1., 0.], [0., 1.]])
        for s in [min_s, max_s]:
            if abs(s - 1.0) < 1e-6:
                continue   # skip identity scale
            g_list.append(_scale(s))
            g_labels.append('Scaling2RBY1')
            g_list.append(c2_2d @ _scale(s))
            g_labels.append('C2Scaling2RBY1')
        g_list.append(c2_2d.copy())
        g_labels.append('C2RBY1')

    # ── Letters dataset ────────────────────────────────────────────────────────
    elif group_str == 'SO2Scaling2RBY1Letters':
        ma      = bounds['max_angle']
        scales  = [s for s in [bounds.get('min_scale', 1.0), bounds.get('max_scale', 1.0)]
                   if abs(s - 1.0) > 1e-6]
        for a in [-ma, ma]:
            g_list.append(_rot(a));              g_labels.append('SO2Letters')
        for s in scales:
            g_list.append(_scale(s));            g_labels.append('Scaling2Letters')
        for s in scales:
            for a in [-ma, ma]:
                g_list.append(_rot(a) @ _scale(s)); g_labels.append('SO2Scaling2Letters')
    elif group_str == 'C2SO2Scaling2RBY1':
        ma     = bounds['max_angle']
        scales = [s for s in [bounds.get('min_scale', 1.0), bounds.get('max_scale', 1.0)]
                  if abs(s - 1.0) > 1e-6]
        c2_2d  = np.array([[-1., 0.], [0., 1.]])
        # Pure rotations (±angle, with and without C2)
        for a in [-ma, ma]:
            g_list.append(_rot(a));               g_labels.append('SO2Letters')
            g_list.append(c2_2d @ _rot(a));       g_labels.append('C2SO2Letters')
        # Pure scales (each scale factor, with and without C2)
        for s in scales:
            g_list.append(_scale(s));             g_labels.append('Scaling2Letters')
            g_list.append(c2_2d @ _scale(s));     g_labels.append('C2Scaling2Letters')
        # Combined rotation × scale (each combination, with and without C2)
        for s in scales:
            for a in [-ma, ma]:
                g = _rot(a) @ _scale(s)
                g_list.append(g);                 g_labels.append('SO2Scaling2Letters')
                g_list.append(c2_2d @ g);         g_labels.append('C2SO2Scaling2Letters')
        # Pure C2 morphological swap
        g_list.append(c2_2d.copy());              g_labels.append('C2Letters')

    else:
        raise ValueError(f"Unknown group: {group_str}")

    return g_list, g_labels

def augment_demonstrations(group, rep_in, rep_out, demonstrations: dict, bounds: dict):
    """
    Augment demonstrations by applying a discretized set of group elements
    directly to the original training data.

    This version returns the whole trajectory to make use of the integration and avoid recomputing augmented states.

    Parameters
    ----------
    group    : Group instance
    rep_in   : input representation
    rep_out  : output representation
    demonstrations : dict with keys train_in, train_out, test_in, test_out, nb_q, nb_x
    bounds   : dict describing the range of group actions, passed to build_group_elements
    """
    for key in ['train_in', 'train_out']:
        # Ensure all trajectories are torch tensors on the correct device
        demonstrations[key] = [traj.to(device) if isinstance(traj, torch.Tensor) else torch.tensor(traj, dtype=torch.float32, device=device) for traj in demonstrations[key]]

    nb_q, nb_x     = demonstrations['nb_q'], demonstrations['nb_x']
    augmented      = copy.deepcopy(demonstrations)
    group_str      = f'{group}'

    # Use the class name for element lookup so custom-named groups still resolve correctly.
    g_list, g_labels = build_group_elements(type(group).__name__, bounds)
    print(f'Augmenting with {len(g_list)} group elements | ' f'train: {len(demonstrations["train_in"])}, test: {len(demonstrations["test_in"])}')

    total_t0 = time.perf_counter()
    for i, (g_, g_label) in enumerate(zip(g_list, g_labels)):
        t0       = time.perf_counter()
        g_tensor = torch.tensor(g_, dtype=torch.float32, device=device)
        new_in, new_out = [], []

        # Always apply to the ORIGINAL demonstrations only
        for traj_in, traj_out in zip(demonstrations['train_in'], demonstrations['train_out']):
            q0 = traj_in[:, :nb_q]
            x0 = traj_in[:, nb_q:nb_q+nb_x]
            v0 = traj_out[:, :nb_q]

            aug_in  = rep_in.act_on( g_tensor, q0=q0, x0=x0, full_traj=True)
            aug_out = rep_out.act_on(g_tensor, q0=q0, v0=v0, full_traj=True)

            # Step 0 of act_on(full_traj=True) is always the original config (q0,
            # zero group action applied).  Skip it to avoid duplicating the original.
            aug_in_list  = [aug_in[t]  for t in range(1, aug_in.shape[0])]
            aug_out_list = [aug_out[t] for t in range(1, aug_out.shape[0])]

            new_in += aug_in_list
            new_out += aug_out_list

        augmented['train_in'].extend(new_in)
        augmented['train_out'].extend(new_out)

        num_new = len(new_in)  # number of new trajectories added for this group element
        # for key in ['train_in', 'train_out', 'test_in', 'test_out']:
        for key in ['train_in', 'train_out']:
            augmented[f'{key}_demo_type'].extend([group_str] * num_new)

        if (i + 1) % 10 == 0 or i == len(g_list) - 1:
            print(f'  [{i+1}/{len(g_list)}] {g_label} — {num_new} new trajs ' f'({time.perf_counter()-t0:.2f}s) | test total: {len(augmented["test_in"])}')

    print(f'Total augmentation time: {time.perf_counter()-total_t0:.2f}s')

    return augmented


# For RBY1 Letters

# Works for individual symmetries but earlier it failed on the combination
# theta_bound = 180   # degrees
# SCALE_BOUND_MIN=0.5
# SCALE_BOUND_MAX=2

# Works for SO2+Scaling2
# theta_bound = 45   # degrees
# SCALE_BOUND_MIN=0.5
# SCALE_BOUND_MAX=1.0


# theta_bound = 90   # degrees
# SCALE_BOUND_MIN=0.5
# SCALE_BOUND_MAX=1.0

# ## RBY1 Letter drawing Setup
# theta_bound = 180   # degrees
# SCALE_BOUND_MIN=0.1
# SCALE_BOUND_MAX=1.0

## RBY1 Pan Grasp Setup
theta_bound = 90   # degrees
SCALE_BOUND_MIN=0.1
SCALE_BOUND_MAX=1.0



# SCALE_BOUND_MIN = -1.5
# theta_bound = 90
# theta_bound = 30   # degrees
# theta_bound = 20   # degrees
# SCALE_BOUND_MIN = 0.4
PERM = np.array([1., -1., -1., 1., -1., 1., -1.])   # RBY1 joint sign flips


## Per-group functions

def _build_C2RBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    G  = C2RBY1(permutator=np.array([1, 1]))
    kw = dict(G=G, robot=robot, is_vf_constant=False, dt=dt, is_taskspace=False)
    return G, C2RBY1RepIn(**kw), C2RBY1RepOut(**kw)

def _build_SO2RBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    _dt = dt_so2 if dt_so2 is not None else dt
    G   = SO2RBY1()
    kw  = dict(G=G, robot=robot, is_vf_constant=False, dt=_dt, is_taskspace=False, task_space=task_space)
    return G, SO2RBY1RepIn(**kw), SO2RBY1RepOut(**kw)

def _build_Scaling2RBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    _dt = dt_scaling2 if dt_scaling2 is not None else dt
    G   = Scaling2RBY1()
    kw  = dict(G=G, robot=robot, is_vf_constant=False, dt=_dt, is_taskspace=False, task_space=task_space)
    return G, Scaling2RBY1RepIn(**kw), Scaling2RBY1RepOut(**kw)

def _build_C2SO2RBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    _dt = dt_so2 if dt_so2 is not None else dt
    G   = C2SO2RBY1()
    kw  = dict(G=G, robot=robot, is_vf_constant=False, dt=_dt, is_taskspace=False, task_space=task_space)
    return G, C2SO2RBY1RepIn(**kw), C2SO2RBY1RepOut(**kw)

def _build_C2Scaling2RBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    _dt = dt_scaling2 if dt_scaling2 is not None else dt
    G   = C2Scaling2RBY1()
    kw  = dict(G=G, robot=robot, is_vf_constant=False, dt=_dt, is_taskspace=False, task_space=task_space)
    return G, C2Scaling2RBY1RepIn(**kw), C2Scaling2RBY1RepOut(**kw)

# Per-arm composite symmetry configurations

def _build_C2LeftC2SO2RightRBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    G  = C2SO2RBY1(name='C2LeftC2SO2RightRBY1')
    kw = dict(G=G, robot=robot, dt=dt, morph=True, perm_vector=PERM,
              dt_so2=dt_so2, dt_scaling2=dt_scaling2,
              left_spec =ArmSymSpec(so2=False, task_space=task_space, center_on_ee=True),
              right_spec=ArmSymSpec(so2=True,  task_space=task_space, center_on_ee=True))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)

def _build_C2SO2LeftC2RightRBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    G  = C2SO2RBY1(name='C2SO2LeftC2RightRBY1')
    kw = dict(G=G, robot=robot, dt=dt, morph=True, perm_vector=PERM,
              dt_so2=dt_so2, dt_scaling2=dt_scaling2,
              left_spec =ArmSymSpec(so2=True,  task_space=task_space, center_on_ee=True),
              right_spec=ArmSymSpec(so2=False, task_space=task_space, center_on_ee=True))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)

def _build_C2LeftC2SO2Scaling2RightRBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    G  = C2SO2Scaling2RBY1(name='C2LeftC2SO2Scaling2RightRBY1')
    kw = dict(G=G, robot=robot, dt=dt, morph=True, perm_vector=PERM,
              dt_so2=dt_so2, dt_scaling2=dt_scaling2,
              left_spec =ArmSymSpec(so2=False, scaling2=False, task_space=task_space, center_on_ee=True),
              right_spec=ArmSymSpec(so2=True,  scaling2=True,  task_space=task_space, center_on_ee=True))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)

def _build_C2SO2Scaling2LeftC2RightRBY1(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    G  = C2SO2Scaling2RBY1(name='C2SO2Scaling2LeftC2RightRBY1')
    kw = dict(G=G, robot=robot, dt=dt, morph=True, perm_vector=PERM,
              dt_so2=dt_so2, dt_scaling2=dt_scaling2,
              left_spec =ArmSymSpec(so2=True,  scaling2=True,  task_space=task_space, center_on_ee=True),
              right_spec=ArmSymSpec(so2=False, scaling2=False, task_space=task_space, center_on_ee=True))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)


def _build_SO2Pan(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    """SO2 augmentation for the pan dataset.

    Both arms rotate in the XY plane using _so2_xyz_theta_vf, with the
    rotation centre set to each arm's initial EE position (center_on_ee=True).
    This prevents the whole trajectory from drifting away from the workspace
    when the origin is far from the actual EE positions.
    """
    _dt = dt_so2 if dt_so2 is not None else dt
    G   = SO2RBY1()
    kw  = dict(G=G, robot=robot, dt=_dt, morph=False, dt_so2=_dt,
               left_spec =ArmSymSpec(so2=True, task_space=task_space, center_on_ee=True),
               right_spec=ArmSymSpec(so2=True, task_space=task_space, center_on_ee=True))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)

def _build_C2SO2Pan(robot, dt, task_space, dt_so2=None, dt_scaling2=None, **kwargs):
    """C2+SO2 augmentation for the pan dataset (both arms symmetric).

    Both arms rotate in the XY plane with center_on_ee=True, plus the
    morphological C2 arm-swap (morph=True) for the C2 component.
    """
    _dt = dt_so2 if dt_so2 is not None else dt
    G   = C2SO2RBY1()
    kw  = dict(G=G, robot=robot, dt=_dt, morph=True, perm_vector=PERM, dt_so2=_dt,
               left_spec =ArmSymSpec(so2=True, task_space=task_space, center_on_ee=True),
               right_spec=ArmSymSpec(so2=True, task_space=task_space, center_on_ee=True))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)

def compute_letter_centers(task: str, demo_folder: str):
    """Compute the convex-hull centroid of the EE letter trajectories for each arm.

    Reads the preprocessed .npz (x_data layout: [y_l, z_l, y_r, z_r]) and returns
    one (2,) numpy array per arm representing the centroid of the convex hull of
    all EE positions across all demonstrations.

    Returns
    -------
    center_left  : np.ndarray, shape (2,)  — [y_c, z_c] for the left arm
    center_right : np.ndarray, shape (2,)  — [y_c, z_c] for the right arm
    """
    from scipy.spatial import ConvexHull

    npz_path = ROOT_DIR / 'demonstrations' / demo_folder / (task + '.npz')
    raw      = np.load(npz_path)
    x_data   = raw['x_data']          # (N, T, 4): [y_l, z_l, y_r, z_r]
    pts      = x_data.reshape(-1, x_data.shape[-1])  # (N*T, 4)

    def hull_centroid(p: np.ndarray) -> np.ndarray:
        try:
            hull = ConvexHull(p)
            return p[hull.vertices].mean(axis=0)
        except Exception:
            return p.mean(axis=0)

    center_left  = hull_centroid(pts[:, :2])
    center_right = hull_centroid(pts[:, 2:4])
    return center_left, center_right

def _build_SO2Letters(robot, dt, task_space, center_left=None, center_right=None,
                      dt_so2=None, dt_scaling2=None):
    _dt = dt_so2 if dt_so2 is not None else dt
    G   = SO2RBY1(name='SO2Letters')
    kw  = dict(G=G, robot=robot, is_vf_constant=False, dt=_dt,
               is_taskspace=False, task_space=task_space,
               center_left=center_left, center_right=center_right)
    return G, SO2RBY1RepIn(**kw), SO2RBY1RepOut(**kw)

def _build_Scaling2Letters(robot, dt, task_space, center_left=None, center_right=None,
                            dt_so2=None, dt_scaling2=None):
    _dt = dt_scaling2 if dt_scaling2 is not None else dt
    G   = Scaling2RBY1(name='Scaling2Letters')
    kw  = dict(G=G, robot=robot, is_vf_constant=False, dt=_dt,
               is_taskspace=False, task_space=task_space,
               center_left=center_left, center_right=center_right)
    return G, Scaling2RBY1RepIn(**kw), Scaling2RBY1RepOut(**kw)

def _build_C2Letters(robot, dt, task_space, center_left=None, center_right=None,
                     dt_so2=None, dt_scaling2=None):
    G  = C2RBY1(permutator=np.array([1, 1]), name='C2Letters')
    kw = dict(G=G, robot=robot, is_vf_constant=False, dt=dt, is_taskspace=False)
    return G, C2RBY1RepIn(**kw), C2RBY1RepOut(**kw)

def _build_SO2Scaling2Letters(robot, dt, task_space, center_left=None, center_right=None,
                               dt_so2=None, dt_scaling2=None):
    G  = SO2Scaling2RBY1Letters(name='SO2Scaling2Letters')
    kw = dict(G=G, robot=robot, dt=dt, morph=False,
              dt_so2=dt_so2, dt_scaling2=dt_scaling2,
              left_spec =ArmSymSpec(so2=True, scaling2=True, task_space=task_space,
                                   center=center_left),
              right_spec=ArmSymSpec(so2=True, scaling2=True, task_space=task_space,
                                   center=center_right))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)

def _build_C2SO2Scaling2Letters(robot, dt, task_space, center_left=None, center_right=None,
                                 dt_so2=None, dt_scaling2=None):
    G  = C2SO2Scaling2RBY1(name='C2SO2Scaling2Letters')
    kw = dict(G=G, robot=robot, dt=dt, morph=True, perm_vector=PERM,
              dt_so2=dt_so2, dt_scaling2=dt_scaling2,
              left_spec =ArmSymSpec(so2=True, scaling2=True, task_space=task_space,
                                   center=center_left),
              right_spec=ArmSymSpec(so2=True, scaling2=True, task_space=task_space,
                                   center=center_right))
    return G, CompositeRepIn(**kw), CompositeRepOut(**kw)


GROUP_CONFIGS = {
    # ── Standard (both-arm) symmetries ───────────────────────────────────────
    'C2RBY1':         dict(discrete=True,  bounds={},
                           build=_build_C2RBY1),
    'SO2RBY1':        dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                           build=_build_SO2RBY1),
    'Scaling2RBY1':   dict(discrete=False, bounds={'min_scale': SCALE_BOUND_MIN, 'max_scale': SCALE_BOUND_MAX},
                           build=_build_Scaling2RBY1),
    'C2SO2RBY1':      dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                           build=_build_C2SO2RBY1),
    'C2Scaling2RBY1': dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound),
                                                   'min_scale': SCALE_BOUND_MIN, 'max_scale': SCALE_BOUND_MAX},
                           build=_build_C2Scaling2RBY1),

    # ── Per-arm composite symmetries ──────────────────────────────────────────
    'C2LeftC2SO2RightRBY1': dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                               build=_build_C2LeftC2SO2RightRBY1),
    'C2SO2LeftC2RightRBY1': dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                               build=_build_C2SO2LeftC2RightRBY1),    

    'C2LeftC2SO2Scaling2RightRBY1': dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                               build=_build_C2LeftC2SO2Scaling2RightRBY1),
    'C2SO2Scaling2LeftC2RightRBY1': dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                               build=_build_C2SO2Scaling2LeftC2RightRBY1),

    # ── Pan dataset (XY-plane rotation, xyz_theta task space) ────────────────
    # Uses _so2_xyz_theta_vf: dx=-(y-y0), dy=(x-x0), dz=0, dyaw=1.
    # center_on_ee=True anchors each arm's rotation at its own initial EE
    # position so the trajectory stays in the workspace after augmentation.
    'C2Pan':    dict(discrete=True,  bounds={},
                    build=_build_C2RBY1,
                    task_space=_PAN_TS, demo_folder=_PAN_FOLDER, task=_PAN_TASK),
    'SO2Pan':   dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                    build=_build_SO2Pan,
                    task_space=_PAN_TS, demo_folder=_PAN_FOLDER, task=_PAN_TASK),
    'C2SO2Pan': dict(discrete=False, bounds={'max_angle': np.deg2rad(theta_bound)},
                    build=_build_C2SO2Pan,
                    task_space=_PAN_TS, demo_folder=_PAN_FOLDER, task=_PAN_TASK),

    # ── Letters dataset (Y–Z task space) ──────────────────────────────────────
    'SO2Letters':         dict(discrete=False,
                               bounds={'max_angle': np.deg2rad(theta_bound)},
                               build=_build_SO2Letters,
                               task_space=_LETTERS_TS, demo_folder=_LETTERS_FOLDER, task=_LETTERS_TASK),
    'Scaling2Letters':    dict(discrete=False,
                               bounds={'min_scale': SCALE_BOUND_MIN, 'max_scale': SCALE_BOUND_MAX},
                               build=_build_Scaling2Letters,
                               task_space=_LETTERS_TS, demo_folder=_LETTERS_FOLDER, task=_LETTERS_TASK),
    'C2Letters':          dict(discrete=True, bounds={},
                               build=_build_C2Letters,
                               task_space=_LETTERS_TS, demo_folder=_LETTERS_FOLDER, task=_LETTERS_TASK),
    'SO2Scaling2Letters': dict(discrete=False,
                               bounds={'max_angle': np.deg2rad(theta_bound),
                                       'min_scale': SCALE_BOUND_MIN, 'max_scale': SCALE_BOUND_MAX},
                               build=_build_SO2Scaling2Letters,
                               task_space=_LETTERS_TS, demo_folder=_LETTERS_FOLDER, task=_LETTERS_TASK),
    'C2SO2Scaling2Letters': dict(discrete=False,
                                 bounds={'max_angle': np.deg2rad(theta_bound),
                                         'min_scale': SCALE_BOUND_MIN, 'max_scale': SCALE_BOUND_MAX},
                                 build=_build_C2SO2Scaling2Letters,
                                 task_space=_LETTERS_TS, demo_folder=_LETTERS_FOLDER, task=_LETTERS_TASK),
}


if __name__ == '__main__':
    conditioning = 'symmetry'
    normalize_conditioning = False

    cfg        = GROUP_CONFIGS[GROUP_NAME]
    # Per-config overrides (letters configs carry their own task_space / folder / task)
    task_space  = cfg.get('task_space',  TASK_SPACE)
    demo_folder = cfg.get('demo_folder', 'rby1_pan_v1')
    task        = cfg.get('task',        'left-rby1-all_right-rby1-all_ndofs-14')
    if IS_TASKSPACE and 'task_space' not in cfg:
        task += '_taskspace'

    # Build robot with the correct task space for this config
    robot = create_rby1_robot(task_space=task_space, ee_joint=False, is_taskspace=IS_TASKSPACE)

    predefined_dataset = None
    data_vis = DataVisualizer(task=task, conditioning=conditioning,
                              normalize_conditioning=normalize_conditioning,
                              task_space=task_space)
    demonstrations, _ = data_vis.construct_demonstrations(
        demo_folder=demo_folder, load_augmented_data=False,
        predefined_dataset=predefined_dataset)

    # ── Compute per-arm letter centers (letters groups only) ─────────────────
    _LETTER_GROUPS = {'SO2Letters', 'Scaling2Letters', 'SO2Scaling2Letters', 'C2SO2Scaling2Letters'}
    if GROUP_NAME in _LETTER_GROUPS:
        center_left, center_right = compute_letter_centers(task, demo_folder)
        print(f'Letter EE centers (convex-hull centroid):')
        print(f'  left  arm: y={center_left[0]:.4f}  z={center_left[1]:.4f}')
        print(f'  right arm: y={center_right[0]:.4f}  z={center_right[1]:.4f}')
    else:
        center_left = center_right = None

    # ── Build group and representations ──────────────────────────────────────
    print(f'DT_SO2={np.rad2deg(DT_SO2):.2f}°  DT_SCALING2={DT_SCALING2:.5f} (dimensionless)')
    G, rep_in, rep_out = cfg['build'](robot, DT_SO2, task_space,
                                      center_left=center_left, center_right=center_right,
                                      dt_so2=DT_SO2, dt_scaling2=DT_SCALING2)

    # ── Augment ───────────────────────────────────────────────────────────────
    if cfg['discrete']:
        augmented = augment_demonstrations_discrete(group=G, rep_in=rep_in, rep_out=rep_out, demonstrations=demonstrations)
    else:
        augmented = augment_demonstrations(group=G, rep_in=rep_in, rep_out=rep_out, demonstrations=demonstrations, bounds=cfg['bounds'])

    # ── Save ──────────────────────────────────────────────────────────────────
    save_name = f'{task}_augmented_config_{G}.npz'
    for key in ['train_in', 'train_out', 'test_in', 'test_out']:
        augmented[key] = [traj.cpu().numpy() if isinstance(traj, torch.Tensor) else traj for traj in augmented[key]]
    np.savez(os.path.join(ROOT_DIR, 'demonstrations', demo_folder, save_name), demonstrations=augmented)
    print(f'Saved → {save_name}')

    










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


# DT = np.deg2rad(1.0) # 
# DT = np.deg2rad(2.0) #
DT = np.deg2rad(5.0) # ~ 0.052


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

    # # Overwrite all demo types to have the name "new_demo_name" # TODO: Just using to generate the C2SO2Scaling2 demos faster for testing purporses isntead of recomputing the SO2Scaling2
    # new_demo_name = f'C2SO2Scaling2Group'
    # for key in ['train_in_demo_type', 'train_out_demo_type', 'test_in_demo_type', 'test_out_demo_type']:
    #     augmented[key] = [new_demo_name if demo_type != 'original' else 'original' for demo_type in augmented[key]]


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

    if group_str == 'SO2':
        max_angle = bounds['max_angle']
        # Set the angles to be the max_angle and -max_angle only
        angles = np.array([-max_angle, max_angle])
        # angles = np.array([max_angle])
        for a in angles:
            g_list.append(_rot(a))
            g_labels.append('SO2')
    elif group_str == 'Scaling2':
        # Use only the minimum and maximum scale factors
        min_s, max_s = bounds['min_scale'], bounds['max_scale']
        scales = np.array([min_s, max_s])
        for s in scales:
            g_list.append(_scale(s))
            g_labels.append('Scaling2')
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

    g_list, g_labels = build_group_elements(group_str, bounds)
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

            # Create a list of torch tensors for the augmented trajectories, to be added to the dataset after the loop
            aug_in_list = [aug_in[t] for t in range(aug_in.shape[0])]
            aug_out_list = [aug_out[t] for t in range(aug_out.shape[0])]

            # new_in.append(aug_in)
            # new_out.append(aug_out)

            new_in += aug_in_list
            new_out += aug_out_list

        augmented['train_in'].extend(new_in)
        augmented['train_out'].extend(new_out)

        # num_new = len(augmented['train_in']) - num_train_traj
        # augmented['test_in'].extend( augmented['train_in'][num_train_traj:])
        # augmented['test_out'].extend(augmented['train_out'][num_train_traj:])
        # augmented['train_in']  = augmented['train_in'][:num_train_traj]
        # augmented['train_out'] = augmented['train_out'][:num_train_traj]

        num_new = len(new_in)  # number of new trajectories added for this group element
        # for key in ['train_in', 'train_out', 'test_in', 'test_out']:
        for key in ['train_in', 'train_out']:
            augmented[f'{key}_demo_type'].extend([g_label] * num_new)

        if (i + 1) % 10 == 0 or i == len(g_list) - 1:
            print(f'  [{i+1}/{len(g_list)}] {g_label} — {num_new} new trajs ' f'({time.perf_counter()-t0:.2f}s) | test total: {len(augmented["test_in"])}')

    print(f'Total augmentation time: {time.perf_counter()-total_t0:.2f}s')

    return augmented


def augment_so2_scaling2_grid(robot, demonstrations: dict, max_angle: float, min_scale: float, dt: float = 0.05):
    """
    Two-step grid augmentation in SO2 × Scaling2 space.

    Step 1: Apply ±max_angle SO2 rotations to the original demos via horizontal-lift
            integration (full_traj=True).  Each integration step is a distinct augmented
            demo, covering the arc [0°, ±max_angle] in increments of dt rad.
            Step 0 (identity) is skipped to avoid duplicating the originals.

    Step 2: Apply a single scaling group element (scale=min_scale) to every SO2-augmented
            demo, again using full_traj=True so that intermediate scales [1.0, …, min_scale]
            all become training demos.  Step 0 (scale=1.0) is skipped to avoid duplicating
            the SO2 demos from Step 1.

    The conditioning vector [angle, scale, reflection] is updated consistently:
    after Step 1 a demo has [θ, 1, 1]; after Step 2 it has [θ, s, 1].
    """
    G       = SO2Scaling2Group()
    rep_in  = SO2Scaling2DualArmConfigTaskRepIn( robot=robot, G=G, is_vf_constant=False, dt=dt)
    rep_out = SO2Scaling2DualArmConfigTaskRepOut(robot=robot, G=G, is_vf_constant=False, dt=dt)

    for key in ['train_in', 'train_out']:
        demonstrations[key] = [
            traj.to(device) if isinstance(traj, torch.Tensor)
            else torch.tensor(traj, dtype=torch.float32, device=device)
            for traj in demonstrations[key]
        ]

    nb_q, nb_x = demonstrations['nb_q'], demonstrations['nb_x']
    augmented  = copy.deepcopy(demonstrations)
    n_orig     = len(demonstrations['train_in'])

    def _rot(a):
        return np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], dtype=np.float32)

    # ── Step 1: SO2 rotations on original demos ───────────────────────────────
    so2_new_in, so2_new_out = [], []

    t0 = time.perf_counter()
    for angle in [-max_angle, max_angle]:
        g_tensor = torch.tensor(_rot(angle), device=device)
        for traj_in, traj_out in zip(demonstrations['train_in'], demonstrations['train_out']):
            q0 = traj_in[:, :nb_q]
            x0 = traj_in[:, nb_q:nb_q + nb_x]
            v0 = traj_out[:, :nb_q]

            aug_in  = rep_in.act_on( g_tensor, q0=q0, x0=x0, full_traj=True)
            aug_out = rep_out.act_on(g_tensor, q0=q0, v0=v0, full_traj=True)

            # Skip t=0 (identity) — duplicate of original
            for t in range(1, aug_in.shape[0]):
                so2_new_in.append(aug_in[t])
                so2_new_out.append(aug_out[t])

    augmented['train_in'].extend(so2_new_in)
    augmented['train_out'].extend(so2_new_out)
    # augmented['train_in_demo_type'].extend( ['SO2'] * len(so2_new_in))
    # augmented['train_out_demo_type'].extend(['SO2'] * len(so2_new_out))
    augmented['train_in_demo_type'].extend( ['SO2Scaling2Group'] * len(so2_new_in))
    augmented['train_out_demo_type'].extend(['SO2Scaling2Group'] * len(so2_new_out))    

    n_so2 = len(so2_new_in)
    print(f'Step 1 (SO2 ±{np.rad2deg(max_angle):.0f}°): {n_so2} demos '
          f'({n_so2 // n_orig} per original × {n_orig} originals) [{time.perf_counter()-t0:.1f}s]')

    # ── Step 2: Scale each SO2-augmented demo from 1.0 down to min_scale ─────
    g_scale = torch.tensor(np.eye(2, dtype=np.float32) * min_scale, device=device)
    s2_new_in, s2_new_out = [], []

    t0 = time.perf_counter()
    for so2_in, so2_out in zip(so2_new_in, so2_new_out):
        q0 = so2_in[:, :nb_q]
        x0 = so2_in[:, nb_q:nb_q + nb_x]
        v0 = so2_out[:, :nb_q]

        aug_in  = rep_in.act_on( g_scale, q0=q0, x0=x0, full_traj=True)
        aug_out = rep_out.act_on(g_scale, q0=q0, v0=v0, full_traj=True)

        # Skip t=0 (scale=1.0) — duplicate of the SO2 demo it came from
        for t in range(1, aug_in.shape[0]):
            s2_new_in.append(aug_in[t])
            s2_new_out.append(aug_out[t])

    augmented['train_in'].extend(s2_new_in)
    augmented['train_out'].extend(s2_new_out)
    augmented['train_in_demo_type'].extend( ['SO2Scaling2Group'] * len(s2_new_in))
    augmented['train_out_demo_type'].extend(['SO2Scaling2Group'] * len(s2_new_out))

    n_grid = len(s2_new_in)
    steps_per_so2 = n_grid // n_so2 if n_so2 else 0
    print(f'Step 2 (Scale 1.0→{min_scale:.1f}): {n_grid} demos '
          f'({steps_per_so2} per SO2 demo × {n_so2} SO2 demos) [{time.perf_counter()-t0:.1f}s]')
    print(f'Total: {len(augmented["train_in"])} demos '
          f'({n_orig} orig + {n_so2} SO2 + {n_grid} SO2×Scale)')

    return augmented


def augment_c2_so2_scaling2_grid(robot, demonstrations: dict, max_angle: float, min_scale: float, dt: float = 0.05):
    """
    Three-step grid augmentation in C2 × SO2 × Scaling2 space.

    Step 1 & 2: Same as augment_so2_scaling2_grid — produces original + SO2 + SO2Scaling2 demos.

    Step 3: Apply the pure C2 reflection to ALL demos produced by steps 1-2 (including
            originals).  Each reflected demo gets the label 'C2SO2Scaling2Group'.

    The conditioning vector [angle, scale, reflection] is updated consistently:
    after reflection the reflection component (index 2) is negated.
    """
    # Steps 1 & 2: build the SO2 × Scaling2 grid
    augmented = augment_so2_scaling2_grid(robot=robot, demonstrations=demonstrations,
                                          max_angle=max_angle, min_scale=min_scale, dt=dt)

    G       = C2SO2Scaling2Group()
    rep_in  = C2SO2Scaling2DualArmConfigTaskRepIn( robot=robot, G=G, is_vf_constant=False, dt=dt)
    rep_out = C2SO2Scaling2DualArmConfigTaskRepOut(robot=robot, G=G, is_vf_constant=False, dt=dt)

    nb_q, nb_x = augmented['nb_q'], augmented['nb_x']

    # Pure reflection element: block-swap the two arms with no rotation / scaling.
    # _decompose detects has_refl via g[0,0] == 0.
    g_refl = torch.tensor([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0]], dtype=torch.float32, device=device)

    # Step 3: reflect ALL demos that exist so far (original + SO2 + SO2Scaling2)
    c2_new_in, c2_new_out = [], []

    t0 = time.perf_counter()
    for traj_in, traj_out in zip(augmented['train_in'], augmented['train_out']):
        if isinstance(traj_in, np.ndarray):
            traj_in  = torch.tensor(traj_in,  dtype=torch.float32, device=device)
            traj_out = torch.tensor(traj_out, dtype=torch.float32, device=device)
        else:
            traj_in  = traj_in.to(device)
            traj_out = traj_out.to(device)

        q0 = traj_in[:, :nb_q]
        x0 = traj_in[:, nb_q:nb_q + nb_x]
        v0 = traj_out[:, :nb_q]

        aug_in  = rep_in.act_on( g_refl, q0=q0, x0=x0, full_traj=False)
        aug_out = rep_out.act_on(g_refl, q0=q0, v0=v0, full_traj=False)

        c2_new_in.append(aug_in)
        c2_new_out.append(aug_out)

    augmented['train_in'].extend(c2_new_in)
    augmented['train_out'].extend(c2_new_out)
    augmented['train_in_demo_type'].extend( ['C2SO2Scaling2Group'] * len(c2_new_in))
    augmented['train_out_demo_type'].extend(['C2SO2Scaling2Group'] * len(c2_new_out))

    n_refl = len(c2_new_in)
    print(f'Step 3 (C2 reflection): {n_refl} demos reflected [{time.perf_counter()-t0:.1f}s]')
    print(f'Total: {len(augmented["train_in"])} demos (SO2Scaling2 grid × 2 with reflection)')

    return augmented


# scale_bound = 1.5 # Maximum working scale factor (before hitting singularities)
# scale_bound = 0.05 # There doesn't seem to be an issue with gowing very small. I even tried 0.02 and it worked fine.

theta_bound = 180 # degrees
# scale_bound = 0.10
scale_bound = 0.09


# theta_bound = 60 # degrees
# scale_bound = 0.2

GROUP_CONFIGS = {
    'SO2': dict(
        bounds={'max_angle': np.deg2rad(theta_bound)},
        # bounds={'max_angle': np.deg2rad(15)},
        # bounds={'max_angle': np.deg2rad(5.0)},
        rep_in_cls=SO2DualArmConfigTaskRepIn,
        rep_out_cls=SO2DualArmConfigTaskRepOut,
        discrete=False,
    ),
    'Scaling2': dict(
        bounds={'min_scale': scale_bound, 'max_scale': 1.0},
        rep_in_cls=Scaling2DualArmConfigTaskRepIn,
        rep_out_cls=Scaling2DualArmConfigTaskRepOut,
        discrete=False,
    ),
    'SO2Scaling2Group': dict(
        bounds={'max_angle': np.deg2rad(theta_bound), 'min_scale': scale_bound, 'max_scale': 1.0},
        rep_in_cls=SO2Scaling2DualArmConfigTaskRepIn,
        rep_out_cls=SO2Scaling2DualArmConfigTaskRepOut,
        discrete=False,
    ),
    'C2SO2Scaling2Group': dict(
        bounds={'max_angle': np.deg2rad(theta_bound), 'min_scale': scale_bound, 'max_scale': 1.0},
        rep_in_cls=C2SO2Scaling2DualArmConfigTaskRepIn,
        rep_out_cls=C2SO2Scaling2DualArmConfigTaskRepOut,
        discrete=False,
        rep_kwargs=dict(is_taskspace=False),
    ),
    'C2': dict(
        bounds={},
        rep_in_cls=C2DualArmConfigTaskRepIn,
        rep_out_cls=C2DualArmConfigTaskRepOut,
        discrete=True,
        rep_kwargs=dict(is_taskspace=False),
    ),
}


if __name__ == '__main__':
    is_taskspace = False
    # robot = create_two_arm_robot(nb_dofs_left=2, nb_dofs_right=2, nb_x_left=2, nb_x_right=2, is_taskspace=is_taskspace)
    robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2)

    conditioning='symmetry'
    normalize_conditioning = False

    # demo_folder     = 'planar_robot'
    demo_folder = 'planar_robot_test_augm_density'
    
    demo_type_left  = 'LASA'; demo_name_left  = 'CShape'
    demo_type_right = 'LASA'; demo_name_right = 'NShape'

    # demo_type_left  = 'LASA'; demo_name_left  = 'NShape'
    # demo_type_right = 'CUSTOM_DEMOS'; demo_name_right = 'point_fullstatic'
    task = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{robot.nb_dofs}'
    if is_taskspace:
        task += '_taskspace'

    # ── Choose augmentation strategy ─────────────────────────────────────────
    # use_grid=True  → two-step SO2 × Scaling2 grid augmentation
    # use_c2=True    → three-step C2 × SO2 × Scaling2 grid (requires use_grid=True)
    # use_grid=False → single-group strategy via GROUP_CONFIGS

    # use_grid = True
    # use_c2   = True

    # use_grid = False
    # use_c2   = False

    use_grid = False
    use_c2   = False

    predefined_dataset = None
    # predefined_dataset = f'left-LASA-CShape_right-LASA-NShape_ndofs-8_augmented_config_SO2Scaling2Group.npz' # TODO: Temp for quick testing of the C2SO2Scaling2 augmentation.

    data_vis = DataVisualizer(task=task, conditioning=conditioning, normalize_conditioning=normalize_conditioning)
    demonstrations, _ = data_vis.construct_demonstrations(demo_folder=demo_folder, load_augmented_data=False, predefined_dataset=predefined_dataset)

    if use_grid and use_c2:
        augmented = augment_c2_so2_scaling2_grid(
            robot          = robot,
            demonstrations = demonstrations,
            max_angle      = np.deg2rad(theta_bound),
            min_scale      = scale_bound,
            dt             = DT,
        )
        save_name = f'{task}_augmented_config_C2SO2Scaling2Group.npz'
    elif use_grid:
        augmented  = augment_so2_scaling2_grid(
            robot        = robot,
            demonstrations = demonstrations,
            max_angle    = np.deg2rad(theta_bound),
            min_scale    = scale_bound,
            dt           = DT,
        )
        save_name = f'{task}_augmented_config_SO2Scaling2Group.npz'
    else:
        # G = C2(permutator=np.array([1, 1]))
        G = SO2()
        # G = Scaling2()
        # G = SO2Scaling2Group()
        # G = C2SO2Scaling2Group()

        cfg        = GROUP_CONFIGS[f'{G}']
        rep_kwargs = cfg.get('rep_kwargs', {})
        dt         = rep_kwargs.pop('dt', DT)
        rep_in     = cfg['rep_in_cls'](robot=robot, G=G, is_vf_constant=False, dt=dt, **rep_kwargs)
        rep_out    = cfg['rep_out_cls'](robot=robot, G=G, is_vf_constant=False, dt=dt, **rep_kwargs)

        if cfg['discrete']:
            augmented = augment_demonstrations_discrete(group=G, rep_in=rep_in, rep_out=rep_out, demonstrations=demonstrations)
        else:
            augmented = augment_demonstrations(group=G, rep_in=rep_in, rep_out=rep_out, demonstrations=demonstrations, bounds=cfg['bounds'])

        save_name = f'{task}_augmented_config_{G}.npz'

    for key in ['train_in', 'train_out', 'test_in', 'test_out']:
        augmented[key] = [traj.cpu().numpy() for traj in augmented[key]]
    np.savez(os.path.join(ROOT_DIR, 'demonstrations', demo_folder, save_name), demonstrations=augmented)
    print(f'Saved → {save_name}')

    





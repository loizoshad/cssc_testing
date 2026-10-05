"""
Case 3 — C2 morphological symmetry (discrete arm swap).

Pre-requisite: run pysymmetries/rby1/augment_data_rby1_symm_conditioning.py
with group_name = 'C2Letters' to generate the augmented dataset.

Run from the repository root:
    python examples/rby1/data_augmentation_letters/rby1_mlp_da_c2_letters.py
"""

import os
import copy
from pathlib import Path
import numpy as np
import random
import torch
import shutil
import matplotlib.pyplot as plt

from utils.groups import *
from utils.utils import *
from utils.networks_pytorch import CustomMLP

cols = shutil.get_terminal_size().columns
np.set_printoptions(precision=8, suppress=True, linewidth=cols)

seed = 42
print('seed:', seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR    = Path(CURRENT_DIR).parent.parent.parent.resolve()


if __name__ == '__main__':
    is_taskspace = False
    task_space   = 'yz'
    robot        = create_rby1_robot(task_space=task_space, ee_joint=False, is_taskspace=is_taskspace)
    demo_folder  = 'rby1_letters'
    task         = 'left-rby1-letters_right-rby1-letters_ndofs-14'

    # ── Network settings ──────────────────────────────────────────────────────
    num_iterations = 10000
    nb_steps       = 20
    stride         = 10
    # nb_steps       = 15
    # stride         = 10
    batch_size     = 250

    train           = False
    load_model_flag = False
    save_model_flag = False
    save_plots      = False

    data_augmentation      = True
    decouple_arms          = True
    conditioning           = 'symmetry'
    normalize_conditioning = False

    # ── Group & representations ───────────────────────────────────────────────
    dt      = np.deg2rad(1.0)
    G       = C2RBY1(permutator=np.array([1, 1]), name='C2Letters')
    rep_in  = C2RBY1RepIn( robot=robot, G=G, is_vf_constant=False, dt=dt, is_taskspace=is_taskspace)
    rep_out = C2RBY1RepOut(robot=robot, G=G, is_vf_constant=False, dt=dt, is_taskspace=is_taskspace)

    model_id = (
        f'RBY1_MLP_DA_{G}_{task}'
        f'_epochs={num_iterations}_nsteps={nb_steps}_stride={stride}'
        f'_{"decoupled" if decouple_arms else "coupled"}'
        f'_conditioning={conditioning}'
        f'_seed={seed}_'
    )

    # ── Load data ─────────────────────────────────────────────────────────────
    predefined_dataset = None
    data_vis = DataVisualizer(
        task=task, conditioning=conditioning,
        normalize_conditioning=normalize_conditioning,
        group=G, task_space=task_space,
    )
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(
        demo_folder=demo_folder, load_augmented_data=True,
        predefined_dataset=predefined_dataset,
    )

    if data_augmentation:
        num_original = sum(1 for t in demonstrations['train_in_demo_type'] if 'original' in t)
        # C2 augmented demos are stored in the test split by augment_demonstrations_discrete.
        # rearrange_demonstrations pools train+test, so passing the quota here promotes
        # them into the training split.
        num_c2_augm = sum(1 for t in demonstrations['test_in_demo_type'] if f'{G}' in t)
        print(f'Demos — original: {num_original}  C2-augmented: {num_c2_augm}')
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(
            demonstrations, demonstrations_normalized,
            train_demo_types=['original', f'{G}'],
            num_train_demos_per_type=[num_original, num_c2_augm],
        )

    # ── Network ───────────────────────────────────────────────────────────────
    network = CustomMLP(
        model_id=model_id, load_model_flag=load_model_flag,
        save_model_flag=save_model_flag,
        demonstrations=demonstrations_normalized,
        robot=robot,
    )
    data_vis.set_network(network)

    # ── Train ─────────────────────────────────────────────────────────────────
    print("Normalization bounds before training:")
    print(f'inp_min: {demonstrations_normalized["inp_min"]}')
    print(f'inp_max: {demonstrations_normalized["inp_max"]}')
    print(f'out_min: {demonstrations_normalized["Dq_min"]}')
    print(f'out_max: {demonstrations_normalized["Dq_max"]}')

    if train:
        network.train_network(
            robot, num_iterations=num_iterations,
            steps=nb_steps, stride=stride,
            verbose=True, batch_size=batch_size,
            verbose_every=500, decouple_arms=decouple_arms,
        )
    if network.save_model_flag:
        network.save_model(model_id=network.model_id)

    data_vis.plot_loss_history(network, save_plots=False)

    # ── Evaluate ──────────────────────────────────────────────────────────────
    horizon          = min(traj.shape[0] for traj in network.demonstrations['train_in'])
    goal_conditioned = conditioning == 'goal'

    # After rearrange_demonstrations both original and C2 demos are in train.
    demo_config_train = dict(
        demo_labels=['original', f'{G}'],
        num_of_demos=[2, 2],
    )
    demonstrations_eval_train = choose_demos_for_evaluation(
        network.demonstrations, config=demo_config_train, use_train=True,
    )
    # demonstrations_eval_test = []
    demo_config_test = dict(
        demo_labels=['original', f'{G}'],
        num_of_demos=[0, 2],
    )
    demonstrations_eval_test = choose_demos_for_evaluation(
        network.demonstrations, config=demo_config_train, use_train=False,
    ) 

    evaluate_full_trajectory(
        network, data_vis, robot, horizon,
        demonstrations_eval_test, demonstrations_eval_train,
        title_prefix='Letters DA — C2',
        save_plots=save_plots, save_trajs=False,
        goal_conditioned=goal_conditioned,
    )
    plt.show()

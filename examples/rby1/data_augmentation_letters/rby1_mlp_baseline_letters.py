"""
Baseline — original data only, no augmentation.

Trains on the 3 original letter demonstrations without any group augmentation.
Uses the same network architecture and symmetry-conditioning input format as the
DA scripts so results are directly comparable.

Run from the repository root:
    python examples/rby1/data_augmentation_letters/rby1_mlp_baseline_letters.py
"""

import os
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
    num_iterations = 100_000
    nb_steps       = 15
    stride         = 10
    batch_size     = 250

    train           = True
    load_model_flag = False
    save_model_flag = True
    save_plots      = False

    decouple_arms          = True
    conditioning           = 'symmetry'   # same format as DA scripts for fair comparison
    normalize_conditioning = False

    model_id = (
        f'RBY1_MLP_DA_BASELINE_{task}'
        f'_epochs={num_iterations}_nsteps={nb_steps}_stride={stride}'
        f'_{"decoupled" if decouple_arms else "coupled"}'
        f'_conditioning={conditioning}'
        f'_seed={seed}_'
    )

    # ── Load original data only ───────────────────────────────────────────────
    data_vis = DataVisualizer(
        task=task, conditioning=conditioning,
        normalize_conditioning=normalize_conditioning,
        group=None, task_space=task_space,
    )
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(
        demo_folder=demo_folder, load_augmented_data=False,
    )
    print(f'Original demos — train: {len(demonstrations["train_in"])}, '
          f'test: {len(demonstrations["test_in"])}')

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

    demo_config = dict(
        demo_labels=['original'],
        num_of_demos=[3],
    )
    demonstrations_eval_train = choose_demos_for_evaluation(
        network.demonstrations, config=demo_config, use_train=True,
    )
    demonstrations_eval_test = choose_demos_for_evaluation(
        network.demonstrations, config=demo_config, use_train=False,
    )
    evaluate_full_trajectory(
        network, data_vis, robot, horizon,
        demonstrations_eval_test, demonstrations_eval_train,
        title_prefix='Letters Baseline (no DA)',
        save_plots=save_plots, save_trajs=False,
        goal_conditioned=goal_conditioned,
    )
    plt.show()

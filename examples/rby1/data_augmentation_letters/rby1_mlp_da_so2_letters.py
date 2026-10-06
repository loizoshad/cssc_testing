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
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.parent.resolve()

if __name__ == '__main__':
    is_taskspace = False
    task_space = 'yz'
    robot = create_rby1_robot(task_space=task_space, ee_joint=False, is_taskspace=is_taskspace)
    demo_folder = 'rby1_letters'
    task = 'left-rby1-letters_right-rby1-letters_ndofs-14'

    # Network settings
    # num_iterations = 100000
    num_iterations = 1000
    nb_steps = 15
    stride = 10

    batch_size = 250

    train = True
    load_model_flag = False
    save_model_flag = True
    save_plots = True

    data_augmentation = True
    decouple_arms = True
    conditioning = 'symmetry'
    normalize_conditioning = False
    n_repeat_terminal = 50   # repeat last state N times → teaches network to stop at goal

    # Train / test split
    so2_train_step = np.deg2rad(15.0)
    scaling_train_values = None

    # Group & representations
    dt = np.deg2rad(1.0)
    G = SO2RBY1(name='SO2Letters')
    rep_in = SO2RBY1RepIn( robot=robot, G=G, is_vf_constant=False, dt=dt, task_space=task_space)
    rep_out = SO2RBY1RepOut(robot=robot, G=G, is_vf_constant=False, dt=dt, task_space=task_space)

    model_id = (
        f'RBY1_MLP_DA_{G}_{task}'
        f'_epochs={num_iterations}_nsteps={nb_steps}_stride={stride}'
        f'_{"decoupled" if decouple_arms else "coupled"}'
        f'_conditioning={conditioning}'
        f'_seed={seed}_'
    )

    # Load data
    predefined_dataset = None
    data_vis = DataVisualizer(task=task, conditioning=conditioning, normalize_conditioning=normalize_conditioning, group=G, task_space=task_space)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(demo_folder=demo_folder, load_augmented_data=True, predefined_dataset=predefined_dataset)

    if data_augmentation:
        num_original = sum(1 for t in demonstrations['train_in_demo_type'] if 'original' in t)
        num_augm = sum(1 for t in demonstrations['train_in_demo_type'] if f'{G}' in t)
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, train_demo_types=['original', f'{G}'], num_train_demos_per_type=[num_original, num_augm])
        demonstrations, demonstrations_normalized = split_train_test(
            demonstrations, demonstrations_normalized,
            so2_train_step=so2_train_step,
            scaling_train_values=scaling_train_values,
            so2_match_tolerance=(so2_train_step / 2 if so2_train_step is not None else np.deg2rad(3)),
            scaling_match_tolerance=0.05,
        )
        print(f'After split — train: {len(demonstrations["train_in"])}, test: {len(demonstrations["test_in"])}')

    # Network
    network = CustomMLP(model_id=model_id, load_model_flag=load_model_flag, save_model_flag=save_model_flag, demonstrations=demonstrations_normalized, robot=robot)
    data_vis.set_network(network)

    # Train
    if train:
        network.train_network(robot, num_iterations=num_iterations, steps=nb_steps, stride=stride, verbose=True, batch_size=batch_size, verbose_every=500, decouple_arms=decouple_arms)
    if network.save_model_flag:
        network.save_model(model_id=network.model_id)

    data_vis.plot_loss_history(network, save_plots=False)

    # Evaluate
    horizon = min(traj.shape[0] for traj in network.demonstrations['train_in'])
    goal_conditioned = conditioning == 'goal'

    demo_config_train = dict(demo_labels=['original', f'{G}'], num_of_demos=[0, 4])
    demonstrations_eval_train = choose_demos_for_evaluation(network.demonstrations, config=demo_config_train, use_train=True)
    demo_config_test = dict(demo_labels=['original', f'{G}'], num_of_demos=[0, 4])
    demonstrations_eval_test = choose_demos_for_evaluation(network.demonstrations, config=demo_config_train, use_train=False)    
    evaluate_full_trajectory(network, data_vis, robot, horizon, demonstrations_eval_test, demonstrations_eval_train, title_prefix='Letters DA — SO2', save_plots=save_plots, save_trajs=False, goal_conditioned=goal_conditioned)
    plt.show()

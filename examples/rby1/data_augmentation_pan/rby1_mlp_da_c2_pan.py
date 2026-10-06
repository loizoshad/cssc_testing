import os
from pathlib import Path
import numpy as np
import random
import torch
import shutil  # Just making things prettier when printing in the console
import matplotlib.pyplot as plt

from utils.groups import *
from utils.utils import *
from utils.networks_pytorch import CustomMLP
from utils.robot_vis import PlanarRobotVisualizer

cols = shutil.get_terminal_size().columns
np.set_printoptions(precision=8, suppress=True, linewidth=cols)
seed = 42

torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.resolve()


if __name__ == '__main__':
    ############################################################################################################
    # Initialize robot
    ############################################################################################################
    is_taskspace = False
    TASK_SPACE = 'xyz_theta'   # [x, y, z, yaw] per arm — planar pan experiment
    robot = create_rby1_robot(task_space=TASK_SPACE, ee_joint=False, is_taskspace=is_taskspace)
    demo_folder = 'rby1_pan_v1'
    task = 'left-rby1-all_right-rby1-all_ndofs-14'
    task = task + '_taskspace' if is_taskspace else task

    ############################################################################################################
    # Network settings
    ############################################################################################################
    # num_iterations = 25000
    num_iterations = 1000

    step = 5.0 # augmentation step in degrees (not used by C2, kept for model_id parity)
    nb_steps = 15
    stride = 10

    batch_size = 250

    train = True
    load_model_flag = False
    save_model_flag = True
    save_plots = True
    save_loss = True

    data_augmentation = True
    decouple_arms = True

    conditioning = 'symmetry'
    normalize_conditioning = False

    G = C2RBY1(permutator=np.array([1, 1]))
    rep_in = C2RBY1RepIn( robot=robot, G=G, is_vf_constant=False, dt=0.05, is_taskspace=is_taskspace)
    rep_out = C2RBY1RepOut(robot=robot, G=G, is_vf_constant=False, dt=0.05, is_taskspace=is_taskspace)

    model_id = (f'RBY1_MLP_DA_{G}_{task}'
                f'_epochs={num_iterations}'
                f'_nsteps={nb_steps}'
                f'_stride={stride}'
                f'_augmstep={step}_')

    if decouple_arms:
        model_id += 'decoupled'
    else:
        model_id += 'coupled'

    model_id += f'_conditioning={conditioning}' if conditioning is not None else '_no_conditioning'
    model_id += f'_normalize_conditioning={normalize_conditioning}' if normalize_conditioning else ''
    model_id += f'_seed={seed}_'

    predefined_dataset = None
    ############################################################################################################
    # Get data and initialize network
    ############################################################################################################
    data_vis = DataVisualizer(task=task, conditioning=conditioning, normalize_conditioning=normalize_conditioning, group=G, task_space=TASK_SPACE)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations( demo_folder=demo_folder, load_augmented_data=True, predefined_dataset=predefined_dataset)

    if data_augmentation:
        num_original = sum(1 for t in demonstrations['train_in_demo_type'] if 'original' in t)
        num_c2 = sum(1 for t in demonstrations['train_in_demo_type'] if 'C2RBY1'  in t)
        print(f' Num of demos per type: original={num_original}, C2RBY1={num_c2}')
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, train_demo_types=['original', f'{G}'], num_train_demos_per_type=[num_original, num_c2])

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

    ############################################################################################################
    # Evaluate network's performance — full trajectory simulation
    ############################################################################################################
    horizon = min(traj.shape[0] for traj in network.demonstrations['train_in'])
    goal_conditioned = conditioning == 'goal'

    demo_config_train = dict(demo_labels=['original', f'{G}'], num_of_demos=[4, 4])

    demonstrations_eval_train = choose_demos_for_evaluation(network.demonstrations, config=demo_config_train, use_train=True)
    demonstrations_eval_test = []

    evaluate_full_trajectory(network, data_vis, robot, horizon, demonstrations_eval_test, demonstrations_eval_train, title_prefix='Full Evaluation', save_plots=save_plots, save_trajs=False, goal_conditioned=goal_conditioned)

    plt.show()

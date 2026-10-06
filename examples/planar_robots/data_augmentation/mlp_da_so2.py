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

if __name__ == '__main__':
    ############################################################################################################
    # Initialize robot
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

    step = 30.0
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

    G = SO2()
    rep_in = SO2DualArmConfigTaskRepIn(robot=robot, G=G, is_vf_constant=False, dt=0.1)
    rep_out = SO2DualArmConfigTaskRepOut(robot=robot, G=G, is_vf_constant=False, dt=0.1)

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
        # # get the number of demos that have "SO2" in their type
        num_so2 = sum([1 for demo_type in demonstrations['train_in_demo_type'] if 'SO2' in demo_type])
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, 
                                                                                      train_demo_types=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'], 
                                                                                      num_train_demos_per_type=[7, num_so2, 0, 0, 0])


    so2_train_step = np.deg2rad(step) # e.g., one training angle every 20°
    scaling_train_values = list(np.arange(1.0, 0.1-0.01, -np.deg2rad(step)))  # e.g., 5 evenly spaced scaling values from 0.1 to 1.0
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
        num_of_demos = [1, 5, 0, 0, 0]
    )
    demo_config_test = dict(
        demo_labels=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
        # num_of_demos = [7, 0, 0, 0, 0]
        num_of_demos = [1, 5, 0, 0, 0]
    )    

    demonstrations_eval_test = choose_demos_for_evaluation(network.demonstrations, config=demo_config_test, use_train = False)
    demonstrations_eval_train = choose_demos_for_evaluation(network.demonstrations, config=demo_config_train, use_train = True)

    evaluate_full_trajectory(network, data_vis, robot, horizon, demonstrations_eval_test, demonstrations_eval_train, title_prefix='Full Evaluation', save_plots=save_plots, save_trajs=False, goal_conditioned=False)
        
    plt.show()






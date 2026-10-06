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

def split_train_test(demonstrations, demonstrations_normalized, test_ratio=0.2, original_train_indices=None):
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

    margin=1e-10
    # margin=0.05
    # margin=0.1

    task_space_loss = False

    G = Scaling2()
    rep_in = Scaling2DualArmConfigTaskRepIn(robot=robot, G=G, is_vf_constant=False, dt=0.1)
    rep_out = Scaling2DualArmConfigTaskRepOut(robot=robot, G=G, is_vf_constant=False, dt=0.1)

    model_id = f'MLP_DA_{G}_{task}_epochs={str(num_iterations)}_nsteps={str(nb_steps)}_stride={stride}' + f'norm_bounds={margin}_'

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
    data_vis = DataVisualizer(task=task, isotropic_normalization=isotropic_normalization, margin=margin, conditioning=conditioning, normalize_conditioning=normalize_conditioning, group=G)
    demonstrations, demonstrations_normalized = data_vis.construct_demonstrations(demo_folder=demo_folder, load_augmented_data=True, predefined_dataset=predefined_dataset)
    if data_augmentation:
        # get the number of demos that have "SO2" in their type
        num_of_scaling2 = sum([1 for demo_type in demonstrations['train_in_demo_type'] if 'Scaling2' in demo_type])
        demonstrations, demonstrations_normalized = data_vis.rearrange_demonstrations(demonstrations, demonstrations_normalized, train_demo_types=[f'{G}', 'original'], num_train_demos_per_type=[num_of_scaling2, 7])

    # Test ratio is determined based on resolution of augmented data we want to have:    
    full_augm_dt = np.deg2rad(5.0) # ~ 0.052
    train_dt = np.deg2rad(30.0)
    train_percentage = full_augm_dt / train_dt # we want reciprocal because the smaller the train_dt, the more augmented data we have, and thus the smaller the percentage of data we want to train on.
    test_ratio = 1 - train_percentage
    demonstrations, demonstrations_normalized = split_train_test(demonstrations, demonstrations_normalized, test_ratio=test_ratio)


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

    ############################################################################################################
    # Evaluate network's performance - 1-step predictions on train and test data
    ############################################################################################################
    data_vis.plot_1step_predictions(network, len(demonstrations['train_in']), save_plots=save_plots)

    # ###########################################################################################################
    # Evaluate network's performance - Full trajectory simulation
    # ###########################################################################################################
    horizon = min([traj.shape[0] for traj in network.demonstrations['train_in']])


    demo_config = dict(
        demo_labels=['original', 'SO2', 'Scaling2', 'SO2Scaling2Group', 'C2SO2Scaling2Group'],
        # num_of_demos = [7, 0, 0, 0, 0]
        num_of_demos = [1, 0, 10, 0, 0]
    )

    demonstrations_eval = choose_demos_for_evaluation(network.demonstrations, config=demo_config)
    demonstrations_eval_ = [demonstrations_eval[0]]
    evaluate_full_trajectory(network, data_vis, robot, horizon, demonstrations_eval, demonstrations_eval_, title_prefix='Full Evaluation', save_plots=save_plots, save_trajs=False, goal_conditioned=False)
    plt.show()







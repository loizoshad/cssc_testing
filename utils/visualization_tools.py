import numpy as np
from scipy.spatial import ConvexHull
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
import copy
import os
import torch

import matplotlib.animation as animation
import matplotlib.animation as animation

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# prioritize cuda, then mps, then cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


###### Some methohds/classes for visualizing the equivariance training process.
from matplotlib.widgets import Slider


def _to_tensor_bounds(state, x_min, x_max):
    """Convert numpy bounds to torch tensors matching state's dtype and device."""
    if isinstance(state, torch.Tensor):
        if isinstance(x_min, np.ndarray):
            x_min = torch.tensor(x_min, dtype=state.dtype, device=state.device)
        if isinstance(x_max, np.ndarray):
            x_max = torch.tensor(x_max, dtype=state.dtype, device=state.device)
    return x_min, x_max

def denormalize_state(state, x_min, x_max):
    """Denormalize state from [-1, 1] using per-variable bounds."""
    x_min, x_max = _to_tensor_bounds(state, x_min, x_max)
    state = ((state / 2) + 0.5) * (x_max - x_min) + x_min
    return state


    




class EquivarianceVisualizer:

    # def __init__(self, online_plotting=False, min_x=-2, max_x=10, min_y=-6, max_y=6):
    # def __init__(self, online_plotting=False, min_x=-4, max_x=12, min_y=-8, max_y=8): # 2-DoF
    def __init__(self, online_plotting=False, min_x=0, max_x=16, min_y=-6, max_y=10): # 2-DoF
        self.online_plotting = online_plotting

        self.data = []
        self.current_iteration = 0

        self.fig, self.axs = plt.subplots(1, 2, figsize=(10, 6))
        plt.subplots_adjust(bottom=0.25)

        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y


        self.vel_left = None
        self.vel_right = None

        self.f_rho_left = None
        self.f_rho_right = None

        self.rho_out_left = None
        self.rho_out_right = None


        # Left plot artists
        self.q_left_scatter = self.axs[0].scatter([], [])
        self.q_right_scatter = self.axs[0].scatter([], [])

        # Right plot artists
        self.rho_q_left_scatter = self.axs[1].scatter([], [])
        self.rho_q_right_scatter = self.axs[1].scatter([], [])

        for ax in self.axs:
            ax.set_xlim(min_x, max_x)
            ax.set_ylim(min_y, max_y)
            ax.set_aspect('equal')

        self.axs[0].set_title("Original")
        self.axs[1].set_title("Symmetric")

        # Slider
        ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
        self.slider = Slider(ax_slider, "Iteration", 0, 1, valinit=0, valstep=1)
        self.slider.on_changed(self._slider_update)

        plt.ion()
        plt.show()

    def add_iteration_offline(self, data):

        self.data.append(data)

        self.slider.valmax = len(self.data) - 1
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)

        if len(self.data) == 1:
            self.update_plot(0)
        else:
            self.update_plot(len(self.data) - 1)

        self.fig.canvas.draw_idle()

    def add_iteration(self, data):

        self.data.append(data)

        self.slider.valmax = len(self.data) - 1
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)

        if len(self.data) == 1:
            self.update_plot(0)
        else:
            self.update_plot(len(self.data) - 1)

        if self.online_plotting:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            plt.pause(0.001)
            # plt.pause(0.01)

    def add_iteration_online_v1(self, data):

        self.data.append(data)

        self.slider.valmax = len(self.data) - 1
        self.slider.ax.set_xlim(self.slider.valmin, self.slider.valmax)

        self.update_plot(len(self.data) - 1)

        if self.online_plotting:
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # plt.pause(0.001)
            plt.pause(0.01)

    def update_plot(self, idx):
        if self.online_plotting:
            self.update_plot_online(idx)
        else:
            self.update_plot_offline(idx)

    def update_plot_offline(self, idx):

        d = self.data[idx]

        # LEFT Plot
        self.q_left_scatter.set_offsets(d["q_left"])
        self.q_right_scatter.set_offsets(d["q_right"])

        if self.vel_left is None:
            self.vel_left = self.axs[0].quiver(
                d["q_left"][:,0], d["q_left"][:,1],
                d["f_left"][:,0], d["f_left"][:,1], color='blue',
                # angles='xy', scale_units='xy', scale=1
            )
            self.vel_right = self.axs[0].quiver(
                d["q_right"][:,0], d["q_right"][:,1],
                d["f_right"][:,0], d["f_right"][:,1], color='blue',
                # angles='xy', scale_units='xy', scale=1
            )
        else:
            self.vel_left.set_offsets(d["q_left"])
            self.vel_left.set_UVC(d["f_left"][:,0], d["f_left"][:,1])

            self.vel_right.set_offsets(d["q_right"])
            self.vel_right.set_UVC(d["f_right"][:,0], d["f_right"][:,1])

        # Right Plot
        self.rho_q_left_scatter.set_offsets(d["rho_q_left"])
        self.rho_q_right_scatter.set_offsets(d["rho_q_right"])

        if self.f_rho_left is None:
            self.f_rho_left = self.axs[1].quiver(
                d["rho_q_left"][:,0], d["rho_q_left"][:,1],
                d["f_rho_left"][:,0], d["f_rho_left"][:,1], color='red',
                # angles='xy', scale_units='xy', scale=1
            )
            self.f_rho_right = self.axs[1].quiver(
                d["rho_q_right"][:,0], d["rho_q_right"][:,1],
                d["f_rho_right"][:,0], d["f_rho_right"][:,1], color='red',
                # angles='xy', scale_units='xy', scale=1
            )
            self.rho_out_left = self.axs[1].quiver(
                d["rho_q_left"][:,0], d["rho_q_left"][:,1],
                d["rho_out_left"][:,0], d["rho_out_left"][:,1], color='blue',
                # angles='xy', scale_units='xy', scale=1
            )
            self.rho_out_right = self.axs[1].quiver(
                d["rho_q_right"][:,0], d["rho_q_right"][:,1],
                d["rho_out_right"][:,0], d["rho_out_right"][:,1], color='blue',
                # angles='xy', scale_units='xy', scale=1
            )
        else:
            self.f_rho_left.set_offsets(d["rho_q_left"])
            self.f_rho_left.set_UVC(d["f_rho_left"][:,0], d["f_rho_left"][:,1])
            self.f_rho_right.set_offsets(d["rho_q_right"])
            self.f_rho_right.set_UVC(d["f_rho_right"][:,0], d["f_rho_right"][:,1])
            self.rho_out_left.set_offsets(d["rho_q_left"])
            self.rho_out_left.set_UVC(d["rho_out_left"][:,0], d["rho_out_left"][:,1])
            self.rho_out_right.set_offsets(d["rho_q_right"])
            self.rho_out_right.set_UVC(d["rho_out_right"][:,0], d["rho_out_right"][:,1])

        self.fig.canvas.draw_idle()

    def update_plot_online(self, idx):

        d = self.data[idx]

        # LEFT Plot
        self.q_left_scatter.set_offsets(d["q_left"])
        self.q_right_scatter.set_offsets(d["q_right"])

        # remove previous arrows
        if self.vel_left is not None:
            self.vel_left.remove()
        if self.vel_right is not None:
            self.vel_right.remove()

        # recreate them
        self.vel_left = self.axs[0].quiver(
            d["q_left"][:,0], d["q_left"][:,1],
            d["f_left"][:,0], d["f_left"][:,1], color='blue', label=r'$f_{\theta}(q, x)$',
            # angles='xy', scale_units='xy', scale=1
        )

        self.vel_right = self.axs[0].quiver(
            d["q_right"][:,0], d["q_right"][:,1],
            d["f_right"][:,0], d["f_right"][:,1], color='blue',
            # angles='xy', scale_units='xy', scale=1
        )


        # Right Plot
        self.rho_q_left_scatter.set_offsets(d["rho_q_left"])
        self.rho_q_right_scatter.set_offsets(d["rho_q_right"])

        if self.f_rho_left is not None:
            self.f_rho_left.remove()
        if self.f_rho_right is not None:
            self.f_rho_right.remove()
        if self.rho_out_left is not None:
            self.rho_out_left.remove()
        if self.rho_out_right is not None:
            self.rho_out_right.remove()

        self.f_rho_left = self.axs[1].quiver(
            d["rho_q_left"][:,0], d["rho_q_left"][:,1],
            d["f_rho_left"][:,0], d["f_rho_left"][:,1], color='red', label=r'$f_{\theta}(\rho_{in}(q, x), g))$',
            # angles='xy', scale_units='xy', scale=1
        )

        self.f_rho_right = self.axs[1].quiver(
            d["rho_q_right"][:,0], d["rho_q_right"][:,1],
            d["f_rho_right"][:,0], d["f_rho_right"][:,1], color='red',
            # angles='xy', scale_units='xy', scale=1
        )

        self.rho_out_left = self.axs[1].quiver(
            d["rho_q_left"][:,0], d["rho_q_left"][:,1],
            d["rho_out_left"][:,0], d["rho_out_left"][:,1], color='blue', label=r'$\rho_{out}(f_{\theta}(q, x), g)$',
            # angles='xy', scale_units='xy', scale=1
        )

        self.rho_out_right = self.axs[1].quiver(
            d["rho_q_right"][:,0], d["rho_q_right"][:,1],
            d["rho_out_right"][:,0], d["rho_out_right"][:,1], color='blue',
            # angles='xy', scale_units='xy', scale=1
        )

        # Update the title of the entire figure with the current iteration and the training equivariant loss
        self.fig.suptitle(rf'$||\rho_{{out}}(f_{{\theta}}(q_{{s}}, x_{{g}}), g) - f_{{\theta}}(\rho_{{in}}((q_{{s}}, x_{{g}}), g))||$ = {d["training_loss_equiv"]:.8f}', fontsize=16)

        # # Add as a title to the figure on the right the MSE between the two batches of vectors at the current iteration. NOTE: We are normalizing them first for this case, but we should not in general.
        # f_rho_left_norm = d["f_rho_left"] / (np.linalg.norm(d["f_rho_left"], axis=1, keepdims=True) + 1e-8)
        # rho_out_left_norm = d["rho_out_left"] / (np.linalg.norm(d["rho_out_left"], axis=1, keepdims=True) + 1e-8)
        # mse_f_rho_left = np.mean(np.linalg.norm(f_rho_left_norm - rho_out_left_norm, axis=1)**2)
        # f_rho_right_norm = d["f_rho_right"] / (np.linalg.norm(d["f_rho_right"], axis=1, keepdims=True) + 1e-8)
        # rho_out_right_norm = d["rho_out_right"] / (np.linalg.norm(d["rho_out_right"], axis=1, keepdims=True) + 1e-8)
        # mse_f_rho_right = np.mean(np.linalg.norm(f_rho_right_norm - rho_out_right_norm, axis=1)**2)
        # self.axs[1].set_title(rf'$MSE_{{l}}$: {mse_f_rho_left:.8f} | $MSE_{{r}}$: {mse_f_rho_right:.8f}', fontsize=14)

        # self.axs[1].set_title(rf'$MSE_{{l}}$: {mse_f_rho_left:.8f} | $MSE_{{r}}$: {mse_f_rho_right:.8f}', fontsize=14)

        # add legends
        self.axs[0].legend(loc='upper left')
        self.axs[1].legend(loc='upper left')
        self.fig.canvas.draw_idle()

    def _slider_update(self, val):
        idx = int(self.slider.val)
        self.update_plot(idx)


def plot_equivariance_sampling_bounds(robot, network, ax=None):
    '''
    robot: the robot object, used to get the forward kinematics and the configuration space bounds
        - The robot is assumed to be a dual-arm robot.
    network: the equivariant network object, used to get the sampling bounds for the equivariance visualization
        - The network is assumed to have the variables:
            - self.equiv_sampling_min_bounds
            - self.equiv_sampling_max_bounds

    The bounds have robot.nb_dofs+robot.nb_x dimensions.
    '''
    #### Step 1: Get sampling bounds of normalized data
    min_bounds, max_bounds = network.get_equivariance_sampling_bounds()

    #### Step 2: Denormalize the bounds to get them in the original scale of the data
    min_q_state = denormalize_state(state=min_bounds[:2*network.n_q], x_min=network.demonstrations["Q_min"], x_max=network.demonstrations["Q_max"])
    max_q_state = denormalize_state(state=max_bounds[:2*network.n_q], x_min=network.demonstrations["Q_min"], x_max=network.demonstrations["Q_max"])
    min_x_goal = denormalize_state(state=min_bounds[2*network.n_q:], x_min=network.demonstrations["X_min"], x_max=network.demonstrations["X_max"])
    max_x_goal = denormalize_state(state=max_bounds[2*network.n_q:], x_min=network.demonstrations["X_min"], x_max=network.demonstrations["X_max"])

    #### Step 3:
    # Create a grid of points between the min and max bounds in the configuration space
    grid_size = 3
    grid_points_q = np.array(np.meshgrid(*[ np.linspace(min_q_state[i], max_q_state[i], grid_size) for i in range(robot.nb_dofs) ])).T.reshape(-1, robot.nb_dofs)

    # Extract the left and right arm joint angles for each point in the grid
    q_left = grid_points_q[:, :robot.nb_dofs_left]
    q_right = grid_points_q[:, robot.nb_dofs_left:]

    # Compute the task space positions for the left and right arms using the forward kinematics
    with torch.no_grad():
        q_state_left_torch = torch.tensor(q_left, dtype=torch.float32, device=device)
        q_state_right_torch = torch.tensor(q_right, dtype=torch.float32, device=device)
        x_state_left = robot.fk_func_left_torch(q_state_left_torch).cpu().numpy()
        x_state_right = robot.fk_func_right_torch(q_state_right_torch).cpu().numpy()    

    # Compute the convex hull of the task space states so that we can then plot a polygon around them
    hull_state_left = ConvexHull(x_state_left[:, :2]) # Only consider x and y for the hull
    hull_state_right = ConvexHull(x_state_right[:, :2]) # Only consider x and y for the hull

    #### Step 4: Compute the convex hull of the task space positions so that we can then plot a polygon around them
    hull_goal_left = ConvexHull(np.array([[min_x_goal[0], min_x_goal[1]], [min_x_goal[0], max_x_goal[1]], [max_x_goal[0], min_x_goal[1]], [max_x_goal[0], max_x_goal[1]]]))
    hull_goal_right = ConvexHull(np.array([[min_x_goal[2], min_x_goal[3]], [min_x_goal[2], max_x_goal[3]], [max_x_goal[2], min_x_goal[3]], [max_x_goal[2], max_x_goal[3]]]))

    if ax is not None:
        # Plot the convex hulls as polygons on the provided axis
        for hull in [hull_state_left, hull_goal_left]:
            hull_points = hull.points[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color='blue' if np.array_equal(hull, hull_state_left) else 'cyan', label='Left Arm Sampling Region' if np.array_equal(hull, hull_state_left) else 'Left Arm Goal Region')
            ax.legend()
        for hull in [hull_state_right, hull_goal_right]:
            hull_points = hull.points[hull.vertices]
            ax.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color='red' if np.array_equal(hull, hull_state_right) else 'orange', label='Right Arm Sampling Region' if np.array_equal(hull, hull_state_right) else 'Right Arm Goal Region')
            ax.legend()
    else:
        # Plot the convex hulls as polygons
        plt.figure(figsize=(8, 8))
        for hull in [hull_state_left, hull_goal_left]:
            hull_points = hull.points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color='blue' if np.array_equal(hull, hull_state_left) else 'cyan', label='Left Arm Sampling Region' if np.array_equal(hull, hull_state_left) else 'Left Arm Goal Region')
            plt.legend()
        for hull in [hull_state_right, hull_goal_right]:
            hull_points = hull.points[hull.vertices]
            plt.fill(hull_points[:, 0], hull_points[:, 1], alpha=0.3, color='red' if np.array_equal(hull, hull_state_right) else 'orange', label='Right Arm Sampling Region' if np.array_equal(hull, hull_state_right) else 'Right Arm Goal Region')
            plt.title('Equivariance Sampling Regions in Task Space')
            plt.xlabel('x')
            plt.ylabel('y')
            plt.grid()
            plt.legend()
            plt.xlim(-20, 20)
            plt.ylim(-20, 20)













def plot_demonstrations(data_vis=None, robot=None, demonstrations=None, demo_indices=None):
    if len(demonstrations['train_in']) == 0 and len(demonstrations['test_in']) == 0:
        print("No demonstrations to plot.")
        return
    if len(demonstrations['train_in']) > 0:
        if demo_indices is None:
            n = max(1, len(demonstrations['train_in']) // 4)
            true_train_traj_Q_X = demonstrations['train_in'][::n]
            true_train_traj_DQ = demonstrations['train_out'][::n]
        else:
            true_train_traj_Q_X = [demonstrations['train_in'][i] for i in demo_indices]
            true_train_traj_DQ = [demonstrations['train_out'][i] for i in demo_indices]
        # Convert to torch tensors for the visualization functions
        true_train_traj_Q_X = [torch.tensor(traj_q, dtype=torch.float32, device=device) for traj_q in true_train_traj_Q_X]
        true_train_traj_DQ = [torch.tensor(traj_dq, dtype=torch.float32, device=device) for traj_dq in true_train_traj_DQ]

        # Get the configuration space trajectories Q
        true_train_traj_Q = [traj_q[:, :robot.nb_dofs] for traj_q in true_train_traj_Q_X]

        # Get task space trajectories X and velocities dX
        true_train_traj_X = data_vis.get_X_from_Q_trajectories_torch(robot.fk_func_left_torch, robot.fk_func_right_torch, true_train_traj_Q_X, robot.nb_dofs_left, robot.nb_dofs_right)
        true_train_traj_dX = data_vis.get_TX_from_TQ_trajectories_torch(robot.jacob0_left_torch, robot.jacob0_right_torch, true_train_traj_Q, true_train_traj_DQ, robot.nb_dofs_left, robot.nb_dofs_right)
        
        # Move to cpu for visualization
        true_train_traj_Q = [traj_q.cpu() for traj_q in true_train_traj_Q]
        true_train_traj_X = [traj_x.cpu() for traj_x in true_train_traj_X]
        true_train_traj_dX = [traj_dX.cpu() for traj_dX in true_train_traj_dX]
    else:
        true_train_traj_Q_X = []
        true_train_traj_DQ = []
        true_train_traj_Q = []
        true_train_traj_X = []
        true_train_traj_dX = []

    if len(demonstrations['test_in']) > 0:
        if demo_indices is None:
            n = max(1, len(demonstrations['test_in']) // 3) # At least 1 trajectory
            true_test_traj_Q_X = demonstrations['test_in'][::n]
            true_test_traj_DQ = demonstrations['test_out'][::n]
        else:
            true_test_traj_Q_X = [demonstrations['test_in'][i] for i in demo_indices]
            true_test_traj_DQ = [demonstrations['test_out'][i] for i in demo_indices]
        # Convert to torch tensors for the visualization functions
        true_test_traj_Q_X = [torch.tensor(traj_q, dtype=torch.float32, device=device) for traj_q in true_test_traj_Q_X]
        true_test_traj_DQ = [torch.tensor(traj_dq, dtype=torch.float32, device=device) for traj_dq in true_test_traj_DQ]

        # Get the configuration space trajectories Q
        true_test_traj_Q = [traj_q[:, :robot.nb_dofs] for traj_q in true_test_traj_Q_X]
        
        # Get task space trajectories X and velocities dX
        true_test_traj_X = data_vis.get_X_from_Q_trajectories_torch(robot.fk_func_left_torch, robot.fk_func_right_torch, true_test_traj_Q_X, robot.nb_dofs_left, robot.nb_dofs_right)
        true_test_traj_dX = data_vis.get_TX_from_TQ_trajectories_torch(robot.jacob0_left_torch, robot.jacob0_right_torch, true_test_traj_Q, true_test_traj_DQ, robot.nb_dofs_left, robot.nb_dofs_right)

        # Move to cpu for visualization
        true_test_traj_Q = [traj_q.cpu() for traj_q in true_test_traj_Q]
        true_test_traj_X = [traj_x.cpu() for traj_x in true_test_traj_X]
        true_test_traj_dX = [traj_dX.cpu() for traj_dX in true_test_traj_dX]
    else:
        true_test_traj_Q_X = []
        true_test_traj_DQ = []
        true_test_traj_Q = []
        true_test_traj_X = []
        true_test_traj_dX = []

    # Plot the true trajectories in task space
    data_vis.plot_trajectories_colored( traj_q = {'train': true_train_traj_Q, 'test': true_test_traj_Q, 'predicted': []}, 
                                    traj_x = {'train': true_train_traj_X, 'test': true_test_traj_X, 'predicted': []}, 
                                    traj_dq = {'train': [], 'test': [], 'pred': []}, 
                                    merged_task_space = True, save_plots=False, title_prefix='', 
                                    color_scheme = {'train': 'green', 'test': 'blue', 'predicted': 'red',},
                                    line_width = {'train': 3, 'test': 3, 'predicted': 1.5})           


    # Plot task space velocities dx for as 2D arrows in the x-y task space plane.
    plt.figure(figsize=(8, 8))

    if len(true_train_traj_X) > 0:
        n = max(1, true_train_traj_X[0].shape[0] // 30)
        true_train_traj_dX_left = [traj_dX[:, :robot.nb_x_left] for traj_dX in true_train_traj_dX]
        true_train_traj_dX_right = [traj_dX[:, robot.nb_x_left:] for traj_dX in true_train_traj_dX]
        # only plot part of the points/vectors for better visualization (can be adjusted as needed)
        true_train_traj_X_sampled = [traj_x[::n] for traj_x in true_train_traj_X]
        true_train_traj_dX_left_sampled = [traj_dX[::n] for traj_dX in true_train_traj_dX_left]
        true_train_traj_dX_right_sampled = [traj_dX[::n] for traj_dX in true_train_traj_dX_right]
        for traj_x, traj_dX_left, traj_dX_right in zip(true_train_traj_X_sampled, true_train_traj_dX_left_sampled, true_train_traj_dX_right_sampled):
            plt.quiver(traj_x[:, 0], traj_x[:, 1], traj_dX_left[:, 0], traj_dX_left[:, 1], color='green', label='Train Traj dX Left' if 'Train Traj dX Left' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.quiver(traj_x[:, 2], traj_x[:, 3], traj_dX_right[:, 0], traj_dX_right[:, 1], color='green', label='Train Traj dX Right' if 'Train Traj dX Right' not in plt.gca().get_legend_handles_labels()[1] else "")

    if len(true_test_traj_X) > 0:
        n = max(1, true_test_traj_X[0].shape[0] // 30)
        true_test_traj_dX_left = [traj_dX[:, :robot.nb_x_left] for traj_dX in true_test_traj_dX]
        true_test_traj_dX_right = [traj_dX[:, robot.nb_x_left:] for traj_dX in true_test_traj_dX]
        # only plot part of the points/vectors for better visualization (can be adjusted as needed)
        true_test_traj_X_sampled = [traj_x[::n] for traj_x in true_test_traj_X]
        true_test_traj_dX_left_sampled = [traj_dX[::n] for traj_dX in true_test_traj_dX_left]
        true_test_traj_dX_right_sampled = [traj_dX[::n] for traj_dX in true_test_traj_dX_right]
        # Plot the sampled task space velocities as quiver plots
        for traj_x, traj_dX_left, traj_dX_right in zip(true_test_traj_X_sampled, true_test_traj_dX_left_sampled, true_test_traj_dX_right_sampled):
            plt.quiver(traj_x[:, 0], traj_x[:, 1], traj_dX_left[:, 0], traj_dX_left[:, 1], color='blue', label='Test Traj dX Left' if 'Test Traj dX Left' not in plt.gca().get_legend_handles_labels()[1] else "")
            plt.quiver(traj_x[:, 2], traj_x[:, 3], traj_dX_right[:, 0], traj_dX_right[:, 1], color='blue', label='Test Traj dX Right' if 'Test Traj dX Right' not in plt.gca().get_legend_handles_labels()[1] else "")

    plt.title('Task Space Velocities dX')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.grid()
    plt.legend()
    plt.show()






def animate_letter_transition(original_letter, vector_field, integrator,
                              num_frames=100, integration_horizon=0.1,
                              xlim=(-4, 4), ylim=(-4, 4),
                              fig=None, ax=None,
                              vf_resolution=20,
                              vf_shift=np.array([0.0, 0.0])):

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    # --- Plot static vector field (background) ---
    plot_vector_field(ax=ax, Vf=vector_field, title='Vector Field (Background)', X_min=np.array([xlim[0], ylim[0]]), X_max=np.array([xlim[1], ylim[1]]), n_points=vf_resolution, color='gray', shift=vf_shift)

    # --- Create animated line once ---
    letter_line, = ax.plot(original_letter[:, 0],
                           original_letter[:, 1],
                           color='blue',
                           linewidth=2,
                           label='Original Letter')

    # Axis formatting (do once)
    ax.set_title('Transition of Original Letter under Vector Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.legend()

    # --- Animation loop ---
    for _ in range(num_frames):

        # Integrate all points (vectorized is cleaner if integrator supports it)
        for i in range(len(original_letter)):
            original_letter[i] = integrator(x=original_letter[i], t=integration_horizon, shift=vf_shift)

        # Update only the line data (no clearing!)
        letter_line.set_data(original_letter[:, 0],
                             original_letter[:, 1])

        plt.pause(0.05)

    plt.show()

def animate_letter_transition_save(original_letter, vector_field, integrator,
                              num_frames=100, integration_horizon=0.1,
                              xlim=(-4, 4), ylim=(-4, 4),
                              fig=None, ax=None,
                              vf_resolution=20,
                              vf_shift=None,
                              gif_name=None,
                              fps=20):
    '''
    BUG: This is still not working properly. The saved animation looks okay, but when I visualize it during runtime using plt.show(), the animation looks wrong.
    '''

    raise NotImplementedError("This function is still not working properly. Needs debugging.")

    if fig is None or ax is None:
        fig, ax = plt.subplots()

    plot_vector_field(ax=ax,
                      Vf=vector_field,
                      title='Vector Field (Background)',
                      X_min=np.array([xlim[0], ylim[0]]),
                      X_max=np.array([xlim[1], ylim[1]]),
                      n_points=vf_resolution,
                      color='gray',
                      shift=vf_shift)

    letter_line, = ax.plot(original_letter[:, 0],
                           original_letter[:, 1],
                           color='blue',
                           linewidth=2,
                           label='Original Letter')

    ax.set_title('Deformation of Letter under Vector Field')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    # ax.legend()

    def update(frame):

        # EXACT SAME SIMULATION STEP
        for i in range(len(original_letter)):
            original_letter[i] = integrator(
                x=original_letter[i],
                t=integration_horizon,
                shift=vf_shift
            )

        letter_line.set_data(original_letter[:, 0],
                             original_letter[:, 1])

        return letter_line,

    ani = animation.FuncAnimation(
        fig,
        update,
        frames=num_frames,
        interval=50,   # purely visual timing
        blit=False
    )

    if gif_name is not None:
        save_path = os.path.join(CURRENT_DIR, gif_name)
        ani.save(save_path, writer='pillow', fps=fps)

    # plt.show()




def plot_vector_field(ax=None, Vf=None, title='Vf', X_min=np.array([-2.0, -2.0]), X_max=np.array([2.0, 2.0]), n_points=20, color='blue', shift=np.array([0.0, 0.0])):
    # Define the task space bounds and resolution for visualization
    # X_min = np.array([-2.0, -2.0])
    # X_max = np.array([2.0, 2.0])
    # n_points = 20 # Per dimension

    x1 = np.linspace(X_min[0], X_max[0], n_points)
    x2 = np.linspace(X_min[1], X_max[1], n_points)
    X1, X2 = np.meshgrid(x1, x2)
    X = np.vstack([X1.flatten(), X2.flatten()]).T

    # Compute the vector field at each point in the grid
    VF = np.array([Vf(x, shift=shift) for x in X])
    VF_X1 = VF[:, 0].reshape(X1.shape)
    VF_X2 = VF[:, 1].reshape(X2.shape)

    # NOTE: Just for visualization purposes and debugging
    mag = np.sqrt(VF_X1**2 + VF_X2**2)
    max_allowed_mag = 1.0
    factor = np.minimum(1.0, max_allowed_mag / (mag + 1e-6))
    VF_X1 = VF_X1 * factor # Clipping the vector field for better visualization (can be removed if not needed or if it disstorts the true results)
    VF_X2 = VF_X2 * factor # Clipping the vector field for better visualization (can be removed if not needed or if it disstorts the true results)


    # Plot the vector field
    if ax is None:
        plt.figure(figsize=(8, 8))
        plt.quiver(X1, X2, VF_X1, VF_X2, color=color)
        # plt.quiver(X1, X2, VF_X1, VF_X2, color=color, scale=0.1, scale_units='xy')
        plt.title(title)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.xlim(X_min[0], X_max[0])
        plt.ylim(X_min[1], X_max[1])
        plt.grid()
        # plt.show()
    else:
        ax.quiver(X1, X2, VF_X1, VF_X2, color=color)
        # ax.quiver(X1, X2, VF_X1, VF_X2, color=color, scale=1.0, scale_units='xy')
        ax.set_title(title)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(X_min[0], X_max[0])
        ax.set_ylim(X_min[1], X_max[1])
        ax.grid()

def plot_VQ_VX(robot, VQ, VX):
    # Create the configuration space grid
    q_min = np.array([-np.pi, -np.pi])
    q_max = np.array([np.pi, np.pi])
    n_points = 20
    q1 = np.linspace(q_min[0], q_max[0], n_points)
    q2 = np.linspace(q_min[1], q_max[1], n_points)
    Q1, Q2 = np.meshgrid(q1, q2)
    Q = np.vstack([Q1.flatten(), Q2.flatten()]).T

    # Construct the task space grid by applying the forward kinematics to the configuration space grid
    X = np.array([robot.fkine(q) for q in Q])

    # Compute the  vector fields in configuration space and task space (VQ takes in the VX method and the q0)
    VQ_values = np.array([VQ(VX, q) for q in Q])
    VX_values = np.array([VX(x) for x in X])
    
    VQ1 = VQ_values[:, 0].reshape(Q1.shape)
    VQ2 = VQ_values[:, 1].reshape(Q2.shape)
    VX1 = VX_values[:, 0].reshape(Q1.shape)
    VX2 = VX_values[:, 1].reshape(Q2.shape)

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(16, 8))
    # Plot VQ
    axs[0].quiver(Q1, Q2, VQ1, VQ2, color='red')
    # axs[0].set_title('Configuration Space Vector Field VQ')
    axs[0].set_title(r'Config Space: $V_{Q} = J^{\dagger} V_{X}$')
    axs[0].set_xlabel('q1')
    axs[0].set_ylabel('q2')
    axs[0].set_xlim(q_min[0], q_max[0])
    axs[0].set_ylim(q_min[1], q_max[1])
    axs[0].grid()
    # Plot VX
    axs[1].quiver(X[:, 0], X[:, 1], VX1, VX2, color='blue')
    # axs[1].set_title('Task Space Vector Field VX')
    axs[1].set_title(r'Task Space: $V_{X}$')
    axs[1].set_xlabel('x')
    axs[1].set_ylabel('y')
    axs[1].set_xlim(X[:, 0].min(), X[:, 0].max())
    axs[1].set_ylim(X[:, 1].min(), X[:, 1].max())
    axs[1].grid()
    # plt.show()



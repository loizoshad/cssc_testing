import matplotlib.pyplot as plt
from typing import Union, Tuple
import numpy as np
import matplotlib.path as mpath
import matplotlib.patches as mpatches


def plot_planar_robot(ax,
                      joint_angles: np.ndarray, link_lengths: Union[int, np.ndarray],
                      width_param: float = 0.2, facecolor: Union[str, list] = 'gray',
                      edgecolor: Union[str, list] = 'white') \
        -> list:
    """
    This function displays a serial planar robot with an arbitrary number of joints.

    Parameters
    ----------
    :param ax: axes instance of the figure
    :param joint_angles: joint angles (numpy array of size nb_dofs)
    :param link_lengths: link lengths (scalar if all links have the same length, numpy array of size nb_dofs otherwise)

    Optional parameters
    -------------------
    :param width_param: link width parameter for the plots
    :param facecolor: color of the links
    :param edgecolor: color of the edges of the links

    Returns
    -------
    :return patch_list: list of patches representing the robot [base_patches, link0_patches, ..., linkn_patches]

    Notes
    -------
    To erase the robot from the figure do:
    for patch_list in patch_list_planar_robot:
    for patch in patch_list:
        patch.remove()

    """
    # Make background white
    ax.patch.set_facecolor('white')
    # No grid
    ax.grid(False)
    # Patch list to return
    patch_list = []

    # Number of DoFs
    nb_dofs = joint_angles.shape[0]
    # Links length as an array
    if np.isscalar(link_lengths):
        link_lengths = np.tile(link_lengths, nb_dofs)

    # Plot basis
    patch_list_basis = plot_robot_basis(ax, width_param, facecolor, edgecolor)
    patch_list.append(patch_list_basis)

    # Plot links
    current_endeff_position = np.zeros(2)
    for i in range(len(joint_angles)):
        patch_list_link, current_endeff_position = plot_robot_link(ax, np.sum(joint_angles[:i+1]), link_lengths[i],
                                                                   current_endeff_position, width_param, facecolor,
                                                                   edgecolor)
        patch_list.append(patch_list_link)

    return patch_list


def plot_robot_basis(ax, width_param: float = 0.05,
                     facecolor: Union[str, list] = 'gray', edgecolor: Union[str, list] = 'white') \
        -> list:
    """
    This function displays the basis of a serial planar robot with an arbitrary number of joints.

    Parameters
    ----------
    :param ax: axes instance of the figure

    Optional parameters
    -------------------
    :param width_param: link width parameter for the plots
    :param facecolor: color of the links
    :param edgecolor: color of the edges of the links

    Returns
    -------
    :return patch_list: list of patches representing the robot base [base_patch, line0_patch, ..., line5_patch]
    """

    nb_segments = 30
    width_param = width_param * 1.2

    # Draw basis
    # Define contour
    t1 = np.linspace(0, np.pi, nb_segments - 2)
    x = np.zeros((nb_segments, 2))
    x[:, 0] = np.append(np.append(width_param * 1.5, width_param * 1.5 * np.cos(t1)), -width_param * 1.5)
    x[:, 1] = np.append(np.append(-width_param * 1.2, width_param * 1.5 * np.sin(t1)), - width_param * 1.2)
    # Draw path
    path = mpath.Path(x)
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=1)
    ax.add_patch(patch)

    # Patch list to return
    patch_list = [patch]

    # Draw 5 bottom lines
    # Lines coordinates
    x2 = np.zeros((5, 2))
    x2[:, 0] = np.linspace(-width_param * 1.2, width_param * 1.2, 5)
    x2[:, 1] = -width_param * 1.2
    x3 = x2 + np.tile(0.25*np.array([-0.5, -1]), (5, 1))
    # Define and draw paths
    for i in range(5):
        x_path = np.array([[x2[i, 0], x2[i, 1]], [x3[i, 0], x3[i, 1]]])
        path = mpath.Path(x_path)
        patch_line = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=facecolor, linewidth=2)
        ax.add_patch(patch_line)
        patch_list.append(patch_line)

    return patch_list


def plot_robot_link(ax,
                    angle: float, link_length: int, base_position: np.ndarray, width_param: float = 0.05,
                    facecolor: Union[str, list] = 'gray', edgecolor: Union[str, list] = 'white', joint_linewidth: float = 2) \
        -> Tuple[list, np.ndarray]:
    """
    This function displays a link of a serial planar robot with an arbitrary number of joints.

    Parameters
    ----------
    :param ax: axes instance of the figure
    :param angle: angle of the link to display
    :param link_length: length of the link
    :param base_position: position of the base of the link (numpy array of size 2)

    Optional parameters
    -------------------
    :param width_param: link width parameter for the plots
    :param facecolor: color of the links
    :param edgecolor: color of the edges of the links

    Returns
    -------
    :return patch_list: list of patches representing the robot link [link_patch, circle0_patch, circle1_patch]
    :return endeffector_position: position of the end of the link (numpy array of size 2)
    """
    # Number of segments
    nb_segments = 30

    # Draw link "bar"
    t1 = np.linspace(0, -np.pi, int(nb_segments/2))
    t2 = np.linspace(np.pi, 0, int(nb_segments/2))
    # Define contour
    x = np.zeros((nb_segments, 2))
    x[:, 0] = np.append(width_param * np.sin(t1), link_length + width_param * np.sin(t2))
    x[:, 1] = np.append(width_param * np.cos(t1), width_param * np.cos(t2))
    x = np.vstack((x, x[0, :]))  # Add first element at the end to have a closed contour
    # Rotate contours
    R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    x = np.dot(R, x.T).T + base_position
    # Draw path
    path = mpath.Path(x)
    patch = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
    ax.add_patch(patch)

    # Draw holes
    # Define circle contour
    endeff_position = np.dot(R, np.array([link_length, 0])) + base_position
    msh = np.zeros((nb_segments, 2))
    msh[:, 0] = np.sin(np.linspace(0, 2 * np.pi, nb_segments))
    msh[:, 1] = np.cos(np.linspace(0, 2 * np.pi, nb_segments))
    msh *= width_param * 0.2
    # Draw first circle
    path = mpath.Path(msh + base_position)
    patch_circle1 = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=joint_linewidth)
    ax.add_patch(patch_circle1)
    # Draw second circle
    path = mpath.Path(msh + endeff_position)
    patch_circle2 = mpatches.PathPatch(path, facecolor=facecolor, edgecolor=edgecolor, linewidth=joint_linewidth)
    ax.add_patch(patch_circle2)

    # Patch list to return
    patch_list = [patch, patch_circle1, patch_circle2]

    return patch_list, endeff_position


class RobotVisualizer:
    def __init__(self, ax, arm_length, ee_joint=False):
        self.ax = ax
        self.arm_length = arm_length
        self.ee_joint = ee_joint

    def draw_robot(self, ax=None, qt=None):
        if ax is None:
            ax = self.ax
        plot_planar_robot(ax, qt, self.arm_length, facecolor='black')

    def draw_demos(self, demos):
        for demo in demos:
            self.ax.plot(demo[:, 0], demo[:, 1], linestyle='dashed', color='gray', alpha=0.5)

    def draw_points(self, ax, points, color='seagreen'):
        ax.scatter(points[:, 0], points[:, 1], color=color, alpha=0.5, s=3)

def animate_robot(ax, visualizer, q_data_left, x_data_left, q_data_right, x_data_right):
    # Plot all arms at the same time. Each data is a list of trajectories
        for t in range(q_data_left[0].shape[0]):
            plt.cla()
            for i in range(len(q_data_left)):
                visualizer.draw_robot(ax, q_data_left[i][t, :])
                visualizer.draw_robot(ax, q_data_right[i][t, :])
                # Draw the desired and current end-effector positions for each arm
                visualizer.draw_points(ax, x_data_left[i][:, :2], color='seagreen')
                visualizer.draw_points(ax, x_data_right[i][:, :2], color='navy')
            plt.xlim(-4, 10)
            plt.ylim(-7, 7)
            plt.pause(0.01)

def plot_demos_single(axs, q_data_left, dq_data_left, x_data_left, dx_data_left, q_data_right, dq_data_right, x_data_right, dx_data_right, x_data_projected_left=None, x_data_projected_right=None, dx_data_projected_left=None, dx_data_projected_right=None):
    nb_dofs = q_data_left.shape[1]
    # print number of axs
    axs = axs.flatten()
    plot_idx = 0
    # Plot joint positions
    for i in range(nb_dofs):
        axs[plot_idx].plot(q_data_left[:, i], color='blue', label='Left arm')
        axs[plot_idx].plot(q_data_right[:, i], color='red', label='Right arm')
        axs[plot_idx].set_title(f'Joint Position {i+1}')
        plot_idx += 1
    # Plot joint velocities
    for i in range(nb_dofs):
        axs[plot_idx].plot(dq_data_left[:, i], color='blue', label='Left arm')
        axs[plot_idx].plot(dq_data_right[:, i], color='red', label='Right arm')
        axs[plot_idx].set_title(f'Joint Velocity {i+1}')
        plot_idx += 1
    # Plot end-effector positions and set the size of the marker
    axs[plot_idx].plot(x_data_left[:, 0], x_data_left[:, 1], color='black', label='Left arm')
    axs[plot_idx].plot(x_data_right[:, 0], x_data_right[:, 1], color='black', label='Right arm')
    # Plot projected end-effector positions (use dashed lines)
    axs[plot_idx].plot(x_data_projected_left[:, 0], x_data_projected_left[:, 1], color='blue', linestyle=(0, (5, 15)), label='Left arm projected')
    axs[plot_idx].plot(x_data_projected_right[:, 0], x_data_projected_right[:, 1], color='red', linestyle=(0, (5, 15)), label='Right arm projected')
    axs[plot_idx].set_title('End-Effector Positions')
    plot_idx += 1
    # Plot end-effector velocities
    for i in range(2):
        axs[plot_idx].plot(dx_data_left[:, i], color='grey', label='Left arm')
        axs[plot_idx].plot(dx_data_right[:, i], color='grey', label='Right arm')
        # Plot projected end-effector velocities (use dashed lines)
        axs[plot_idx].plot(dx_data_projected_left[:, i], color='blue', linestyle=(0, (5, 15)), label='Left arm projected')
        axs[plot_idx].plot(dx_data_projected_right[:, i], color='red', linestyle=(0, (5, 15)), label='Right arm projected')
        axs[plot_idx].set_title(f'End-Effector Velocity {"x" if i==0 else "y"}')
        plot_idx += 1

def plot_demos(axs, q_data_left, dq_data_left, x_data_left, dx_data_left, q_data_right, dq_data_right, x_data_right, dx_data_right, x_data_projected_left=None, x_data_projected_right=None, dx_data_projected_left=None, dx_data_projected_right=None):
    # Loop through each demo in eaceh list and plot them (left arm are with red color and right arm with blue color)
    for i in range(len(q_data_left)):
        q_left = q_data_left[i]
        dq_left = dq_data_left[i]
        x_left = x_data_left[i]
        dx_left = dx_data_left[i]

        q_right = q_data_right[i]
        dq_right = dq_data_right[i]
        x_right = x_data_right[i]
        dx_right = dx_data_right[i]

        x_projected_left = x_data_projected_left[i]
        x_projected_right = x_data_projected_right[i]
        dx_projected_left = dx_data_projected_left[i]
        dx_projected_right = dx_data_projected_right[i]

        plot_demos_single(axs, q_left, dq_left, x_left, dx_left, q_right, dq_right, x_right, dx_right, x_projected_left, x_projected_right, dx_projected_left, dx_projected_right)

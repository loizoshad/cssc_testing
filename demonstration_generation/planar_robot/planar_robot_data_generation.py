import os
from pathlib import Path
import numpy as np
import scipy.io as sp
from spatialmath import SE3
import matplotlib.pyplot as plt

from robots.planar_robot.visualization import RobotVisualizer, animate_robot, plot_demos
from robots.planar_robot.robots import PlanarManipulator_nDoF
from robots.planar_robot.utils import get_damped_least_squares_inverse

from dataclasses import dataclass

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent



#################################################################################
# Auxiliary classes and methods
#################################################################################
@dataclass
class ArmSetup:
    q0: np.ndarray        # initial joint configuration of the arm
    scale_bound: float    # the raw demos are rescaled to [-scale_bound, scale_bound]
    offset: np.ndarray    # the rescaled demos are shifted by fkine(q0) - offset

@dataclass
class ArmDemos:
    robot: PlanarManipulator_nDoF
    q: list               # per demo: (T, nb_dofs) joint positions
    dq: list              # per demo: (T, nb_dofs) joint velocities (finite differences)
    x: list               # per demo: (T, 2) end-effector positions reached by the robot
    dx: list              # per demo: (T, 2) end-effector velocities (finite differences)

def arm_setup(side: str, nb_dofs: int, demo_name: str, is_taskspace: bool = False) -> ArmSetup:
    """
    Description
    -----------    
    Initialize the arm's configuration and placement of the LASA demo.
    TODO: Right now for each arm and different letters we have hardcoded the initial configuration and placement of the demo, but we should make it more general in the future.
    """
    left = side == 'left'
    if nb_dofs == 4:
        q0 = np.array([np.pi/4, -np.pi/6, -np.pi/8, -np.pi/10]) if left else np.array([-np.pi/4, np.pi/6, np.pi/8, np.pi/10])
        offset = np.array([5.0, -1.5]) if left else np.array([5.0, 1.5])
        return ArmSetup(q0=q0, scale_bound=1.5, offset=offset)
    if nb_dofs == 2:
        q0 = np.array([np.pi/4, -np.pi/6]) if left else np.array([-np.pi/4, np.pi/6])
        if is_taskspace:
            offset = np.array([-5.0, -4.0]) if left else np.array([-6.5, 4.0])
        else:
            # PShape (left arm) / SShape (right arm) are placed closer to the base.
            x_offset = 1.4 if demo_name == ('PShape' if left else 'SShape') else 1.9
            offset = np.array([x_offset, 0.9 if left else -0.9])
        return ArmSetup(q0=q0, scale_bound=1.0, offset=offset)
    raise ValueError(f'No arm setup defined for nb_dofs={nb_dofs}')

def generate_arm_demos(setup: ArmSetup, demo_type: str, demo_name: str, nb_dofs: int, arm_length: float, dt: float, base: SE3 = SE3(x=0.0, y=0.0, z=0), is_taskspace: bool = False) -> ArmDemos:
    """
    Description
    ---------
    Track the LASA demos with one planar robot arm and return the resulting end-effector and joint-space trajectories.
    """
    robot = PlanarManipulator_nDoF(nb_dofs, base, arm_length, is_taskspace=is_taskspace)
    controller = TaskSpaceController(robot, ControllerConfig(dt=dt), is_taskspace=is_taskspace)
    x0 = robot.fkine(setup.q0)

    demos = load_raw_dataset(demo_type, demo_name)
    print(f"Loaded {len(demos)} for demo: {demo_name}")
    demos = preprocess_dataset(demos_data=demos, scale_bound=setup.scale_bound, shift=x0 - setup.offset)

    q, dq, x, dx = construct_demos(robot, controller, demos, setup.q0)
    return ArmDemos(robot=robot, q=q, dq=dq, x=x, dx=dx)

@dataclass
class RobotConfig:
    nb_dofs: int
    arm_length: float
    base: SE3
    q0: np.ndarray

@dataclass
class ControllerConfig:
    dt: float      # Sampling time
    K: np.ndarray = 100  # Controller gain

class TaskSpaceController:
    def __init__(self, robot, control_config: ControllerConfig, dim_task=2, is_taskspace=False):
        self.robot = robot
        self.dt = control_config.dt
        self.dim_task = dim_task
        self.is_taskspace = is_taskspace

    def step(self, qt, xd):
        # Current end-effector position
        xt = self.robot.fkine(qt)[:self.dim_task]
        # Desired end-effector velocity
        dx = xd - xt
        # Jacobian at current configuration
        jacobian = self.robot.jacob0(qt)
        # Damped least squares inverse of the Jacobian
        
        if self.is_taskspace:
            pinvJ = jacobian
        else:
            pinvJ = get_damped_least_squares_inverse(jacobian, damping_factor=1e-3)
        
        # Compute joint velocities
        dqt = pinvJ @ dx
        # Update joint positions
        # qt_next = qt + dqt * self.dt
        qt_next = qt + dqt #* self.dt
        return qt_next, dqt, dx
    
def load_raw_dataset(demo_type, demo_name):
    # letter_path = os.path.join(ROOT_DIR / 'demonstrations' / 'planar_robot' / demo_type / (demo_name + '.mat'))
    letter_path = os.path.join(ROOT_DIR / 'demonstrations' / 'LASA' / (demo_name + '.mat'))
    letter_data = sp.loadmat(letter_path)['demos']

    # letter_data = sp.loadmat(path + '/../../demonstrations/' + demo_type + '/' + demo_name + '.mat')['demos']
    num_demos = len(letter_data[0])
    demos_data = [letter_data[0, i][0, 0][0].T for i in range(num_demos)]

    return demos_data

def normalize_and_shift_raw_dataset(demos_data, bound, shift, max_num_demos=100):
    # Normalize dataset
    stacked_demos_data = np.vstack(demos_data)

    min_data = np.min(stacked_demos_data, axis=0)
    max_data = np.max(stacked_demos_data, axis=0)
    demos_data = [(((demo - min_data) / (max_data - min_data)) - 0.5) * 2.0 * bound for demo in demos_data]
    # Shift dataset
    stacked_demos_data += shift
    demos_data = [demo + shift for demo in demos_data]

    return demos_data, stacked_demos_data

def rollout_trajectory(robot, controller, q0, demo):
    T = demo.shape[0] # Number of time steps
    nb_dofs = q0.shape[0] # Number of DoFs

    q_history = np.zeros((T, nb_dofs))
    x_history = np.zeros((T, 2))

    qt = q0.copy()
    for t in range(T):
        xd = demo[t, :]
        qt_next, _, _ = controller.step(qt, xd)

        # Log data
        q_history[t, :] = qt
        x_history[t, :2] = robot.fkine(qt)[:2]
        qt = qt_next

    n_trimmed_entries = 10
    n_trimmed_end_entries = 1 
    q_history = q_history[n_trimmed_entries:-n_trimmed_end_entries, :]
    x_history = x_history[n_trimmed_entries:-n_trimmed_end_entries, :]

    return q_history, x_history

def merge_and_save_data(q_data_left, dq_data_left, x_data_left, q_data_right, dq_data_right, x_data_right, demo_type_left, demo_name_left, demo_type_right, demo_name_right, nb_dofs, is_taskspace=False):
    q_data = []
    dq_data = []
    x_data = []

    num_demos = len(q_data_left)

    for i in range(num_demos):
        q_merged = np.hstack((q_data_left[i], q_data_right[i]))
        dq_merged = np.hstack((dq_data_left[i], dq_data_right[i]))
        x_merged = np.hstack((x_data_left[i], x_data_right[i]))

        q_data.append(q_merged)
        dq_data.append(dq_merged)
        x_data.append(x_merged)

    # Save data
    demos_name = f'left-{demo_type_left}-{demo_name_left}_right-{demo_type_right}-{demo_name_right}_ndofs-{2*nb_dofs}'


    # If taskspace attach _taskspace to the name of the dataset
    if is_taskspace:
        demos_name = demos_name + '_taskspace'

    np.savez(ROOT_DIR / 'demonstrations' / 'planar_robot' / 'planar_robot_lasa' / demos_name, q_data=q_data, dq_data=dq_data, x_data=x_data)
    # np.savez(ROOT_DIR / 'demonstrations/planar_robot' / demo_name, q_data=q_data, dq_data=dq_data, x_data=x_data) 


def construct_demos(robot, controller, demos, q0):
    q_data = []
    dq_data = []
    x_data = []
    dx_data = []

    for demo in demos:
        q_history, x_history = rollout_trajectory(robot, controller, q0, demo)
        q_data.append(q_history)
        x_data.append(x_history)

    # For dq we can compute it by finite differences from the q_data
    for q_history in q_data:
        dq_history = np.diff(q_history, axis=0) / controller.dt
        # Pad the first entry of dq_history with zeros to maintain the same length as q_history
        dq_history = np.vstack((np.zeros((1, q_history.shape[1])), dq_history))
        dq_data.append(dq_history)

    # For dx we can compute it by finite differences from the x_data
    for x_history in x_data:
        dx_history = np.diff(x_history, axis=0) / controller.dt
        # Pad the first entry of dx_history with zeros to maintain the same length as x_history
        dx_history = np.vstack((np.zeros((1, x_history.shape[1])), dx_history))
        dx_data.append(dx_history)

    return q_data, dq_data, x_data, dx_data

def preprocess_dataset(demos_data, scale_bound, shift, window_size=5, smoothen_coeff=0.5):
    # Normalize and shift dataset
    demos_data, _ = normalize_and_shift_raw_dataset(demos_data, bound=scale_bound, shift=shift)

    # Remove the first n and last m entries of each trajectory to avoid large velocities at the beginning and end of the trajectories

    # Remove the first n and last m entries of each trajectory again to avoid large velocities at the beginning and end of the trajectories after smoothing
    # # NOTE: Good
    # n = 15
    # m = 20

    # n = 15
    # m = 2

    # n = 5
    # m = 2

    n = 10
    m = 2   
    
    # n = 100
    # m = 2     

    demos_data = [demo[n:-m, :] for demo in demos_data]

    return demos_data


def get_taskspace_from_config(q_traj, dq_traj, robot):
    # Apply forward kinematics on each trajectory in q_traj and jacobian for dq
    x_traj = []
    dx_traj = []

    for i, (q_single_traj, dq_single_traj) in enumerate(zip(q_traj, dq_traj)):
        x_single_traj = np.array([robot.fkine(q)[:2] for q in q_single_traj])
        dx_single_traj = np.array([robot.jacob0(q) @ dq for q, dq in zip(q_single_traj, dq_single_traj)])

        x_traj.append(x_single_traj)
        dx_traj.append(dx_single_traj)

    return x_traj, dx_traj


#################################################################################
# Main code
#################################################################################
def main():
    is_taskspace = False
    arm_length = 3.0
    nb_dofs = 4
    dt = 0.01
    save_data_flag = True
    animate_robot_flag = False

    #################################
    # Left arm (Robot 1)
    #################################
    demo_type = 'LASA'    
    demo_name_left = 'CShape'
    demo_name_right = 'NShape'

    left_arm_demos = generate_arm_demos(arm_setup('left', nb_dofs, demo_name_left, is_taskspace), demo_type, demo_name_left, nb_dofs, arm_length, dt, is_taskspace=is_taskspace)
    right_arm_demos = generate_arm_demos(arm_setup('right', nb_dofs, demo_name_right, is_taskspace), demo_type, demo_name_right, nb_dofs, arm_length, dt, is_taskspace=is_taskspace)

    ##################################################################
    # Visualize demos
    ##################################################################
    num_plots = nb_dofs * 2 + 4 # Joint positions, joint velocities, end-effector positions, end-effector velocities
    max_cols = 4
    num_cols = min(num_plots, max_cols)
    num_rows = (num_plots + num_cols - 1) // num_cols
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(4*num_cols, 4*num_rows))
    x_data_projected_left, dx_data_projected_left = get_taskspace_from_config(left_arm_demos.q, left_arm_demos.dq, left_arm_demos.robot)
    x_data_projected_right, dx_data_projected_right = get_taskspace_from_config(right_arm_demos.q, right_arm_demos.dq, right_arm_demos.robot)
    plot_demos(axs, left_arm_demos.q, left_arm_demos.dq, left_arm_demos.x, left_arm_demos.dx, right_arm_demos.q, right_arm_demos.dq, right_arm_demos.x, right_arm_demos.dx, x_data_projected_left, x_data_projected_right, dx_data_projected_left, dx_data_projected_right)
    plt.tight_layout()
    plt.show()

    # Animate the robot
    if animate_robot_flag:
        fig, ax = plt.subplots(figsize=(8, 8))
        visualizer = RobotVisualizer(ax, arm_length, ee_joint=False)
        animate_robot(ax, visualizer, left_arm_demos.q, left_arm_demos.x, right_arm_demos.q, right_arm_demos.x)

    # Save data
    if save_data_flag:
        merge_and_save_data(left_arm_demos.q, left_arm_demos.dq, left_arm_demos.x, right_arm_demos.q, right_arm_demos.dq, right_arm_demos.x, demo_type, demo_name_left, demo_type, demo_name_right, nb_dofs, is_taskspace=is_taskspace)


if __name__ == "__main__":
    main()







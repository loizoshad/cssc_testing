import numpy as np

def get_damped_least_squares_inverse(jacobian: np.ndarray, damping_factor: float = 0.01) -> np.ndarray:
    """
    Description
    -----------
    This function return the damped least-squares (DLS) inverse of the Jacobian.
    This function was implemented based on get_damped_least_squares_inverse of robot.py in pyrobolearn
    (https://github.com/robotlearn/pyrobolearn).

    Parameters
    ----------
    :param jacobian: Jacobian matrix (numpy array of dimension [nb_task_vars, nb_dofs])

    Optional parameters
    -------------------
    :param damping_factor: damping factor

    Returns
    -------
    :return: DLS inverse of the Jacobian
    """
    damping = damping_factor ** 2 * np.identity(jacobian.shape[0])
    return jacobian.T.dot(np.linalg.inv(jacobian.dot(jacobian.T) + damping))

def get_weighted_damped_least_squares_inverse(jacobian: np.ndarray, weight_matrix: np.ndarray, damping_factor: float = 0.01) -> np.ndarray:
    """
    Description
    -----------    
    This function return the damped least-squares (DLS) inverse of the Jacobian.
    This function was implemented based on get_damped_least_squares_inverse of robot.py in pyrobolearn
    (https://github.com/robotlearn/pyrobolearn).

    Parameters
    ----------
    :param jacobian: Jacobian matrix (numpy array of dimension [nb_task_vars, nb_dofs])
    :param weight_matrix: Weight matrix (numpy array of dimension [nb_dofs, nb_dofs])

    Optional parameters
    -------------------
    :param damping_factor: damping factor

    Returns
    -------
    :return: DLS inverse of the Jacobian
    """
    damping = damping_factor ** 2 * np.identity(jacobian.shape[0])
    return weight_matrix.dot(jacobian.T).dot(np.linalg.inv(jacobian.dot(weight_matrix.dot(jacobian.T)) + damping))


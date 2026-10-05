import numpy as np
import torch
from torch.nn import Module


def joint_derivative_Jacobian_np(J: np.ndarray) -> np.ndarray:
    nb_rows = J.shape[0]
    nb_cols = J.shape[1]

    # Check if the robot is planar, and augment the Jacobian
    if nb_rows == 2:
        J = np.vstack((J, np.zeros((3, nb_cols)), np.ones((1, nb_cols))))
    elif nb_rows == 3:
        J = np.vstack((J[:2, :], np.zeros((3, nb_cols)), J[2, :]))
    elif nb_rows != 6:
        raise NotImplementedError

    J_grad = np.zeros((6, nb_cols, nb_cols))
    for i in range(nb_cols):
        for j in range(nb_cols):
            J_i = J[:, i]
            J_j = J[:, j]
            if j < i:
                J_grad[0:3, i, j] = np.cross(J_j[3:6], J_i[0:3])
                J_grad[3:6, i, j] = np.cross(J_j[3:6], J_i[3:6])
            elif j > i:
                J_grad[0:3, i, j] = -np.cross(J_j[0:3], J_i[3:6])
            else:
                J_grad[0:3, i, j] = np.cross(J_i[3:6], J_i[0:3])

    if nb_rows == 2:
        return J_grad[0:2, :, :]
    if nb_rows == 3:
        return np.concatenate((J_grad[0:2, :, :], J_grad[-1, :, :]), axis=0)
    else:
        return J_grad


def joint_derivative_Jacobian(J: torch.Tensor) -> torch.Tensor:
    nb_rows = J.shape[0]
    nb_cols = J.shape[1]

    # Check if the robot is planar, and augment the Jacobian
    if nb_rows == 2:
        J = torch.vstack((J, torch.zeros((3, nb_cols)), torch.ones((1, nb_cols))))
    elif nb_rows == 3:
        J = torch.vstack((J[:2, :], torch.zeros((3, nb_cols)), J[2, :]))
    elif nb_rows != 6:
        raise NotImplementedError

    J_grad = torch.zeros((6, nb_cols, nb_cols), dtype=J.dtype)
    for i in range(nb_cols):
        for j in range(nb_cols):
            J_i = J[:, i]
            J_j = J[:, j]
            if j < i:
                J_grad[0:3, i, j] = torch.cross(J_j[3:6], J_i[0:3])
                J_grad[3:6, i, j] = torch.cross(J_j[3:6], J_i[3:6])
            elif j > i:
                J_grad[0:3, i, j] = -torch.cross(J_j[0:3], J_i[3:6])
            else:
                J_grad[0:3, i, j] = torch.cross(J_i[3:6], J_i[0:3])

    if nb_rows == 2:
        return J_grad[0:2, :, :]
    if nb_rows == 3:
        return torch.cat((J_grad[0:2, :, :], J_grad[-1, :, :]), dim=0)
    else:
        return J_grad


class PlanarRobotKinematics(Module):
    def __init__(self, robot):
        self.robot = robot

        self.fk_torch = PlanarRobotForwardKinematics()
        self.jacobian_torch = PlanarRobotJacobian()

        super().__init__()

    def fk_function(self, q):
        return self.robot.fkine(q).t[..., :2]

    def jacobian_function(self, q):
        return self.robot.jacob0(q)[:2]

    def fk_and_jacobian_function(self, q):
        return self.fk_function(q), self.jacobian_function(q)

    def fkine(self, q):
        return self.fk_torch.apply(q, self.fk_and_jacobian_function)

    def jacob0(self, q):
        return self.jacobian_torch.apply(q, self.jacobian_function)


class PlanarRobotForwardKinematics(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, fk_and_jacobian_function):
        # ctx.save_for_backward(x)
        # ctx.fk_function = fk_function
        # ctx.jacobian_function = jacobian_function
        fk_res, jac_res = fk_and_jacobian_function(x.cpu().detach().numpy())
        ctx.save_for_backward(torch.as_tensor(jac_res, dtype=x.dtype))
        return torch.as_tensor(fk_res, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        # x = ctx.saved_tensors[0]
        # x = ctx.saved_tensors[0]
        # jacobian = torch.as_tensor(ctx.jacobian_function(x.cpu().detach().numpy()), dtype=x.dtype)
        jacobian = ctx.saved_tensors[0]
        # grad output = dloss/dq, we must return dloss/dx = dloss/df * df/dq
        return torch.einsum('i,ij->j', grad_output, jacobian), None, None


class PlanarRobotJacobian(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, jacobian_function):
        # ctx.save_for_backward(x)
        # ctx.fk_function = fk_function
        # ctx.jacobian_function = jacobian_function
        jac_res = jacobian_function(x.cpu().detach().numpy())
        ctx.save_for_backward(torch.as_tensor(jac_res, dtype=x.dtype))
        return torch.as_tensor(jac_res, dtype=x.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        jacobian = ctx.saved_tensors[0]
        djacobian = joint_derivative_Jacobian(J=jacobian)
        # grad output = dloss/dJ, we must return dloss/dx = dloss/dJ * dJ/dq
        return torch.einsum('ij,ijk->k', grad_output, djacobian), None, None


# class ExternalForwardKinematics(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x, fk_function, jacobian_function):
#         ctx.save_for_backward(x)
#         ctx.fk_function = fk_function
#         ctx.jacobian_function = jacobian_function
#         fk_res = fk_function(x.cpu().detach().numpy())
#         return torch.as_tensor(fk_res, dtype=x.dtype)
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         x = ctx.saved_tensors[0]
#         jacobian = torch.as_tensor(ctx.jacobian_function(x.cpu().detach().numpy()), dtype=x.dtype)
#         # grad output = dloss/dmetric, we must return dloss/dx = dloss/dmetric * dmetric/dx
#         return torch.einsum('i,ij->j', grad_output, jacobian), None, None

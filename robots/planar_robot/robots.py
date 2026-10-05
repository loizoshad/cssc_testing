import numpy as np
from pathlib import Path

import roboticstoolbox as rtb
from spatialmath import SE3
import torch
from torch.func import jacrev, vmap, jvp


'''
References

[1] Modern Robotics Mecahnicas, Planning, And Control, KM Lynch, FC Park 2017.
'''

class PlanarManipulator_nDoF:
    def __init__(self, nb_dofs, base: SE3 = SE3(x=0.0, y=0.0, z=0), arm_length=3, ee_joint=False, nb_x=3, is_taskspace=False):
        self.nb_dofs = nb_dofs
        self.ee_joint = ee_joint
        self.nb_x = nb_x
        self.arm_length = arm_length
        self.is_taskspace = is_taskspace
        self.base_pose = base
        ### Create robot
        if not self.ee_joint:
            links_list = [rtb.RevoluteDH(d=0., a=arm_length, alpha=0.) for i in range(nb_dofs)]
        else:
            links_list = [rtb.RevoluteDH(d=0., a=arm_length, alpha=0.) for i in range(nb_dofs-1)]
            links_list.append(rtb.RevoluteDH(d=0., a=0.0, alpha=0.))
        self.robot = rtb.DHRobot(links_list, base=base)

    def fkine(self, q):
        if self.is_taskspace:
            return q
        
        pos = self.robot.fkine(q).t[0:2]
        if not self.ee_joint:
            return pos
        
        # Extract the orientation angle from the transformation matrix
        R = self.robot.fkine(q).R
        theta = np.arctan2(R[1, 0], R[0, 0])     
        return np.hstack((pos, theta))

    def fkine_torch(self, q):
        if self.is_taskspace:
            return q
        
        batch_size, n_links = q.shape
        device = q.device
        dtype = q.dtype

        # Robot base transform (constant), expand to batch
        T = torch.tensor(self.robot.base.A, dtype=dtype, device=device).unsqueeze(0).expand(batch_size, 4, 4)

        # Get lengths of each link
        a_list = [link.a for link in self.robot.links]
        a = torch.tensor(a_list, dtype=dtype, device=device)

        zeros = torch.zeros(batch_size, dtype=dtype, device=device)
        ones = torch.ones(batch_size, dtype=dtype, device=device)

        # Iterate through the transform of each link (batched)
        for i in range(n_links):
            qi = q[:, i]
            c = torch.cos(qi)
            s = torch.sin(qi)
            ai = a[i]

            # Build rows without inplace operations
            row0 = torch.stack([c, -s, zeros, ai * c], dim=1)
            row1 = torch.stack([s,  c, zeros, ai * s], dim=1)
            row2 = torch.stack([zeros, zeros, ones, zeros], dim=1)
            row3 = torch.stack([zeros, zeros, zeros, ones], dim=1)

            Ai = torch.stack([row0, row1, row2, row3], dim=1)

            # FK accumulation
            T = torch.bmm(T, Ai)

        pos = T[:, 0:2, 3]  # get only xy
        if not self.ee_joint:
            return pos

        # planar yaw from rotation
        theta = torch.atan2(T[:, 1, 0], T[:, 0, 0])
        return torch.cat([pos, theta.unsqueeze(1)], dim=1)

    def jacob0(self, q):
        if self.is_taskspace:
            return np.eye(2, 2)
        
        J_full = self.robot.jacob0(q)
        if not self.ee_joint:
            return J_full[0:2, :]
        J_reduced = np.zeros((3, J_full.shape[1])) # The reduced Jacobian only considers planar motion (x, y, theta)
        J_reduced[0:2, :] = J_full[0:2, :]
        J_reduced[2, :] = J_full[5, :]  # The angular velocity around z-axis
        return J_reduced
        
    def jacob0_torch(self, q):
        '''
        This method computes the Jacobian of the forward kinematics using torch.func.jacrev. 
        It is implemented in a way to ensure that the Jacobian is computed independently for each sample in the batch, resulting in a tensor of shape (batch_size, output_dim, input_dim).
        '''
        def fk_single(q):
            return self.fkine_torch(q.unsqueeze(0))[0]
        
        J = vmap(jacrev(fk_single))(q)
        return J

    def fkine_torch_and_jacob0_torch(self, q):
        '''
        returns 
            - x = f(q)
            - J(q)
        '''
        def fk_single(q_single):
            return self.fkine_torch(q_single.unsqueeze(0))[0]
        
        x = vmap(fk_single)(q) # (batch_size, dim(out))
        # Vectorized Jacobian
        J = vmap(jacrev(fk_single))(q) # (batch_size, dim(out), dim(in))

        return x, J

    def get_VX_VTX_batch(self, vf_X, x0, v0):
        '''
        Vf_X is the velocity field evaluated at x0, which gives us the "velocity" of the base point.
        vf_TX = Jac(Vf_X) @ dx/dt.
        '''
        def single_vf_X(x):
            return vf_X(x.unsqueeze(0))[0]
        
        vf_X_eval = vf_X(x0) # shape (B, 2)

        # Constructing Jacobian of vf_X
        jac_vf_X_single = jacrev(single_vf_X)

        # Evaluate Jacobian
        jac_vf_X_eval = vmap(jac_vf_X_single)(x0) # TODO: When linear, this is constant across samples, as it is equivalent to the inf. gen. of the group.
        dx_dt = v0.unsqueeze(-1) # This is the actual velocity vector that we want to transport along the flow.

        # The acceleration field is given by Jac_{vf} @ dx/dt. But should it be dx/dt, or should it be the velocity field evalutated at x?
        # vf_TX_eval = torch.bmm(jac_vf_X_eval, dx_dt).squeeze(-1)
        vf_TX_eval = torch.bmm(jac_vf_X_eval, vf_X_eval.unsqueeze(-1)).squeeze(-1)

        return vf_X_eval, vf_TX_eval

    def fkine_single(self, q):
        return self.fkine_torch(q.unsqueeze(0))[0]

    def jacob0_torch(self, q):
        '''
        This method computes the Jacobian of the forward kinematics using torch.func.jacrev. 
        It is implemented in a way to ensure that the Jacobian is computed independently for each sample in the batch, resulting in a tensor of shape (batch_size, output_dim, input_dim).
        '''
        J = vmap(jacrev(self.fkine_single))(q)
        return J

    def hor_lift_VX(self, vf_X, q):
        if q.dim() == 1:
            q = q.unsqueeze(0)
        
        def V_Q_batch(q):
            x = self.fkine_torch(q)              # (B, ...)
            J = self.jacob0_torch(q)             # (B, m, n)
            J_pinv = torch.pinverse(J)
            # J_pinv = torch.linalg.pinv(J)        # or damped version

            vf = vf_X(x)                         # (B, m)
            V = torch.bmm(J_pinv, vf.unsqueeze(-1)).squeeze(-1)
            return V
        V_Q = V_Q_batch(q)

        return V_Q

    def hor_lift_VX_VTX(self, vf_X, q, vq):
        '''
        This is an attempt to speed up the lifting.

        - Makes use of jvp instead of jacrev
        - removed the vmap and directly batched it.
        '''
        if q.dim() == 1:
            q = q.unsqueeze(0)
        
        def V_Q_batch(q):
            x = self.fkine_torch(q)              # (B, ...)
            J = self.jacob0_torch(q)             # (B, m, n)
            J_pinv = torch.pinverse(J)
            # J_pinv = torch.linalg.pinv(J)        # or damped version

            vf = vf_X(x)                         # (B, m)
            V = torch.bmm(J_pinv, vf.unsqueeze(-1)).squeeze(-1)
            return V

        V_Q = V_Q_batch(q)
        _, V_TQ = jvp(V_Q_batch, (q,), (vq,))

        return V_Q, V_TQ

class TwoArmsPlanarManipulator:
    def __init__(self, nb_dofs_left, nb_dofs_right, 
                 base_left: SE3 = SE3(x=0.0, y=0.0, z=0), base_right: SE3 = SE3(x=0.0, y=0.0, z=0), 
                 arm_length=3, ee_joint=False, nb_x_left=3, nb_x_right=3, is_taskspace=False):
        self.robot_left_arm = PlanarManipulator_nDoF(nb_dofs_left, base_left, arm_length, ee_joint, nb_x_left, is_taskspace=is_taskspace)
        self.robot_right_arm = PlanarManipulator_nDoF(nb_dofs_right, base_right, arm_length, ee_joint, nb_x_right, is_taskspace=is_taskspace)









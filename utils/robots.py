import numpy as np
import time
from pathlib import Path

import roboticstoolbox as rtb
from spatialmath import SE3
import torch
import torch.nn.functional as F
from typing import Optional
from torch.func import jacrev, vmap, jvp

import pytorch_kinematics as pk

# use this to silence the URDF parser's warnings about missing inertial and visual elements in the RBY1 URDF. This does not affect the functionality of the kinematics chains for our purposes.
from pytorch_kinematics.urdf_parser_py.xml_reflection import core as _urdf_xml
_urdf_xml.on_error = lambda message: None

# ── RBY1 FK chains (built once, shared across all RBY1SingleArm instances) ─────
_RBY1_URDF_PATH = Path(__file__).parent.parent / "urdf" / "rby1a" / "model.urdf"
_rby1_device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
_rby1_urdf_data = _RBY1_URDF_PATH.read_text()
_rby1_chain_right = pk.build_serial_chain_from_urdf(_rby1_urdf_data, end_link_name="ee_right", root_link_name="link_torso_5").to(device=_rby1_device)
_rby1_chain_left = pk.build_serial_chain_from_urdf(_rby1_urdf_data, end_link_name="ee_left", root_link_name="link_torso_5").to(device=_rby1_device)

# _rby1_chain_right = pk.build_serial_chain_from_urdf(_rby1_urdf_data, end_link_name="ee_right_tip", root_link_name="link_torso_5").to(device=_rby1_device)
# _rby1_chain_left = pk.build_serial_chain_from_urdf(_rby1_urdf_data, end_link_name="ee_left_tip", root_link_name="link_torso_5").to(device=_rby1_device)

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


_RBY1_TASK_SPACES = {
    'xy':               2,   # x, y position
    'yz':               2,   # y, z position  (letter-writing plane — X barely moves)
    'xy_theta':         3,   # x, y position + yaw angle (z-axis rotation in torso frame)
    'xyz':              3,   # x, y, z position
    'xyz_theta':        4,   # x, y, z position + yaw angle
    'yz_in_xyz_theta':  4,   # x, y, z position + yaw; VF acts only on y,z (x,theta frozen)
    'yz_in_6d':         6,   # [x,y,z,roll,pitch,yaw]; full 6×7 Jacobian; VF moves only y,z
    'full':             12,  # pos(3) + rotation matrix flattened row-major (9)
}


class RBY1SingleArm:
    def __init__(self, arm: str, task_space: str = 'full', ee_joint=False, is_taskspace=False):
        """
        Args:
            arm:        'left' or 'right'.
            task_space: one of 'xy', 'xy_theta', 'xyz', 'xyz_theta', 'yz_in_xyz_theta', 'full'.
                        Controls the dimensionality and content of FK output and Jacobian rows.
                          'xy'              → [x, y]               nb_x = 2
                          'yz'              → [y, z]               nb_x = 2
                          'xy_theta'        → [x, y, yaw]          nb_x = 3
                          'xyz'             → [x, y, z]            nb_x = 3
                          'xyz_theta'       → [x, y, z, yaw]       nb_x = 4
                          'yz_in_xyz_theta' → [x, y, z, yaw]            nb_x = 4
                                             (4-row Jacobian; VF only moves y,z — x,yaw frozen)
                          'yz_in_6d'        → [x, y, z, roll, pitch, yaw]  nb_x = 6
                                             (full 6×7 Jacobian [J_lin; J_ang]; VF only moves
                                              y,z — all translation and rotation DOF frozen)
                          'full'            → [x, y, z, R_flat(9)]        nb_x = 12
                        yaw is the z-axis rotation in the link_torso_5 frame: atan2(R[1,0], R[0,0]).
        """
        assert arm in ('left', 'right'), "arm must be 'left' or 'right'"
        assert task_space in _RBY1_TASK_SPACES, \
            f"task_space must be one of {list(_RBY1_TASK_SPACES)}, got '{task_space}'"
        self.arm = arm
        self.task_space = task_space
        self.arm_length = None  # not meaningful for 3-D arms; kept for interface compat
        self.ee_joint = ee_joint
        self.is_taskspace = is_taskspace
        self.nb_x = _RBY1_TASK_SPACES[task_space] if not is_taskspace else 2
        self.nb_dofs = 7 if not self.is_taskspace else 2
        self._chain = _rby1_chain_left if arm == 'left' else _rby1_chain_right
        self._device = _rby1_device

    # ── numpy wrappers (required by RobotHandler) ────────────────────────────
    def fkine(self, q: np.ndarray) -> np.ndarray:
        if self.is_taskspace:
            return q
        q_t = torch.tensor(q, dtype=torch.float32, device=self._device).unsqueeze(0)
        return self.fkine_torch(q_t)[0].cpu().numpy()

    def jacob0(self, q: np.ndarray) -> np.ndarray:
        if self.is_taskspace:
            return np.eye(self.nb_x, self.nb_dofs)
        q_t = torch.tensor(q, dtype=torch.float32, device=self._device).unsqueeze(0)
        return self.jacob0_torch(q_t)[0].cpu().numpy()

    # ── internal helpers ──────────────────────────────────────────────────────
    def _fk_matrix(self, q: torch.Tensor):
        """q: (B, 7) → homogeneous transform (B, 4, 4)"""
        return self._chain.forward_kinematics(q).get_matrix()

    def _extract_x(self, m: torch.Tensor) -> torch.Tensor:
        """m: (B, 4, 4) → x: (B, nb_x) for the selected task_space"""
        pos = m[:, :3, 3]        # (B, 3)
        R   = m[:, :3, :3]       # (B, 3, 3)
        ts  = self.task_space
        if ts == 'full':
            return torch.cat([pos, R.reshape(-1, 9)], dim=1)
        elif ts == 'xy':
            return pos[:, :2]
        elif ts == 'yz':
            return pos[:, 1:3]                              # y, z only
        elif ts == 'xyz':
            return pos
        elif ts == 'yz_in_6d':
            # Full 6D pose: [x, y, z, roll, pitch, yaw].
            # Only y (idx 1) and z (idx 2) are used by the VF; roll/pitch/yaw complete the
            # representation that pairs with the 6×7 Jacobian [J_lin; J_ang].
            roll  = torch.atan2( R[:, 2, 1],  R[:, 2, 2]).unsqueeze(1)                   # (B,1)
            pitch = torch.atan2(-R[:, 2, 0],
                                 torch.sqrt(R[:, 2, 1]**2 + R[:, 2, 2]**2 + 1e-8)).unsqueeze(1)
            yaw   = torch.atan2( R[:, 1, 0],  R[:, 0, 0]).unsqueeze(1)                   # (B,1)
            return torch.cat([pos, roll, pitch, yaw], dim=1)                              # (B,6)
        else:  # 'xy_theta', 'xyz_theta', or 'yz_in_xyz_theta'
            yaw = torch.atan2(R[:, 1, 0], R[:, 0, 0]).unsqueeze(1)  # (B, 1)
            if ts == 'xy_theta':
                return torch.cat([pos[:, :2], yaw], dim=1)
            else:  # 'xyz_theta' or 'yz_in_xyz_theta' — both return [x, y, z, theta]
                return torch.cat([pos, yaw], dim=1)

    def _build_J_rot(self, J_ang: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
        """
        Builds the (B, 9, 7) Jacobian of the flattened rotation matrix.
            dR/dq_i = skew(J_ang[:, :, i]) @ R  → reshaped to (B, 9)
        Only needed for task_spaces that include rotation information.
        """
        B, _, n = J_ang.shape
        device, dtype = J_ang.device, J_ang.dtype
        wx = J_ang[:, 0, :]   # (B, 7)
        wy = J_ang[:, 1, :]   # (B, 7)
        wz = J_ang[:, 2, :]   # (B, 7)
        zeros = torch.zeros(B, n, device=device, dtype=dtype)
        J_rot = torch.zeros(B, 9, n, device=device, dtype=dtype)
        for i in range(n):
            skew_i = torch.stack([
                torch.stack([zeros[:, i], -wz[:, i],  wy[:, i]], dim=1),
                torch.stack([ wz[:, i],  zeros[:, i], -wx[:, i]], dim=1),
                torch.stack([-wy[:, i],   wx[:, i],  zeros[:, i]], dim=1),
            ], dim=1)                       # (B, 3, 3)
            J_rot[:, :, i] = torch.bmm(skew_i, R).reshape(B, 9)
        return J_rot

    def _extract_J(self, J_lin: torch.Tensor, J_ang: torch.Tensor,
                   R: torch.Tensor) -> torch.Tensor:
        """
        Assembles the (B, nb_x, 7) Jacobian for the selected task_space.

        J_lin: (B, 3, 7) — linear velocity Jacobian (= ∂pos/∂q in torso frame)
        J_ang: (B, 3, 7) — angular velocity Jacobian in torso frame
        R:     (B, 3, 3) — end-effector rotation matrix in torso frame

        Yaw Jacobian (∂yaw/∂q_i):
            yaw = atan2(R[1,0], R[0,0])
            ∂yaw/∂q_i = (R[0,0]·∂R[1,0]/∂q_i − R[1,0]·∂R[0,0]/∂q_i) / (R[0,0]² + R[1,0]²)
        Row-major indices: R[0,0]=0, R[1,0]=3 in the 9-element flat vector.
        """
        ts = self.task_space
        if ts == 'full':
            J_rot = self._build_J_rot(J_ang, R)           # (B, 9, 7)
            return torch.cat([J_lin, J_rot], dim=1)        # (B, 12, 7)
        elif ts == 'xy':
            return J_lin[:, :2, :]                         # (B, 2, 7)
        elif ts == 'yz':
            return J_lin[:, 1:3, :]                        # (B, 2, 7)  — y and z rows
        elif ts == 'xyz':
            return J_lin                                   # (B, 3, 7)
        elif ts == 'yz_in_6d':
            # Full 6×7 geometric Jacobian: [J_lin (3×7); J_ang (3×7)].
            # J_ang maps dq → angular velocity ω in the torso frame.
            # With VF = [0, vy, vz, 0, 0, 0] the pseudoinverse enforces ω≈0,
            # keeping the end-effector orientation fixed during augmentation.
            return torch.cat([J_lin, J_ang], dim=1)        # (B, 6, 7)
        else:  # 'xy_theta', 'xyz_theta', or 'yz_in_xyz_theta' — need the yaw row
            J_rot = self._build_J_rot(J_ang, R)           # (B, 9, 7)
            R00 = R[:, 0, 0].unsqueeze(1)                  # (B, 1)
            R10 = R[:, 1, 0].unsqueeze(1)                  # (B, 1)
            denom = R00 ** 2 + R10 ** 2                    # (B, 1)
            # ∂yaw/∂q = (R00 · J_rot[row=3] − R10 · J_rot[row=0]) / denom
            J_yaw = (R00 * J_rot[:, 3, :] - R10 * J_rot[:, 0, :]) / denom  # (B, 7)
            J_yaw = J_yaw.unsqueeze(1)                     # (B, 1, 7)
            if ts == 'xy_theta':
                return torch.cat([J_lin[:, :2, :], J_yaw], dim=1)   # (B, 3, 7)
            else:  # 'xyz_theta' or 'yz_in_xyz_theta' — full [x,y,z] + yaw
                return torch.cat([J_lin, J_yaw], dim=1)             # (B, 4, 7)

    # ── torch implementations ─────────────────────────────────────────────────
    def fkine_torch(self, q: torch.Tensor) -> torch.Tensor:
        if self.is_taskspace:
            return q
        """q: (B, 7) → x: (B, nb_x)"""
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(self._device)
        m = self._fk_matrix(q)
        return self._extract_x(m)

    def fkine_single(self, q: torch.Tensor) -> torch.Tensor:
        """q: (7,) → x: (nb_x,)"""
        return self.fkine_torch(q.unsqueeze(0))[0]

    def jacob0_torch(self, q: torch.Tensor) -> torch.Tensor:
        """
        q: (B, 7) → J: (B, nb_x, 7)

        Uses the geometric Jacobian from pytorch_kinematics and converts it
        analytically to match the selected task_space output representation.
        """
        if self.is_taskspace:
            return torch.eye(2, 2, device=self._device)
        if q.dim() == 1:
            q = q.unsqueeze(0)
        q = q.to(self._device)
        m      = self._fk_matrix(q)
        J_geom = self._chain.jacobian(q)        # (B, 6, 7)
        R      = m[:, :3, :3]                   # (B, 3, 3)
        J_lin  = J_geom[:, :3, :]               # (B, 3, 7)
        J_ang  = J_geom[:, 3:, :]               # (B, 3, 7)
        return self._extract_J(J_lin, J_ang, R)

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


class TwoArmsRBY1:
    def __init__(self, task_space: str = 'full', ee_joint=False, is_taskspace=False):
        self.robot_left_arm  = RBY1SingleArm('left',  task_space=task_space, ee_joint=ee_joint, is_taskspace=is_taskspace)
        self.robot_right_arm = RBY1SingleArm('right', task_space=task_space, ee_joint=ee_joint, is_taskspace=is_taskspace)











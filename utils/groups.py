'''
This assumes we are acting on symmetry conditioned input vectors
'''

import numpy as np
from abc import abstractmethod
from dataclasses import dataclass
import torch
from utils.vector_fields import so2_R2_vf, scaling2_R2_vf

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

############################################################################################################
# Helpers: integration along infinitesimal generators
############################################################################################################

def _n_steps(magnitude: torch.Tensor, dt: float) -> int:
    """Number of Euler steps needed to integrate magnitude with step size dt."""
    if torch.abs(magnitude) < 1e-6:
        return 1
    return int(torch.ceil((torch.abs(magnitude) + 1e-6) / dt).item())

def _signed_dt(magnitude: torch.Tensor, dt: float) -> float:
    return torch.sign(magnitude).item() * dt

def act_on_Q(robot, q0, dt, theta, VX, n_steps: int = None):
    if n_steps is None:
        n_steps = _n_steps(theta, dt)
    ddt = _signed_dt(theta, dt)
    q_traj = torch.zeros((n_steps, *q0.shape), dtype=q0.dtype, device=q0.device)
    q_traj[0] = q0
    for i in range(1, n_steps):
        q_traj[i] = q_traj[i - 1] + robot.hor_lift_VX(VX, q_traj[i - 1]) * ddt
    return q_traj

def act_on_TQ(robot, theta, q0, vq0, dt, VX, return_q_traj=False, n_steps: int = None):
    if n_steps is None:
        n_steps = _n_steps(theta, dt)
    ddt = _signed_dt(theta, dt)
    q_traj  = torch.zeros((n_steps, *q0.shape),  dtype=q0.dtype,  device=q0.device)
    vq_traj = torch.zeros((n_steps, *vq0.shape), dtype=vq0.dtype, device=vq0.device)
    q_traj[0], vq_traj[0] = q0, vq0
    for i in range(1, n_steps):
        vf_Q, vf_TQ = robot.hor_lift_VX_VTX(VX, q_traj[i - 1], vq_traj[i - 1])
        q_traj[i]  = q_traj[i - 1]  + vf_Q  * ddt
        vq_traj[i] = vq_traj[i - 1] + vf_TQ * ddt
    return (q_traj, vq_traj) if return_q_traj else vq_traj

def act_on_Q_dual_arm(robot, g, q0, dt, VX, n_steps: int = None):
    nb_q = robot.nb_dofs
    q_l = act_on_Q(robot=robot.robot_left,  theta=g, q0=q0[:, :nb_q//2], dt=dt, VX=VX, n_steps=n_steps)
    q_r = act_on_Q(robot=robot.robot_right, theta=g, q0=q0[:, nb_q//2:], dt=dt, VX=VX, n_steps=n_steps)
    return torch.cat([q_l, q_r], dim=-1)

def act_on_TQ_dual_arm(robot, g, q0, vq0, dt, VX, return_q_traj=False, n_steps: int = None):
    nb_q = robot.nb_dofs
    out_l = act_on_TQ(robot=robot.robot_left,  theta=g, q0=q0[:, :nb_q//2], vq0=vq0[:, :nb_q//2], dt=dt, VX=VX, return_q_traj=True, n_steps=n_steps)
    out_r = act_on_TQ(robot=robot.robot_right, theta=g, q0=q0[:, nb_q//2:], vq0=vq0[:, nb_q//2:], dt=dt, VX=VX, return_q_traj=True, n_steps=n_steps)
    q_t  = torch.cat([out_l[0], out_r[0]], dim=-1)
    vq_t = torch.cat([out_l[1], out_r[1]], dim=-1)
    return (q_t, vq_t) if return_q_traj else vq_t

def evolve_X_v1(x0: torch.Tensor, magnitude: torch.Tensor, dt: float, mode: str) -> torch.Tensor:
    """
    Integrate the task-space trajectory under a linear group action.

    Parameters
    ----------
    x0        : (batch, nb_x)  initial task-space positions
    magnitude : scalar tensor  rotation angle (rad) or scale - 1
    dt        : step size
    mode      : 'rotation' | 'scaling'

    Returns
    -------
    x_traj : (n_steps, batch, nb_x)
    """
    n   = max(2, _n_steps(magnitude, dt))
    ddt = _signed_dt(magnitude, dt)

    if mode == 'rotation':
        c, s = torch.cos(torch.tensor(ddt)), torch.sin(torch.tensor(ddt))
        rot  = torch.tensor([[c, -s], [s, c]], dtype=x0.dtype, device=x0.device)
        A    = torch.block_diag(rot, rot)          # (nb_x, nb_x)
    elif mode == 'scaling':
        A = torch.eye(x0.shape[-1], dtype=x0.dtype, device=x0.device) * (1 + ddt)
    else:
        raise ValueError(f"Unknown mode: {mode}")

    x_traj = torch.zeros((n, *x0.shape), dtype=x0.dtype, device=x0.device)
    x_traj[0] = x0
    for i in range(1, n):
        x_traj[i] = (A @ x_traj[i - 1].unsqueeze(-1)).squeeze(-1)
    return x_traj

def evolve_X(x0: torch.Tensor, magnitude: torch.Tensor, dt: float, mode: str, n_steps: int = None) -> torch.Tensor:
    if mode == 'reflection':
        n_steps = 2
    elif n_steps is None:
        n_steps = max(2, _n_steps(magnitude, dt))
    ddt = _signed_dt(magnitude, dt)

    # Initialise every step to x0; only the component relevant to the mode is then overwritten.
    x_traj = x0.unsqueeze(0).expand(n_steps, *x0.shape).clone()

    # steps: (n_steps, 1, ..., 1) — broadcasts over all x0 dims except the last (nb_x)
    n_extra = x0.dim() - 1
    steps = torch.arange(n_steps, dtype=x0.dtype, device=x0.device).view(n_steps, *([1] * n_extra))

    # Linearly interpolate the conditioning from x0 to x0+magnitude so that
    # the final step (index n_steps-1) lands exactly at the target value.
    # Using steps/max(n_steps-1,1)*magnitude is more accurate than steps*ddt
    # (which overshoots by up to one dt step).
    frac = steps / max(n_steps - 1, 1)   # shape (n_steps, 1, ..., 1)

    if mode == 'rotation':
        x_traj[..., 0] = x0[..., 0] + frac * magnitude
    elif mode == 'scaling':
        x_traj[..., 1] = x0[..., 1] + frac * magnitude
    elif mode == 'reflection':
        # step 0: identity (already x0); step 1: flip reflection component
        x_traj[1, ..., 2] = x0[..., 2] * magnitude   # magnitude is +1 or -1

    return x_traj

############################################################################################################
# Groups
############################################################################################################

class Group:
    lie_algebra        = NotImplemented
    discrete_generators = NotImplemented
    z_scale    = None
    is_orthogonal  = None
    is_permutation = None
    d = NotImplemented

    def __init__(self, *args, **kwargs):
        self._name = kwargs.pop('name', None)
        if self.d is NotImplemented:
            for attr in [self.lie_algebra, self.discrete_generators]:
                if attr is not NotImplemented and len(attr):
                    self.d = attr[0].shape[-1]
                    break
        if self.lie_algebra        is NotImplemented: self.lie_algebra        = np.zeros((0, self.d, self.d))
        if self.discrete_generators is NotImplemented: self.discrete_generators = np.zeros((0, self.d, self.d))
        self.args = args

    def __str__(self):  return self._name if self._name is not None else self.__class__.__name__
    def __repr__(self): return self.__str__()


class SO2(Group):
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.zeros((1, n, n))
        self.lie_algebra[0] = np.array([[0, -1], [1, 0]])
        super().__init__(**kwargs)



class Scaling2(Group):
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.eye(n)[None]
        super().__init__(n, **kwargs)


class SO2Scaling2Group(Group):
    def __init__(self, **kwargs):
        self.lie_algebra = np.concatenate([SO2().lie_algebra, Scaling2().lie_algebra], axis=0)
        super().__init__(**kwargs)


class C2(Group):
    def __init__(self, permutator: np.ndarray = None, **kwargs):
        n = permutator.shape[0]
        perm_diag = np.diag(permutator)
        Z = np.zeros((n, n))
        self.discrete_generators = np.array([np.block([[Z, perm_diag], [perm_diag, Z]])])
        assert np.allclose(self.discrete_generators[0] @ self.discrete_generators[0], np.eye(2 * n)), \
            "Invalid permutator: gs^2 ≠ e"
        super().__init__(**kwargs)


class C2SO2Scaling2Group(Group):
    def __init__(self, **kwargs):
        Z2 = np.zeros((2, 2))
        so2_lie  = np.array([np.block([[g, Z2], [Z2, g]]) for g in SO2().lie_algebra])
        sc2_lie  = np.array([np.block([[g, Z2], [Z2, g]]) for g in Scaling2().lie_algebra])
        self.lie_algebra = np.concatenate([so2_lie, sc2_lie], axis=0)
        self.discrete_generators = C2(permutator=np.array([1, 1])).discrete_generators
        super().__init__(**kwargs)


class OneParameterGroup(Group):
    def __init__(self, name: str, **kwargs):
        self.lie_algebra = np.eye(1)[None]
        self.name = name
        super().__init__(**kwargs)

    def __str__(self): return self._name if self._name is not None else self.name

############################################################################################################
# Base Representation
############################################################################################################

class CustomBaseRep:
    def __init__(self, G=None):
        self.G = G
        if G is not None:
            self.is_permutation = G.is_permutation
        super().__init__()
        self._base_attrs = {k for k in self.__dict__ if k not in ('_size', 'is_permutation', 'is_orthogonal')}

    def __str__(self):  return self.__class__.__name__
    def __call__(self, G): return self.__class__(G)
    def __hash__(self):
        return hash((type(self), tuple((k, v) for k, v in self.__dict__.items() if k in self._base_attrs)))


class TrivialRep(CustomBaseRep):
    def __init__(self, G=None, dim=2):
        super().__init__(G)
        self._dim = dim

    def size(self):
        self._size = self._dim
        return self._size

    def rho(self, M):
        return np.eye(self.size())


class CustomBaseRepHorLift(CustomBaseRep):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, VX=None):
        self.robot       = robot
        self.nb_q        = robot.nb_dofs
        self.nb_x        = robot.nb_x
        self.is_vf_constant = is_vf_constant
        self.is_linear   = False
        self.dt          = dt
        if VX is not None:
            self.VX = VX
        super().__init__(G)

    @abstractmethod
    def size(self): pass

    @abstractmethod
    def act_on(self, g, **kwargs): pass

############################################################################################################
# C2 Representations
############################################################################################################

class C2DualArmConfigTaskRepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, is_taskspace=False):
        self.is_taskspace = is_taskspace
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def _decompose(self, g):
        has_refl = torch.isclose(g[0, 0], torch.tensor(0., device=g.device))
        perm_Q = torch.tensor([1., -1.], device=g.device) if self.is_taskspace \
                 else -torch.ones(self.nb_q // 2, device=g.device)
        if has_refl:
            refl_Q = torch.zeros(self.nb_q, self.nb_q, device=g.device)
            refl_Q[:self.nb_q//2, self.nb_q//2:] = torch.diag(perm_Q)
            refl_Q[self.nb_q//2:, :self.nb_q//2] = torch.diag(perm_Q)
        else:
            refl_Q = torch.eye(self.nb_q, device=g.device)
        return refl_Q, has_refl

    def _apply_refl_to_conditioning(self, x, has_refl):
        if not has_refl:
            return x
        x = x.clone()
        if x.shape[-1] == 3:
            # Symmetry conditioning: flip only the reflection component (index 2)
            x[..., 2] = -x[..., 2]
        else:
            # Task-space / goal conditioning: block-swap with sign flip
            n    = x.shape[-1]
            perm = torch.tensor([1., -1.], device=x.device)
            R    = torch.zeros(n, n, device=x.device)
            R[:n//2, n//2:] = torch.diag(perm)
            R[n//2:, :n//2] = torch.diag(perm)
            x = (R @ x.unsqueeze(-1)).squeeze(-1)
        return x

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        refl_Q, has_refl = self._decompose(g)
        q0 = (refl_Q @ q0.unsqueeze(-1)).squeeze(-1)
        x0 = self._apply_refl_to_conditioning(x0, has_refl)
        return torch.cat([q0, x0], dim=-1)


class C2DualArmConfigTaskRepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, is_taskspace=False):
        self.is_taskspace = is_taskspace
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def _decompose(self, g):
        has_refl = torch.isclose(g[0, 0], torch.tensor(0., device=g.device))
        perm_Q = torch.tensor([1., -1.], device=g.device) if self.is_taskspace \
                 else -torch.ones(self.nb_q // 2, device=g.device)
        if has_refl:
            refl_Q = torch.zeros(self.nb_q, self.nb_q, device=g.device)
            refl_Q[:self.nb_q//2, self.nb_q//2:] = torch.diag(perm_Q)
            refl_Q[self.nb_q//2:, :self.nb_q//2] = torch.diag(perm_Q)
        else:
            refl_Q = torch.eye(self.nb_q, device=g.device)
        return refl_Q

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        refl_Q = self._decompose(g)
        return (refl_Q @ v0.unsqueeze(-1)).squeeze(-1)

############################################################################################################
# SO2 Representations
############################################################################################################

class SO2DualArmConfigTaskRepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05):
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX(self, x, shift=None):
        return so2_R2_vf(x=x, shift=shift)

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta = torch.atan2(g[1, 0], g[0, 0])
        n_steps = max(2, _n_steps(theta, self.dt))

        q_t = act_on_Q_dual_arm(robot=self.robot, g=theta, q0=q0, dt=self.dt, VX=self.VX, n_steps=n_steps)  # (n, batch, nb_q)
        x_t = evolve_X(x0=x0, magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_steps)                # (n, batch, nb_x)

        if full_traj:
            return torch.cat([q_t, x_t], dim=-1)   # (n, batch, nb_q+nb_x)
        return torch.cat([q_t[-1], x_t[-1]], dim=-1)

class SO2DualArmConfigTaskRepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05):
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX(self, x, shift=None):
        return so2_R2_vf(x=x, shift=shift)

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta = torch.atan2(g[1, 0], g[0, 0])
        n_steps = max(2, _n_steps(theta, self.dt))
        v_t = act_on_TQ_dual_arm(robot=self.robot, g=theta, q0=q0, vq0=v0, dt=self.dt, VX=self.VX, n_steps=n_steps)
        return v_t if full_traj else v_t[-1]

############################################################################################################
# Scaling2 Representations
############################################################################################################

class Scaling2DualArmConfigTaskRepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05):
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX(self, x, shift=None):
        return scaling2_R2_vf(x=x, shift=shift)

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        scale = g[0, 0] - 1
        n_steps = max(2, _n_steps(scale, self.dt))

        q_t = act_on_Q_dual_arm(robot=self.robot, g=scale, q0=q0, dt=self.dt, VX=self.VX, n_steps=n_steps)
        x_t = evolve_X(x0=x0, magnitude=scale, dt=self.dt, mode='scaling', n_steps=n_steps)

        if full_traj:
            return torch.cat([q_t, x_t], dim=-1)
        return torch.cat([q_t[-1], x_t[-1]], dim=-1)


class Scaling2DualArmConfigTaskRepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05):
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX(self, x, shift=None):
        return scaling2_R2_vf(x=x, shift=shift)

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        scale = g[0, 0] - 1
        n_steps = max(2, _n_steps(scale, self.dt))
        v_t = act_on_TQ_dual_arm(robot=self.robot, g=scale, q0=q0, vq0=v0, dt=self.dt, VX=self.VX, n_steps=n_steps)
        return v_t if full_traj else v_t[-1]

############################################################################################################
# SO2 x Scaling2 Representations  —  the key rewrite
############################################################################################################

class SO2Scaling2DualArmConfigTaskRepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05):
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX_so2(self, x, shift=None):     return so2_R2_vf(x=x, shift=shift)
    def VX_scaling2(self, x, shift=None): return scaling2_R2_vf(x=x, shift=shift)

    def _decompose(self, g):
        theta = torch.atan2(g[1, 0], g[0, 0])
        scale = torch.sqrt((_det2x2(g))) - 1
        return theta, scale

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta, scale = self._decompose(g)

        # Single source of truth for step counts
        n_s = max(2, _n_steps(scale, self.dt)) if torch.abs(scale) >= 1e-6 else 1
        n_r = max(2, _n_steps(theta, self.dt)) if torch.abs(theta) >= 1e-6 else 1

        q_scaled = act_on_Q_dual_arm(robot=self.robot, g=scale, q0=q0, dt=self.dt, VX=self.VX_scaling2, n_steps=n_s)
        x_scaled = evolve_X(x0=x0, magnitude=scale, dt=self.dt, mode='scaling', n_steps=n_s)

        if not full_traj:
            q_out = act_on_Q_dual_arm(robot=self.robot, g=theta, q0=q_scaled[-1], dt=self.dt, VX=self.VX_so2, n_steps=n_r)[-1]
            x_out = evolve_X(x0=x_scaled[-1], magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_r)[-1]
            return torch.cat([q_out, x_out], dim=-1)

        q_out_list, x_out_list = [], []
        for i in range(n_s):
            q_out_list.append(act_on_Q_dual_arm(robot=self.robot, g=theta, q0=q_scaled[i], dt=self.dt, VX=self.VX_so2, n_steps=n_r))
            x_out_list.append(evolve_X(x0=x_scaled[i], magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_r))

        return torch.cat([torch.cat([q, x], dim=-1) for q, x in zip(q_out_list, x_out_list)], dim=0)   # (n_s * n_r, T, nb_q+nb_x)

class SO2Scaling2DualArmConfigTaskRepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05):
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX_so2(self, x, shift=None):     return so2_R2_vf(x=x, shift=shift)
    def VX_scaling2(self, x, shift=None): return scaling2_R2_vf(x=x, shift=shift)

    def _decompose(self, g):
        theta = torch.atan2(g[1, 0], g[0, 0])
        scale = torch.sqrt((_det2x2(g))) - 1
        return theta, scale

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta, scale = self._decompose(g)

        n_s = max(2, _n_steps(scale, self.dt)) if torch.abs(scale) >= 1e-6 else 1
        n_r = max(2, _n_steps(theta, self.dt)) if torch.abs(theta) >= 1e-6 else 1

        q_scaled, v_scaled = act_on_TQ_dual_arm(robot=self.robot, g=scale, q0=q0, vq0=v0, dt=self.dt, VX=self.VX_scaling2, return_q_traj=True, n_steps=n_s)

        if not full_traj:
            return act_on_TQ_dual_arm(robot=self.robot, g=theta, q0=q_scaled[-1], vq0=v_scaled[-1], dt=self.dt, VX=self.VX_so2, n_steps=n_r)[-1]

        return torch.cat([act_on_TQ_dual_arm(robot=self.robot, g=theta, q0=q_scaled[i], vq0=v_scaled[i], dt=self.dt, VX=self.VX_so2, n_steps=n_r) for i in range(n_s)], dim=0)   # (n_s * n_r, T, nb_q)

############################################################################################################
# C2 x SO2 x Scaling2 Representations
############################################################################################################

class C2SO2Scaling2DualArmConfigTaskRepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, is_taskspace=False):
        self.is_taskspace = is_taskspace
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX_so2(self, x, shift=None):     return so2_R2_vf(x=x, shift=shift)
    def VX_scaling2(self, x, shift=None): return scaling2_R2_vf(x=x, shift=shift)

    def _decompose(self, g):
        has_refl = torch.isclose(g[0, 0], torch.tensor(0., device=g.device))
        cont     = g[2:, :2] if has_refl else g[:2, :2]
        theta    = torch.atan2(cont[1, 0], cont[0, 0])
        scale    = torch.sqrt((_det2x2(cont))) - 1
        perm_Q   = torch.tensor([1., -1.], device=g.device) if self.is_taskspace \
                   else -torch.ones(self.nb_q // 2, device=g.device)
        if has_refl:
            refl_Q = torch.zeros(self.nb_q, self.nb_q, device=g.device)
            refl_Q[:self.nb_q//2, self.nb_q//2:] = torch.diag(perm_Q)
            refl_Q[self.nb_q//2:, :self.nb_q//2] = torch.diag(perm_Q)
        else:
            refl_Q = torch.eye(self.nb_q, device=g.device)
        return theta, scale, refl_Q, has_refl

    def _apply_refl_to_conditioning(self, x, has_refl):
        """Update the conditioning tensor for a C2 reflection.
        Symmetry conditioning (nb_x=3): flip the reflection component at index 2.
        Task-space / goal conditioning (nb_x=4): apply the block-swap with sign flip.
        """
        if not has_refl:
            return x
        x = x.clone()
        if x.shape[-1] == 3:
            x[..., 2] = -x[..., 2]
        else:
            n    = x.shape[-1]
            perm = torch.tensor([1., -1.], device=x.device)
            R    = torch.zeros(n, n, device=x.device)
            R[:n//2, n//2:] = torch.diag(perm)
            R[n//2:, :n//2] = torch.diag(perm)
            x = (R @ x.unsqueeze(-1)).squeeze(-1)
        return x

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta, scale, refl_Q, has_refl = self._decompose(g)

        n_s = max(2, _n_steps(scale, self.dt)) if torch.abs(scale) >= 1e-6 else 1
        n_r = max(2, _n_steps(theta, self.dt)) if torch.abs(theta) >= 1e-6 else 1

        q_scaled = act_on_Q_dual_arm(robot=self.robot, g=scale, q0=q0, dt=self.dt, VX=self.VX_scaling2, n_steps=n_s)
        x_scaled = evolve_X(x0=x0, magnitude=scale, dt=self.dt, mode='scaling', n_steps=n_s)

        if not full_traj:
            q_out = act_on_Q_dual_arm(robot=self.robot, g=theta, q0=q_scaled[-1], dt=self.dt, VX=self.VX_so2, n_steps=n_r)[-1]
            x_out = evolve_X(x0=x_scaled[-1], magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_r)[-1]
            q_out = (refl_Q @ q_out.unsqueeze(-1)).squeeze(-1)
            x_out = self._apply_refl_to_conditioning(x_out, has_refl)
            return torch.cat([q_out, x_out], dim=-1)

        q_out_list, x_out_list = [], []
        for i in range(n_s):
            q_out_list.append(act_on_Q_dual_arm(robot=self.robot, g=theta, q0=q_scaled[i], dt=self.dt, VX=self.VX_so2, n_steps=n_r))
            x_out_list.append(evolve_X(x0=x_scaled[i], magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_r))

        q_traj = torch.cat(q_out_list, dim=0)   # (n_s*n_r, T, nb_q)
        x_traj = torch.cat(x_out_list, dim=0)   # (n_s*n_r, T, nb_x)

        # Apply reflection uniformly to every step in the trajectory
        q_traj = (refl_Q @ q_traj.unsqueeze(-1)).squeeze(-1)
        x_traj = self._apply_refl_to_conditioning(x_traj, has_refl)

        return torch.cat([q_traj, x_traj], dim=-1)   # (n_s*n_r, T, nb_q+nb_x)


class C2SO2Scaling2DualArmConfigTaskRepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, is_taskspace=False):
        self.is_taskspace = is_taskspace
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX_so2(self, x, shift=None):     return so2_R2_vf(x=x, shift=shift)
    def VX_scaling2(self, x, shift=None): return scaling2_R2_vf(x=x, shift=shift)

    def _decompose(self, g):
        has_refl = torch.isclose(g[0, 0], torch.tensor(0., device=g.device))
        cont     = g[2:, :2] if has_refl else g[:2, :2]
        theta    = torch.atan2(cont[1, 0], cont[0, 0])
        scale    = torch.sqrt((_det2x2(cont))) - 1
        perm_Q   = torch.tensor([1., -1.], device=g.device) if self.is_taskspace \
                   else -torch.ones(self.nb_q // 2, device=g.device)
        if has_refl:
            refl_Q = torch.zeros(self.nb_q, self.nb_q, device=g.device)
            refl_Q[:self.nb_q//2, self.nb_q//2:] = torch.diag(perm_Q)
            refl_Q[self.nb_q//2:, :self.nb_q//2] = torch.diag(perm_Q)
        else:
            refl_Q = torch.eye(self.nb_q, device=g.device)
        return theta, scale, refl_Q

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta, scale, refl_Q = self._decompose(g)

        n_s = max(2, _n_steps(scale, self.dt)) if torch.abs(scale) >= 1e-6 else 1
        n_r = max(2, _n_steps(theta, self.dt)) if torch.abs(theta) >= 1e-6 else 1

        q_scaled, v_scaled = act_on_TQ_dual_arm(
            robot=self.robot, g=scale, q0=q0, vq0=v0,
            dt=self.dt, VX=self.VX_scaling2, return_q_traj=True, n_steps=n_s
        )

        if not full_traj:
            v_out = act_on_TQ_dual_arm(robot=self.robot, g=theta, q0=q_scaled[-1], vq0=v_scaled[-1], dt=self.dt, VX=self.VX_so2, n_steps=n_r)[-1]
            return (refl_Q @ v_out.unsqueeze(-1)).squeeze(-1)

        v_out_list = [
            act_on_TQ_dual_arm(robot=self.robot, g=theta, q0=q_scaled[i], vq0=v_scaled[i], dt=self.dt, VX=self.VX_so2, n_steps=n_r)
            for i in range(n_s)
        ]
        v_traj = torch.cat(v_out_list, dim=0)   # (n_s*n_r, T, nb_q)
        return (refl_Q @ v_traj.unsqueeze(-1)).squeeze(-1)
    








############################################################################################################
# RB-Y1 Symmetries
############################################################################################################
class C2RBY1(Group):
    def __init__(self, permutator: np.ndarray = None, **kwargs):
        n = permutator.shape[0]
        perm_diag = np.diag(permutator)
        Z = np.zeros((n, n))
        self.discrete_generators = np.array([np.block([[Z, perm_diag], [perm_diag, Z]])])
        assert np.allclose(self.discrete_generators[0] @ self.discrete_generators[0], np.eye(2 * n)), \
            "Invalid permutator: gs^2 ≠ e"
        super().__init__(**kwargs)



class C2RBY1RepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, is_taskspace=False):
        self.is_taskspace = is_taskspace
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def _decompose(self, g):
        has_refl = torch.isclose(g[0, 0], torch.tensor(0., device=g.device))
        if self.is_taskspace:
            perm_Q = torch.tensor([1., -1.], device=g.device)
        else:
            perm_Q = torch.tensor([1., -1., -1., 1., -1., 1., -1.], device=g.device)
        
        if has_refl:
            refl_Q = torch.zeros(self.nb_q, self.nb_q, device=g.device)
            refl_Q[:self.nb_q//2, self.nb_q//2:] = torch.diag(perm_Q)
            refl_Q[self.nb_q//2:, :self.nb_q//2] = torch.diag(perm_Q)
        else:
            refl_Q = torch.eye(self.nb_q, device=g.device)
        return refl_Q, has_refl

    def _apply_refl_to_conditioning(self, x, has_refl):
        if not has_refl:
            return x
        x = x.clone()
        if x.shape[-1] == 3:
            # Symmetry conditioning: flip only the reflection component (index 2)
            x[..., 2] = -x[..., 2]
        else:
            # Task-space / goal conditioning: block-swap with sign flip
            n    = x.shape[-1]
            perm = torch.tensor([1., -1.], device=x.device)
            R    = torch.zeros(n, n, device=x.device)
            R[:n//2, n//2:] = torch.diag(perm)
            R[n//2:, :n//2] = torch.diag(perm)
            x = (R @ x.unsqueeze(-1)).squeeze(-1)
        return x

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        refl_Q, has_refl = self._decompose(g)
        q0 = (refl_Q @ q0.unsqueeze(-1)).squeeze(-1)
        x0 = self._apply_refl_to_conditioning(x0, has_refl)
        return torch.cat([q0, x0], dim=-1)

class C2RBY1RepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05, is_taskspace=False):
        self.is_taskspace = is_taskspace
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def _decompose(self, g):
        has_refl = torch.isclose(g[0, 0], torch.tensor(0., device=g.device))
        if self.is_taskspace:
            perm_Q = torch.tensor([1., -1.], device=g.device)
        else:
            perm_Q = torch.tensor([1., -1., -1., 1., -1., 1., -1.], device=g.device)

        if has_refl:
            refl_Q = torch.zeros(self.nb_q, self.nb_q, device=g.device)
            refl_Q[:self.nb_q//2, self.nb_q//2:] = torch.diag(perm_Q)
            refl_Q[self.nb_q//2:, :self.nb_q//2] = torch.diag(perm_Q)
        else:
            refl_Q = torch.eye(self.nb_q, device=g.device)
        return refl_Q

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        refl_Q = self._decompose(g)
        return (refl_Q @ v0.unsqueeze(-1)).squeeze(-1)

class C2SO2Scaling2RBY1(Group):
    def __init__(self, **kwargs):
        Z2 = np.zeros((2, 2))
        so2_lie  = np.array([np.block([[g, Z2], [Z2, g]]) for g in SO2().lie_algebra])
        sc2_lie  = np.array([np.block([[g, Z2], [Z2, g]]) for g in Scaling2().lie_algebra])
        self.lie_algebra = np.concatenate([so2_lie, sc2_lie], axis=0)
        self.discrete_generators = C2(permutator=np.array([1, 1])).discrete_generators
        super().__init__(**kwargs)


class SO2Scaling2RBY1Letters(Group):
    """SO2 × Scaling2 symmetry in the Y–Z plane, no morphological C2.

    Used for the letters dataset (Case 4: rotation + uniform scaling without
    arm swap).  2×2 Lie algebra; CompositeRepIn decodes group elements.
    """
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.zeros((2, n, n))
        self.lie_algebra[0] = np.array([[0., -1.], [1., 0.]])   # SO2 generator
        self.lie_algebra[1] = np.eye(n)                          # Scaling2 generator
        self.discrete_generators = np.zeros((0, n, n))           # no discrete part
        super().__init__(**kwargs)


############################################################################################################
# RBY1 SO2 — task-space vector fields and rep classes
############################################################################################################

class SO2RBY1(Group):
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.zeros((1, n, n))
        self.lie_algebra[0] = np.array([[0, -1], [1, 0]])
        super().__init__(**kwargs)

# One vector field per supported task-space.  All follow the same pattern:
#   SO2 generator in the XY plane: (−y, x, 0..., [1 if yaw present])
# z is invariant under rotation about the z-axis; yaw co-rotates at unit rate.

def _so2_xyz_vf(x, shift=None):
    if shift is not None:
        x = x.clone()
        x[..., :2] = x[..., :2] - shift[..., :2]
    vf = torch.zeros_like(x)
    vf[..., 0] = -x[..., 1]
    vf[..., 1] =  x[..., 0]
    return vf

def _so2_xy_theta_vf(x, shift=None):
    if shift is not None:
        x = x.clone()
        x[..., :2] = x[..., :2] - shift[..., :2]
    vf = torch.zeros_like(x)
    vf[..., 0] = -x[..., 1]
    vf[..., 1] =  x[..., 0]
    vf[..., 2] =  1.0
    return vf

def _so2_xyz_theta_vf(x, shift=None):
    if shift is not None:
        x = x.clone()
        x[..., :2] = x[..., :2] - shift[..., :2]
    vf = torch.zeros_like(x)
    vf[..., 0] = -x[..., 1]
    vf[..., 1] =  x[..., 0]
    vf[..., 3] =  1.0
    return vf

def _so2_yz_in_xyz_theta_vf(x, shift=None):
    """SO2 rotation in the Y–Z plane; X (dim 0) and theta (dim 3) have zero velocity.

    x layout: [x, y, z, theta]
    shift (if given): (B, 2) — [c_y, c_z] — rotation center in the Y–Z subspace.
    Infinitesimal generator: dy/dt = -(z-c_z), dz/dt = (y-c_y).
    """
    vf = torch.zeros_like(x)
    y = x[..., 1]
    z = x[..., 2]
    if shift is not None:
        y = y - shift[..., 0]   # shift[..., 0] = c_y
        z = z - shift[..., 1]   # shift[..., 1] = c_z
    vf[..., 1] = -z
    vf[..., 2] =  y
    return vf


def _so2_yz_in_6d_vf(x, shift=None):
    """SO2 rotation in the Y–Z plane; ALL other DOF (x, ωx, ωy, ωz) have zero velocity.

    x layout: [x, y, z, roll, pitch, yaw]  (indices 0–5)
    shift (if given): (B, 2) — [c_y, c_z] — rotation centre in Y–Z subspace.
    Pairs with the full 6×7 geometric Jacobian [J_lin; J_ang], which forces the
    end-effector to keep its orientation completely fixed during augmentation.
    """
    vf = torch.zeros_like(x)
    y = x[..., 1]
    z = x[..., 2]
    if shift is not None:
        y = y - shift[..., 0]
        z = z - shift[..., 1]
    vf[..., 1] = -z
    vf[..., 2] =  y
    return vf


_SO2_VF = {
    'xy':               so2_R2_vf,
    'yz':               so2_R2_vf,    # same 2-D VF; x_data stores [y,z] so dim0/dim1 already correct
    'xyz':              _so2_xyz_vf,
    'xy_theta':         _so2_xy_theta_vf,
    'xyz_theta':        _so2_xyz_theta_vf,
    'yz_in_xyz_theta':  _so2_yz_in_xyz_theta_vf,  # rotation in Y-Z; x,theta unchanged
    'yz_in_6d':         _so2_yz_in_6d_vf,         # rotation in Y-Z; all orientation DOF frozen
}


def _rot2d_traj(q2d, ddt, n_steps):
    """
    Analytically rotate a 2-D dual-arm trajectory at each integration step.

    q2d     : (T, 4)  — [x_l, y_l, x_r, y_r]
    ddt     : signed step size (rad)
    n_steps : number of steps

    Returns (n_steps, T, 4) where result[i] = R(i·ddt) @ q2d.
    Exact to floating-point precision — no Euler integration error.
    """
    device, dtype = q2d.device, q2d.dtype
    angles = torch.arange(n_steps, dtype=dtype, device=device) * ddt
    c, s   = torch.cos(angles), torch.sin(angles)          # (n_steps,)
    xl, yl = q2d[:, 0], q2d[:, 1]                          # (T,)
    xr, yr = q2d[:, 2], q2d[:, 3]
    xl_r = c[:, None] * xl[None] - s[:, None] * yl[None]
    yl_r = s[:, None] * xl[None] + c[:, None] * yl[None]
    xr_r = c[:, None] * xr[None] - s[:, None] * yr[None]
    yr_r = s[:, None] * xr[None] + c[:, None] * yr[None]
    return torch.stack([xl_r, yl_r, xr_r, yr_r], dim=-1)  # (n_steps, T, 4)


class SO2RBY1RepIn(CustomBaseRepHorLift):
    """
    Input representation for SO2 on the RBY1 dual-arm robot.

    is_taskspace=True  — rotates the 2-D task-space positions analytically
                         (exact, no integration error).
    is_taskspace=False — integrates the horizontal lift using the SO2 vector
                         field appropriate for `task_space`:
                           'xy'        → (−y,  x)
                           'xyz'       → (−y,  x, 0)
                           'xy_theta'  → (−y,  x, 1)
                           'xyz_theta' → (−y,  x, 0, 1)

    center_left, center_right : optional (2,) array-like
        Fixed task-space point around which each arm rotates.
        When provided the VF is shifted so the arm rotates around that center
        instead of the origin.  Different centers per arm are supported.
    """

    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy',
                 center_left=None, center_right=None):
        self.is_taskspace   = is_taskspace
        self._task_space    = task_space
        self._center_left   = center_left
        self._center_right  = center_right
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX(self, x, shift=None):
        return _SO2_VF[self._task_space](x, shift)

    def _make_vx(self, center):
        """Build a VF callable, optionally shifted to a fixed center."""
        base_vf = _SO2_VF[self._task_space]
        if center is None:
            return base_vf
        center_arr = np.asarray(center, dtype=np.float32)
        def vx_with_center(x, _c=center_arr):
            shift = torch.tensor(_c, dtype=x.dtype, device=x.device).unsqueeze(0)
            return base_vf(x, shift=shift)
        return vx_with_center

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta   = torch.atan2(g[1, 0], g[0, 0])
        n_steps = max(2, _n_steps(theta, self.dt))
        ddt     = _signed_dt(theta, self.dt)

        if self.is_taskspace:
            q_t = _rot2d_traj(q0, ddt, n_steps)
        else:
            nb_arm = self.robot.nb_dofs // 2
            VX_l = self._make_vx(self._center_left)
            VX_r = self._make_vx(self._center_right)
            q_l  = act_on_Q(robot=self.robot.robot_left,  theta=theta, q0=q0[:, :nb_arm],
                            dt=self.dt, VX=VX_l, n_steps=n_steps)
            q_r  = act_on_Q(robot=self.robot.robot_right, theta=theta, q0=q0[:, nb_arm:],
                            dt=self.dt, VX=VX_r, n_steps=n_steps)
            q_t  = torch.cat([q_l, q_r], dim=-1)

        x_t = evolve_X(x0=x0, magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_steps)

        if full_traj:
            return torch.cat([q_t, x_t], dim=-1)
        return torch.cat([q_t[-1], x_t[-1]], dim=-1)


class SO2RBY1RepOut(CustomBaseRepHorLift):
    """Output (velocity) representation for SO2 on the RBY1 dual-arm robot.

    center_left, center_right : optional (2,) array-like
        Fixed task-space point around which each arm rotates (see SO2RBY1RepIn).
    """

    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy',
                 center_left=None, center_right=None):
        self.is_taskspace  = is_taskspace
        self._task_space   = task_space
        self._center_left  = center_left
        self._center_right = center_right
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX(self, x, shift=None):
        return _SO2_VF[self._task_space](x, shift)

    def _make_vx(self, center):
        """Build a VF callable, optionally shifted to a fixed center."""
        base_vf = _SO2_VF[self._task_space]
        if center is None:
            return base_vf
        center_arr = np.asarray(center, dtype=np.float32)
        def vx_with_center(x, _c=center_arr):
            shift = torch.tensor(_c, dtype=x.dtype, device=x.device).unsqueeze(0)
            return base_vf(x, shift=shift)
        return vx_with_center

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        theta   = torch.atan2(g[1, 0], g[0, 0])
        n_steps = max(2, _n_steps(theta, self.dt))
        ddt     = _signed_dt(theta, self.dt)

        if self.is_taskspace:
            # Velocities transform identically to positions under SO2.
            v_t = _rot2d_traj(v0, ddt, n_steps)
        else:
            nb_arm = self.robot.nb_dofs // 2
            VX_l   = self._make_vx(self._center_left)
            VX_r   = self._make_vx(self._center_right)
            out_l  = act_on_TQ(robot=self.robot.robot_left,  theta=theta,
                               q0=q0[:, :nb_arm], vq0=v0[:, :nb_arm],
                               dt=self.dt, VX=VX_l, return_q_traj=False, n_steps=n_steps)
            out_r  = act_on_TQ(robot=self.robot.robot_right, theta=theta,
                               q0=q0[:, nb_arm:], vq0=v0[:, nb_arm:],
                               dt=self.dt, VX=VX_r, return_q_traj=False, n_steps=n_steps)
            v_t    = torch.cat([out_l, out_r], dim=-1)

        return v_t if full_traj else v_t[-1]


############################################################################################################
# RBY1 SO2×C2 — combined morphological (C2) + rotational (SO2) symmetry
############################################################################################################

def _c2rby1_perm_matrix(nb_q: int, is_taskspace: bool, device) -> torch.Tensor:
    """Build the nb_q×nb_q C2 permutation matrix for RBY1 (swaps arms with sign flips)."""
    if is_taskspace:
        perm_Q = torch.tensor([1., -1.], device=device)
    else:
        perm_Q = torch.tensor([1., -1., -1., 1., -1., 1., -1.], device=device)
    refl_Q = torch.zeros(nb_q, nb_q, device=device)
    refl_Q[:nb_q//2, nb_q//2:] = torch.diag(perm_Q)
    refl_Q[nb_q//2:, :nb_q//2] = torch.diag(perm_Q)
    return refl_Q


class C2SO2RBY1(Group):
    """Combined C2 (morphological) × SO2 (rotational) symmetry for RBY1.

    Group elements are encoded as 2×2 matrices:
      det > 0  →  pure SO2 rotation
      det < 0  →  C2 reflection composed with an SO2 rotation
    The 2D C2 marker used here is x-reflection: diag(-1, 1).
    """
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.zeros((1, n, n))
        self.lie_algebra[0] = np.array([[0., -1.], [1., 0.]])
        self.discrete_generators = np.array([[[-1., 0.], [0., 1.]]])
        super().__init__(**kwargs)


class C2SO2RBY1RepIn(CustomBaseRepHorLift):
    """Input (config + task) representation for SO2×C2 on the RBY1 dual-arm robot."""

    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy'):
        self.is_taskspace = is_taskspace
        self._task_space  = task_space
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX(self, x, shift=None):
        return _SO2_VF[self._task_space](x, shift)

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3:
            g = g.squeeze(0)

        is_refl = (_det2x2(g)) < 0
        if is_refl:
            # Extract the pure rotation part: g_rot = C2_2d @ g  (det > 0)
            c2_2d = torch.tensor([[-1., 0.], [0., 1.]], dtype=g.dtype, device=g.device)
            g_rot = c2_2d @ g
            theta = torch.atan2(g_rot[1, 0], g_rot[0, 0])
        else:
            theta = torch.atan2(g[1, 0], g[0, 0])

        if is_refl:
            refl_Q  = _c2rby1_perm_matrix(self.nb_q, self.is_taskspace, g.device)
            q_t     = (refl_Q @ q0.unsqueeze(-1)).squeeze(-1)
            x_t     = x0.clone()
            x_t[..., 2] = -x_t[..., 2]   # flip reflection conditioning component
        else:
            q_t = q0
            x_t = x0

        # SO2 integration step (skip if identity rotation)
        if torch.abs(theta) > 1e-6:
            n_steps = max(2, _n_steps(theta, self.dt))
            ddt     = _signed_dt(theta, self.dt)
            if self.is_taskspace:
                q_t = _rot2d_traj(q_t, ddt, n_steps)
            else:
                q_t = act_on_Q_dual_arm(robot=self.robot, g=theta, q0=q_t,
                                         dt=self.dt, VX=self.VX, n_steps=n_steps)
            x_t = evolve_X(x0=x0, magnitude=theta, dt=self.dt, mode='rotation', n_steps=n_steps)
        else:
            q_t = q_t.unsqueeze(0)
            x_t = x_t.unsqueeze(0)

        result = torch.cat([q_t, x_t], dim=-1)
        return result if full_traj else result[-1]


class C2SO2RBY1RepOut(CustomBaseRepHorLift):
    """Output (velocity) representation for SO2×C2 on the RBY1 dual-arm robot."""

    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy'):
        self.is_taskspace = is_taskspace
        self._task_space  = task_space
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX(self, x, shift=None):
        return _SO2_VF[self._task_space](x, shift)

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3:
            g = g.squeeze(0)

        is_refl = (_det2x2(g)) < 0
        if is_refl:
            c2_2d = torch.tensor([[-1., 0.], [0., 1.]], dtype=g.dtype, device=g.device)
            g_rot = c2_2d @ g
            theta = torch.atan2(g_rot[1, 0], g_rot[0, 0])
        else:
            theta = torch.atan2(g[1, 0], g[0, 0])

        if is_refl:
            refl_Q = _c2rby1_perm_matrix(self.nb_q, self.is_taskspace, g.device)
            v_t    = (refl_Q @ v0.unsqueeze(-1)).squeeze(-1)
            # Reflect q as well
            q_t   = (refl_Q @ q0.unsqueeze(-1)).squeeze(-1)
        else:
            v_t = v0
            q_t = q0

        if torch.abs(theta) > 1e-6:
            n_steps = max(2, _n_steps(theta, self.dt))
            ddt     = _signed_dt(theta, self.dt)
            if self.is_taskspace:
                v_t = _rot2d_traj(v_t, ddt, n_steps)
            else:
                v_t = act_on_TQ_dual_arm(robot=self.robot, g=theta, q0=q_t, vq0=v_t,
                                          dt=self.dt, VX=self.VX, n_steps=n_steps)
        else:
            v_t = v0.unsqueeze(0)

        return v_t if full_traj else v_t[-1]


############################################################################################################
# RBY1 Scaling2 — task-space vector fields and rep classes
############################################################################################################

class Scaling2RBY1(Group):
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.eye(n)[None]
        super().__init__(n, **kwargs)


def _scaling2_xyz_vf(x, shift=None):
    vf = torch.zeros_like(x)
    vf[..., 0] = x[..., 0]
    vf[..., 1] =  x[..., 1]
    # vf[..., 1] = x[..., 2] # Here decide if we want to scale z as well.
    return vf

def _scaling2_xy_theta_vf(x, shift=None):
    vf = torch.zeros_like(x)
    vf[..., 0] =  x[..., 0]
    vf[..., 1] =  x[..., 1]
    vf[..., 2] =  0.0
    return vf

def _scaling2_xyz_theta_vf(x, shift=None):
    vf = torch.zeros_like(x)
    vf[..., 0] = x[..., 0]
    vf[..., 1] = x[..., 1]
    # vf[..., 1] = x[..., 2] # Here decide if we want to scale z as well.
    vf[..., 3] = 0.0
    return vf

def _scaling2_yz_in_xyz_theta_vf(x, shift=None):
    """Scaling2 (radial expansion) in the Y–Z plane; X (dim 0) and theta (dim 3) have zero velocity.

    x layout: [x, y, z, theta]
    shift (if given): (B, 2) — [c_y, c_z] — scaling center in the Y–Z subspace.
    Infinitesimal generator: dy/dt = (y-c_y), dz/dt = (z-c_z).
    """
    vf = torch.zeros_like(x)
    y = x[..., 1]
    z = x[..., 2]
    if shift is not None:
        y = y - shift[..., 0]   # shift[..., 0] = c_y
        z = z - shift[..., 1]   # shift[..., 1] = c_z
    vf[..., 1] = y
    vf[..., 2] = z
    return vf


def _scaling2_yz_in_6d_vf(x, shift=None):
    """Scaling2 (radial expansion) in the Y–Z plane; ALL other DOF have zero velocity.

    x layout: [x, y, z, roll, pitch, yaw]  (indices 0–5)
    shift (if given): (B, 2) — [c_y, c_z] — scaling centre in Y–Z subspace.
    Pairs with the full 6×7 geometric Jacobian [J_lin; J_ang], which forces the
    end-effector to keep its orientation completely fixed during augmentation.
    """
    vf = torch.zeros_like(x)
    y = x[..., 1]
    z = x[..., 2]
    if shift is not None:
        y = y - shift[..., 0]
        z = z - shift[..., 1]
    vf[..., 1] = y
    vf[..., 2] = z
    return vf


_SCALING2_VF = {
    'xy':               scaling2_R2_vf,
    'yz':               scaling2_R2_vf,  # same 2-D VF; x_data stores [y,z] so dim0/dim1 already correct
    'xyz':              _scaling2_xyz_vf,
    'xy_theta':         _scaling2_xy_theta_vf,
    'xyz_theta':        _scaling2_xyz_theta_vf,
    'yz_in_xyz_theta':  _scaling2_yz_in_xyz_theta_vf,  # scaling in Y-Z; x,theta unchanged
    'yz_in_6d':         _scaling2_yz_in_6d_vf,         # scaling in Y-Z; all orientation DOF frozen
}

def _scale2d_traj(q2d, ddt, n_steps):
    """
    Analytically scale a 2-D dual-arm trajectory at each integration step.

    q2d     : (T, 4)  — [x_l, y_l, x_r, y_r]
    ddt     : signed step size (rad)
    n_steps : number of steps

    Returns (n_steps, T, 4) where result[i] = scaling @ q2d.
    Exact to floating-point precision — no Euler integration error.
    """
    print(f'WARNING: This is not tested yet!')
    device, dtype = q2d.device, q2d.dtype
    scales = 1 + torch.arange(n_steps, dtype=dtype, device=device) * ddt
    xl, yl = q2d[:, 0], q2d[:, 1]
    xr, yr = q2d[:, 2], q2d[:, 3]
    xl_s = scales[:, None] * xl[None]
    yl_s = scales[:, None] * yl[None]
    xr_s = scales[:, None] * xr[None]
    yr_s = scales[:, None] * yr[None]
    return torch.stack([xl_s, yl_s, xr_s, yr_s], dim=-1)  # (n_steps, T, 4)

class Scaling2RBY1RepIn(CustomBaseRepHorLift):
    """
    Input representation for Scaling2 on the RBY1 dual-arm robot.

    is_taskspace=True  — scales the 2-D task-space positions analytically
                         (exact, no integration error).
    is_taskspace=False — integrates the horizontal lift using the Scaling2 vector
                         field appropriate for the chosen `task_space`.

    center_left, center_right : optional (2,) array-like
        Fixed task-space point from which each arm scales radially.
        The flow of dx/dt = x − c is x(t) = c + (x0 − c)·exp(t), i.e.
        radial scaling from c instead of the origin.
        Different centers per arm are supported.
    """
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy',
                 center_left=None, center_right=None):
        self.is_taskspace  = is_taskspace
        self._task_space   = task_space
        self._center_left  = center_left
        self._center_right = center_right
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX(self, x, shift=None):
        return _SCALING2_VF[self._task_space](x, shift)

    def _make_vx(self, center):
        """Build a VF callable, optionally shifted to a fixed center."""
        base_vf = _SCALING2_VF[self._task_space]
        if center is None:
            return base_vf
        center_arr = np.asarray(center, dtype=np.float32)
        def vx_with_center(x, _c=center_arr):
            shift = torch.tensor(_c, dtype=x.dtype, device=x.device).unsqueeze(0)
            return base_vf(x, shift=shift)
        return vx_with_center

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        scale   = torch.sqrt((_det2x2(g))) - 1
        n_steps = max(2, _n_steps(scale, self.dt))
        ddt     = _signed_dt(scale, self.dt)

        if self.is_taskspace:
            q_t = _scale2d_traj(q0, ddt, n_steps)
        else:
            nb_arm = self.robot.nb_dofs // 2
            VX_l = self._make_vx(self._center_left)
            VX_r = self._make_vx(self._center_right)
            q_l  = act_on_Q(robot=self.robot.robot_left,  theta=scale, q0=q0[:, :nb_arm],
                            dt=self.dt, VX=VX_l, n_steps=n_steps)
            q_r  = act_on_Q(robot=self.robot.robot_right, theta=scale, q0=q0[:, nb_arm:],
                            dt=self.dt, VX=VX_r, n_steps=n_steps)
            q_t  = torch.cat([q_l, q_r], dim=-1)

        x_t = evolve_X(x0=x0, magnitude=scale, dt=self.dt, mode='scaling', n_steps=n_steps)

        if full_traj:
            return torch.cat([q_t, x_t], dim=-1)
        return torch.cat([q_t[-1], x_t[-1]], dim=-1)


class Scaling2RBY1RepOut(CustomBaseRepHorLift):
    """Output (velocity) representation for Scaling2 on the RBY1 dual-arm robot.

    center_left, center_right : optional (2,) array-like
        Fixed task-space point from which each arm scales (see Scaling2RBY1RepIn).
    """

    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy',
                 center_left=None, center_right=None):
        self.is_taskspace  = is_taskspace
        self._task_space   = task_space
        self._center_left  = center_left
        self._center_right = center_right
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX(self, x, shift=None):
        return _SCALING2_VF[self._task_space](x, shift)

    def _make_vx(self, center):
        """Build a VF callable, optionally shifted to a fixed center."""
        base_vf = _SCALING2_VF[self._task_space]
        if center is None:
            return base_vf
        center_arr = np.asarray(center, dtype=np.float32)
        def vx_with_center(x, _c=center_arr):
            shift = torch.tensor(_c, dtype=x.dtype, device=x.device).unsqueeze(0)
            return base_vf(x, shift=shift)
        return vx_with_center

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        scale   = torch.sqrt((_det2x2(g))) - 1
        n_steps = max(2, _n_steps(scale, self.dt))
        ddt     = _signed_dt(scale, self.dt)

        if self.is_taskspace:
            # Velocities transform identically to positions under scaling.
            v_t = _scale2d_traj(v0, ddt, n_steps)
        else:
            nb_arm = self.robot.nb_dofs // 2
            VX_l   = self._make_vx(self._center_left)
            VX_r   = self._make_vx(self._center_right)
            out_l  = act_on_TQ(robot=self.robot.robot_left,  theta=scale,
                               q0=q0[:, :nb_arm], vq0=v0[:, :nb_arm],
                               dt=self.dt, VX=VX_l, return_q_traj=False, n_steps=n_steps)
            out_r  = act_on_TQ(robot=self.robot.robot_right, theta=scale,
                               q0=q0[:, nb_arm:], vq0=v0[:, nb_arm:],
                               dt=self.dt, VX=VX_r, return_q_traj=False, n_steps=n_steps)
            v_t    = torch.cat([out_l, out_r], dim=-1)

        return v_t if full_traj else v_t[-1]


############################################################################################################
# RBY1 SO2×C2 — combined morphological (C2) + rotational (SO2) symmetry
############################################################################################################

class C2Scaling2RBY1(Group):
    """Combined Scaling2 (scaling) × C2 (morphological) symmetry for RBY1.

    Group elements are encoded as 2×2 matrices:
      det < 0  →  C2 reflection composed with an SO2 rotation
    The 2D C2 marker used here is x-reflection: diag(-1, 1).
    """
    def __init__(self, **kwargs):
        n = 2
        self.lie_algebra = np.zeros((1, n, n))
        self.lie_algebra[0] = np.eye(n)
        self.discrete_generators = np.array([[[-1., 0.], [0., 1.]]])
        super().__init__(**kwargs)


class C2Scaling2RBY1RepIn(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy'):
        self.is_taskspace = is_taskspace
        self._task_space  = task_space
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q + self.nb_x

    def VX(self, x, shift=None):
        return _SCALING2_VF[self._task_space](x, shift)

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3:
            g = g.squeeze(0)

        is_refl = (_det2x2(g)) < 0
        if is_refl:
            # Extract the pure scaling part
            c2_2d = torch.tensor([[-1., 0.], [0., 1.]], dtype=g.dtype, device=g.device)
            g_scale = c2_2d @ g
            scale = torch.sqrt((_det2x2(g_scale))) - 1            
        else:
            scale = torch.sqrt((_det2x2(g))) - 1

        if is_refl:
            refl_Q  = _c2rby1_perm_matrix(self.nb_q, self.is_taskspace, g.device)
            q_t     = (refl_Q @ q0.unsqueeze(-1)).squeeze(-1)
            x_t     = x0.clone()
            x_t[..., 2] = -x_t[..., 2]   # flip reflection conditioning component
        else:
            q_t = q0
            x_t = x0

        # Scaling2 integration step (skip if identity rotation)
        if torch.abs(scale) > 1e-6:
            n_steps = max(2, _n_steps(scale, self.dt))
            ddt     = _signed_dt(scale, self.dt)
            if self.is_taskspace:
                q_t = _scale2d_traj(q_t, ddt, n_steps)
            else:
                q_t = act_on_Q_dual_arm(robot=self.robot, g=scale, q0=q_t,
                                         dt=self.dt, VX=self.VX, n_steps=n_steps)
            x_t = evolve_X(x0=x0, magnitude=scale, dt=self.dt, mode='scaling', n_steps=n_steps)            
        else:
            q_t = q_t.unsqueeze(0)
            x_t = x_t.unsqueeze(0)

        result = torch.cat([q_t, x_t], dim=-1)
        return result if full_traj else result[-1]


class C2Scaling2RBY1RepOut(CustomBaseRepHorLift):
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 is_taskspace=False, task_space='xy'):
        self.is_taskspace = is_taskspace
        self._task_space  = task_space
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self):
        return self.nb_q

    def VX(self, x, shift=None):
        return _SCALING2_VF[self._task_space](x, shift)

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3:
            g = g.squeeze(0)

        is_refl = (_det2x2(g)) < 0
        if is_refl:
            # Extract the pure scaling part
            c2_2d = torch.tensor([[-1., 0.], [0., 1.]], dtype=g.dtype, device=g.device)
            g_scale = c2_2d @ g
            scale = torch.sqrt((_det2x2(g_scale))) - 1
        else:
            scale = torch.sqrt((_det2x2(g))) - 1

        if is_refl:
            refl_Q = _c2rby1_perm_matrix(self.nb_q, self.is_taskspace, g.device)
            v_t    = (refl_Q @ v0.unsqueeze(-1)).squeeze(-1)
            # Reflect q as well
            q_t   = (refl_Q @ q0.unsqueeze(-1)).squeeze(-1)
        else:
            v_t = v0
            q_t = q0

        if torch.abs(scale) > 1e-6:
            n_steps = max(2, _n_steps(scale, self.dt))
            ddt     = _signed_dt(scale, self.dt)
            if self.is_taskspace:
                v_t = _scale2d_traj(v_t, ddt, n_steps)
            else:
                v_t = act_on_TQ_dual_arm(robot=self.robot, g=scale, q0=q_t, vq0=v_t,
                                          dt=self.dt, VX=self.VX, n_steps=n_steps)                
        else:
            v_t = v0.unsqueeze(0)

        return v_t if full_traj else v_t[-1]


############################################################################################################
# Composite per-arm Representations (modular, generalized)
############################################################################################################

@dataclass
class ArmSymSpec:
    """Continuous symmetry specification for one arm of a dual-arm robot.

    so2          : arm participates in SO2 (rotation) symmetry
    scaling2     : arm participates in Scaling2 (isotropic scaling) symmetry
    task_space   : key into _SO2_VF / _SCALING2_VF selecting the horizontal lift VF
    center_on_ee : if True, the SO2 VF is centered at the arm's initial EE position
                   (computed via FK from q0) rather than the origin
    center       : optional fixed (2,) array-like [c0, c1] in task-space coordinates.
                   When set, both SO2 (rotation) and Scaling2 vector fields are centered
                   at this point (rotation around it; radial scaling from it).
                   Takes precedence over center_on_ee.
    """
    so2:          bool   = False
    scaling2:     bool   = False
    task_space:   str    = 'xy'
    center_on_ee: bool   = False
    center:       object = None


def _det2x2(m: torch.Tensor) -> torch.Tensor:
    """Determinant of a 2×2 matrix without torch.det (works on MPS)."""
    return m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]


def _decompose_2x2_g(g: torch.Tensor):
    """
    Decompose a 2×2 group element into (is_refl, theta, scale).

    Handles pure SO2, pure Scaling2, C2×SO2, C2×Scaling2, and products thereof.
      is_refl : det(g) < 0  — C2 morphological swap is present
      theta   : SO2 rotation angle (0 when no rotation component)
      scale   : scale − 1   (0 when no scaling component)
    """
    is_refl = _det2x2(g) < 0
    c2_2d   = torch.tensor([[-1., 0.], [0., 1.]], dtype=g.dtype, device=g.device)
    g_cont  = c2_2d @ g if is_refl else g
    theta   = torch.atan2(g_cont[1, 0], g_cont[0, 0])
    scale   = torch.sqrt(torch.abs(_det2x2(g_cont))) - 1
    return is_refl, theta, scale


def _build_perm_matrix(nb_q: int, perm_vector, device: torch.device) -> torch.Tensor:
    """
    Build the nb_q × nb_q C2 arm-swap permutation matrix.

    perm_vector : 1-D array-like of length nb_q//2 with per-joint sign flips.
                  None → all joints flip sign (default planar-robot convention).
    """
    if perm_vector is not None:
        perm_Q = torch.as_tensor(perm_vector, dtype=torch.float32, device=device)
    else:
        perm_Q = -torch.ones(nb_q // 2, device=device)
    R = torch.zeros(nb_q, nb_q, device=device)
    R[:nb_q//2, nb_q//2:] = torch.diag(perm_Q)
    R[nb_q//2:, :nb_q//2] = torch.diag(perm_Q)
    return R


class CompositeRepIn(CustomBaseRepHorLift):
    """
    Modular input representation for dual-arm robots.

    Each arm can independently participate in SO2 and/or Scaling2 symmetry.
    The morphological C2 arm-swap (controlled by `morph`) is always applied
    first when det(g) < 0, and always acts on both arms jointly.

    Group elements are the same 2×2 matrices produced by `build_group_elements`
    — identical interface to all existing RBY1 rep classes.

    Parameters
    ----------
    left_spec, right_spec : ArmSymSpec
        Which continuous symmetries each arm participates in.
        ArmSymSpec() (all False) leaves that arm unchanged by continuous actions.
    morph : bool
        Apply the C2 morphological arm-swap when det(g) < 0.
    perm_vector : array-like of shape (nb_q//2,), optional
        Per-joint sign flips for the C2 swap.  None → all −1.
    is_taskspace : bool
        Reserved for future Cartesian task-space path; must be False for now.
    """
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 left_spec: ArmSymSpec = None, right_spec: ArmSymSpec = None,
                 morph: bool = True, perm_vector=None, is_taskspace: bool = False,
                 dt_so2: float = None, dt_scaling2: float = None):
        self.left_spec     = left_spec  or ArmSymSpec()
        self.right_spec    = right_spec or ArmSymSpec()
        self.morph         = morph
        self._perm_vec     = perm_vector
        self.is_taskspace  = is_taskspace
        self._dt_so2       = dt_so2      if dt_so2      is not None else dt
        self._dt_scaling2  = dt_scaling2 if dt_scaling2 is not None else dt
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self): return self.nb_q + self.nb_x

    def _vx(self, spec: ArmSymSpec, mode: str):
        return _SO2_VF[spec.task_space] if mode == 'so2' else _SCALING2_VF[spec.task_space]

    def _vx_for(self, spec: ArmSymSpec, mode: str, q_arm, arm_robot):
        """Return VF, optionally shifted to a fixed center or the arm's initial EE position.

        Priority:
          1. spec.center  — fixed (2,) array set before augmentation (e.g. convex-hull centroid).
                            Applies to BOTH so2 and scaling modes.
          2. center_on_ee — legacy per-call FK-based shift; SO2 only.
          3. Default      — VF centred at origin.
        """
        base_vf = self._vx(spec, mode)
        if spec.center is not None:
            center_arr = np.asarray(spec.center, dtype=np.float32)
            def vx_fixed_center(x, _c=center_arr):
                shift = torch.tensor(_c, dtype=x.dtype, device=x.device).unsqueeze(0)
                return base_vf(x, shift=shift)
            return vx_fixed_center
        if mode == 'so2' and spec.center_on_ee:
            with torch.no_grad():
                shift = arm_robot.fkine_torch(q_arm[0:1])[..., :2]  # (1, 2)
            return lambda x, _s=shift: base_vf(x, shift=_s)
        return base_vf

    def act_on(self, g, q0=None, x0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        is_refl, theta, scale = _decompose_2x2_g(g)
        nb_arm = self.nb_q // 2

        # Correct semi-direct product order: continuous actions (scale → rotate)
        # are applied to the ORIGINAL arms using their ORIGINAL specs, and the
        # morphological C2 swap is applied LAST.  Applying C2 first and swapping
        # specs produces the wrong horizontal-lift trajectory because hor_lift
        # depends on the current joint configuration.
        q_l = q0[..., :nb_arm]
        q_r = q0[..., nb_arm:]

        l_spec = self.left_spec
        r_spec = self.right_spec

        has_sl = l_spec.scaling2 and torch.abs(scale) > 1e-6
        has_sr = r_spec.scaling2 and torch.abs(scale) > 1e-6
        has_rl = l_spec.so2      and torch.abs(theta) > 1e-6
        has_rr = r_spec.so2      and torch.abs(theta) > 1e-6

        n_s = max(2, _n_steps(scale, self._dt_scaling2)) if (has_sl or has_sr) else 1
        n_r = max(2, _n_steps(theta, self._dt_so2))      if (has_rl or has_rr) else 1

        # Pre-compute VFs (with optional center shift for both rotation and scaling)
        vx_sl = self._vx_for(l_spec, 'sc2', q_l, self.robot.robot_left)  if has_sl else None
        vx_sr = self._vx_for(r_spec, 'sc2', q_r, self.robot.robot_right) if has_sr else None
        vx_rl = self._vx_for(l_spec, 'so2', q_l, self.robot.robot_left)  if has_rl else None
        vx_rr = self._vx_for(r_spec, 'so2', q_r, self.robot.robot_right) if has_rr else None

        if not full_traj:
            # ── final state only ──────────────────────────────────────────
            # Phase 1: scaling
            if has_sl:
                q_l = act_on_Q(robot=self.robot.robot_left,  q0=q_l, dt=self._dt_scaling2,
                               theta=scale, VX=vx_sl, n_steps=n_s)[-1]
            if has_sr:
                q_r = act_on_Q(robot=self.robot.robot_right, q0=q_r, dt=self._dt_scaling2,
                               theta=scale, VX=vx_sr, n_steps=n_s)[-1]
            # Phase 2: rotation
            if has_rl:
                q_l = act_on_Q(robot=self.robot.robot_left,  q0=q_l, dt=self._dt_so2,
                               theta=theta, VX=vx_rl, n_steps=n_r)[-1]
            if has_rr:
                q_r = act_on_Q(robot=self.robot.robot_right, q0=q_r, dt=self._dt_so2,
                               theta=theta, VX=vx_rr, n_steps=n_r)[-1]
            # Phase 3: C2 morphological swap (LAST)
            if self.morph and is_refl:
                R     = _build_perm_matrix(self.nb_q, self._perm_vec, q_l.device)
                q_cat = torch.cat([q_l, q_r], dim=-1)
                q_cat = q_cat @ R.t()
                q_l, q_r = q_cat[..., :nb_arm], q_cat[..., nb_arm:]

            x_out = x0
            if has_sl or has_sr:
                x_out = evolve_X(x0=x_out, magnitude=scale, dt=self._dt_scaling2, mode='scaling', n_steps=n_s)[-1]
            if has_rl or has_rr:
                x_out = evolve_X(x0=x_out, magnitude=theta,  dt=self._dt_so2, mode='rotation', n_steps=n_r)[-1]
            if self.morph and is_refl:
                x_out = x_out.clone()
                x_out[..., 2] = -x_out[..., 2]

            return torch.cat([q_l, q_r, x_out], dim=-1)

        # ── full trajectory ───────────────────────────────────────────────
        # Phase 1: scaling
        q_l_s = act_on_Q(robot=self.robot.robot_left,  q0=q_l, dt=self._dt_scaling2,
                         theta=scale, VX=vx_sl, n_steps=n_s) if has_sl \
               else q_l.unsqueeze(0).expand(n_s, *q_l.shape)
        q_r_s = act_on_Q(robot=self.robot.robot_right, q0=q_r, dt=self._dt_scaling2,
                         theta=scale, VX=vx_sr, n_steps=n_s) if has_sr \
               else q_r.unsqueeze(0).expand(n_s, *q_r.shape)
        x_s   = evolve_X(x0=x0, magnitude=scale, dt=self._dt_scaling2, mode='scaling', n_steps=n_s) if (has_sl or has_sr) \
               else x0.unsqueeze(0).expand(n_s, *x0.shape)

        # Phase 2: rotation, applied on top of each scaling step
        q_l_list, q_r_list, x_list = [], [], []
        for i in range(n_s):
            q_l_list.append(act_on_Q(robot=self.robot.robot_left,  q0=q_l_s[i], dt=self._dt_so2,
                                     theta=theta, VX=vx_rl, n_steps=n_r)
                            if has_rl else q_l_s[i].unsqueeze(0).expand(n_r, *q_l_s[i].shape))
            q_r_list.append(act_on_Q(robot=self.robot.robot_right, q0=q_r_s[i], dt=self._dt_so2,
                                     theta=theta, VX=vx_rr, n_steps=n_r)
                            if has_rr else q_r_s[i].unsqueeze(0).expand(n_r, *q_r_s[i].shape))
            x_list.append(evolve_X(x0=x_s[i], magnitude=theta, dt=self._dt_so2, mode='rotation', n_steps=n_r)
                          if (has_rl or has_rr) else x_s[i].unsqueeze(0).expand(n_r, *x_s[i].shape))

        q_l_out = torch.cat(q_l_list, dim=0)  # (n_s*n_r, T, nb_arm)
        q_r_out = torch.cat(q_r_list, dim=0)
        x_out   = torch.cat(x_list,   dim=0)  # (n_s*n_r, T, nb_x)

        # Phase 3: C2 morphological swap (LAST) — applied to every step in the orbit traj
        if self.morph and is_refl:
            R     = _build_perm_matrix(self.nb_q, self._perm_vec, q_l_out.device)
            q_cat = torch.cat([q_l_out, q_r_out], dim=-1)  # (n_s*n_r, T, nb_q)
            q_cat = q_cat @ R.t()
            q_l_out = q_cat[..., :nb_arm]
            q_r_out = q_cat[..., nb_arm:]
            x_out = x_out.clone()
            x_out[..., 2] = -x_out[..., 2]

        return torch.cat([q_l_out, q_r_out, x_out], dim=-1)


class CompositeRepOut(CustomBaseRepHorLift):
    """
    Modular output (velocity) representation for dual-arm robots.

    Mirrors CompositeRepIn but acts on the tangent bundle (q, v) via act_on_TQ.
    """
    def __init__(self, G, robot=None, is_vf_constant=False, dt=0.05,
                 left_spec: ArmSymSpec = None, right_spec: ArmSymSpec = None,
                 morph: bool = True, perm_vector=None, is_taskspace: bool = False,
                 dt_so2: float = None, dt_scaling2: float = None):
        self.left_spec     = left_spec  or ArmSymSpec()
        self.right_spec    = right_spec or ArmSymSpec()
        self.morph         = morph
        self._perm_vec     = perm_vector
        self.is_taskspace  = is_taskspace
        self._dt_so2       = dt_so2      if dt_so2      is not None else dt
        self._dt_scaling2  = dt_scaling2 if dt_scaling2 is not None else dt
        super().__init__(G, robot, is_vf_constant, dt)

    def size(self): return self.nb_q

    def _vx(self, spec: ArmSymSpec, mode: str):
        return _SO2_VF[spec.task_space] if mode == 'so2' else _SCALING2_VF[spec.task_space]

    def _vx_for(self, spec: ArmSymSpec, mode: str, q_arm, arm_robot):
        """Return VF, optionally shifted to a fixed center or the arm's initial EE position.

        Priority:
          1. spec.center  — fixed (2,) array set before augmentation (e.g. convex-hull centroid).
                            Applies to BOTH so2 and scaling modes.
          2. center_on_ee — legacy per-call FK-based shift; SO2 only.
          3. Default      — VF centred at origin.
        """
        base_vf = self._vx(spec, mode)
        if spec.center is not None:
            center_arr = np.asarray(spec.center, dtype=np.float32)
            def vx_fixed_center(x, _c=center_arr):
                shift = torch.tensor(_c, dtype=x.dtype, device=x.device).unsqueeze(0)
                return base_vf(x, shift=shift)
            return vx_fixed_center
        if mode == 'so2' and spec.center_on_ee:
            with torch.no_grad():
                shift = arm_robot.fkine_torch(q_arm[0:1])[..., :2]  # (1, 2)
            return lambda x, _s=shift: base_vf(x, shift=_s)
        return base_vf

    def act_on(self, g, q0=None, v0=None, full_traj=False):
        if g.dim() == 3: g = g.squeeze(0)
        is_refl, theta, scale = _decompose_2x2_g(g)
        nb_arm = self.nb_q // 2

        # Correct semi-direct product order: continuous actions (scale → rotate)
        # are applied to the ORIGINAL arms; C2 swap is applied LAST.
        q_l, q_r = q0[..., :nb_arm], q0[..., nb_arm:]
        v_l, v_r = v0[..., :nb_arm], v0[..., nb_arm:]

        l_spec = self.left_spec
        r_spec = self.right_spec

        has_sl = l_spec.scaling2 and torch.abs(scale) > 1e-6
        has_sr = r_spec.scaling2 and torch.abs(scale) > 1e-6
        has_rl = l_spec.so2      and torch.abs(theta) > 1e-6
        has_rr = r_spec.so2      and torch.abs(theta) > 1e-6

        n_s = max(2, _n_steps(scale, self._dt_scaling2)) if (has_sl or has_sr) else 1
        n_r = max(2, _n_steps(theta, self._dt_so2))      if (has_rl or has_rr) else 1

        # Pre-compute VFs (with optional center shift for both rotation and scaling)
        vx_sl = self._vx_for(l_spec, 'sc2', q_l, self.robot.robot_left)  if has_sl else None
        vx_sr = self._vx_for(r_spec, 'sc2', q_r, self.robot.robot_right) if has_sr else None
        vx_rl = self._vx_for(l_spec, 'so2', q_l, self.robot.robot_left)  if has_rl else None
        vx_rr = self._vx_for(r_spec, 'so2', q_r, self.robot.robot_right) if has_rr else None

        if not full_traj:
            # ── final state only ──────────────────────────────────────────
            # Phase 1: scaling
            if has_sl:
                q_l, v_l = act_on_TQ(robot=self.robot.robot_left,  theta=scale, q0=q_l, vq0=v_l,
                                     dt=self._dt_scaling2, VX=vx_sl,
                                     return_q_traj=True, n_steps=n_s)
                q_l, v_l = q_l[-1], v_l[-1]
            if has_sr:
                q_r, v_r = act_on_TQ(robot=self.robot.robot_right, theta=scale, q0=q_r, vq0=v_r,
                                     dt=self._dt_scaling2, VX=vx_sr,
                                     return_q_traj=True, n_steps=n_s)
                q_r, v_r = q_r[-1], v_r[-1]
            # Phase 2: rotation
            if has_rl:
                q_l, v_l = act_on_TQ(robot=self.robot.robot_left,  theta=theta, q0=q_l, vq0=v_l,
                                     dt=self._dt_so2, VX=vx_rl,
                                     return_q_traj=True, n_steps=n_r)
                q_l, v_l = q_l[-1], v_l[-1]
            if has_rr:
                q_r, v_r = act_on_TQ(robot=self.robot.robot_right, theta=theta, q0=q_r, vq0=v_r,
                                     dt=self._dt_so2, VX=vx_rr,
                                     return_q_traj=True, n_steps=n_r)
                q_r, v_r = q_r[-1], v_r[-1]
            # Phase 3: C2 swap (LAST)
            if self.morph and is_refl:
                R     = _build_perm_matrix(self.nb_q, self._perm_vec, v_l.device)
                v_cat = torch.cat([v_l, v_r], dim=-1)
                v_cat = v_cat @ R.t()
                v_l, v_r = v_cat[..., :nb_arm], v_cat[..., nb_arm:]

            return torch.cat([v_l, v_r], dim=-1)

        # ── full trajectory ───────────────────────────────────────────────
        # Phase 1: scaling
        if has_sl:
            q_l_s, v_l_s = act_on_TQ(robot=self.robot.robot_left,  theta=scale, q0=q_l, vq0=v_l,
                                      dt=self._dt_scaling2, VX=vx_sl,
                                      return_q_traj=True, n_steps=n_s)
        else:
            q_l_s = q_l.unsqueeze(0).expand(n_s, *q_l.shape)
            v_l_s = v_l.unsqueeze(0).expand(n_s, *v_l.shape)

        if has_sr:
            q_r_s, v_r_s = act_on_TQ(robot=self.robot.robot_right, theta=scale, q0=q_r, vq0=v_r,
                                      dt=self._dt_scaling2, VX=vx_sr,
                                      return_q_traj=True, n_steps=n_s)
        else:
            q_r_s = q_r.unsqueeze(0).expand(n_s, *q_r.shape)
            v_r_s = v_r.unsqueeze(0).expand(n_s, *v_r.shape)

        # Phase 2: rotation on top of each scaling step
        v_l_list, v_r_list = [], []
        for i in range(n_s):
            if has_rl:
                _, v_l_r = act_on_TQ(robot=self.robot.robot_left,  theta=theta,
                                     q0=q_l_s[i], vq0=v_l_s[i], dt=self._dt_so2,
                                     VX=vx_rl, return_q_traj=True, n_steps=n_r)
            else:
                v_l_r = v_l_s[i].unsqueeze(0).expand(n_r, *v_l_s[i].shape)
            if has_rr:
                _, v_r_r = act_on_TQ(robot=self.robot.robot_right, theta=theta,
                                     q0=q_r_s[i], vq0=v_r_s[i], dt=self._dt_so2,
                                     VX=vx_rr, return_q_traj=True, n_steps=n_r)
            else:
                v_r_r = v_r_s[i].unsqueeze(0).expand(n_r, *v_r_s[i].shape)
            v_l_list.append(v_l_r)
            v_r_list.append(v_r_r)

        v_l_out = torch.cat(v_l_list, dim=0)  # (n_s*n_r, T, nb_arm)
        v_r_out = torch.cat(v_r_list, dim=0)

        # Phase 3: C2 swap (LAST) — applied to every step in the orbit trajectory
        if self.morph and is_refl:
            R     = _build_perm_matrix(self.nb_q, self._perm_vec, v_l_out.device)
            v_cat = torch.cat([v_l_out, v_r_out], dim=-1)  # (n_s*n_r, T, nb_q)
            v_cat = v_cat @ R.t()
            v_l_out = v_cat[..., :nb_arm]
            v_r_out = v_cat[..., nb_arm:]

        return torch.cat([v_l_out, v_r_out], dim=-1)












































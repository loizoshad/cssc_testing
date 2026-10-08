import numpy as np
import math
import random
import torch
import time

import os
import shutil

# from utils.timer import _Timer, _fmt_s
from utils.utils import *

cols = shutil.get_terminal_size().columns
np.set_printoptions(precision = 3, suppress = True, linewidth=cols)

# prioritize cuda, then mps, then cpu
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = Path(CURRENT_DIR).parent.parent.resolve()

# The NETWORKS_DIR is: ../networks relative to CURRENT_DIR
NETWORKS_DIR = Path(CURRENT_DIR).parent / 'networks'
NETWORKS_DIR_RBY1 = Path(CURRENT_DIR).parent / 'networks_rby1'


class CustomNetworks:
    def __init__(self, model_id = 'default', load_model_flag=False, save_model_flag=True, demonstrations=None, robot=None):
        self.model_id = model_id
        self.save_model_flag = save_model_flag
        self.load_model_flag = load_model_flag
        self.demonstrations = demonstrations
        self.goal_conditioned = False # TODO
        self.dt = 1.0 # This is the time step for evolving through the dynamics of the system. NOT for the group actions.
        self.robot = robot
        
        self.n_q = self.demonstrations["nb_q"]//2 # Both arms are assumed to have the same DoFs and task space dimension.
        self.n_x = self.demonstrations["nb_x"]//2 # Both arms are assumed to have the same DoFs and task space dimension.

        # Initialize Neural Network losses
        self.mse_loss = torch.nn.MSELoss().to(device)
        self.loss_history = []

        # Convert trajectory data to device tensors once.
        if not isinstance(self.demonstrations['train_in'][0], torch.Tensor):
            self.demonstrations['train_in']  = [torch.tensor(x, dtype=torch.float32, device=device) for x in self.demonstrations['train_in']]
            self.demonstrations['train_out'] = [torch.tensor(x, dtype=torch.float32, device=device) for x in self.demonstrations['train_out']]
            self.demonstrations['test_in']   = [torch.tensor(x, dtype=torch.float32, device=device) for x in self.demonstrations['test_in']]
            self.demonstrations['test_out']  = [torch.tensor(x, dtype=torch.float32, device=device) for x in self.demonstrations['test_out']]

        # Pre-convert normalisation bounds to device tensors so denormalize_state never triggers a CPU→device transfer during the training loop.
        for key in ('Q_min', 'Q_max', 'X_min', 'X_max', 'Dq_min', 'Dq_max', 'inp_min', 'inp_max'):
            if key in self.demonstrations and isinstance(self.demonstrations[key], np.ndarray):
                self.demonstrations[key] = torch.tensor(self.demonstrations[key], dtype=torch.float32, device=device)

    def save_model(self, model_id = None, intermediate_save = False, current_iteration = None):
        if model_id is None:
            model_id = self.model_id
        
        if not intermediate_save and self.save_model_flag:
            net_dir = NETWORKS_DIR_RBY1 if 'RBY1' in self.model_id else NETWORKS_DIR
            torch.save(self.model.state_dict(), os.path.join(net_dir, 'models', model_id + '.pt'))
            print(f'[INFO] Model saved to {model_id}_{self.__class__.__name__}.pt')

        # Save intermediate models sometimes
        if intermediate_save and self.save_model_flag:
            net_dir = NETWORKS_DIR_RBY1 if 'RBY1' in self.model_id else NETWORKS_DIR
            torch.save(self.model.state_dict(), os.path.join(net_dir, 'models', f'{model_id}' + '.pt'))
            print(f'[INFO] Intermediate model saved to {model_id}.pt')

    def load_model(self, model_id = None):
        if model_id is None:
            model_id = self.model_id
        net_dir = NETWORKS_DIR_RBY1 if 'RBY1' in self.model_id else NETWORKS_DIR
        self.model.load_state_dict(torch.load(os.path.join(net_dir, 'models', model_id + '.pt')))
        self.model.eval()
        print(f'[INFO] Model loaded from models/{model_id}.pt')

    def choose_batch_indices(self, train_in_trajs, batch_size, steps, num_trajectories):
        max_step = train_in_trajs[0].shape[0] - steps
        traj_index = torch.randint(0, num_trajectories, (batch_size,), device=device)
        step_index  = torch.randint(0, max_step,         (batch_size,), device=device)
        return traj_index, step_index

    def get_batch(self, train_in_trajs, train_out_trajs, traj_index, step_index, steps):
        if not hasattr(self, '_traj_stack_cache'):
            self._train_in_stack  = torch.stack(train_in_trajs)   # (N_traj, T, d_in)
            self._train_out_stack = torch.stack(train_out_trajs)  # (N_traj, T, d_out)
            self._traj_stack_cache = True

        offsets = torch.arange(steps, device=device)       # (steps,)
        idx = step_index[:, None] + offsets                # (B, steps)

        inp_batch         = self._train_in_stack[traj_index[:, None],  idx]
        desired_out_batch = self._train_out_stack[traj_index[:, None], idx]

        return inp_batch, desired_out_batch
    
    def train_network(self, robot, num_iterations=200, verbose=True, steps=5, stride=1, batch_size=10, profile=False, profile_every=5, verbose_every=1, decouple_arms=False, task_space_loss=False):
        '''
        The difference in the multistep training is that the loss is computed over multiple steps of the dynamics
        It is also important to not mix up data points from different trajectories during multistep training.
        '''
        start_time = time.perf_counter()

        train_in_trajs = self.demonstrations['train_in']
        train_out_trajs = self.demonstrations['train_out']
        num_trajectories = len(train_in_trajs)

        n_save = 500
        # n_save = 200
        # n_save = 100
        t_batch = 0.0
        t_train = 0.0

        for iteration in range(num_iterations):
            t0 = time.perf_counter()
            if decouple_arms:
                # traj_index, step_index_left, step_index_right = self.choose_batch_indices_decoupled(train_in_trajs, batch_size, steps, num_trajectories)
                # inp_batch, desired_out_batch = self.get_batch_decoupled(train_in_trajs, train_out_trajs, traj_index, step_index_left, step_index_right, steps)
                # inp_batch, desired_out_batch = self.get_batch_decoupled_goal_conditioned(train_in_trajs, train_out_trajs, steps, stride)
                inp_batch, desired_out_batch = self.get_batch_decoupled_symmetry_conditioned(train_in_trajs, train_out_trajs, steps, stride)
            else:
                if stride != 1:
                    raise NotImplementedError("The current implementation of the get_batch method does not support a stride > 1")
                traj_index, step_index = self.choose_batch_indices(train_in_trajs, batch_size, steps, num_trajectories)
                inp_batch, desired_out_batch = self.get_batch(train_in_trajs, train_out_trajs, traj_index, step_index, steps)
            t1 = time.perf_counter()

            train_loss = self.train_step(inp_batch, desired_out_batch, robot, steps=steps, stride=stride, task_space_loss=task_space_loss)
            t2 = time.perf_counter()

            t_batch += t1 - t0
            t_train += t2 - t1

            if ((iteration + 1) % verbose_every == 0 or iteration == 0) and verbose:
                loss_val = train_loss.item()
                self.loss_history.append(loss_val)
                print(f"iteration {iteration+1}/{num_iterations}, Training Loss: {loss_val:.8f}")

            if (iteration + 1) % n_save == 0 and self.save_model_flag:
                self.save_model(model_id=f'{self.model_id}iter{iteration+1}', intermediate_save=True, current_iteration=iteration+1)

        total = time.perf_counter() - start_time
        print(f'[INFO] Training completed in {total:.2f}s | batch={t_batch:.2f}s ({100*t_batch/total:.0f}%) | train_step={t_train:.2f}s ({100*t_train/total:.0f}%) | loss={train_loss.item():.8f}')

    def choose_batch_indices_decoupled(self, train_in_trajs, batch_size, steps, num_trajectories):
        max_step = train_in_trajs[0].shape[0] - steps # All demos have the same length.
        traj_index       = torch.randint(0, num_trajectories, (batch_size,), device=device)
        step_index_left  = torch.randint(0, max_step,         (batch_size,), device=device)
        step_index_right = torch.randint(0, max_step,         (batch_size,), device=device)
        return traj_index, step_index_left, step_index_right

    def get_batch_decoupled_original(self, train_in_trajs, train_out_trajs, steps, stride):
        '''
        Independently samples TRAJECTORY index AND starting time-step for each arm,
        then concatenates them into a single batch element.
        This gives O(n^2 * T^2) effective training pairs instead of O(n * T^2).

        For n=7 demos, T≈970: ~46 M unique pairs vs ~6.6 M with shared traj_idx.

        Input layout:  [q_left (n_q) | q_right (n_q) | conditioning (n_x*2)]
        Output layout: [dq_left(n_q) | dq_right(n_q)]
        '''
        raise NotImplementedError("The stride is not implemented yet")
        batch_size = 250
        n = len(train_in_trajs)
        # Independent trajectory index per arm
        traj_idx_left  = [random.randint(0, n - 1) for _ in range(batch_size)]
        traj_idx_right = [random.randint(0, n - 1) for _ in range(batch_size)]
        inp = torch.zeros((batch_size, steps, train_in_trajs[0].shape[1]), dtype=torch.float32, device=device)
        out = torch.zeros((batch_size, steps, train_out_trajs[0].shape[1]), dtype=torch.float32, device=device)

        step_idx_left  = [random.randint(0, train_in_trajs[tl].shape[0] - steps - 1) for tl in traj_idx_left]
        step_idx_right = [random.randint(0, train_in_trajs[tr].shape[0] - steps - 1) for tr in traj_idx_right]
        for j, (tl, tr, sli, sri) in enumerate(zip(traj_idx_left, traj_idx_right, step_idx_left, step_idx_right)):
            # --- joint positions: left arm from (tl, sli), right arm from (tr, sri) ---
            inp[j, :, :self.n_q]           = train_in_trajs[tl][sli:sli+steps, :self.n_q]
            inp[j, :, self.n_q:2*self.n_q] = train_in_trajs[tr][sri:sri+steps, self.n_q:2*self.n_q]
            # --- conditioning (goal / symmetry): constant per demo, take from left ---
            inp[j, :, 2*self.n_q:] = train_in_trajs[tl][sli:sli+steps, 2*self.n_q:]

            # --- joint velocities: left from (tl, sli), right from (tr, sri) ---
            out[j, :, :self.n_q]           = train_out_trajs[tl][sli:sli+steps, :self.n_q]
            out[j, :, self.n_q:2*self.n_q] = train_out_trajs[tr][sri:sri+steps, self.n_q:2*self.n_q]

        return inp, out

    def get_batch_decoupled_goal_conditioned(self, train_in_trajs, train_out_trajs, steps, stride):
        '''
        Independently samples TRAJECTORY index AND starting time-step for each arm,
        then concatenates them into a single batch element.
        This gives O(n^2 * T^2) effective training pairs instead of O(n * T^2).

        For n=7 demos, T≈970: ~46 M unique pairs vs ~6.6 M with shared traj_idx.

        Input layout:  [q_left (n_q) | q_right (n_q) | conditioning (n_x*2)]
        Output layout: [dq_left(n_q) | dq_right(n_q)]
        '''
        batch_size = 250
        n = len(train_in_trajs)
        window = steps * stride  # span in original trajectory indices
        # Independent trajectory index per arm
        traj_idx_left  = [random.randint(0, n - 1) for _ in range(batch_size)]
        traj_idx_right = [random.randint(0, n - 1) for _ in range(batch_size)]
        inp = torch.zeros((batch_size, steps, train_in_trajs[0].shape[1]), dtype=torch.float32, device=device)
        out = torch.zeros((batch_size, steps, train_out_trajs[0].shape[1]), dtype=torch.float32, device=device)

        step_idx_left  = [random.randint(0, train_in_trajs[tl].shape[0] - window - 1) for tl in traj_idx_left]
        step_idx_right = [random.randint(0, train_in_trajs[tr].shape[0] - window - 1) for tr in traj_idx_right]
        for j, (tl, tr, sli, sri) in enumerate(zip(traj_idx_left, traj_idx_right, step_idx_left, step_idx_right)):
            # --- joint positions: left arm from (tl, sli), right arm from (tr, sri) ---
            inp[j, :, :self.n_q]           = train_in_trajs[tl][sli:sli+window:stride, :self.n_q]
            inp[j, :, self.n_q:2*self.n_q] = train_in_trajs[tr][sri:sri+window:stride, self.n_q:2*self.n_q]
            # --- conditioning (goal / symmetry): constant per demo, take from left ---
            inp[j, :, 2*self.n_q:] = train_in_trajs[tl][sli:sli+window:stride, 2*self.n_q:]

            # --- joint velocities: left from (tl, sli), right from (tr, sri) ---
            out[j, :, :self.n_q]           = train_out_trajs[tl][sli:sli+window:stride, :self.n_q]
            out[j, :, self.n_q:2*self.n_q] = train_out_trajs[tr][sri:sri+window:stride, self.n_q:2*self.n_q]

        return inp, out
    
    def get_batch_decoupled_symmetry_conditioned(self, train_in_trajs, train_out_trajs, steps, stride):
        '''
        Independently samples TRAJECTORY index AND starting time-step for each arm,
        then concatenates them into a single batch element.
        This gives O(n^2 * T^2) effective training pairs instead of O(n * T^2).

        For n=7 demos, T≈970: ~46 M unique pairs vs ~6.6 M with shared traj_idx.

        Input layout:  [q_left (n_q) | q_right (n_q) | conditioning (n_x*2)]
        Output layout: [dq_left(n_q) | dq_right(n_q)]
        '''
        batch_size = 250
        n = len(train_in_trajs)
        window = steps * stride  # span in original trajectory indices
        # Independent trajectory index per arm
        traj_idx_left  = [random.randint(0, n - 1) for _ in range(batch_size)]
        traj_idx_right = traj_idx_left  # symmetry conditioning: same trajectory index for both arms
        inp = torch.zeros((batch_size, steps, train_in_trajs[0].shape[1]), dtype=torch.float32, device=device)
        out = torch.zeros((batch_size, steps, train_out_trajs[0].shape[1]), dtype=torch.float32, device=device)

        step_idx_left  = [random.randint(0, train_in_trajs[tl].shape[0] - window - 1) for tl in traj_idx_left]
        step_idx_right = [random.randint(0, train_in_trajs[tr].shape[0] - window - 1) for tr in traj_idx_right]
        for j, (tl, tr, sli, sri) in enumerate(zip(traj_idx_left, traj_idx_right, step_idx_left, step_idx_right)):
            # --- joint positions: left arm from (tl, sli), right arm from (tr, sri) ---
            inp[j, :, :self.n_q]           = train_in_trajs[tl][sli:sli+window:stride, :self.n_q]
            inp[j, :, self.n_q:2*self.n_q] = train_in_trajs[tr][sri:sri+window:stride, self.n_q:2*self.n_q]
            # --- conditioning (goal / symmetry): constant per demo, take from left ---
            inp[j, :, 2*self.n_q:] = train_in_trajs[tl][sli:sli+window:stride, 2*self.n_q:]

            # --- joint velocities: left from (tl, sli), right from (tr, sri) ---
            out[j, :, :self.n_q]           = train_out_trajs[tl][sli:sli+window:stride, :self.n_q]
            out[j, :, self.n_q:2*self.n_q] = train_out_trajs[tr][sri:sri+window:stride, self.n_q:2*self.n_q]

        return inp, out

    def get_batch_decoupled_v2(self, train_in_trajs, train_out_trajs, steps):
        '''
        Inspired by the mlp_robot_lasa_generic.py script.
        Independently samples a starting time-step for the left arm (sli) and
        the right arm (sri), then concatenates them into a single batch element.
        This gives O(T^2) effective training pairs per trajectory instead of O(T).

        Input layout:  [q_left (n_q) | q_right (n_q) | conditioning (n_x*2)]
        Output layout: [dq_left(n_q) | dq_right(n_q)]
        '''
        batch_size = 250
        n = len(train_in_trajs)
        traj_idx = [random.randint(0, n - 1) for _ in range(batch_size)]
        inp = torch.zeros((batch_size, steps, train_in_trajs[0].shape[1]), dtype=torch.float32, device=device)
        out = torch.zeros((batch_size, steps, train_out_trajs[0].shape[1]), dtype=torch.float32, device=device)

        step_idx_left  = [random.randint(0, train_in_trajs[ti].shape[0] - steps - 1) for ti in traj_idx]
        step_idx_right = [random.randint(0, train_in_trajs[ti].shape[0] - steps - 1) for ti in traj_idx]
        for j, (ti, sli, sri) in enumerate(zip(traj_idx, step_idx_left, step_idx_right)):
            # --- joint positions: left arm from sli, right arm from sri ---
            inp[j, :, :self.n_q]           = train_in_trajs[ti][sli:sli+steps, :self.n_q]
            inp[j, :, self.n_q:2*self.n_q] = train_in_trajs[ti][sri:sri+steps, self.n_q:2*self.n_q]
            # --- conditioning (goal / symmetry): constant per demo, either step works ---
            inp[j, :, 2*self.n_q:] = train_in_trajs[ti][sli:sli+steps, 2*self.n_q:]

            # --- joint velocities: left from sli, right from sri ---
            out[j, :, :self.n_q]           = train_out_trajs[ti][sli:sli+steps, :self.n_q]
            out[j, :, self.n_q:2*self.n_q] = train_out_trajs[ti][sri:sri+steps, self.n_q:2*self.n_q]

        return inp, out    
        
    def get_batch_decoupled_v1(self, train_in_trajs, train_out_trajs, traj_index, step_index_left, step_index_right, steps):
        # Pre-stack all trajectories once (safe because all trajectories share the same length)
        if not hasattr(self, '_traj_stack_cache'):
            self._train_in_stack  = torch.stack(train_in_trajs)   # (N_traj, T, d_in)
            self._train_out_stack = torch.stack(train_out_trajs)  # (N_traj, T, d_out)
            self._traj_stack_cache = True

        # For each trajectory in the batch, we have a different random starting index for the left and right arm, so we need to index them separately and then concatenate the results to form the input and output batches.
        offsets = torch.arange(steps, device=device)  # (steps,)
        li = step_index_left[:, None]  + offsets      # (B, steps)
        ri = step_index_right[:, None] + offsets      # (B, steps)
        ti = traj_index[:, None]                      # (B, 1) — broadcasts with li/ri

        left_win  = self._train_in_stack[ti, li]      # (B, steps, d_in)
        right_win = self._train_in_stack[ti, ri]      # (B, steps, d_in)

        inp_batch = torch.cat([
            left_win[...,  :self.n_q],
            right_win[..., self.n_q:2*self.n_q],
            left_win[...,  2*self.n_q:2*self.n_q+self.n_x],
            right_win[..., 2*self.n_q+self.n_x:],
        ], dim=-1)

        out_left_win  = self._train_out_stack[ti, li]  # (B, steps, d_out)
        out_right_win = self._train_out_stack[ti, ri]

        desired_out_batch = torch.cat([
            out_left_win[...,  :self.n_q],
            out_right_win[..., self.n_q:2*self.n_q],
        ], dim=-1)

        return inp_batch, desired_out_batch

    def train_step(self, inp, desired_out, robot, steps=5, stride=1, task_space_loss=False):
        self.optimizer.zero_grad()
        lossval = self.loss(inp, desired_out, robot, steps=steps, stride=stride, task_space_loss=task_space_loss)
        lossval.backward()
        self.optimizer.step()
        return lossval

    def loss(self, inp_0, desired_out, robot, steps=5, stride=1, task_space_loss=False, λ_pos=1e5, λ_task=1e3):
        loss = self.compute_imitation_loss(inp_0, desired_out, robot, steps=steps, stride=stride, task_space_loss=task_space_loss, λ_pos=λ_pos, λ_task=λ_task)
        return loss
    
    def compute_imitation_loss(self, inp_0, desired_out, robot, steps=5, stride=1, task_space_loss=False, λ_vel=1e1, λ_pos=1e5, λ_task=1e3):
        '''
        Computes theh imitation loss over multiple steps by propagating the dynamics using the network predictions and comparing to the desired trajectory over multiple steps.

        loss_imitation = λ_vel*loss_velocity + λ_pos*loss_position

        loss_velocity = MSE(f(q_t_normalized), dq_bar_dot_bar). dq_bar_dot_bar normalized desired velocity, where the velocity itself is the finite difference between the two consequtive normalized q states.
        loss_position = MSE(q_t+1_normalized,
                            q_t_normalized + denorm(f(q_t_normalized))*dt). The denorm of f(q_t_normalized) is done w.r.t. Dq_min and Dq_max.
        '''
        inp_t = inp_0[:, 0, :]  # (B, d_in) — first state in the sequence
        inp_x = inp_t[:, 2*self.n_q:]  # task-space target is constant throughout the rollout

        loss = 0

        for i in range(steps - 1):
            pred_Vq = self.model(inp_t)
            pred_Vq_denorm = denormalize_state(state=pred_Vq, x_min=self.demonstrations["Dq_min"], x_max=self.demonstrations["Dq_max"])

            next_q_t_norm = inp_t[:, :2*self.n_q] + pred_Vq_denorm * (self.dt * stride)
            inp_t = torch.cat([next_q_t_norm, inp_x], dim=1)

            loss += λ_vel*self.mse_loss(pred_Vq, desired_out[:, i])
            loss += λ_pos*self.mse_loss(next_q_t_norm, inp_0[:, i+1, :2*self.n_q])

        loss = loss / (steps - 1)

        if task_space_loss:
            λ_task = 5e-1
            loss += λ_task * self.compute_taskspace_imitation_loss(inp_0, desired_out, robot, steps=steps)

        return loss

    def compute_taskspace_imitation_loss(self, inp_0, desired_out, robot, steps=5, λ_vel=1e1, λ_pos=1e5):
        '''
        Computes the imitation loss in task space over multiple steps.

        At each step the predicted joint velocities are integrated to get the next joint configuration,
        which is then mapped to task space via differentiable FK. Both task-space velocity and
        position losses are computed.

        loss_taskspace = λ_vel * MSE(x_dot_pred, x_dot_desired) + λ_pos * MSE(x_next_pred, x_next_desired)

        where x = fk(q) is the Cartesian end-effector position.
        '''
        inp_t = inp_0[:, 0, :]
        inp_x = inp_t[:, 2*self.n_q:]

        _q_min = torch.tensor(self.demonstrations["Q_min"], dtype=torch.float32, device=inp_0.device)
        _q_max = torch.tensor(self.demonstrations["Q_max"], dtype=torch.float32, device=inp_0.device)

        def to_task(q_norm):
            """Denormalize (B, 2*n_q) normalized joint angles → (B, 2*n_x) task-space positions via FK."""
            q_phys = (q_norm + 1.0) / 2.0 * (_q_max - _q_min) + _q_min
            x_left  = robot.fk_func_left_torch(q_phys[:, :self.n_q])
            x_right = robot.fk_func_right_torch(q_phys[:, self.n_q:])
            return torch.cat([x_left, x_right], dim=1)

        loss = 0

        for i in range(steps - 1):
            pred_Vq = self.model(inp_t)
            pred_Vq_denorm = denormalize_state(state=pred_Vq, x_min=self.demonstrations["Dq_min"], x_max=self.demonstrations["Dq_max"])

            next_q_t_norm = inp_t[:, :2*self.n_q] + pred_Vq_denorm * self.dt
            inp_t = torch.cat([next_q_t_norm, inp_x], dim=1)

            # Task-space positions for predicted and desired next state
            x_next_pred    = to_task(next_q_t_norm)
            x_next_desired = to_task(inp_0[:, i+1, :2*self.n_q])
            loss += λ_pos * self.mse_loss(x_next_pred, x_next_desired)

            # Task-space velocity: finite difference in task space over dt
            x_curr         = to_task(inp_0[:, i, :2*self.n_q])
            Vx_pred        = (x_next_pred    - x_curr) / self.dt
            Vx_desired     = (x_next_desired - x_curr) / self.dt
            loss += λ_vel * self.mse_loss(Vx_pred, Vx_desired)

        loss = loss / (steps - 1)
        return loss

    def forward_denorm(self, inp_batch=None, use_grad=True):
        '''
        Description
        -----------
        This method performs a single forward pass through the network and returns the output in the denormalized scale (scale of the original demonstrations).
        It expects the input to be in the denormalized scale (scale of the original demonstrations).

        inp_denorm -> inp_norm -> pred_Vq -> pred_Vq_denorm -> pred_Vq_denorm_denorm
        '''
        ### Step 1: Normalize the input data.
        inp_q_norm = normalize_state(state=inp_batch[:, :2*self.n_q], x_min=self.demonstrations["Q_min"], x_max=self.demonstrations["Q_max"])
        if self.goal_conditioned:
            inp_x_norm = normalize_state(state=inp_batch[:, 2*self.n_q:], x_min=self.demonstrations["X_min"], x_max=self.demonstrations["X_max"])
        else:
            # just copy it
            inp_x_norm = inp_batch[:, 2*self.n_q:]
        inp_batch = torch.hstack([inp_q_norm, inp_x_norm])

        
        ### Step 2: Pass through the network
        if not use_grad:
            with torch.no_grad():
                pred_Vq = self.model(inp_batch)
        else:
            pred_Vq = self.model(inp_batch)

        ### Step 3: Denormalize the output using the network output scales. (denormalize_state(..., x_min=Dq_min, x_max=Dq_max))
        pred_Vq_denorm = denormalize_state(state=pred_Vq, x_min=self.demonstrations["Dq_min"], x_max=self.demonstrations["Dq_max"])

        ### Step 4: Further denormalize the denormalized output to bring it back to the scale of the original data. (denormalize_state_derivative(..., x_min=Q_min, x_max=Q_max))
        pred_Vq_denorm_denorm = denormalize_state_derivative(state_dot=pred_Vq_denorm, x_min=self.demonstrations["Q_min"], x_max=self.demonstrations["Q_max"])

        return pred_Vq_denorm_denorm

    def forward_denorm_multistep(self, horizon=1, inp_batch=None, return_traj=False, use_grad=True):
        '''
        Description
        -----------
        This method performs a multistep forward pass through the network integrating along its own predictions (using Euler integration).
        It expects the input to be in the denormalized scale (scale of the original demonstrations).
        The output is in the denormalized scale (scale of the original demonstrations).
        '''
        # Ensure inp_batch has a batch dimension
        if inp_batch.dim() == 1:
            inp_batch = inp_batch.unsqueeze(0) # Add a batch dimension if the input is a single data point
            
        ### Main Loop
        inp_t = inp_batch

        # Placeholders for the entire predicted trajectory (both the states and the velocities) if needed for visualization purposes.
        predicted_qs = [inp_t[:, :2*self.n_q]]
        predicted_dqs = []
        
        dt = 1
        horizon = int(horizon / dt)
        for _ in range(horizon):
            ### Step 1: Forward pass through the network to get the predicted velocity in the denormalized scale.
            pred_Vq_denorm_denorm = self.forward_denorm(inp_t, use_grad=use_grad)

            ### Step 2: Propagate the dynamics using Euler integration in the denormalized scale.
            inp_q_t = inp_t[:, :2*self.n_q]
            # if self.goal_conditioned
            inp_x_t = inp_t[:, 2*self.n_q:]
            next_q_t = inp_q_t + pred_Vq_denorm_denorm * dt
            inp_t = torch.hstack([next_q_t, inp_x_t])

            ### Step 3: Append the predicted next state and velocity to the trajectory lists.
            predicted_qs.append(next_q_t)
            predicted_dqs.append(pred_Vq_denorm_denorm)
            
        predicted_qs = torch.stack(predicted_qs, dim=1) # Shape: (batch_size, horizon+1, 2*n_q)
        predicted_dqs = torch.stack(predicted_dqs, dim=1) # Shape: (batch_size, horizon, 2*n_q)

        if return_traj:
            return predicted_qs, predicted_dqs
        else:
            return predicted_qs[:, -1, :2*self.n_q], predicted_dqs[:, -1, :2*self.n_q] # Return the final predicted state and velocity after the multistep integration.


class MLP(torch.nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim):
        super().__init__()
        # Select activation function
        self.activation = torch.nn.GELU()
        
        # Initialize dynamical system decoder layers: phi
        self.l_1 = torch.nn.Linear(in_dim, hidden_dim)
        self.norm_l_1 = torch.nn.LayerNorm(hidden_dim)
        self.l_2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.norm_l_2 = torch.nn.LayerNorm(hidden_dim)
        self.l_3 = torch.nn.Linear(hidden_dim, out_dim)
        
    def forward(self, x):
        l_1 = self.activation(self.norm_l_1(self.l_1(x)))
        l_2 = self.activation(self.norm_l_2(self.l_2(l_1)))
        l_3 = self.l_3(l_2)
        
        return l_3

class CustomMLP(CustomNetworks):
    def __init__(self, rep_in=None, rep_out=None, group=None, model_id='default', load_model_flag=False, save_model_flag=True,
                 demonstrations=None, robot=None):

        super().__init__(model_id, load_model_flag, save_model_flag,
                         demonstrations=demonstrations, robot=robot)

        inp_dim = demonstrations['train_in'][0].shape[1]
        out_dim = demonstrations['train_out'][0].shape[1]
        self.model = MLP(in_dim=inp_dim, out_dim=out_dim, hidden_dim=300).to(device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-4)

        self.rep_in = rep_in
        self.rep_out = rep_out
        self.G = group

        if self.load_model_flag:
            self.load_model(model_id=self.model_id)
































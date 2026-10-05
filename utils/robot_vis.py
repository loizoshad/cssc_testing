from typing import Union
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.path as mpath
import matplotlib.patches as mpatches
from matplotlib.patches import Circle
from matplotlib.colors import to_rgba
import torch



# ── Color palette ─────────────────────────────────────────────────────────────
COLORS = {
    'left_arm':   '#7BA7C7',   # muted steel blue
    'right_arm':  '#C49A7A',   # muted terracotta
    'left_ee':    '#3D6E8F',   # darker blue for EE trace
    'right_ee':   '#8F5A35',   # darker terracotta for EE trace
    'gt_arm':     '#AAAAAA',   # light grey for GT robot body
    'gt_ee_left': '#555555',   # dark grey for GT left EE trace
    'gt_ee_right':'#777777',   # slightly lighter for GT right EE trace
}

# # ── Color palette ─────────────────────────────────────────────────────────────
# # Use these instead of tab10 — perceptually distinct, print-safe
# COLORS = {
#     'left_arm':   '#4A90D9',   # steel blue  — robot body
#     'right_arm':  '#E07B4A',   # burnt sienna — robot body  
#     'left_ee':    '#1A5FA8',   # deep blue   — EE trace (darker → reads on top)
#     'right_ee':   '#B84A15',   # deep orange — EE trace
#     'gt_arm':     '#888888',   # neutral grey for GT shadows
#     'gt_ee':      '#555555',   # darker grey for GT EE trace
# }


def _robot_basis_patches(ax, width: float, facecolor, edgecolor,
                         base_pos: np.ndarray = None) -> list:
    """Draw the robot base. base_pos offsets the whole base."""
    if base_pos is None:
        base_pos = np.zeros(2)
    w  = width * 1.2
    t1 = np.linspace(0, np.pi, 28)
    x  = np.zeros((30, 2))
    x[:, 0] = np.append(np.append(w * 1.5, w * 1.5 * np.cos(t1)), -w * 1.5)
    x[:, 1] = np.append(np.append(-w * 1.2, w * 1.5 * np.sin(t1)), -w * 1.2)
    x += base_pos
    patches = [mpatches.PathPatch(mpath.Path(x), facecolor=facecolor,
                                  edgecolor=edgecolor, linewidth=0.8)]
    ax.add_patch(patches[0])

    xs = np.linspace(-w * 1.2, w * 1.2, 5)
    for xi in xs:
        base = np.array([xi, -w * 1.2]) + base_pos
        tip  = base + np.array([-0.125, -0.25])
        p    = mpatches.PathPatch(mpath.Path([base, tip]),
                                  facecolor=facecolor, edgecolor=facecolor,
                                  linewidth=1.5)
        ax.add_patch(p)
        patches.append(p)
    return patches


def _robot_link_patches(ax, cumulative_angle: float, link_length: float,
                        base_pos: np.ndarray, width: float,
                        facecolor, edgecolor,
                        draw_joint_circles: bool = True) -> tuple[list, np.ndarray]:
    """Draw one robot link. Optionally skip joint circles for cleaner shadows."""
    n  = 30
    t1 = np.linspace(0, -np.pi, n // 2)
    t2 = np.linspace(np.pi,  0, n // 2)
    x  = np.zeros((n, 2))
    x[:, 0] = np.append(width * np.sin(t1), link_length + width * np.sin(t2))
    x[:, 1] = np.append(width * np.cos(t1), width * np.cos(t2))
    x  = np.vstack((x, x[0]))

    R  = np.array([[np.cos(cumulative_angle), -np.sin(cumulative_angle)],
                   [np.sin(cumulative_angle),  np.cos(cumulative_angle)]])
    x  = (R @ x.T).T + base_pos
    ee = R @ np.array([link_length, 0.0]) + base_pos

    link_patch = mpatches.PathPatch(mpath.Path(x), facecolor=facecolor,
                                    edgecolor=edgecolor, linewidth=1.2)
    ax.add_patch(link_patch)
    patches = [link_patch]

    if draw_joint_circles:
        circle_pts = np.column_stack([
            np.sin(np.linspace(0, 2 * np.pi, n)),
            np.cos(np.linspace(0, 2 * np.pi, n)),
        ]) * width * 0.25
        for center in [base_pos, ee]:
            c = mpatches.PathPatch(mpath.Path(circle_pts + center),
                                   facecolor=edgecolor, edgecolor=facecolor,
                                   linewidth=1.0)
            ax.add_patch(c)
            patches.append(c)

    return patches, ee


def draw_planar_arm(ax, joint_angles: np.ndarray,
                    link_lengths: Union[float, np.ndarray],
                    base_pos: np.ndarray = None,
                    width: float = 0.18,
                    facecolor='black', edgecolor='white',
                    draw_base: bool = True,
                    draw_joint_circles: bool = True) -> list:
    if base_pos is None:
        base_pos = np.zeros(2)
    if np.isscalar(link_lengths):
        link_lengths = np.full(len(joint_angles), link_lengths)

    all_patches = []
    if draw_base:
        all_patches.append(_robot_basis_patches(ax, width, facecolor, edgecolor, base_pos))

    pos = base_pos.copy()
    for i in range(len(joint_angles)):
        patches, pos = _robot_link_patches(
            ax,
            cumulative_angle=float(np.sum(joint_angles[:i+1])),
            link_length=link_lengths[i],
            base_pos=pos,
            width=width,
            facecolor=facecolor,
            edgecolor=edgecolor,
            draw_joint_circles=draw_joint_circles,
        )
        all_patches.append(patches)
    return all_patches


class PlanarRobotVisualizer:
    def __init__(self, robot, colors: dict = None):
        self.robot        = robot
        self.link_lengths = robot.arm_length
        self.n_q_left     = robot.nb_dofs_left
        self.n_q_right    = robot.nb_dofs_right
        self.n_x_left     = robot.nb_x_left
        self.n_x_right    = robot.nb_x_right
        self.colors       = colors or COLORS

    def _to_numpy(self, x):
        return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

    def _split_traj(self, traj: np.ndarray) -> tuple:
        n_q = self.n_q_left + self.n_q_right
        n_x = self.n_x_left + self.n_x_right
        q_left  = traj[:, :self.n_q_left]
        q_right = traj[:, self.n_q_left:n_q]
        if traj.shape[1] >= n_q + n_x:
            x_left  = traj[:, n_q:n_q + self.n_x_left]
            x_right = traj[:, n_q + self.n_x_left:n_q + n_x]
        else:
            x_left  = np.array([self.robot.fk_func_left(q)[:2]  for q in q_left])
            x_right = np.array([self.robot.fk_func_right(q)[:2] for q in q_right])
        return q_left, q_right, x_left, x_right

    def _style(self, ax, title: str = '', legend: bool = False):
        ax.set_title(title, fontsize=13, pad=8)
        ax.set_aspect('equal')
        ax.margins(0.12)
        ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        if legend:
            ax.legend(fontsize=9, loc='upper right',
                      framealpha=0.9, edgecolor='none')

    # ── Primitives ────────────────────────────────────────────────────────────

    def draw_arm_pose(self, ax, q: np.ndarray,
                      facecolor='black', edgecolor='white',
                      width: float = 0.18, alpha: float = 1.0,
                      draw_base: bool = True,
                      draw_joint_circles: bool = True):
        fc = (*to_rgba(facecolor)[:3], alpha)
        ec = (*to_rgba(edgecolor)[:3], alpha)
        draw_planar_arm(ax, q, self.link_lengths, width=width,
                        facecolor=fc, edgecolor=ec,
                        draw_base=draw_base,
                        draw_joint_circles=draw_joint_circles)

    def draw_arm_shadows(self, ax, q_traj: np.ndarray,
                         n_shadows: int = 6,
                         facecolor='black', edgecolor='white',
                         width: float = 0.18,
                         draw_base: bool = False):
        """
        Ghost poses along the trajectory.
        Base is hidden by default — much cleaner for shadows.
        """
        T       = q_traj.shape[0]
        indices = np.linspace(0, T - 1, n_shadows, dtype=int)
        for rank, t in enumerate(indices):
            alpha = 0.08 + 0.55 * (rank / max(n_shadows - 1, 1))
            self.draw_arm_pose(ax, q_traj[t],
                               facecolor=facecolor, edgecolor=edgecolor,
                               width=width, alpha=alpha,
                               draw_base=draw_base,
                               draw_joint_circles=False)   # cleaner without circles
        # Draw final pose fully with base + circles
        self.draw_arm_pose(ax, q_traj[-1],
                           facecolor=facecolor, edgecolor=edgecolor,
                           width=width, alpha=0.9,
                           draw_base=draw_base,
                           draw_joint_circles=True)

    def draw_ee_trace(self, ax, x_traj: np.ndarray,
                      color='purple', alpha: float = 0.9,
                      linewidth: float = 2.0, label: str = None):
        ax.plot(x_traj[:, 0], x_traj[:, 1],
                color=color, alpha=alpha, linewidth=linewidth,
                label=label, zorder=4)

    def draw_ee_trace_faded(self, ax, x_traj: np.ndarray,
                            color='purple', linewidth: float = 2.0,
                            linestyle: str = '-',
                            label: str = None):
        T = x_traj.shape[0]
        for t in range(T - 1):
            alpha = 0.2 + 0.8 * (t / T)
            ax.plot(x_traj[t:t+2, 0], x_traj[t:t+2, 1],
                    color=color, alpha=alpha, linewidth=linewidth,
                    linestyle=linestyle, zorder=4)
        if label:
            ax.plot([], [], color=color, linewidth=linewidth,
                    linestyle=linestyle, label=label)

    def draw_start_end_markers(self, ax, x_traj: np.ndarray,
                                color='purple', size: float = 50.0):
        ax.scatter(*x_traj[0,  :2], s=size,       color=color,
                   zorder=6, marker='o', edgecolors='white', linewidths=0.8)
        ax.scatter(*x_traj[-1, :2], s=size * 1.5, color=color,
                   zorder=6, marker='*', edgecolors='white', linewidths=0.5)

    def draw_ee_point(self, ax, x: np.ndarray,
                      color='seagreen', radius: float = 0.02):
        ax.add_patch(Circle(x[:2], radius, color=color, alpha=0.9, zorder=6))

    # ── Trajectory-level ──────────────────────────────────────────────────────

    def draw_trajectory(self, ax, traj,
                        color_left: str = None, color_right: str = None,
                        label: str = None,
                        n_shadows: int = 6,
                        faded_trace: bool = True,
                        show_start_end: bool = True,
                        arm_width: float = 0.18,
                        draw_base: bool = False):
        traj = self._to_numpy(traj)
        q_left, q_right, x_left, x_right = self._split_traj(traj)

        c_left  = color_left  or self.colors['left_arm']
        c_right = color_right or self.colors['right_arm']
        ce_left  = self.colors['left_ee']
        ce_right = self.colors['right_ee']

        # Robot shadows — muted arm color, no base, no joint circles
        self.draw_arm_shadows(ax, q_left,  n_shadows=n_shadows,
                              facecolor=c_left,  width=arm_width,
                              draw_base=draw_base)
        self.draw_arm_shadows(ax, q_right, n_shadows=n_shadows,
                              facecolor=c_right, width=arm_width,
                              draw_base=draw_base)

        # EE traces — darker color, drawn on top (higher zorder)
        trace = self.draw_ee_trace_faded if faded_trace else self.draw_ee_trace
        trace(ax, x_left[:, :2],  color=ce_left,
              label=f'{label} (L)' if label else None)
        trace(ax, x_right[:, :2], color=ce_right,
              label=f'{label} (R)' if label else None)

        if show_start_end:
            self.draw_start_end_markers(ax, x_left[:, :2],  color=ce_left)
            self.draw_start_end_markers(ax, x_right[:, :2], color=ce_right)

    # ── Top-level ─────────────────────────────────────────────────────────────

    def plot_trajectories_v1(self, trajectories: list,
                          color_pairs: list = None,
                          labels: list = None,
                          n_shadows: int = 6,
                          faded_trace: bool = True,
                          show_start_end: bool = True,
                          arm_width: float = 0.18,
                          draw_base: bool = False,
                          title: str = 'Robot Trajectories',
                          ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 8))

        cmap = plt.cm.get_cmap('tab10')
        if color_pairs is None:
            color_pairs = [(cmap(i * 2 % 10), cmap((i * 2 + 1) % 10))
                           for i in range(len(trajectories))]

        for i, traj in enumerate(trajectories):
            c_left, c_right = color_pairs[i]
            self.draw_trajectory(
                ax, traj,
                color_left=c_left, color_right=c_right,
                label=labels[i] if labels else None,
                n_shadows=n_shadows,
                faded_trace=faded_trace,
                show_start_end=show_start_end,
                arm_width=arm_width,
                draw_base=draw_base,
            )

        self._style(ax, title=title, legend=labels is not None)
        return ax


    def draw_trajectory(self, ax, traj,
                        traj_predicted=None,
                        label: str = None,
                        n_shadows: int = 6,
                        faded_trace: bool = True,
                        show_start_end: bool = True,
                        arm_width: float = 0.18,
                        draw_base: bool = False):
        """
        Draw a nominal trajectory (always) and optionally a predicted one on top.
        Nominal → grey robot shadows + dashed grey EE trace.
        Predicted → colored robot shadows + solid colored EE trace.
        """
        traj = self._to_numpy(traj)
        q_left, q_right, x_left, x_right = self._split_traj(traj)

        # ── Nominal (grey, dashed) ────────────────────────────────────────────
        self.draw_arm_shadows(ax, q_left,  n_shadows=n_shadows,
                            facecolor=self.colors['gt_arm'], width=arm_width,
                            draw_base=draw_base)
        self.draw_arm_shadows(ax, q_right, n_shadows=n_shadows,
                            facecolor=self.colors['gt_arm'], width=arm_width,
                            draw_base=draw_base)
        self.draw_ee_trace_faded(ax, x_left[:, :2],
                                color=self.colors['gt_ee_left'],
                                linestyle='--', label='nominal (L)')
        self.draw_ee_trace_faded(ax, x_right[:, :2],
                                color=self.colors['gt_ee_right'],
                                linestyle='--', label='nominal (R)')
        if show_start_end:
            self.draw_start_end_markers(ax, x_left[:, :2],
                                        color=self.colors['gt_ee_left'])
            self.draw_start_end_markers(ax, x_right[:, :2],
                                        color=self.colors['gt_ee_right'])

        # ── Predicted (colored, solid) ────────────────────────────────────────
        if traj_predicted is not None:
            traj_predicted = self._to_numpy(traj_predicted)
            q_left_p, q_right_p, x_left_p, x_right_p = self._split_traj(traj_predicted)

            self.draw_arm_shadows(ax, q_left_p,  n_shadows=n_shadows,
                                facecolor=self.colors['left_arm'], width=arm_width,
                                draw_base=draw_base)
            self.draw_arm_shadows(ax, q_right_p, n_shadows=n_shadows,
                                facecolor=self.colors['right_arm'], width=arm_width,
                                draw_base=draw_base)

            trace = self.draw_ee_trace_faded if faded_trace else self.draw_ee_trace
            trace(ax, x_left_p[:, :2],
                color=self.colors['left_ee'],
                label=f'{label} (L)' if label else 'predicted (L)')
            trace(ax, x_right_p[:, :2],
                color=self.colors['right_ee'],
                label=f'{label} (R)' if label else 'predicted (R)')
            if show_start_end:
                self.draw_start_end_markers(ax, x_left_p[:, :2],
                                            color=self.colors['left_ee'])
                self.draw_start_end_markers(ax, x_right_p[:, :2],
                                            color=self.colors['right_ee'])



    def draw_gt_trajectory_v1(self, ax, traj,
                            n_shadows: int = 6,
                            faded_trace: bool = True,
                            show_start_end: bool = True,
                            arm_width: float = 0.18,
                            draw_base: bool = True):
        """
        Draw the ground truth / nominal trajectory.
        Uses neutral grey tones so it reads as a reference without
        competing with the predicted trajectory colors.
        """
        traj = self._to_numpy(traj)
        q_left, q_right, x_left, x_right = self._split_traj(traj)

        c_arm     = self.colors['gt_arm']
        c_ee_left  = self.colors['gt_ee_left']
        c_ee_right = self.colors['gt_ee_right']

        # Robot shadows in grey
        self.draw_arm_shadows(ax, q_left,  n_shadows=n_shadows,
                            facecolor=c_arm, width=arm_width,
                            draw_base=draw_base)
        self.draw_arm_shadows(ax, q_right, n_shadows=n_shadows,
                            facecolor=c_arm, width=arm_width,
                            draw_base=draw_base)

        # # EE traces — dashed so predictions (solid) read on top
        # trace_fn = self.draw_ee_trace_faded if faded_trace else self.draw_ee_trace
        # trace_fn(ax, x_left[:, :2],  color=c_ee_left,  label='GT (L)')
        # trace_fn(ax, x_right[:, :2], color=c_ee_right, label='GT (R)')
        #
        self.draw_ee_trace_faded(ax, x_left[:, :2],  color=c_ee_left,
                                linestyle='--', label='GT (L)')
        self.draw_ee_trace_faded(ax, x_right[:, :2], color=c_ee_right,
                                linestyle='--', label='GT (R)')    

        if show_start_end:
            self.draw_start_end_markers(ax, x_left[:, :2],  color=c_ee_left)
            self.draw_start_end_markers(ax, x_right[:, :2], color=c_ee_right)

    def draw_gt_trajectory(self, ax, traj,
                            n_shadows: int = 6,
                            faded_trace: bool = True,
                            show_start_end: bool = True,
                            arm_width: float = 0.18,
                            draw_base: bool = True):
        """
        Draw the ground truth / nominal trajectory in neutral grey.
        Shows both joint space (robot shadows) and task space (EE trace).
        """
        traj = self._to_numpy(traj)
        q_left, q_right, x_left, x_right = self._split_traj(traj)

        c_arm      = self.colors['gt_arm']
        c_ee_left  = self.colors['gt_ee_left']
        c_ee_right = self.colors['gt_ee_right']

        # Joint space: robot shadows in grey
        self.draw_arm_shadows(ax, q_left,  n_shadows=n_shadows,
                            facecolor=c_arm, width=arm_width,
                            draw_base=draw_base)
        self.draw_arm_shadows(ax, q_right, n_shadows=n_shadows,
                            facecolor=c_arm, width=arm_width,
                            draw_base=draw_base)

        # Task space: dashed EE traces in grey
        self.draw_ee_trace_faded(ax, x_left[:, :2],  color=c_ee_left,
                                linestyle='--', label='GT (L)')
        self.draw_ee_trace_faded(ax, x_right[:, :2], color=c_ee_right,
                                linestyle='--', label='GT (R)')

        if show_start_end:
            self.draw_start_end_markers(ax, x_left[:, :2],  color=c_ee_left)
            self.draw_start_end_markers(ax, x_right[:, :2], color=c_ee_right)











#################################### V1

# # ─────────────────────────────────────────────────────────────────────────────
# # Low-level robot drawing (kinematically correct)
# # ─────────────────────────────────────────────────────────────────────────────

# def _robot_basis_patches(ax, width: float, facecolor, edgecolor) -> list:
#     """Draw the robot base and return its patches."""
#     w  = width * 1.2
#     t1 = np.linspace(0, np.pi, 28)
#     x  = np.zeros((30, 2))
#     x[:, 0] = np.append(np.append(w * 1.5, w * 1.5 * np.cos(t1)), -w * 1.5)
#     x[:, 1] = np.append(np.append(-w * 1.2, w * 1.5 * np.sin(t1)), -w * 1.2)
#     patches = [mpatches.PathPatch(mpath.Path(x), facecolor=facecolor,
#                                   edgecolor=edgecolor, linewidth=1)]
#     ax.add_patch(patches[0])

#     xs = np.linspace(-w * 1.2, w * 1.2, 5)
#     for xi in xs:
#         base = np.array([xi, -w * 1.2])
#         tip  = base + np.array([-0.125, -0.25])
#         p    = mpatches.PathPatch(mpath.Path([base, tip]),
#                                   facecolor=facecolor, edgecolor=facecolor, linewidth=2)
#         ax.add_patch(p)
#         patches.append(p)
#     return patches


# def _robot_link_patches(ax, cumulative_angle: float, link_length: float,
#                         base_pos: np.ndarray, width: float,
#                         facecolor, edgecolor) -> tuple[list, np.ndarray]:
#     """Draw one robot link and return (patches, end_effector_position)."""
#     n  = 30
#     t1 = np.linspace(0, -np.pi, n // 2)
#     t2 = np.linspace(np.pi,  0, n // 2)
#     x  = np.zeros((n, 2))
#     x[:, 0] = np.append(width * np.sin(t1), link_length + width * np.sin(t2))
#     x[:, 1] = np.append(width * np.cos(t1), width * np.cos(t2))
#     x  = np.vstack((x, x[0]))

#     R    = np.array([[np.cos(cumulative_angle), -np.sin(cumulative_angle)],
#                      [np.sin(cumulative_angle),  np.cos(cumulative_angle)]])
#     x    = (R @ x.T).T + base_pos
#     ee   = R @ np.array([link_length, 0.0]) + base_pos

#     link_patch = mpatches.PathPatch(mpath.Path(x), facecolor=facecolor,
#                                     edgecolor=edgecolor, linewidth=2)
#     ax.add_patch(link_patch)

#     circle_pts = np.column_stack([
#         np.sin(np.linspace(0, 2 * np.pi, n)),
#         np.cos(np.linspace(0, 2 * np.pi, n)),
#     ]) * width * 0.2

#     c1 = mpatches.PathPatch(mpath.Path(circle_pts + base_pos),
#                              facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
#     c2 = mpatches.PathPatch(mpath.Path(circle_pts + ee),
#                              facecolor=facecolor, edgecolor=edgecolor, linewidth=2)
#     ax.add_patch(c1)
#     ax.add_patch(c2)
#     return [link_patch, c1, c2], ee


# def draw_planar_arm(ax, joint_angles: np.ndarray,
#                     link_lengths: Union[float, np.ndarray],
#                     base_pos: np.ndarray = None,
#                     width: float = 0.2,
#                     facecolor='black', edgecolor='white') -> list:
#     """
#     Draw a full planar arm (base + all links) and return all patches.

#     Parameters
#     ----------
#     base_pos : (2,) position of the arm base in world frame. Defaults to origin.
#     """
#     if base_pos is None:
#         base_pos = np.zeros(2)
#     if np.isscalar(link_lengths):
#         link_lengths = np.full(len(joint_angles), link_lengths)

#     all_patches = [_robot_basis_patches(ax, width, facecolor, edgecolor)]
#     pos = base_pos.copy()
#     for i, (angle, length) in enumerate(zip(joint_angles, link_lengths)):
#         patches, pos = _robot_link_patches(
#             ax, cumulative_angle=float(np.sum(joint_angles[:i+1])),
#             link_length=length, base_pos=pos,
#             width=width, facecolor=facecolor, edgecolor=edgecolor,
#         )
#         all_patches.append(patches)
#     return all_patches


# # ─────────────────────────────────────────────────────────────────────────────
# # Visualizer class
# # ─────────────────────────────────────────────────────────────────────────────

# class PlanarRobotVisualizer:
#     def __init__(self, robot):
#         self.robot       = robot
#         self.link_lengths = robot.arm_length   # scalar or array
#         self.n_q_left    = robot.nb_dofs_left
#         self.n_q_right   = robot.nb_dofs_right
#         self.n_x_left    = robot.nb_x_left
#         self.n_x_right   = robot.nb_x_right

#     # ── Helpers ──────────────────────────────────────────────────────────────

#     def _to_numpy(self, x):
#         return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else np.asarray(x)

#     def _split_traj(self, traj: np.ndarray) -> tuple:
#         """Split (T, n_q+n_x) into (q_left, q_right, x_left, x_right)."""
#         n_q = self.n_q_left + self.n_q_right
#         n_x = self.n_x_left + self.n_x_right
#         q_left  = traj[:, :self.n_q_left]
#         q_right = traj[:, self.n_q_left:n_q]
#         if traj.shape[1] >= n_q + n_x:
#             x_left  = traj[:, n_q:n_q + self.n_x_left]
#             x_right = traj[:, n_q + self.n_x_left:n_q + n_x]
#         else:
#             x_left  = np.array([self.robot.fk_func_left(q)[:2]  for q in q_left])
#             x_right = np.array([self.robot.fk_func_right(q)[:2] for q in q_right])
#         return q_left, q_right, x_left, x_right

#     def _style(self, ax, title: str = '', legend: bool = False):
#         ax.set_title(title, fontsize=14)
#         ax.set_aspect('equal')
#         ax.margins(0.1)
#         ax.set_xticks([]); ax.set_yticks([])
#         for spine in ax.spines.values():
#             spine.set_visible(False)
#         if legend:
#             ax.legend(fontsize=9, loc='upper right')

#     # ── Atomic drawing primitives ─────────────────────────────────────────────

#     def draw_arm_pose(self, ax, q: np.ndarray,
#                     facecolor='black', edgecolor='white',
#                     width: float = 0.2, alpha: float = 1.0):
#         """Draw a single arm pose using the physically correct kinematics."""
#         fc = (*to_rgba(facecolor)[:3], alpha)
#         ec = (*to_rgba(edgecolor)[:3], alpha)
#         draw_planar_arm(ax, q, self.link_lengths,
#                         width=width, facecolor=fc, edgecolor=ec)

#     def draw_arm_shadows(self, ax, q_traj: np.ndarray,
#                          n_shadows: int = 8,
#                          facecolor='black', edgecolor='white',
#                          width: float = 0.2):
#         """Draw n evenly-spaced ghost poses along a joint trajectory."""
#         T       = q_traj.shape[0]
#         indices = np.linspace(0, T - 1, n_shadows, dtype=int)
#         for rank, t in enumerate(indices):
#             alpha = 0.1 + 0.7 * (rank / max(n_shadows - 1, 1))
#             self.draw_arm_pose(ax, q_traj[t], facecolor=facecolor,
#                                edgecolor=edgecolor, width=width, alpha=alpha)

#     def draw_ee_trace(self, ax, x_traj: np.ndarray,
#                       color='purple', alpha: float = 0.8,
#                       linewidth: float = 1.5, label: str = None):
#         """Draw end-effector trace as a plain line."""
#         ax.plot(x_traj[:, 0], x_traj[:, 1],
#                 color=color, alpha=alpha, linewidth=linewidth, label=label)

#     def draw_ee_trace_faded(self, ax, x_traj: np.ndarray,
#                              color='purple', linewidth: float = 1.5,
#                              label: str = None):
#         """Draw end-effector trace with opacity increasing over time."""
#         T = x_traj.shape[0]
#         for t in range(T - 1):
#             alpha = 0.15 + 0.85 * (t / T)
#             ax.plot(x_traj[t:t+2, 0], x_traj[t:t+2, 1],
#                     color=color, alpha=alpha, linewidth=linewidth)
#         if label:
#             ax.plot([], [], color=color, linewidth=linewidth, label=label)

#     def draw_start_end_markers(self, ax, x_traj: np.ndarray,
#                                 color='purple', size: float = 60.0):
#         """Mark start (○) and end (★) of an EE trace."""
#         ax.scatter(*x_traj[0,  :2], s=size,       color=color, zorder=5, marker='o')
#         ax.scatter(*x_traj[-1, :2], s=size * 1.3, color=color, zorder=5, marker='*')

#     def draw_ee_point(self, ax, x: np.ndarray,
#                       color='seagreen', radius: float = 0.02):
#         """Draw a filled circle at a single EE position."""
#         ax.add_patch(Circle(x[:2], radius, color=color, alpha=0.9, zorder=6))

#     # ── Trajectory-level convenience ──────────────────────────────────────────

#     def draw_trajectory(self, ax, traj,
#                         color_left='orange', color_right='purple',
#                         label: str = None,
#                         n_shadows: int = 8,
#                         faded_trace: bool = True,
#                         show_start_end: bool = True,
#                         arm_width: float = 0.2):
#         """Draw all visual elements for a single (T, n_q+n_x) trajectory."""
#         traj = self._to_numpy(traj)
#         q_left, q_right, x_left, x_right = self._split_traj(traj)

#         self.draw_arm_shadows(ax, q_left,  n_shadows=n_shadows,
#                               facecolor=color_left,  width=arm_width)
#         self.draw_arm_shadows(ax, q_right, n_shadows=n_shadows,
#                               facecolor=color_right, width=arm_width)

#         trace = self.draw_ee_trace_faded if faded_trace else self.draw_ee_trace
#         trace(ax, x_left[:, :2],  color=color_left,
#               label=f'{label} (L)' if label else None)
#         trace(ax, x_right[:, :2], color=color_right,
#               label=f'{label} (R)' if label else None)

#         if show_start_end:
#             self.draw_start_end_markers(ax, x_left[:, :2],  color=color_left)
#             self.draw_start_end_markers(ax, x_right[:, :2], color=color_right)

#     # ── Top-level ─────────────────────────────────────────────────────────────

#     def plot_trajectories(self, trajectories: list,
#                           color_pairs: list = None,
#                           labels: list = None,
#                           n_shadows: int = 8,
#                           faded_trace: bool = True,
#                           show_start_end: bool = True,
#                           arm_width: float = 0.2,
#                           title: str = 'Robot Trajectories',
#                           ax: plt.Axes = None) -> plt.Axes:
#         """Plot a list of trajectories with shadow/trace effects."""
#         if ax is None:
#             _, ax = plt.subplots(figsize=(8, 8))

#         cmap = plt.cm.get_cmap('tab10')
#         if color_pairs is None:
#             color_pairs = [(cmap(i * 2 % 10), cmap((i * 2 + 1) % 10))
#                            for i in range(len(trajectories))]

#         for i, traj in enumerate(trajectories):
#             c_left, c_right = color_pairs[i]
#             self.draw_trajectory(
#                 ax, traj,
#                 color_left=c_left, color_right=c_right,
#                 label=labels[i] if labels else None,
#                 n_shadows=n_shadows,
#                 faded_trace=faded_trace,
#                 show_start_end=show_start_end,
#                 arm_width=arm_width,
#             )

#         self._style(ax, title=title, legend=labels is not None)
#         return ax
    




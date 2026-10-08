import argparse
import random
from pathlib import Path

from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from robots.planar_robot.visualization import plot_robot_basis, plot_robot_link
from utils.utils import DataVisualizer, create_two_arm_robot
from evaluation import planar_robot_rmse_table as table
from evaluation import planar_robot_so2_density as density
from evaluation.planar_robot_rmse_table import device, to_task_space, check_inputs

ROOT_DIR = Path(__file__).resolve().parent.parent

# ─── per-category figures (traj_<category>.pdf) ──────────────────────────────────────────────────
# A category's example trajectory is drawn from a pool of at most POOL_CAP trajectories (seeded shuffle),
# picked by index or as the one whose mean (angle, scale, reflection) is closest to the target.
POOL_CAP, POOL_SEED = 2, 42
EXAMPLES = {
    'original':           dict(index=4),
    'SO2':                dict(angle_deg=45.0, scale=1.0, refl=+1),
    'SO2Scaling2Group':   dict(angle_deg=-120.0, scale=0.1, refl=+1),
    'C2SO2Scaling2Group': dict(angle_deg=-60.0, scale=0.5, refl=-1),
}
# What each paper figure shows. pred_steps: draw only the first N rollout steps, ending in an arrow.
FIGURES = {
    'original':           dict(arms=True,  hide=[],                pred_steps={}),
    'SO2':                dict(arms=False, hide=[],                pred_steps={'Baseline': 500}),
    'SO2Scaling2Group':   dict(arms=False, hide=['Baseline'],        pred_steps={'SO2': 80}),
    'C2SO2Scaling2Group': dict(arms=False, hide=['Baseline', 'SO2'], pred_steps={'SO2Scaling2': 70}),
}
POLICY_STYLES = {   # policy name (planar_robot_rmse_table.POLICIES) -> line style
    'Baseline':      dict(color='#E07B00', linestyle='-',  linewidth=2.4),
    'SO2':           dict(color='#17B2FF', linestyle='--', linewidth=2.4),
    'SO2Scaling2':   dict(color='#7BB36A', linestyle='-.', linewidth=2.4),
    'C2SO2Scaling2': dict(color='#BC34D1', linestyle='--', linewidth=2.4),
}
ARM_STYLE = dict(link_face='#4A4A4A', link_edge='#000000', base_face='#3A3A3A', base_edge='#000000',
                 width_param=0.23, joint_linewidth=1.5, alpha=0.80, zorder=0.8)
BOUNDS_PAD = 0.30   # padding (fraction of the extent) around the union of all example trajectories
CATEGORY_STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '#D0D0D0', 'grid.linewidth': 0.5,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.6,
    'font.family': 'STIXGeneral', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 9, 'legend.framealpha': 0.92, 'text.usetex': False, 'mathtext.fontset': 'stix',
}

# ─── multi-angle letter figure ───────────────────────────────────────────────────────────────────
LETTER_STEPS = [5.0, 10.0, 45.0, 90.0]      # subset of planar_robot_so2_density.STEPS (keeps those steps' colors)
LETTER_ANGLES = [-180.0, -150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0, 180.0]
LETTER_ANGLE_TOL = 3.0                      # a test trajectory matches an angle within this tolerance [deg]
LETTER_STYLE = {
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
    'axes.grid': True, 'grid.color': '#B2A5A5', 'grid.linewidth': 0.6,
    'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.8,
    'lines.linewidth': 3.5, 'font.family': 'STIXGeneral', 'font.size': 11, 'axes.labelsize': 12, 'axes.titlesize': 13,
    'legend.fontsize': 10, 'legend.framealpha': 0.9, 'text.usetex': False, 'mathtext.fontset': 'stix',
}


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  Drawing helpers
# ════════════════════════════════════════════════════════════════════════════════════════════════
def rollout_task_space(net, traj, robot, horizon=None):
    """Roll out from the trajectory's first state; (horizon+1, 4) end-effector positions."""
    horizon = traj.shape[0] - 1 if horizon is None else horizon
    initial_state = torch.tensor(np.asarray(traj)[[0]], dtype=torch.float32).to(device)
    with torch.no_grad():
        pred_q, _ = net.forward_denorm_multistep(horizon=horizon, inp_batch=initial_state, return_traj=True, use_grad=False)
    return to_task_space(pred_q[0, :, :robot.nb_dofs], robot)


def draw_arm(ax, q, arm_length, base, draw_base):
    s = ARM_STYLE
    patches = list(plot_robot_basis(ax, s['width_param'], s['base_face'], s['base_edge'])) if draw_base else []
    pos, angle = np.array(base, dtype=float), 0.0
    for qi in q:
        angle += float(qi)
        link, pos = plot_robot_link(ax, angle, arm_length, pos, s['width_param'], s['link_face'], s['link_edge'],
                                    joint_linewidth=s['joint_linewidth'])
        patches += link
    for p in patches:
        p.set_alpha(s['alpha'])
        p.set_zorder(s['zorder'])


def add_arrow(ax, p0, p1, color, linewidth, bounds, zorder):
    """Small arrowhead from p0 towards p1 (marks a rollout that continues beyond what is drawn)."""
    d = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)
    if np.linalg.norm(d) > 1e-8:
        d /= np.linalg.norm(d)
    length = 0.07 * (np.hypot(bounds[1] - bounds[0], bounds[3] - bounds[2]) if bounds is not None else 0.5)
    ax.annotate('', xy=(p0[0] + d[0] * length, p0[1] + d[1] * length), xytext=(p0[0], p0[1]),
                arrowprops=dict(arrowstyle='->', color=color, lw=linewidth * 0.85, mutation_scale=10), zorder=zorder + 2)


def draw_trajectory(ax, xy, color, linestyle, linewidth, zorder, bounds=None, truncated=False, alpha=1.0):
    """
    (T, 2) trajectory with a start dot and an end cross. If `bounds` is given, the trajectory is cut
    where it first leaves them; cut or `truncated` trajectories end in an arrow instead of a cross.
    """
    clipped = False
    if bounds is not None and len(xy) > 1 and not truncated:
        x_min, x_max, y_min, y_max = bounds
        inside = (xy[:, 0] >= x_min) & (xy[:, 0] <= x_max) & (xy[:, 1] >= y_min) & (xy[:, 1] <= y_max)
        first_out = int(np.argmax(~inside)) if not inside.all() else 0
        if first_out > 0:
            add_arrow(ax, xy[first_out - 1].copy(), xy[first_out].copy(), color, linewidth, bounds, zorder)
            xy, clipped = xy[:first_out], True
    if len(xy) < 2:
        return
    ax.plot(xy[:, 0], xy[:, 1], color=color, linestyle=linestyle, linewidth=linewidth, zorder=zorder,
            solid_capstyle='round', alpha=alpha)
    ax.scatter(xy[0, 0], xy[0, 1], s=50, c=color, alpha=alpha, zorder=zorder + 1, marker='o', edgecolors='white', linewidths=1.0)
    if truncated:
        add_arrow(ax, xy[-2], xy[-1], color, linewidth, bounds, zorder)
    elif not clipped:
        ax.scatter(xy[-1, 0], xy[-1, 1], s=100, c=color, alpha=alpha, zorder=zorder + 1, marker='x', linewidths=1.8)


def style_axes(ax):
    ax.set_aspect('equal', adjustable='box')
    ax.grid(True, color='#D0D0D0', linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  traj_<category>.pdf
# ════════════════════════════════════════════════════════════════════════════════════════════════
def pick_examples(pool):
    rng = random.Random(POOL_SEED)
    pool = dict(pool)
    for category in ('SO2Scaling2Group', 'C2SO2Scaling2Group'):
        if len(pool[category]) > POOL_CAP:
            shuffled = list(pool[category])
            rng.shuffle(shuffled)
            pool[category] = shuffled[:POOL_CAP]
    examples = {}
    for category, target in EXAMPLES.items():
        if 'index' in target:
            examples[category] = pool[category][target['index']]
            continue
        def distance(traj):
            angle, scale, refl = np.mean(traj[:, -3]), np.mean(traj[:, -2]), np.mean(traj[:, -1])
            return (((angle - np.deg2rad(target['angle_deg'])) / np.pi) ** 2 + (scale - target['scale']) ** 2
                    + (refl - target['refl']) ** 2) ** 0.5
        examples[category] = min(pool[category], key=distance)
    return examples


def figure_bounds(task_trajs):
    xs = np.concatenate([t[:, c] for t in task_trajs for c in (0, 2)])
    ys = np.concatenate([t[:, c] for t in task_trajs for c in (1, 3)])
    pad_x, pad_y = max((xs.max() - xs.min()) * BOUNDS_PAD, 1e-2), max((ys.max() - ys.min()) * BOUNDS_PAD, 1e-2)
    return float(xs.min() - pad_x), float(xs.max() + pad_x), float(ys.min() - pad_y), float(ys.max() + pad_y)


def plot_category(category, gt, reference, policies, robot, bounds, path):
    settings = FIGURES[category]
    gt_x = to_task_space(gt[:, :robot.nb_dofs], robot)
    with plt.rc_context(CATEGORY_STYLE):
        fig, ax = plt.subplots(figsize=(4.0, 4.0))
        ax.set_aspect('equal', adjustable='box')
        ax.set_xlim(bounds[0], bounds[1])
        ax.set_ylim(bounds[2], bounds[3])
        style_axes(ax)
        if settings['arms']:   # robot at the first state of the example
            q0 = gt[0, :robot.nb_dofs]
            for side, q, draw_base in (('left', q0[:robot.nb_dofs_left], True), ('right', q0[robot.nb_dofs_left:], False)):
                arm = robot.robot_left if side == 'left' else robot.robot_right
                draw_arm(ax, q, robot.arm_length, arm.base_pose.t[:2], draw_base)
        if reference is not None:   # the untransformed original demo, dotted
            ref_x = to_task_space(reference[:, :robot.nb_dofs], robot)
            draw_trajectory(ax, ref_x[:, :2], '#747070', ':', 1.9, zorder=1, alpha=0.9)
            draw_trajectory(ax, ref_x[:, 2:], '#8D8989', ':', 1.9, zorder=1, alpha=0.9)
        draw_trajectory(ax, gt_x[:, :2], '#444444', '-', 1.8, zorder=2)
        draw_trajectory(ax, gt_x[:, 2:], '#444444', '-', 1.8, zorder=2)
        for name, net in policies.items():
            if name in settings['hide']:
                continue
            style, pred_x = POLICY_STYLES[name], rollout_task_space(net, gt, robot)
            steps = settings['pred_steps'].get(name)
            truncated = steps is not None and steps < len(pred_x)
            pred_x = pred_x[:steps] if truncated else pred_x
            for cols in (slice(0, 2), slice(2, 4)):
                draw_trajectory(ax, pred_x[:, cols], style['color'], style['linestyle'], style['linewidth'], zorder=3,
                                bounds=None if truncated else bounds, truncated=truncated)
        plt.tight_layout(pad=0.4)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def make_category_figures(out_dir, robot, bar):
    bar.set_postfix_str('loading test pool')
    pool = table.build_test_pool()
    bar.update()
    examples = pick_examples(pool)
    policies = {}
    for name, (ckpt, dataset) in table.POLICIES.items():
        bar.set_postfix_str(f'loading {name}')
        policies[name] = table.load_policy(ckpt, dataset, robot)
        bar.update()
    all_x = [to_task_space(t[:, :robot.nb_dofs], robot) for t in examples.values()]
    bounds = figure_bounds(all_x + [all_x[0]])   # all examples + the original reference (== the 'original' example)
    paths = []
    for category in FIGURES:
        bar.set_postfix_str(f'plotting traj_{category}')
        reference = None if category == 'original' else examples['original']
        paths.append(Path(out_dir) / f'traj_{category}.pdf')
        plot_category(category, examples[category], reference, policies, robot, bounds, paths[-1])
        bar.update()
    return paths


# ════════════════════════════════════════════════════════════════════════════════════════════════
#  traj_at_..._letter_C.pdf
# ════════════════════════════════════════════════════════════════════════════════════════════════
def make_letter_figure(out_dir, robot, bar):
    bar.set_postfix_str('loading SO2 1-degree dataset')
    data_vis = DataVisualizer(task=density.TASK, conditioning='symmetry', normalize_conditioning=False)
    demos, demos_norm = data_vis.construct_demonstrations(demo_folder=density.DATASET_FOLDER,
                                                          load_augmented_data=True, predefined_dataset=density.DATASET)
    test = [(t[density.N_TRIM:], round(float(np.rad2deg(np.mean(t[:, -3]))), 1))
            for t, kind in zip(demos['test_in'], demos['test_in_demo_type']) if kind == 'original' or 'SO2' in kind]
    colors = {step: density.COLORS[list(density.STEPS).index(step)] for step in LETTER_STEPS}
    nets = {step: density.load_policy(step, demos_norm, robot) for step in LETTER_STEPS}
    bar.update()
    left = slice(0, 2)   # task-space columns of the left arm (the letter C)
    with plt.rc_context(LETTER_STYLE):
        fig, ax = plt.subplots(figsize=(3.8, 3.8))
        for target in LETTER_ANGLES:
            bar.set_postfix_str(f'plotting letter C at {target:+.0f} deg')
            gt = next(t for t, angle in test if abs(angle - target) <= LETTER_ANGLE_TOL)   # first match in file order
            style_axes(ax)
            gt_x = to_task_space(gt[:, :robot.nb_dofs], robot)[:, left]
            ax.plot(gt_x[:, 0], gt_x[:, 1], color='#888888', linewidth=1.6, linestyle='--', alpha=0.85, zorder=1)
            ax.scatter(gt_x[0, 0], gt_x[0, 1], s=50, c='#888888', zorder=4, marker='o', edgecolors='white', linewidths=1.0)
            ax.scatter(gt_x[-1, 0], gt_x[-1, 1], s=90, c='#888888', zorder=4, marker='x', linewidths=1.8)
            for step, net in nets.items():
                x = rollout_task_space(net, gt, robot)[:, left]
                ax.plot(x[:, 0], x[:, 1], color=colors[step], linewidth=2.5, linestyle='-', zorder=2)
                ax.scatter(x[0, 0], x[0, 1], s=50, c=colors[step], zorder=5, marker='o', edgecolors='white', linewidths=1.0)
                ax.scatter(x[-1, 0], x[-1, 1], s=90, c=colors[step], zorder=5, marker='x', linewidths=1.8)
            bar.update()
        plt.tight_layout(pad=0.4)
        tags = '_'.join(f'{a:+.0f}' for a in LETTER_ANGLES)
        path = Path(out_dir) / f'traj_at_{tags}deg_letter_C.pdf'
        fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return path


def required_files():
    """Checkpoints and datasets these figures need (not stored in git)."""
    return table.required_files() + density.required_files(LETTER_STEPS)


def make_all(out_dir=ROOT_DIR / 'results' / 'planar_paper' / 'figures', verbose=True):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    robot = create_two_arm_robot(nb_dofs_left=4, nb_dofs_right=4, nb_x_left=2, nb_x_right=2, ee_joint=False)
    # one bar step per dataset load, policy load and drawn figure / letter angle; the postfix names the running step
    total = 1 + len(table.POLICIES) + len(FIGURES) + 1 + len(LETTER_ANGLES)
    with tqdm(total=total, desc='Figures', disable=not verbose) as bar:
        return make_category_figures(out_dir, robot, bar) + [make_letter_figure(out_dir, robot, bar)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out', default=str(ROOT_DIR / 'results' / 'planar_paper' / 'figures'), help='output folder (default: %(default)s)')
    args = p.parse_args()
    check_inputs(required_files())
    for path in make_all(args.out):
        print(f'saved -> {path}')


if __name__ == '__main__':
    matplotlib.use('Agg')
    main()

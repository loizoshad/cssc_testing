"""
Best-checkpoint search and 4×4 evaluation table for RBY1 letter networks.

Workflow
────────
1. Load each of the 4 datasets with its own DataVisualizer + train/test split
   (exact same hyperparameters as the corresponding training scripts) so that
   normalization bounds are consistent with how each network was trained.

2. Split each augmented test set into:
     val   → used to rank checkpoints for the CORRESPONDING network only
     eval  → used in the final 4×4 RMSE table (cross-network evaluation)
   For the baseline the 3 original demos serve as both val and eval.

3. For each network scan all saved checkpoints (every-500-iter intermediates +
   final), evaluate on the network's validation set, and save the top-K
   rankings to a text file.

4. Load the best checkpoint of each network and compute mean RMSE ± std on
   all 4 evaluation datasets → 4×4 table saved as a text file.

Run from the repository root:
    python examples/rby1/data_augmentation_letters/rby1_letters_evaluate_checkpoints.py
"""

import os, sys, copy, random, glob, time
from pathlib import Path

import numpy as np
import torch

# ── repo root on sys.path ─────────────────────────────────────────────────────
CURRENT_DIR = Path(os.path.abspath(__file__)).parent
ROOT_DIR    = CURRENT_DIR.parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from utils.groups import *
from utils.utils  import *
from utils.networks_pytorch import CustomMLP, NETWORKS_DIR_RBY1

device = torch.device("cuda" if torch.cuda.is_available() else "mps"  if torch.backends.mps.is_available() else "cpu" )

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION — edit here
# ─────────────────────────────────────────────────────────────────────────────

# Fixed random seed for the val / eval demo split (reproducible)
SPLIT_SEED = 42

# Number of test demos used as validation (per augmented dataset).
# The remaining test demos become the evaluation set.
N_VAL_DEMOS = 30

# Save the top-K checkpoints per network to text.
TOP_K = 100

# Output directory for rankings and table files.
RESULTS_DIR = ROOT_DIR / 'results_rby1' / 'checkpoint_eval'

# Common fields shared by all letter training scripts
_TASK       = 'left-rby1-letters_right-rby1-letters_ndofs-14'
_DEMO_FOLDER = 'rby1_letters'
_SEED       = 42
_DECOUPLED  = True
_COND       = 'symmetry'

# ─────────────────────────────────────────────────────────────────────────────
# Per-network dataset / split configuration
# ─────────────────────────────────────────────────────────────────────────────
# Each entry mirrors exactly the training script it corresponds to.

def _build_dataset_configs(robot, task_space='yz'):
    dt = np.deg2rad(1.0)

    # ── Baseline ──────────────────────────────────────────────────────────────
    baseline_cfg = dict(
        name='baseline',
        display='Baseline',
        load_augmented=False,
        group=None,
        rearrange=False,
        so2_train_step=None,
        scaling_train_values=None,
        so2_match_tolerance=np.deg2rad(3),
        scaling_match_tolerance=0.05,
        num_iterations=100_000,
        nb_steps=15,
        stride=10,
    )
    baseline_cfg['model_id'] = (
        f'RBY1_MLP_DA_BASELINE_{_TASK}'
        f'_epochs={baseline_cfg["num_iterations"]}'
        f'_nsteps={baseline_cfg["nb_steps"]}'
        f'_stride={baseline_cfg["stride"]}'
        f'_{"decoupled" if _DECOUPLED else "coupled"}'
        f'_conditioning={_COND}'
        f'_seed={_SEED}_'
    )

    # ── SO2 ───────────────────────────────────────────────────────────────────
    G_so2 = SO2RBY1(name='SO2Letters')
    so2_cfg = dict(
        name='so2',
        display='SO2',
        load_augmented=True,
        group=G_so2,
        rearrange=True,
        so2_train_step=np.deg2rad(15.0),
        scaling_train_values=None,
        so2_match_tolerance=np.deg2rad(7.5),
        scaling_match_tolerance=0.05,
        num_iterations=100_000,
        nb_steps=15,
        stride=10,
    )
    so2_cfg['model_id'] = (
        f'RBY1_MLP_DA_{G_so2}_{_TASK}'
        f'_epochs={so2_cfg["num_iterations"]}'
        f'_nsteps={so2_cfg["nb_steps"]}'
        f'_stride={so2_cfg["stride"]}'
        f'_{"decoupled" if _DECOUPLED else "coupled"}'
        f'_conditioning={_COND}'
        f'_seed={_SEED}_'
    )

    # ── SO2 × Scaling2 ────────────────────────────────────────────────────────
    G_so2s2 = SO2Scaling2RBY1Letters(name='SO2Scaling2Letters')
    so2s2_cfg = dict(
        name='so2scaling2',
        display='SO2×Scaling2',
        load_augmented=True,
        group=G_so2s2,
        rearrange=True,
        so2_train_step=np.deg2rad(15.0),
        scaling_train_values=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        so2_match_tolerance=np.deg2rad(7.5),
        scaling_match_tolerance=0.05,
        num_iterations=100_000,
        nb_steps=15,
        stride=10,
    )
    so2s2_cfg['model_id'] = (
        f'RBY1_MLP_DA_{G_so2s2}_{_TASK}'
        f'_epochs={so2s2_cfg["num_iterations"]}'
        f'_nsteps={so2s2_cfg["nb_steps"]}'
        f'_stride={so2s2_cfg["stride"]}'
        f'_{"decoupled" if _DECOUPLED else "coupled"}'
        f'_conditioning={_COND}'
        f'_seed={_SEED}_'
    )

    # ── C2 × SO2 × Scaling2 ───────────────────────────────────────────────────
    G_c2so2s2 = C2SO2Scaling2RBY1(name='C2SO2Scaling2Letters')
    c2so2s2_cfg = dict(
        name='c2so2scaling2',
        display='C2×SO2×Scaling2',
        load_augmented=True,
        group=G_c2so2s2,
        rearrange=True,
        so2_train_step=np.deg2rad(15.0),
        scaling_train_values=[0.1, 0.2, 0.4, 0.6, 0.8, 1.0],
        so2_match_tolerance=np.deg2rad(7.5),
        scaling_match_tolerance=0.05,
        num_iterations=100_000,
        nb_steps=15,
        stride=10,
    )
    c2so2s2_cfg['model_id'] = (
        f'RBY1_MLP_DA_{G_c2so2s2}_{_TASK}'
        f'_epochs={c2so2s2_cfg["num_iterations"]}'
        f'_nsteps={c2so2s2_cfg["nb_steps"]}'
        f'_stride={c2so2s2_cfg["stride"]}'
        f'_{"decoupled" if _DECOUPLED else "coupled"}'
        f'_conditioning={_COND}'
        f'_seed={_SEED}_'
    )

    return [baseline_cfg, so2_cfg, so2s2_cfg, c2so2s2_cfg]


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(cfg, robot, task_space='yz'):
    """
    Load and split one dataset exactly as the corresponding training script does.

    Returns
    -------
    demonstrations      : raw (unnormalised) dict
    demonstrations_norm : normalised dict  (normalization tied to THIS dataset)
    """
    data_vis = DataVisualizer(
        task=_TASK,
        conditioning=_COND,
        normalize_conditioning=False,
        group=cfg['group'],
        task_space=task_space,
    )
    demonstrations, demonstrations_norm = data_vis.construct_demonstrations(
        demo_folder=_DEMO_FOLDER,
        load_augmented_data=cfg['load_augmented'],
    )

    if cfg['rearrange']:
        G_str = str(cfg['group'])
        num_original = sum(1 for t in demonstrations['train_in_demo_type'] if 'original' in t)
        num_augm     = sum(1 for t in demonstrations['train_in_demo_type'] if G_str in t)
        demonstrations, demonstrations_norm = data_vis.rearrange_demonstrations(
            demonstrations, demonstrations_norm,
            train_demo_types=['original', G_str],
            num_train_demos_per_type=[num_original, num_augm],
        )
        if cfg['so2_train_step'] is not None or cfg['scaling_train_values'] is not None:
            demonstrations, demonstrations_norm = split_train_test(
                demonstrations, demonstrations_norm,
                so2_train_step=cfg['so2_train_step'],
                scaling_train_values=cfg['scaling_train_values'],
                so2_match_tolerance=cfg['so2_match_tolerance'],
                scaling_match_tolerance=cfg['scaling_match_tolerance'],
            )

    n_train = len(demonstrations['train_in'])
    n_test  = len(demonstrations['test_in'])
    print(f"  [{cfg['display']}] train={n_train}, test={n_test}")
    return demonstrations, demonstrations_norm


# ─────────────────────────────────────────────────────────────────────────────
# Helpers: denormalise demos, compute RMSE
# ─────────────────────────────────────────────────────────────────────────────

def demos_to_tensors(raw_demos, device):
    """
    Convert a list of raw (physical-scale) demo arrays / tensors to a list of
    float32 tensors on `device`.  No normalisation or de-normalisation is applied
    — the data in ``demonstrations['train_in']`` / ``demonstrations['test_in']``
    is already in physical joint-angle units, which is exactly what
    ``forward_denorm_multistep`` expects.
    """
    out = []
    for d in raw_demos:
        t = d if isinstance(d, torch.Tensor) else torch.tensor(d, dtype=torch.float32)
        out.append(t.to(device))
    return out


def compute_rmse_on_demos(network, demos_denorm, horizon):
    """
    Roll out the network from each demo's initial condition and compute RMSE
    against the ground-truth joint trajectory.

    Parameters
    ----------
    network      : CustomMLP (weights already loaded, normalization bounds set)
    demos_denorm : list of Tensor(T, nb_q + nb_x) in DENORMALISED (physical) scale
    horizon      : int — number of integration steps

    Returns
    -------
    rmse_per_demo : list of float
    mean_rmse     : float
    std_rmse      : float
    """
    if len(demos_denorm) == 0:
        return [], float('nan'), float('nan')

    nb_q = 2 * network.n_q    # total joints (14); network stores per-arm DoFs as n_q

    # Stack initial conditions → batch forward pass
    init_conds = torch.stack([d[0] for d in demos_denorm]).to(device)  # (N, nb_q+nb_x)
    with torch.no_grad():
        pred_qs, _ = network.forward_denorm_multistep(
            horizon=horizon,
            inp_batch=init_conds,
            return_traj=True,
            use_grad=False,
        )
    # pred_qs: (N, horizon+1, 2*n_q)  — but network.forward returns 2*n_q cols

    rmse_list = []
    for i, demo in enumerate(demos_denorm):
        T_gt = min(demo.shape[0], horizon + 1)
        gt_q   = demo[:T_gt, :nb_q].to(device)           # (T_gt, nb_q)
        pr_q   = pred_qs[i, :T_gt, :nb_q]                # (T_gt, nb_q)
        rmse   = torch.sqrt(torch.mean((pr_q - gt_q) ** 2)).item()
        rmse_list.append(rmse)

    arr = np.array(rmse_list)
    return rmse_list, float(arr.mean()), float(arr.std())


# ─────────────────────────────────────────────────────────────────────────────
# Validation / evaluation split
# ─────────────────────────────────────────────────────────────────────────────

def make_val_eval_split(demos_denorm, n_val, seed=SPLIT_SEED):
    """
    Randomly split demos_denorm into (val_demos, eval_demos).
    For the baseline (n_val ≥ len), everything goes to both val and eval.
    """
    rng = random.Random(seed)
    indices = list(range(len(demos_denorm)))
    rng.shuffle(indices)
    if n_val >= len(demos_denorm):
        # Not enough data to separate → use all for both
        return demos_denorm, demos_denorm
    val_idx  = indices[:n_val]
    eval_idx = indices[n_val:]
    val_demos  = [demos_denorm[i] for i in val_idx]
    eval_demos = [demos_denorm[i] for i in eval_idx]
    return val_demos, eval_demos


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint discovery
# ─────────────────────────────────────────────────────────────────────────────

def find_checkpoints(model_id_prefix):
    """
    Return all checkpoint paths whose stem starts with model_id_prefix.

    Intermediate checkpoints: {prefix}iter{N}.pt
    Final checkpoint:         {prefix}.pt  (treated as iter=max+500)
    """
    models_dir = NETWORKS_DIR_RBY1 / 'models'
    # Intermediate: must have 'iter' after the prefix
    pattern_inter = str(models_dir / f'{model_id_prefix}iter*.pt')
    inter_paths   = sorted(glob.glob(pattern_inter))
    # Final: exact match
    final_path = models_dir / f'{model_id_prefix}.pt'

    def iter_num(p):
        stem = Path(p).stem
        tag  = stem[len(model_id_prefix):]          # e.g. "iter1500"
        return int(tag.replace('iter', ''))

    checkpoint_list = []
    for p in inter_paths:
        try:
            n = iter_num(p)
            checkpoint_list.append((n, p))
        except ValueError:
            pass

    if final_path.exists():
        # Assign it an iteration number just beyond the last intermediate
        last_n = max((n for n, _ in checkpoint_list), default=0)
        checkpoint_list.append((last_n + 500, str(final_path)))

    checkpoint_list.sort(key=lambda x: x[0])
    print(f"    Found {len(checkpoint_list)} checkpoints for prefix '{model_id_prefix[:60]}...'")
    return checkpoint_list


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    task_space = 'yz'
    robot = create_rby1_robot(task_space=task_space, ee_joint=False, is_taskspace=False)

    # ── 1. Build dataset configs ──────────────────────────────────────────────
    configs = _build_dataset_configs(robot, task_space=task_space)
    config_by_name = {c['name']: c for c in configs}
    NAMES   = [c['name']    for c in configs]
    DISPLAY = [c['display'] for c in configs]

    # ── 2. Load datasets ──────────────────────────────────────────────────────
    print("\n═══ Loading datasets ═══")
    datasets = {}
    for cfg in configs:
        print(f"\n  Loading dataset: {cfg['display']}")
        dems, dems_norm = load_dataset(cfg, robot, task_space=task_space)
        datasets[cfg['name']] = dict(
            demonstrations=dems,
            demonstrations_norm=dems_norm,
        )

    # ── 3. Build one CustomMLP shell per network (normalization tied to its dataset)
    print("\n═══ Building network shells ═══")
    network_shells = {}
    for cfg in configs:
        ds = datasets[cfg['name']]
        net = CustomMLP(
            model_id=cfg['model_id'],
            load_model_flag=False,
            save_model_flag=False,
            demonstrations=ds['demonstrations_norm'],
            robot=robot,
        )
        network_shells[cfg['name']] = net
        print(f"  [{cfg['display']}] in_dim={net.model.l_1.in_features}, "
              f"Q_min={ds['demonstrations_norm']['Q_min'][:3]}...")

    # ── 4. Denormalise test demos and split into val / eval ───────────────────
    print("\n═══ Preparing validation / evaluation splits ═══")
    val_demos  = {}   # demos_denorm used to find best ckpt for the SAME network
    eval_demos = {}   # demos_denorm used in the final 4×4 table

    for name, ds in datasets.items():
        if name == 'baseline':
            # All 3 original train demos (no test split for baseline)
            raw_demos = ds['demonstrations']['train_in']
        else:
            raw_demos = ds['demonstrations']['test_in']

        # demonstrations['train_in'] / ['test_in'] hold raw (physical-scale) data.
        # forward_denorm_multistep expects physical-scale input → cast only.
        demos_dn = demos_to_tensors(raw_demos, device)

        if name == 'baseline':
            val_demos[name]  = demos_dn   # 3 demos → both val and eval
            eval_demos[name] = demos_dn
        else:
            v, e = make_val_eval_split(demos_dn, n_val=N_VAL_DEMOS)
            val_demos[name]  = v
            eval_demos[name] = e

        print(f"  [{name}] total={len(demos_dn)}  val={len(val_demos[name])}  "
              f"eval={len(eval_demos[name])}")

    # ── 5. Determine horizon per eval dataset ─────────────────────────────────
    # Use min trajectory length across eval demos (so every demo fits entirely)
    eval_horizons = {}
    for name in NAMES:
        if eval_demos[name]:
            h = min(d.shape[0] for d in eval_demos[name]) - 1
        else:
            h = 200
        eval_horizons[name] = h
        print(f"  [{name}] eval horizon = {h}")

    # ── 6. Scan and rank checkpoints for each network ─────────────────────────
    print("\n═══ Ranking checkpoints (validation pass) ═══")
    best_ckpts = {}   # best_ckpts[net_name] = path to best checkpoint .pt file

    for cfg in configs:
        name    = cfg['name']
        display = cfg['display']
        net     = network_shells[name]
        v_demos = val_demos[name]         # validation demos for THIS network
        h       = eval_horizons[name]     # horizon

        print(f"\n  ── {display} ──")
        ckpts = find_checkpoints(cfg['model_id'])
        if not ckpts:
            print(f"  [WARNING] No checkpoints found for {display}. Skipping.")
            best_ckpts[name] = None
            continue

        scores = []   # list of (iteration, rmse_mean, rmse_std, path)
        t0 = time.perf_counter()
        for k, (iteration, ckpt_path) in enumerate(ckpts):
            # Load weights
            state_dict = torch.load(ckpt_path, map_location=device, weights_only=True)
            net.model.load_state_dict(state_dict)
            net.model.eval()

            # Evaluate on validation demos
            _, mean_rmse, std_rmse = compute_rmse_on_demos(net, v_demos, h)
            scores.append((iteration, mean_rmse, std_rmse, ckpt_path))

            if (k + 1) % 20 == 0 or k == 0:
                elapsed = time.perf_counter() - t0
                print(f"    [{k+1}/{len(ckpts)}] iter={iteration:>7}  "
                      f"val_rmse={mean_rmse:.6f}  ({elapsed:.1f}s elapsed)")

        # Sort by mean RMSE (ascending)
        scores.sort(key=lambda x: x[1])

        # Save top-K rankings
        rank_file = RESULTS_DIR / f'rankings_{name}.txt'
        with open(rank_file, 'w') as f:
            f.write(f"Checkpoint rankings for: {display}\n")
            f.write(f"Validation set: {len(v_demos)} demos, horizon={h}\n")
            f.write(f"{'Rank':<6} {'Iter':>8} {'Val RMSE':>12} {'Std':>10}  Path\n")
            f.write("-" * 100 + "\n")
            for rank, (iteration, mean_rmse, std_rmse, path) in enumerate(scores[:TOP_K], 1):
                f.write(f"{rank:<6} {iteration:>8} {mean_rmse:>12.6f} {std_rmse:>10.6f}  {path}\n")
        print(f"  Saved top-{TOP_K} rankings → {rank_file}")

        best_ckpts[name] = scores[0][3]   # path to best checkpoint
        best_iter  = scores[0][0]
        best_rmse  = scores[0][1]
        print(f"  Best checkpoint: iter={best_iter}, val_rmse={best_rmse:.6f}")
        print(f"  → {best_ckpts[name]}")

    # ── 7. Final 4×4 evaluation table ─────────────────────────────────────────
    print("\n═══ Final evaluation (best checkpoints × eval datasets) ═══")

    # table[net_name][dataset_name] = (mean_rmse, std_rmse)
    table = {n: {} for n in NAMES}

    for net_name, cfg in zip(NAMES, configs):
        net  = network_shells[net_name]
        ckpt = best_ckpts[net_name]
        display_net = cfg['display']

        if ckpt is None:
            print(f"  [SKIP] {display_net}: no checkpoint available.")
            for ds_name in NAMES:
                table[net_name][ds_name] = (float('nan'), float('nan'))
            continue

        # Load best weights for this network
        state_dict = torch.load(ckpt, map_location=device, weights_only=True)
        net.model.load_state_dict(state_dict)
        net.model.eval()
        print(f"\n  Network: {display_net}  (ckpt: {Path(ckpt).name})")

        for ds_name in NAMES:
            e_demos = eval_demos[ds_name]
            h       = eval_horizons[ds_name]
            display_ds = config_by_name[ds_name]['display']

            if len(e_demos) == 0:
                print(f"    vs {display_ds}: no eval demos.")
                table[net_name][ds_name] = (float('nan'), float('nan'))
                continue

            _, mean_rmse, std_rmse = compute_rmse_on_demos(net, e_demos, h)
            table[net_name][ds_name] = (mean_rmse, std_rmse)
            print(f"    vs {display_ds:20s}: RMSE = {mean_rmse:.6f} ± {std_rmse:.6f}  "
                  f"(n={len(e_demos)}, h={h})")

    # ── 8. Print and save the 4×4 table ──────────────────────────────────────
    print("\n═══ 4×4 RMSE Table (rows=networks, cols=eval datasets) ═══")

    col_w = 24
    header = f"{'Network':<22}" + "".join(f"{d:>{col_w}}" for d in DISPLAY)
    sep    = "─" * (22 + col_w * len(DISPLAY))
    print(header)
    print(sep)
    for net_name, display_net in zip(NAMES, DISPLAY):
        row = f"{display_net:<22}"
        for ds_name in NAMES:
            mean_r, std_r = table[net_name][ds_name]
            if np.isnan(mean_r):
                cell = "n/a"
            else:
                cell = f"{mean_r:.4f}±{std_r:.4f}"
            row += f"{cell:>{col_w}}"
        print(row)

    table_file = RESULTS_DIR / 'evaluation_table.txt'
    with open(table_file, 'w') as f:
        f.write("4×4 RMSE Table  (rows = network, cols = evaluation dataset)\n")
        f.write("Values: mean ± std  (joint-angle RMSE in radians)\n\n")
        f.write(header + "\n")
        f.write(sep + "\n")
        for net_name, display_net in zip(NAMES, DISPLAY):
            row = f"{display_net:<22}"
            for ds_name in NAMES:
                mean_r, std_r = table[net_name][ds_name]
                if np.isnan(mean_r):
                    cell = "n/a"
                else:
                    cell = f"{mean_r:.4f}±{std_r:.4f}"
                row += f"{cell:>{col_w}}"
            f.write(row + "\n")
        f.write("\nBest checkpoints used:\n")
        for name, cfg in zip(NAMES, configs):
            ckpt = best_ckpts.get(name)
            f.write(f"  {cfg['display']:<25} {Path(ckpt).name if ckpt else 'N/A'}\n")

    print(f"\nSaved evaluation table → {table_file}")
    print("\nDone.")


if __name__ == '__main__':
    main()

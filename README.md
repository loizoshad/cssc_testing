# CSSC

## Abstract:

Robots exhibit a rich variety of symmetries arising from their mechanical structure and the properties of their tasks.
Although many robotics problems exhibit several symmetries simultaneously, existing approaches typically treat them in isolation, failing to exploit their combined potential. This paper introduces cross-space symmetry compositions, a framework for learning robot policies that are jointly equivariant to multiple symmetries across configuration and task spaces. Leveraging the differential-geometric structure of the forward kinematics map, we both descend symmetries from configuration to task space and lift symmetries from task to configuration space, enabling their composition within a unified representation space. We validate our framework on simulated and real-world experiments on a dual-arm robot, demonstrating that jointly leveraging multiple symmetries yields improved generalization.

For more information, please visit the [project website](https://sites.google.com/view/cross-space-symmetries).

## 1. Setup

```bash
git clone https://github.com/loizoshad/cssc_testing && cd cssc_testing
conda env create -p ./miniconda_env -f environment.yml
conda activate ./miniconda_env
```

## 2. Download trained policies and datasets

Needed to reproduce the paper results (not needed to train your own policies).
The code expects these networks to be under `networks/models/` and these demonstrations under `demonstrations/planar_robot/paper*/`.

<u>CAUTION</u>:, if you unzip the data for reproducing the paper in the root directory of the project, your existing `demonstrations`, `networks`, and `networks_rby1'` folders will be replaced with the downloaded ones if you run this.
```bash
curl -L -o files_for_paper_reproduction.zip https://github.com/loizoshad/cssc_testing/releases/download/data-v1/files_for_paper_reproduction.zip
echo "f26ad015aa2cd56122ca5c0702027264a81ef93cd71693f6dacd90f7193165c1 files_for_paper_reproduction.zip" | shasum -a 256 -c
unzip -q files_for_paper_reproduction.zip && rm files_for_paper_reproduction.zip
```

## 3. Reproduce the paper results (planar dual arm)

| Command | Reproduces | <nobr> Runtime (min) on MPS </nobr> | Output |
|---|---|---|---|
| `planar_robot_rmse_table` | Tab. II | ~3| `results/planar_paper/cross_eval_rmse.csv` |
| `planar_robot_writte_letter_plots` | Fig. 4 & 5b | ~1 | `results/planar_paper/figures/` |
| `lanar_robot_so2_density` | Fig. 5a | ~30 | `results/planar_paper/so2_density/` |

```bash
python -m evaluation.planar_robot_rmse_table
python -m evaluation.planar_robot_writte_letter_plots
python -m evaluation.planar_robot_so2_density
```

## 4. Generate, augment and train your own (planar dual arm)

```bash
# 1. Demonstrations: track LASA letters with the two arms. You can choose within the script which letters to trace. Saves the demos in: demonstrations/planar_robot/planar_robot_lasa/
python -m demonstration_generation.planar_robot.planar_robot_data_generation

# 2. Augment them with a symmetry group (default: SO(2)). You can choose within the script what symmetries to use. Saves the augmented demos in the same folder, *_augmented_config_<group>.npz
python -m data_augmentation.planar_robots.augment_data_planar_robot_symm_conditioning

# 3. Train a policy. For example you can check the baseline (no symmetries) and the SO(2)-augmented cases using the two examples below. They save the network checkpoints in: networks/models/

# no augmentation
python -m examples.planar_robots.mlp_baseline
# with the SO(2)-augmented data
python -m examples.planar_robots.data_augmentation.mlp_da_so2
```

Other groups: `mlp_da_scaling`, `mlp_da_so2_scaling`, `mlp_da_c2_so2_scaling` (augment with the matching
group first). Training runs for 100k iterations and saves a checkpoint every 500.

## 5. RB-Y1 robot

The raw demonstrations are in the repository (`demonstrations/rby1_*`) for the two tasks: (1) letter-drawing, (2) pan-grasping.

```bash
# Letter drawing
python -m demonstration_generation.rby1.make_rby1_letters_dataset

# Pan grasping
python -m demonstration_generation.rby1.make_rby1_dataset
```

To augment datasets, you can use the script

```bash
python -m data_augmentation.rby1.augment_data_rby1_symm_conditioning
```

The correspondinbg training scripts are in `examples/rby1/`.

## Cite information

If you used our paper in your work, please cite it as:

```bibtex
@article{hadjiloizou2026symmetries,
		title={Symmetries {H}ere and {T}here, {C}ombined {E}verywhere: {C}ross-space {S}ymmetry {C}ompositions in {R}obotics}, 
		author={Hadjiloizou, Loizos and P{\'e}rez-Dattari, Rodrigo and Jaquier, No{\'e}mie},
		journal={{IEEE} Robotics and Automation Letters},
		year={2026}
}
```
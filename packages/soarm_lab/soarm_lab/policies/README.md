# Policies

## Bundled policy provenance

`reach-v3` and `lift-v3` come from the SO-101 PPO training work in the
[Machinekind digital twin](https://github.com/machinekind/twin). The training
code uses MuJoCo simulation and domain randomization. Git records both checkpoint
additions under the author `przemeknowak781`, dated 2026-09-24.

### Upstream checkpoints

| Policy | Original checkpoint | Training history recorded by the commit |
| --- | --- | --- |
| `lift-v3` | [`assets/policies/lift-v3/policy.pt`](https://github.com/machinekind/twin/blob/3c3bc2808151d2f6c430d2d61c22878e30266116/assets/policies/lift-v3/policy.pt) | [Fine-tuning update](https://github.com/machinekind/twin/commit/3c3bc2808151d2f6c430d2d61c22878e30266116) that added `lift-v3` and support for saving and restoring critic weights. |
| `reach-v3` | [`assets/policies/reach-v3/policy.pt`](https://github.com/machinekind/twin/blob/13cfb693119b3929dab35b0fa9d4d013cf857a39/assets/policies/reach-v3/policy.pt) | [Fine-tuned from `reach-v2`](https://github.com/machinekind/twin/commit/13cfb693119b3929dab35b0fa9d4d013cf857a39) with a penalty for wrist rotation away from the home pose. |

The [reach metadata](reach-v3/meta.json) and [lift metadata](lift-v3/meta.json)
record task settings, network dimensions, training steps, randomization and
evaluation results. Their `notes` fields identify the preceding checkpoint by
its creation timestamp. These files match the metadata at the respective
upstream commits. They are training summaries, not a complete record of every
earlier training run.

### Import and format conversion

[Commit `5acba3408df83eb26a4c10e5276b0a83c10b76fa`](https://github.com/machinekind/so-arm101/commit/5acba3408df83eb26a4c10e5276b0a83c10b76fa)
imported both policy folders from the twin into this package. The imported
`policy.pt` and `meta.json` files match the upstream checkpoints above byte for
byte.

[Commit `c1c5b29ada65168c2f5f194dddf842480679e88b`](https://github.com/machinekind/so-arm101/commit/c1c5b29ada65168c2f5f194dddf842480679e88b)
converted the PyTorch checkpoints to `policy.npz` for NumPy inference, retaining
the actor, critic and observation-normalization arrays. The bundled NPZ files
are unchanged from that conversion, and the metadata is unchanged from the
original import. The [policy API](../rl/policy.py) documents the NPZ format.

The policies are distributed under the license stated in the
[soarm-lab README](../../README.md#license).


## Train a policy

Training uses PPO in JAX on MuJoCo Warp. Install the `train` extra for Hydra,
JAX, optax and MuJoCo Warp. Linux installs JAX with CUDA 12 support.
Use `device=cpu` for a small local run.

```bash
uv run --extra train soarm-lab train task=reach
uv run --extra train soarm-lab train task=lift ppo.iterations=50 device=cpu
uv run --extra train soarm-lab train task=pickplace station=$HOME/stations/desk1
uv run --extra train soarm-lab train --cfg job
uv run --extra train soarm-lab train --help
```

Training ends with CPU evaluation. Policies are saved in
`workspace/policies/<name>/` in a checkout. Set `SOARM_LAB_POLICIES` or `out`
to choose another directory. Each folder holds `policy.npz`, `meta.json`,
`config.json`, `experiment.yaml` and training progress. `experiment.yaml`
records the composed settings. Policy inference uses NumPy and needs no Hydra.
Commands accept a policy name or a path to `policy.npz`. Local names take
precedence over bundled names.

The task group selects `reach`, `lift` or `pickplace`. Task settings use keys
such as `task.episode_steps`. PPO settings use `ppo.num_envs`, `ppo.iterations`
and `ppo.seed`. The root `seed` sets `ppo.seed` unless it is overridden directly.
The other run keys are `name`, `out`, `station`, `eval_episodes`, `stop_file`,
`init` and `device`.

Set `randomization.spread=0` to train on the station's measured dynamics with
zero spread. This also skips randomized evaluation. A positive spread scales
the randomization around those measurements. Export a measured station with
`twin export ~/stations/desk1` and pass `station=$HOME/stations/desk1`.

## Experiments

An experiment selects config groups and overrides their settings.
The standard experiments are `reach` with 150 iterations, `lift` with 300,
and `pickplace` with 300. `lift_finetune` starts from the bundled lift policy
and selects `ppo=fine_tune`.

```bash
uv run --extra train soarm-lab train experiment=lift_finetune
uv run --extra train soarm-lab train init=lift-v3 ppo=fine_tune
uv run --extra train soarm-lab train init=lift-v3 ppo=fine_tune +task_overrides.episode_steps=250
EXPERIMENTS="reach lift" OVERRIDES="ppo.iterations=50" bash jobs/train.sh
```

`init` requires `ppo=fine_tune` and uses the policy's stored task.
An explicitly selected task must have the same name. `task_overrides` changes that stored task. The fine-tuning PPO group
sets `init_std=0.25` and `critic_warmup=30`.

`jobs/train.sh` runs `reach` when `EXPERIMENTS` is empty.
`EXPERIMENTS` and `OVERRIDES` are space-separated lists. Each experiment sets its
output name. Set `OVERRIDES="name=x"` to change it. The dispatcher also reads
`GPUS` and `ON_FAILURE`.

## Train one policy on several blocks

`lift_blocks` fine-tunes `lift-v3` on three task variants, one per block shape.
`ppo.widen_init` enables appended descriptor inputs with zero initial actor and critic weights.

```bash
uv run --extra train soarm-lab train experiment=lift_blocks
uv run soarm-lab eval lift_blocks --variant cylinder
```

Training saves nominal and randomized CPU evaluations for each variant.
The [Classic World catalog](../blocks/classic_world.py) holds measured dimensions.
Masses are estimated from volume at 0.7 g/cm³ until the blocks are weighed.
To add a catalog, create a module under [blocks](../blocks) with named `BlockSpec`
entries in `BLOCKS`, then register that mapping in `blocks.__init__._CATALOGS`.
Set the experiment's `task_overrides.variants` to the ordered names and
`task_overrides.block` to its default variant. `ppo.num_envs` must be divisible by
the number of variants. Training samples each block's position, yaw and side lengths at random.

## Start some episodes with a standing block

`task_overrides.standing_fraction` is the probability that an episode starts with
the block on its end. It defaults to 0. The `lift_standing` experiment sets it to
0.25. A cube always lies. The policy sees a standing block's rotation with yaw
folded by its end face's symmetry.

## Train with the stand's pose errors

`lift_late_pose` extends `lift_blocks` with pose errors and a gripper closing time.
The following keys control them:

- `randomization.pose_bias_m` bounds the constant xy pose bias per episode [m].
- `randomization.pose_delay` sets the inclusive delay range as `[max]` or `[min, max]` [policy ticks].
- `randomization.pose_period` bounds the refresh period from 1 through this value [policy ticks].
- `randomization.latch_m` freezes pose updates within this true TCP-to-object distance [m].
- `randomization.grip_close_s` sets the nominal full gripper closing time [s].
- `randomization.block_tolerance_mm` sets the side length tolerance t [mm].
  Each world samples side changes uniformly in [-t, +t] mm each episode.
  Mass and inertia follow the scaled dimensions. `lift_blocks` sets 2 mm, which `lift_late_pose` inherits.
  The policy's descriptor keeps the catalog dimensions, as on the arm.
  Values must be above 0 and at most 5 mm. Use `null` to disable it.

Scale sharing preserves radial and yaw symmetry. Cylinders stay round.
Blocks with quarter-turn yaw symmetry share x and y scales. Their height varies independently.
Size tolerance rejects blocks with several parts, an off-centre part or a part that is not axis-aligned.

For dispatched training, an exported station must sit in the checkout as `stations/<name>`
(gitignored), because the dispatcher's sync skips `workspace/`.
Pass it as `OVERRIDES="station=stations/<name>"`:

```bash
twin export stations/desk1
EXPERIMENTS=lift_late_pose OVERRIDES="station=stations/desk1" bash jobs/train.sh
```

## Add an experiment

Add a YAML file under [`conf/experiment`](../conf/experiment) with a global
package directive. Select groups in `defaults` and set the values for the run.
For example, `short_lift.yaml` can contain:

```yaml
# @package _global_
defaults:
  - override /task: lift

name: short_lift
ppo:
  iterations: 50
```

Run it with `soarm-lab train experiment=short_lift`. Command-line overrides
apply after the experiment. Unknown config keys are rejected.

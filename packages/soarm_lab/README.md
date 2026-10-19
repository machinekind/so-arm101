# soarm-lab

Simulation, reinforcement learning and inverse kinematics for the SO-101 arm.
Train PPO policies for reaching, lifting and pick-and-place, evaluate them in
MuJoCo, and run trained policies or inverse kinematics through the
[Machinekind digital twin](https://github.com/machinekind/twin).

The lab uses the robot model from [soarm](../soarm/README.md). It can simulate a
bundled station or a physical setup exported by the twin, including its table,
cameras and measured servo dynamics.

## Get started

From the repository root, follow the [setup requirements](../../README.md#get-started):

```bash
uv sync
uv run soarm-lab policies
uv run soarm-lab eval lift-v3 --rand
```

Bundled policies are `reach-v3` and `lift-v3`. Evaluation runs on CPU using MuJoCo;
policy inference uses NumPy. Neither requires PyTorch or a GPU.
See [policy provenance](soarm_lab/policies/README.md) for their upstream
checkpoints, training history and conversion to NumPy.

## Train a policy

GPU training uses MuJoCo Warp and PyTorch PPO. It requires a CUDA-capable GPU,
a CUDA-enabled PyTorch installation, and both the `train` and `torch` extras:

```bash
uv run --package soarm-lab --extra train --extra torch soarm-lab train --task reach
```

Tasks are `reach`, `lift` and `pickplace`. Use `--init lift-v3` to fine-tune the
bundled lift policy. Training ends with CPU evaluation; `--help` lists training options.

Trained policies are saved under `workspace/policies/<name>/` in a checkout.
Set `SOARM_LAB_POLICIES` to choose another directory. Commands accept a policy
name or a path to `policy.npz`; local names take precedence over bundled ones.

## Use a measured station

Export the setup from the twin's environment, then evaluate on it from this workspace:

```bash
twin export ~/stations/desk1
uv run soarm-lab eval lift-v3 --station ~/stations/desk1
```

The export contains `scene.xml`, mesh `assets/` and `station.json` with the setup's
measurements. Pass the same `--station` option when training to use the measured
servo dynamics as the centre of domain randomization. Without it, commands use
the bundled `sim` station. [Station](soarm_lab/station.py) also exposes a Python API.

## Policies and environments

Each policy folder contains `policy.npz` (weights and observation normalization)
and `meta.json` (task, network configuration and evaluation results). The
[Policy API](soarm_lab/rl/policy.py) loads and runs these files with NumPy.
To convert an older PyTorch checkpoint:

```bash
uv run --package soarm-lab --extra torch soarm-lab convert path/to/policy.pt
```

Importing `soarm_lab.rl` registers Gymnasium environments `SoArmLab/Reach-v0`,
`SoArmLab/Lift-v0` and `SoArmLab/PickPlace-v0`:

```python
import gymnasium as gym
import soarm_lab.rl

with gym.make("SoArmLab/Lift-v0") as env:
    observation, info = env.reset()
```

## Run in the twin

With this package installed alongside the twin, its Strategies tab offers:

- `policy`: runs a bundled or trained policy, with tracking and episode limits.
- `ik`: moves the tool centre point toward a goal without training, keeping the gripper fixed.

Both send joint targets through the twin's safety supervisor. Read the
[safety rules](../../docs/safety.md) before running on hardware.
Settings can be supplied through the panel or the twin's configuration:

```yaml
policy:
  policy: reach-v3
ik:
  goal_x: 0.22
  goal_y: 0.0
  goal_z: 0.12
  speed_mps: 0.10
  end_on_reach: false
```

Goals use metres in the robot's base frame. Enable the panel's 3D goal handle
to follow a moving goal; `end_on_reach: false` keeps IK following until stopped.
Lift and pick-and-place require cube poses in `obs.objects["cube"]`. The Strategies
panel supplies goals but no object poses, so those tasks need a runner connected
to an object tracker; without a cube pose, they stop after `wait_for_cube_s`.

See [policy settings](soarm_lab/strategies/policy.py),
[IK settings](soarm_lab/strategies/ik.py) and
[conventions](../../docs/conventions.md) for the full interfaces.

## License

[Apache-2.0](LICENSE), with project attribution in [NOTICE](NOTICE).
The bundled `reach-v3` and `lift-v3` policies are distributed under the same license.
Their `meta.json` files record training settings and evaluation results.

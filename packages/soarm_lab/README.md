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
policy inference uses NumPy. Neither requires JAX, PyTorch or a GPU.
See [policy provenance](soarm_lab/policies/README.md) for their upstream
checkpoints, training history and conversion to NumPy.

## Train a policy

GPU training is PPO in JAX on MuJoCo Warp. It requires a CUDA-capable GPU and the
`train` extra, which installs JAX (with CUDA 12 on Linux), optax and MuJoCo Warp.
It does not use PyTorch:

```bash
uv run --package soarm-lab --extra train soarm-lab train --task reach
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
PyTorch is needed only to convert a checkpoint of the old PyTorch trainer (`policy.pt`)
to `policy.npz`. It comes with the `torch` extra:

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
- `touch`: closes the gripper, moves it above an object from `obs.objects` and down until the arm
  stalls on it; built for coarse, late poses such as the LLM detector's.
- `dream`: runs Dream Labs' Dream-1 model from a prompt and the cameras; see [Dream Labs](#dream-labs).

All send joint targets through the twin's safety supervisor. Read the
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
touch:
  label: cube
  touch_z: 0.03
  max_pose_age_s: 30.0
```

Goals use metres in the robot's base frame. Enable the panel's 3D goal handle
to follow a moving goal; `end_on_reach: false` keeps IK following until stopped.
Lift, pick-and-place and touch require object poses in `obs.objects`. The Strategies
panel passes the twin's perception to every run, so they need a running detector;
without a pose, lift and pick-and-place stop after `wait_for_cube_s` and touch after
`wait_for_pose_s`. Touch works with the `cube` or the `llm` detector. Lift and
pick-and-place need `cube`: they count success only from fresh poses, and the LLM
detector's poses are seconds old ("ostatnie widziane"). Lift accepts those only with
`allow_last_seen: true` and a `max_pose_age_s` above their age. Pick-and-place never
does: it counts only a cube the cameras see lying ("kamery") or the simulation's truth.

`touch` takes x and y from the first usable pose of its `label` and touches at
`touch_z` above the table, because the LLM detector's poses lie on the table top,
1 to 3 cm off, with no size or orientation. Set the detector's `objects` to the
same label. Its poses arrive as "ostatnie widziane", seconds old, so
`max_pose_age_s` is 30 s by default. Success means the arm came down and stalled
on something in the column above the pose; a descent that reaches `touch_z` in free
air is reported as a miss. The hand touches first, because the approach is tilted, so a
touch pushes a light object by a few millimetres. Near the base the hand would hit
the arm's own shoulder; the run then ends before it does. The arm keeps to the
panel's speed cap and is measured against the supervisor's command, so a low cap
slows the run but never reads as a touch.

See [policy settings](soarm_lab/strategies/policy.py),
[IK settings](soarm_lab/strategies/ik.py),
[touch settings](soarm_lab/strategies/touch.py) and
[conventions](../../docs/conventions.md) for the full interfaces.

## Dream Labs

The `dream` strategy runs [Dream-1](https://dreamlabs.ai/research/dream-1/), Dream Labs' robot
model, over the network: camera frames, joint state and a text prompt go out, joint targets come
back. `soarm-lab dream` runs it without the twin, in the simulation or on the arm.

Install the `dream` extra, which brings the `dreamlabs` library with its IK (Python 3.11 to 3.13).
It conflicts with the `train` extra, so the two never install together. Create an API key on the
[Dream Labs platform](https://platform.dreamlabs.ai) and put it in `DREAM_API_KEY`, in
`~/.dreamlabs/credentials`, or in the repository's git-ignored `.env`:

```bash
uv sync --extra dream
uv run --env-file .env soarm-lab dream models
uv run --env-file .env soarm-lab dream sim --prompt "pick up the cube" --record /tmp/dream.mp4
uv run --env-file .env soarm-lab dream arm --prompt "pick up the cube" \
    --camera scene=/dev/video2 --camera wrist=/dev/video4 --speed 0.4 --record ~/run.mp4
```

`models` lists the checkpoints the key may run (`--checkpoint` picks one, default
`dreamlabs/dream-1-beta`). `sim` uses the bundled station with a cube. `arm` drives the arm through
the `feetech` backend and reads OpenCV cameras by index or device path; `--rotate scene=180` turns
a camera mounted upside down. While it runs, a sentence typed and Enter is a new prompt; `q` and
Enter, or Ctrl-C, stops and holds the arm with torque on.

Cameras: the scene camera sits behind and above the arm, looking down at the table as a head
camera would, and the wrist camera looks along the fingers. Dream-1 is sensitive to both views.

Guards: the command ramps from the measured pose at most `--speed` times `robot.yaml`'s `max_vel`
and `--max-step-deg` per tick; targets stay within the robot's limits and the servos' own; the arm
holds while a camera frame is missing or older than a second; the run ends on a servo fault, when
the arm lags its target by 25 degrees, and when no chunk came for two minutes.

Caveats. Dream-1 is a bimanual model trained on YAM arms. It runs on an SO-101 zero-shot through
dream's model of the XLeRobot, two SO-101s. With one arm, this arm is the right one and every tick
also carries a dummy left arm (dream's rest pose, gripper open, the scene frame as its wrist view);
the left answer is ignored. Round trips take about a second, so the arm moves in short bursts. The
Dream client uploads every observation tick, state and frames, to Dream Labs' recorder in the
background. Opening the policy prepares dream's IK, up to a minute the first time on a machine.

In the twin, the strategy's settings come from the `dream:` section:

```yaml
dream:
  prompt: pick up the cube
  checkpoint: dreamlabs/dream-1-beta
  scene_camera: a            # the twin's camera that sees the table; empty: the first camera
  wrist_camera: ""           # the arm's wrist camera; empty: none, its slot gets fill_frame
  fill_frame: scene          # or black: what fills a camera slot no camera has
  max_step_deg: 4.0          # the most a joint target moves per tick [deg]
```

## License

[Apache-2.0](LICENSE), with project attribution in [NOTICE](NOTICE).
The bundled `reach-v3` and `lift-v3` policies are distributed under the same license.
Their `meta.json` files record training settings and evaluation results.

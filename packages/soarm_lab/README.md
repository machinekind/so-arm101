# soarm-lab

Training for the SO-101: RL tasks, a CPU and a GPU environment, PPO, evaluation, the base policies,
and three strategies for the digital twin (`machinekind_twin`): `policy` runs a policy, `ik` drives the TCP by inverse kinematics,
`dream` runs Dream Labs' Dream-1 model (see [Dream Labs](#dream-labs)).
It imports the twin's plugin contracts, `machinekind_twin_api`, and never the twin.
The twin hands it the measured station as data and runs its strategies as plugins.
It imports `soarm` for the robot folder.

## Stack

The training stack is JAX and MuJoCo Warp. The JAX trainer is not written yet.
Until it is, `soarm-lab train` runs the legacy PPO trainer in torch.

Running a policy needs numpy only. The strategies, `soarm-lab eval` and `soarm-lab policies` never import torch.
Policies are `policy.npz` files.

| Install | Brings | For |
|---|---|---|
| `pip install soarm-lab` | numpy, PyYAML, MuJoCo, Gymnasium, `soarm`, `machinekind-twin-api` | the strategies, evaluation, the CPU environments |
| `pip install "soarm-lab[train]"` | MuJoCo Warp, JAX | GPU training; the JAX trainer will use it |
| `pip install "soarm-lab[torch]"` | torch | the legacy torch trainer and `soarm-lab convert` |
| `pip install "soarm-lab[dream]"` | `dreamlabs` (the Dream client and dream's IK), pyserial | the `dream` strategy and `soarm-lab dream` |

The legacy trainer needs both extras and a CUDA build of torch: `pip install "soarm-lab[train,torch]"`.
The torch extra stays until the PPO trainer is rewritten in JAX. It may also be needed on Windows.
A command that needs torch without it fails with "install soarm-lab[torch]".

## Contents

| Path | Holds |
|---|---|
| `soarm_lab/station.py` | `Station`: an exported station, read with plain MuJoCo; `Box`, `CameraView` |
| `soarm_lab/stations/sim/` | the bundled simulated station: `scene.xml` and `station.json`, meshes from `soarm` |
| `soarm_lab/robot.py` | `RobotSpec`: the fields of `soarm`'s `robot.yaml` the lab uses |
| `soarm_lab/kinematics.py` | `RobotKinematics`: degrees and gripper 0..100 <-> MJCF, TCP forward kinematics, IK |
| `soarm_lab/sim.py` | `Scene`: the compiled station scene with the task's objects; renders |
| `soarm_lab/rl/` | tasks (`reach`, `lift`, `pickplace`), `ArmEnv` (Gymnasium, CPU), `Policy` (numpy), evaluation, domain randomization; the legacy torch trainer: `BatchEnv` (MuJoCo Warp, GPU), PPO, `torch_policy` |
| `soarm_lab/policies/` | base policies `reach-v3` and `lift-v3` (`policy.npz` + `meta.json`) |
| `soarm_lab/strategies/` | strategies for the twin, one module per approach; `policy.py`: `PolicyStrategy`, a policy as a strategy; `ik.py`: `IKStrategy`, the TCP to a goal by IK; `dream.py`: `DreamStrategy`, Dream-1 over the network |
| `soarm_lab/cli.py` | the `soarm-lab` command |
| `soarm_lab/dream_run.py` | `soarm-lab dream`: Dream-1 without the twin, in the simulation or on the arm |
| `soarm_lab/paths.py` | where trained policies go |

## Get a station

The twin measures the station: the table and where the arm stands on it, the cameras, the arm's dynamics.
Export it from the twin's environment:

```bash
twin export ~/stations/desk1
```

The folder holds `scene.xml` (the MuJoCo scene: table, lights, the arm under the prefix `robot/`, cameras
with their full intrinsics, the arm's physics options), `assets/` (the arm's meshes) and `station.json`
(the station record: table, cameras, identified dynamics, home pose).

```python
from soarm_lab.station import Box, Station

station = Station.load("~/stations/desk1")      # or Station.load("sim"): the bundled one
model = station.build(objects=[Box("cube", (0.015, 0.015, 0.015), (0.2, 0.0))], grasp_sensors=["cube"])
```

`build` loads `scene.xml` through `mujoco.MjSpec`, adds the task's objects and a contact sensor between each
object in `grasp_sensors` and each jaw, and compiles.
A folder without `assets/` takes the meshes from `soarm`'s robot folder.
`station.dynamics`, the servo dynamics the twin identified on this arm, is the centre of the domain randomization.

## Train and evaluate

```bash
soarm-lab policies                                   # base policies and the ones trained here
soarm-lab eval lift-v3 --rand                        # CPU evaluation in plain MuJoCo
soarm-lab eval workspace/policies/reach-x/policy.npz --station ~/stations/desk1
soarm-lab train --task reach --station ~/stations/desk1   # PPO on the GPU, then CPU evaluation
soarm-lab train --task lift --init lift-v3           # fine-tune a base policy on this station
```

Training runs thousands of MuJoCo Warp worlds.
Today it is the legacy torch trainer, so it needs both extras and a CUDA build of torch:

```bash
pip install "soarm-lab[train,torch]"     # in the workspace: uv sync --extra train
```

Trained policies go to `workspace/policies/<name>/`: relative to the repository root in a checkout,
to the current folder otherwise, or to `SOARM_LAB_POLICIES`.
A policy name is looked up there first, then among the base policies.

## The policy file

A policy is a folder with two files.
`meta.json` holds the task the policy was trained on, its layer sizes and its results.
`policy.npz` holds the weights in float32 and loads with `np.load(allow_pickle=False)`:

| Key | Shape | Holds |
|---|---|---|
| `format` | () | 1 |
| `sizes` | (L + 1,) | obs_dim, the hidden sizes, act_dim |
| `obs_mean`, `obs_var` | (obs_dim,) | the running mean and variance of the observations from training |
| `obs_count` | () | the number of observations behind them |
| `obs_clip` | () | the clip of the normalized observation (5) |
| `actor.{i}.weight` | (sizes[i+1], sizes[i]) | layer i of the actor, the layout of torch's `nn.Linear` |
| `actor.{i}.bias` | (sizes[i+1],) | |
| `critic.{i}.weight`, `critic.{i}.bias` | | the critic, sizes obs_dim, hidden, 1; optional, kept for fine-tuning |

The action is `clip(actor(clip((obs - obs_mean) / sqrt(obs_var + 1e-8), -obs_clip, obs_clip)), -1, 1)`.
Every actor layer is `x @ weight.T + bias`, and every layer but the last is followed by ELU.
`soarm_lab.rl.policy.Policy` reads, runs and writes the file with numpy.

A torch checkpoint, `policy.pt`, is the older policy format.
`soarm-lab eval` and the strategy refuse it and name the command that converts it:

```bash
pip install "soarm-lab[torch]"
soarm-lab convert workspace/policies/reach-x/policy.pt    # writes policy.npz and meta.json next to it
```

The Gymnasium environments register on import:

```python
import gymnasium as gym, soarm_lab.rl
env = gym.make("SoArmLab/Lift-v0")
```

## The policy in the twin

`pyproject.toml` registers the strategies:

```toml
[project.entry-points."machinekind_twin.strategies"]
policy = "soarm_lab.strategies.policy:PolicyStrategy"
ik = "soarm_lab.strategies.ik:IKStrategy"
dream = "soarm_lab.strategies.dream:DreamStrategy"
```

With `soarm-lab` installed next to the twin, the panel's Strategies tab lists `policy` and `ik`.
Its form comes from `PolicySettings`, prefilled from the `policy:` section of the twin's config file:

```yaml
policy:
  policy: lift-v3            # a base policy, one trained here, or a path to policy.npz
  max_tracking_deg: 25.0     # end the run, holding the measured pose, when a joint lags this much
  episode_limit: true        # end after a training episode's number of ticks
  wait_for_cube_s: 3.0       # lift, pickplace: how long to wait for the first cube pose
  goal_x: 0.22               # the goal [m, base frame] when the twin gives none
  goal_y: 0.0
  goal_z: 0.12
```

The twin's `StrategyRunner` claims the arm and calls the strategy at the policy's rate (20 Hz).
Every target goes through the twin's safety supervisor, in simulation and on hardware.
The strategy subclasses `machinekind_twin_api.Strategy`. It takes a `RunInfo` and an `Observation` and returns an `Action`.
It reads these fields of the observation:

- `obs.joints`: measured joints;
- `obs.command`: the supervisor's command, for a gripper that already squeezes a cube;
- `obs.goal`: the goal (reach: the TCP target; pickplace: the place target);
- `obs.objects["cube"]`: the cube's pose in the base frame and its `source`.

`run.limits` narrows the targets to the hardware's limits.
`run.folder` gives the robot description.
A policy trained on another robot refuses the run in `start`.

The twin's Strategies tab passes neither a goal nor objects today.
Reach then drives to the goal in the settings.
Lift and pickplace wait `wait_for_cube_s` for a cube and then end the run.
They need a runner that passes `objects`, e.g. the twin's cube tracker as an `ObjectPose` with its source.
`tests/strategies/test_lift_from_cameras.py` builds one.

## Inverse kinematics in the twin

The strategy `ik` drives the TCP to a goal point with `RobotKinematics` alone: no policy, no training.
Its form comes from `IKSettings`, prefilled from the `ik:` section of the twin's config file:

```yaml
ik:
  goal_x: 0.22               # the goal [m, base frame] when the twin gives none
  goal_y: 0.0
  goal_z: 0.12
  hz: 20.0                   # ticks per second
  speed_mps: 0.10            # highest TCP speed along the path
  tol_m: 0.005               # the measured TCP is at the goal within this
  end_on_reach: true         # end the run once at the goal; false: keep following a goal the panel drags
  max_tracking_deg: 25.0     # end the run, on the measured pose, when a joint lags this much
```

Every tick a point moves from the TCP of the last target straight toward the goal, by at most `speed_mps / hz`.
`RobotKinematics.ik_step`, started from the last target, turns it into joint targets, clipped to `run.limits`.
The gripper never moves.

`start` refuses a goal the IK cannot reach within the limits.
A goal dragged during the run is not checked again; when the arm cannot follow it, the note says so.
The run ends as a success once the last target puts the TCP on the goal and the measured TCP is within `tol_m`.

## Dream Labs

The strategy `dream` runs Dream-1, Dream Labs' robot model, over the network: camera frames and a prompt go
out, joint targets come back. `soarm-lab dream` runs it without the twin, in the simulation or on the arm.

Install the `dream` extra. It brings the `dreamlabs` library: the Dream client and dream's IK (`dream_ik`),
which an SO-101 needs (Python 3.11 to 3.13; jax, MuJoCo, OpenCV). The extra conflicts with `train`, so the
two never install together.

```bash
uv sync --extra dream                 # a plain `uv sync` removes it again; `uv run` keeps it
```

The API key comes from `DREAM_API_KEY` or `~/.dreamlabs/credentials` (the four lines the Dream Labs platform
prints; keep the file `chmod 600`). `DREAM_API_KEY=...` in the repository's `.env`, which git ignores, also
works: `uv run --env-file .env soarm-lab dream ...`. Nothing here prints the key.

```bash
uv run soarm-lab dream models                                    # the checkpoints the key may run
uv run soarm-lab dream sim --prompt "pick up the cube" --record /tmp/dream.mp4
uv run soarm-lab dream arm --prompt "pick up the cube" --camera scene=0 --camera wrist=1 --speed 0.3
```

`--checkpoint` picks the model (default `dreamlabs/dream-1-beta`). `sim` runs the bundled station with a cube
on the table: a station camera in the scene slot, the arm's `robot/wrist_cam` in the wrist slot. `arm` drives
the arm through the `feetech` backend (the port as `soarm-panel` finds it, or `--port`) and reads OpenCV
cameras by index or device path (on macOS the terminal needs camera access). While it runs, a line typed and
Enter is a new prompt: the arm holds and a fresh episode starts. `q` and Enter, or Ctrl-C, stops: the arm
holds where it is measured, torque on.

The run's guards: the command ramps from the measured pose at most `--speed` x `robot.yaml`'s `max_vel` and
`--max-step-deg` per tick (4 at 20 Hz); targets stay within `robot.yaml`'s limits and the servos' own; the arm
holds while a camera frame is missing or older than 1 s; the run ends on a servo fault, when the arm lags its
target by 25 deg, and when no chunk came for 120 s.

Dream-1 is a bimanual model trained on YAM arms. It runs on an SO-101 zero-shot through dream's model of the
XLeRobot, two SO-101s (`ik` arm `so101_faas`). The client then hides the arm's state from the model: it acts on
the cameras and the prompt, and its answer is a hand path the client solves back to joints with dream's IK.
With one arm, this arm is the right one, and every tick also carries a dummy left arm: dream's left rest pose,
the gripper open, and the scene frame (or black, `--fill black`) as the left wrist camera. The left answer is
ignored. The joints go out as `deg2rad` of this repository's degrees: dream's SO-101 URDF has the same joint
frames, zero and directions as the MJCF here. Frames go out at 320x176, JPEG quality 85.

The Dream client uploads every observation tick it is handed (the state and the frames) to Dream Labs'
recorder, in the background, under one robot id per run. Opening the policy prepares dream's IK, which takes
up to a minute the first time on a machine (it is cached in `~/.cache/dream_ik`), and a model that is not
running yet answers only after it starts.

In the twin, `dream` reads its form from `DreamSettings`, prefilled from the `dream:` section:

```yaml
dream:
  prompt: pick up the cube
  checkpoint: dreamlabs/dream-1-beta
  scene_camera: a            # the twin's camera that sees the table; empty: the first camera
  wrist_camera: ""           # the arm's wrist camera; empty: none, its slot gets fill_frame
  fill_frame: scene          # or black: what fills a camera slot no camera has
  max_step_deg: 4.0          # the most a joint target moves per tick [deg]
  max_wait_s: 120.0          # end the run when no chunk came for this long
```

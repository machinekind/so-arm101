# soarm-lab

Training for the SO-101: RL tasks, a CPU and a GPU environment, PPO, evaluation, the base policies,
and a strategy that runs a policy in the digital twin (`lerobot_mp`).
It does not import the twin. The twin hands it the measured station as data and runs its strategy as a plugin.
It imports `soarm` for the robot folder.

## Contents

| Path | Holds |
|---|---|
| `soarm_lab/station.py` | `Station`: an exported station, read with plain MuJoCo; `Box`, `CameraView` |
| `soarm_lab/stations/sim/` | the bundled simulated station: `scene.xml` and `station.json`, meshes from `soarm` |
| `soarm_lab/robot.py` | `RobotSpec`: the fields of `soarm`'s `robot.yaml` the lab uses |
| `soarm_lab/kinematics.py` | `RobotKinematics`: degrees and gripper 0..100 <-> MJCF, TCP forward kinematics, IK |
| `soarm_lab/sim.py` | `Scene`: the compiled station scene with the task's objects; renders |
| `soarm_lab/rl/` | tasks (`reach`, `lift`, `pickplace`), `TwinEnv` (Gymnasium, CPU), `BatchEnv` (MuJoCo Warp, GPU), PPO, `Policy`, evaluation, domain randomization |
| `soarm_lab/policies/` | base policies `reach-v3` and `lift-v3` (`policy.pt` + `meta.json`) |
| `soarm_lab/strategy.py` | `PolicyStrategy`: a policy as a strategy of the twin |
| `soarm_lab/cli.py` | the `soarm-lab` command |
| `soarm_lab/paths.py` | where trained policies go |

The code in `rl/`, `kinematics.py` and the logic of `strategy.py` come from the twin (`twin/rl/`, `twin/kinematics.py`).
Their comments stay in Polish, as in the twin.

## Get a station

The twin measures the station: the table and where the arm stands on it, the cameras, the arm's dynamics.
Export it from the twin's environment:

```bash
lerobot-twin export ~/stations/desk1
```

The folder holds `scene.xml` (the MuJoCo scene: table, lights, the arm under the prefix `robot/`, cameras
with their full intrinsics, the arm's physics options), `assets/` (the arm's meshes) and `station.json`
(the twin's station record: table, cameras, identified dynamics, home pose).

```python
from soarm_lab.station import Box, Station

station = Station.load("~/stations/desk1")      # or Station.load("sim"): the bundled one
model = station.build(objects=[Box("cube", (0.015, 0.015, 0.015), (0.2, 0.0))], grasp_sensors=["cube"])
```

`build` loads `scene.xml` through `mujoco.MjSpec`, adds the task's objects and the jaw contact sensors as the
twin's scene does, and compiles. A folder without `assets/` takes the meshes from `soarm`'s robot folder.
`station.dynamics` (the twin's sysid result) is the centre of the domain randomization.

## Train and evaluate

```bash
soarm-lab policies                                   # base policies and the ones trained here
soarm-lab eval lift-v3 --rand                        # CPU evaluation in plain MuJoCo
soarm-lab eval workspace/policies/reach-x/policy.pt --station ~/stations/desk1
soarm-lab train --task reach --station ~/stations/desk1   # PPO on the GPU, then CPU evaluation
soarm-lab train --task lift --init lift-v3           # fine-tune a base policy on this station
```

Training runs thousands of MuJoCo Warp worlds and needs the `train` extra and a CUDA build of torch:

```bash
pip install "soarm-lab[train]"     # in the workspace: uv sync --extra train
```

Trained policies go to `workspace/policies/<name>/`: relative to the repository root in a checkout,
to the current folder otherwise, or to `SOARM_LAB_POLICIES`.
A policy name is looked up there first, then among the base policies.

The Gymnasium environments register on import:

```python
import gymnasium as gym, soarm_lab.rl
env = gym.make("SoArmLab/Lift-v0")
```

## The policy in the twin

`pyproject.toml` registers the strategy:

```toml
[project.entry-points."lerobot_mp.strategies"]
policy = "soarm_lab.strategy:PolicyStrategy"
```

With `soarm-lab` installed next to the twin, the panel's Strategies tab lists `policy`.
Its form comes from `PolicySettings`, prefilled from the `policy:` section of the twin's config file:

```yaml
policy:
  policy: lift-v3            # a base policy, one trained here, or a path to policy.pt
  max_tracking_deg: 25.0     # end the run, holding the measured pose, when a joint lags this much
  episode_limit: true        # end after a training episode's number of ticks
  wait_for_cube_s: 3.0       # lift, pickplace: how long to wait for the first cube pose
  goal_x: 0.22               # the goal [m, base frame] when the twin gives none
  goal_y: 0.0
  goal_z: 0.12
```

The twin's `StrategyRunner` claims the arm and calls the strategy at the policy's rate (20 Hz).
Every target goes through the twin's safety supervisor, in simulation and on hardware.
The strategy reads the observation by its attributes:

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
`tests/test_lab_lift_from_cameras.py` builds one.

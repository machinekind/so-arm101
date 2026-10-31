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

### Lift a block

One lift policy can learn several wooden blocks (Classic World, maple, 2.5 cm unit):
`cube`, `cuboid5`, `cuboid85` and `cylinder`. Their sizes and masses are placeholders
until measured. The table is in [rl/objects.py](soarm_lab/rl/objects.py), and its
`MEASURED` flag says whether the numbers are measured. With `--kinds`, the policy also
sees the block's kind and half sizes, 7 inputs after the 33 of the cube policy. Each
kind gets an equal share of the worlds, so `--envs` must divide by the number of kinds:

```bash
uv run --package soarm-lab --extra train soarm-lab train --init lift-v3 --kinds cube,cuboid5,cuboid85,cylinder
```

`--init lift-v3` widens the bundled cube policy to the new inputs with zero weights,
so training starts from a policy that already grasps a cube. Success counts the
block's lowest point above the table, so a block stood on its end is not lifted.
After training, the CPU evaluation runs once per kind, and `soarm-lab policies` shows
the mean and the worst kind. `soarm-lab eval <policy> --kind cylinder` evaluates one
kind. In the twin, the `policy` strategy's `object_kind` setting names the block the
detector sees; the block's size comes from the table, not from the detector.

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
- `touch_jev`: `touch` plus a decision step, in which TypeSafe's Jev model says after each descent
  whether the gripper touched the object and where to aim after a miss; see
  [Jev in the touch loop](#jev-in-the-touch-loop).
- `grasp`: `touch`'s probe on a block of the Classic World set, then rise, open, straddle, close and
  lift; the jaw rule verifies the hold, and the strategy grasps the block again when it drops out of
  the jaws; see [Grasp a block](#grasp-a-block).
- `grasp_jev`: `grasp` plus Jev after the probe, the close and the lift, which picks the next step; see
  [Jev in the grasp loop](#jev-in-the-grasp-loop).
- `grasp_wrist`: `grasp` plus a look through the wrist camera at each hover: the block's colour blob is
  fitted with its known shape, the jaws are centred on it and turned across its short side; see
  [Wrist camera](#wrist-camera).
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
touch_jev:
  label: cube
  model: jev-1.14.0
  min_confidence: 0.8
  max_attempts: 3
grasp:
  object_kind: cube
  grasp_source: geometry   # or library
  open_margin_m: 0.008
  squeeze_m: 0.008
  grasp_depth: 0.5
  lift_m: 0.04
  carry_s: 10.0
  wait_s: 20.0
  time_limit_s: 120.0
grasp_jev:
  object_kind: cube
  model: jev-1.14.0
  min_confidence: 0.8
  send_frames: false
  wrist_camera: wrist
grasp_wrist:
  object_kind: cuboid5
  wrist_camera: wrist
  hue_deg: -1.0
  centre_tol_m: 0.003
  look_before_straddle: true
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
[touch settings](soarm_lab/strategies/touch.py),
[touch_jev settings](soarm_lab/strategies/touch_jev.py),
[grasp settings](soarm_lab/strategies/grasp.py),
[grasp_jev settings](soarm_lab/strategies/grasp_jev.py),
[grasp_wrist settings](soarm_lab/strategies/wrist.py) and
[conventions](../../docs/conventions.md) for the full interfaces.

## Jev in the touch loop

`touch_jev` runs `touch` and asks [Jev](https://docs.typesafe.ai), TypeSafe's decision model, about
each descent: did the gripper touch the object, and if it missed, where should it aim next. After a
miss the arm rises straight up, moves to the new hover point and descends again, within the same run.

Put your TypeSafe API key in `TYPESAFE_API_KEY` in the twin's environment, for example in the
repository's git-ignored `.env`, and start the twin with it:

```bash
uv run --env-file .env twin ui
```

Without the key the strategy refuses the run before the arm moves. The model is a pinned version,
`jev-1.14.0`, because the confidence thresholds belong to one version; an alias such as `jev-latest`
moves with new releases.

What goes to `api.typesafe.ai`: one request per descent, and a second only after a timeout, an
overload or a dropped connection. Each holds the model, a JSON state and two questions. The state says
how the descent ended, how high it stopped against the object's top, where the latest pose lies
relative to the touch point and whether it is a new look, the earlier descents, and the corrections
that would repeat a point already tried. It holds no joint values, not the contact
geometry's own verdict, and never the key. Camera frames go out only with `send_frames: true`, as PNG
images; TypeSafe does not yet guarantee image input for production, and Jev refuses a frame unless both
of its sides lie within 280 to 1568 px.

| Jev's answer | The run |
| --- | --- |
| touched, confidence at least `min_confidence` (0.8) | ends as a success |
| blocked above the object | ends |
| missed, on the last of `max_attempts` (3) | ends |
| missed, correction confidence at least `min_correction_confidence` (0.6) | rises, aims at the corrected point, descends again |
| missed, with a less certain correction, or one that gives no new reachable point | ends for review |
| any verdict below `min_confidence` | ends for review |
| no answer within `answer_timeout_s`, an overload or a dropped connection | asks once more after 0.5 s, then ends for review |
| a refused key or request, an unusable answer | ends for review |

A run that ends for review leaves the arm holding the pose it stopped in, and its stop reason ends with
"do oceny czlowieka". An arm that never came down is not asked about; it ends as in `touch`. When Jev's
verdict differs from the contact geometry's, the stop reason adds "geometria: ...". A failed call is
never read as an answer.

Every call gives one record per question (`judges.Decision`): the answer as received or the error, the
exact state sent, the confidence and the response time. The strategy keeps them in `judging.decisions`,
writes one log line per call, and puts a summary in the panel's note and stop reason. The twin API this
workspace pins (0.1.2) has no place for them in an `Action`. Once the pin moves to 0.1.3 they go to
`Action.decisions`, and the twin's episode log keeps them.

Caveats:

- Confidence measures how concentrated Jev's probabilities are, not the chance that the answer is
  right. The thresholds are TypeSafe's starting points (0.8 to act alone, 0.6 as a floor) and are not
  yet tuned for this task.
- Jev leans toward the first option of a question. "Missed" comes first, so that bias costs one more
  descent rather than a false success.
- Without frames Jev sees only what the geometry measured, so it mostly repeats `touch`'s own verdict.
  The frames are the A/B that can add information.
- `need_contact` only changes the geometry label recorded next to each answer; the model decides.
- Model waits count toward `time_limit_s`, 90 s by default.

Each record holds the exact state, the exact questions and their version (`touch_jev/1`), so the same
input can be put to Claude or to a person to compare accuracy. A Claude judge would implement the
`Judge` protocol in [judges.py](soarm_lab/strategies/judges.py); it is not built yet. It needs the
process handling of the twin's `machinekind_twin_llm.cli` copied here, not imported (that package is
only in the `twin` group and loads OpenCV), a fake CLI for its tests, and an `answer_timeout_s` near
its 60 s limit, because the CLI answers in seconds where Jev answers in about 0.1 s.

## Jev in the grasp loop

`grasp_jev` runs `grasp` and asks Jev at three points of every attempt, through the same client, key,
retry and records as `touch_jev`. Without `TYPESAFE_API_KEY` it refuses the run before the arm moves.
`grasp` itself never asks.

1. **The probe**, when the closed tips have stalled on the block: `touch_jev`'s two questions (touched,
   correction) on its state, with the block centre as the touch point. A confident "touched" goes on into
   the rise, the open jaw and the straddle. A "touched" on a probe that met nothing, or that stalled more
   than `contact_band_m` above the probe's bottom, ends the run for review: the straddle needs the
   measured top. A miss is corrected one `correction_m` step as in `touch_jev`.
2. **The close**: after a held close on the first tick of the lift, after an empty close or a jaw tip on
   the block while the arm holds where it stopped (phase `judge`).
3. **The lift**, on the first tick of the hold check at the top.

At the close and the lift the state holds facts computed in code, in named buckets: how the close ended;
the measured jaw gap against the block's short side (`closed`: a jaw commanded tighter than the block
reached its command; `narrower_than_block`; `block_wide`, within 3 mm of the short side; `open`); whether
the jaw stands and whether it moved since the close; how far the arm lifted (`not_lifted`, `partly`,
`lifted`); whether the wrist camera shows the block between the jaws (`unknown` until a blob is wired in,
see below); the twin's latest pose of the block against the jaws (`at_the_jaws`, `near_the_jaws`,
`elsewhere`) with its source, annotated sources named apart (`last_seen_dropped` for "ostatnie widziane
(upuszczona)"); whether the jaws' heading came from the block's rotation or was a guess; the earlier
attempts; and `available_next`, the steps the geometry allows. Two questions go out: `held`
(nothing_in_jaws, block_held, held_by_edge) and `next` (lift, regrasp_wider, regrasp_rotated, give_up).

| Jev's answer | The run |
| --- | --- |
| `held` below `min_confidence` (0.8), or `next` below `min_correction_confidence` (0.6) | ends for review |
| a `next` not in `available_next` | ends for review |
| regrasp on the last of `max_attempts` (3) | ends |
| lift | goes on; the lift's time limit and the 1 s hold count from the answer, not from the ask |
| give_up | ends with the block where it is: a held block stays in the jaws |
| regrasp_wider | the jaw opens 16 mm wider on the moving jaw's side for every later straddle |
| regrasp_rotated | the jaws turn 90 degrees at the same block centre |
| no answer within `answer_timeout_s`, an overload or a dropped connection | asks once more after 0.5 s, then ends for review |
| a refused key or request, an unusable answer | ends for review |

The `held` answer never moves the arm. A disagreement with the jaw rule is counted, and a review or stop
reason says "szczeka: trzyma" or "szczeka: pusta". Lift is available only after a held close.
regrasp_rotated is available only for a heading that was a guess on a block a turn changes, so never on
the cube, and regrasp_wider only while the jaw can open that far.

The wider opening goes all to the moving jaw's side. The SO-101 closes one jaw and the block always ends
against the fixed tip, so a fixed tip farther out pushes the block farther across the table: in the twin
a 16 mm push ran the cube up onto the fixed tip. After a tip on the block the regrasp probes half the
short side toward the moving jaw first, as `grasp`'s own retry does; after an empty close it probes the
same block centre again. A block in the jaws is first put back. On the table the jaw opens there (phase
`let_go`). In the air the arm first lowers it to the grasp point (phase `lower`), and a stall on the way
counts as arrival, so nothing presses into the table. The jaw opens before the arm rises and closes only
at the hover, and the next probe aims at the centre between the fingertips.

A block that drops out of the jaws goes to `grasp`'s recovery (see [Recovery](#recovery)), also while
it is lowered: a close or lift question in flight or in its retry pause is cancelled and recorded as
such, and the regrasp's probe is asked about as any other. The plan's ambiguous row, a held block the
cameras see elsewhere, reaches Jev through the state's `block_sighting`.

With `send_frames: true` only the wrist camera's frame goes out, named by `wrist_camera` as in robot.yaml.
The MJCF camera renders 1920x1080; the frame is halved by integer striding to 960x540 to fit Jev's 1568
px. A frame below 280 px, a missing one, or one older than 1 s is left out with its reason in
`frames_left_out`: a stalled stream keeps serving its last image with a growing age. The state then
always carries `camera_views`, `frames_left_out`, `frames_scaled` and `frames_age_s`.

`GraspJevStrategy(..., blob=fn)` takes a function of the wrist frame that gives the share of the area
between the jaws the block's colour fills, 0..1; 0.15 or more is `yes`. Nothing passes one yet: the twin
builds the strategy from its settings alone, so the state says `unknown` until a blob from `grasp_wrist`'s
colour mask (`wrist.segment`) is wired in.

Every call gives one record per question, with `point` (probe, close, lift) in its state beside the
questions' version (`touch_jev/1` at the probe, `grasp_jev/1` otherwise), and one "Ocena chwytu" log line.
A run takes up to three answers per attempt, each up to `answer_timeout_s` plus a retry, so
`time_limit_s` defaults to 180 s. test_grasp_jev_in_twin.py runs the strategy in the twin with a judge
that answers from the jaws' contact sensors.

## Grasp a block

`grasp` picks up one block of the Classic World set (`rl/objects.py`: `cube`, `cuboid5`, `cuboid85`,
`cylinder`) that the detector names, lifts it 4 cm and checks that the jaw holds it. It starts as
`touch`: the closed jaw goes straight down onto the block until the arm stalls. Then:

1. The arm rises to the hover with the jaw closed.
2. It moves 16 mm sideways and opens the jaw to the block's short side plus twice `open_margin_m`
   (41 mm for the 25 mm blocks), so the open jaw is centred on the block.
3. It goes straight down until the tips' centres are `grasp_depth` of the block's height below its top.
4. It closes the jaw to the short side minus `squeeze_m` (17 mm) and waits until the jaw stands.
5. When the jaw holds, it lifts `lift_m` straight up and checks the hold for `hold_s`.
6. It carries the block there for `carry_s` (10 s) while it watches the jaw, then ends with the block
   held, or with `release: true` opens the jaw.

Every number comes from the block and the model, not from a table. `rl.objects.object_spec` gives the
block's size. `RobotKinematics.jaw_gap` turns an opening in metres into a gripper command,
`jaw_centre` gives where the fingertips' midpoint sits against the TCP (15.6 mm toward the moving jaw
when closed, centred when open), and `jaw_reach` gives how far the jaw's sole reaches below the TCP
(6.6 mm). With `touch_z: 0` the probe's bottom lies 10 mm under the block's top plus that reach. The TCP
height at the stall minus the reach is the block's measured top, and the grasp height follows from it.

The jaw rule (`strategies/jaw.py`) decides whether the jaw holds the block: it stands still, wider than
commanded and wider than closed by more than `jaw.margin`. A jaw that has not started closing also
looks like that, so the close counts only after the jaw has left its opening. From then on the jaw is
watched on every tick (see [Recovery](#recovery)). The twin's tracker uses the same thresholds,
so it reports the block "w dloni" on the tick the strategy counts it held. The SO-101 closes only one
jaw: the moving jaw pushes the block `open_margin_m` across the table onto the fixed jaw's tip.

Once the pose is taken the hand points straight down, with the jaws across the block's short side (the
pose's yaw, folded by the block's symmetry) or, when the pose has no rotation, on their current heading.
The hand turns toward that orientation at 90 deg/s. The SO-101 reaches a top-down hand only in a band.
From the rest pose, for a pose without rotation and the 4 cm hover, a block straight ahead must lie at x
0.14 to 0.27 m. At 10 cm to the right (y -0.10) the band is 0.12 to 0.25 m, and at 10 cm to the left
(y +0.10) 0.03 to 0.25 m. With `touch`'s 6 cm hover the band shrinks to 0.18 to 0.24 m straight ahead,
so `hover_m` and `lift_m` default to 4 cm. Beyond the band `wrist_flex` stops at its 95 degree limit,
and the run ends at once with "chwytak nie siega tam pionowo".

When an open jaw tip lands on the block during the straddle, the arm holds where it stopped, rises,
closes the jaw once its tips are 10 mm clear of the block, and probes again half the short side along
the jaws, toward the moving jaw first. If that probe misses, the block lies on the other side, and the
next probe goes half the short side the other way. A tip stall costs one extra probe when the first
shift picks the right side and two when it does not; `max_attempts` (3) covers both. A close that finds
nothing ends the run: after a straddle that met nothing, the jaw alone cannot say whether the probe
pushed the block away or it sits at the jaws' edge.

| How the run ends | Stop reason |
| --- | --- |
| the jaw holds for `hold_s` at the top and through `carry_s` | blok w szczekach, podniesiony o N mm, niesiony T s (szczeka G mm) |
| no top-down hand at the block | punkt (...) m: chwytak nie siega tam pionowo, ... poza zasiegiem ramienia, ...: chwytak nie ustawi sie pionowo (blad N st.), ...: ramie uderzyloby samo w siebie |
| the probe misses, is blocked or stuck | touch's reasons, such as brak kontaktu w punkcie dotyku - obiektu tam nie ma |
| tip stalls or misses beyond `max_attempts` | wyczerpane proby chwytu (N): ... |
| the hand is not top-down at a hover | chwytak nie ustawil sie pionowo nad blokiem (N st.) |
| the jaw does not open | szczeka nie otworzyla sie w 3 s (A mm zamiast B mm) |
| the arm does not stand over the block | ramie nie stanelo nad blokiem (N mm od punktu) |
| the jaw does not start closing | szczeka nie ruszyla po rozkazie zamkniecia (G mm) |
| the jaw does not stand after the close | szczeka nie staje 3 s po zamknieciu - serwo albo odczyt |
| the jaw closes on nothing | chwyt pusty - blok przesuniety albo obok szczek |
| the arm does not reach the lift point 3 s after the lift's travel | ramie nie podnioslo bloku (N mm z 40 mm) |
| the jaw does not stand at the top | szczeka nie stoi po podniesieniu - blok sie rusza w szczekach |
| the block drops out of the jaws | see [Recovery](#recovery) |

The lag guard, the time limit (`time_limit_s`, 120 s) and the panel's Stop end a run as in `touch`.

The block stays in the jaws after "zadanie wykonane". The last target squeezes, the twin's runner sends
the final tick's target before it disengages, and disengaging keeps the supervisor's command. A later
STOP or Home keeps the squeeze through the twin's `hold_measured`. The operator releases the block. With
`release: true` the strategy opens the jaw to the straddle opening after `carry_s` instead and ends with
"blok oddany" once the jaw stands open; a block nobody takes falls to the table.

A run on the twin panel's simulated block currently ends with "brak kontaktu w punkcie dotyku - obiektu
tam nie ma". The twin builds the block with MuJoCo's default contact, because its `scene.Box` has no
contact settings. The arm's push sinks the 11 g cube 8 mm into the table instead of stalling, and the
probe reaches its bottom as if in free air. Until the twin's blocks get a contact of their own, set
`need_contact: false` for a simulated block. A probe that reaches its bottom then counts, the grasp
height comes from the block's nominal top, and the run lifts the cube. A missing block then shows only
at the close ("chwyt pusty"). Keep `need_contact` on for the real arm, where the stall is the only sign
that the block is there. test_grasp_in_twin.py runs both cases at the default contact. Its other runs
make the block's contact as stiff as the 5 ms step allows. The cube then still gives about 2.5 mm under
the probe, so the measured top is that much low. The real pads' sole and the real jaw's opening per unit
are not measured yet; `touch_z` above zero overrides the probe height.

### Recovery

From the close until the release or the end of the run, every tick of the lift, the check and the carry
compares the jaw rule with the block's pose in `obs.objects`, as step 6 of the
[grasp plan](../../docs/grasp-plan.md) has it:

| Jaw rule | The block's pose | Meaning | Action |
| --- | --- | --- | --- |
| squeezing | any, "w dloni" as a rule | held | carry on |
| squeezing | "kamery" on the table, more than 6 cm from the fingertips | a second block or a tracker error | carry on; the carry's note says so |
| squeezing, creeping closed | any | the block turns or slides in the jaws | the check's hold starts again; counted while carrying |
| empty on two ticks in a row | any | dropped | rise to the safe height and open the jaw |
| (jaw open) | seen on the table since the drop, within `recover_m` | dropped where the cameras see it | grasp it again there |
| (jaw open) | seen farther than `recover_m` from where it fell | taken away | stop for a person |
| (jaw open) | not seen since the drop for `wait_s` | taken away, or the cameras cannot tell | stop for a person |

A sighting is a plain "kamery" or "ostatnie widziane" pose no older than `max_pose_age_s`, lying at rest
height (the block's largest half size plus 1 cm) and captured 0.3 s or more after the drop. The LLM
detector's answers arrive seconds old and the twin shows them as "ostatnie widziane", so on that path the
plain last-seen pose is the sighting. Annotated poses never count: "ostatnie widziane (upuszczona)" is the
tracker's guess under the hand, and "(1 kamera widzi ja gdzie indziej)" keeps the old position. Neither
does "w dloni", nor a pose captured before the drop. In simulation the scene's own pose counts when it
lies at rest height.

After a drop the arm rises straight up to the probe's hover height, the safe height, or stays where it is
when that is higher, and opens the jaw to 0.50 rad (55 mm between the tips). The twin's tracker accepts a
single camera's detection next to a standing hand only when three detections within 1.5 s agree and the
measured jaw stands wider than its `open` threshold, 0.40 rad, without squeezing. The opened jaw passes
that gate, and a block still hanging in the tips falls where the cameras see it. The seeded detector
(every 0.2 s) and the cube detector (every 0.1 to 0.33 s) settle within a second. The LLM detector alone
answers every 10 s, so three answers never fall within 1.5 s: with one camera it never gets a sighting
next to the hand, and the run stops after `wait_s`. Recovery on that path needs a second camera or the
seeded detector.

The regrasp goes through `retarget`. The jaw closes at the hover before the probe, the probe measures the
block's top again, and the grasp repeats. Its descent counts against `max_attempts` together with the
sideways retries, so a demonstration with two snatches needs `max_attempts` of 4 or more. A snatch is not
a miss: a block seen farther than `recover_m` (15 cm) from where it fell is never chased. A regrasp whose
probe meets nothing ends as `touch` does.

| How a run ends after a drop or a release | Stop reason |
| --- | --- |
| seen farther than `recover_m` from where it fell | blok zabrany: lezy N cm od miejsca upadku - czekam na czlowieka |
| nothing seen for `wait_s`; the cube and seeded detectors drop the pose 6 s after the drop | blok zabrany ze szczeki: kamery go nie widza od N s - czekam na czlowieka |
| the LLM detector alone: its pose lowered under the hand is held 30 s or more | blok zabrany ze szczeki: zrodlo 'ostatnie widziane (upuszczona)' od N s - czekam na czlowieka |
| a pose above the table, as in a person's hand | blok zabrany ze szczeki: zrodlo 'kamery', N cm nad blatem od N s - czekam na czlowieka |
| only a pose captured before the drop | blok zabrany ze szczeki: zrodlo 'ostatnie widziane' sprzed upadku od N s - czekam na czlowieka |
| the block lies out of reach | blok upadl poza zasiegiem: punkt (...) m poza zasiegiem ramienia - czekam na czlowieka |
| `max_attempts` spent | wyczerpane proby chwytu (N): blok wypadl ze szczeki |
| the jaw has not closed at the hover | szczeka nie zamknela sie przed zejsciem w 3 s |
| `release: true`, the jaw opened | blok oddany: szczeka otwarta (G mm) |
| `release: true`, the jaw did not open | szczeka nie otworzyla sie po oddaniu bloku w 3 s (G mm zamiast O mm) |

A run that stops for a person leaves the arm at the safe height with the jaw open: the twin's runner
sends the last target and disengages, and the supervisor holds that command.

`carry_s` (10 s) is the window in which a demonstrator can knock the block out or take it; 0 ends the run
at the verified lift. `wait_s` (20 s) covers two LLM answers when two cameras see the block. The default
`time_limit_s` (120 s) fits a grasp of about 10 s, its carry, one drop with its wait, a regrasp and a
second carry, about 65 s in all. `release()` is the seam a place task calls to end the watch and open the
jaw; it returns the tick's answer, so a `carrying` override can return it.
test_grasp_from_cameras.py pulls the cube out of the jaws mid-carry in the twin, the cube seen only by
two simulated cameras, and the strategy grasps it again where it lies.

Later steps of the [grasp plan](../../docs/grasp-plan.md) plug in through these methods: wrist-camera
centring through `before_descent` and `before_straddle` (`grasp_wrist`, below), the Jev judge through
`after_touch`, `after_grasp_failed` and `carrying` (`grasp_jev`, above), a place task through `carrying`
and `release()`, and the grasp library through `candidates` ([Grasp library](#grasp-library), below).

### Wrist camera

`grasp_wrist` is `grasp` with a look through the wrist camera at each hover: at the probe hover before the
descent, and again at the straddle hover with the jaw open. The LLM detector's pose is 1 to 3 cm off and
has no yaw. The first look moves the aim onto the block and turns the jaws across its short side before
the jaw touches it; the second sees where the probe pushed it.

The camera rides on the gripper (robot.yaml `cameras: wrist`, the MJCF's `wrist_cam`): 53 mm behind the
TCP along the approach, 55 mm to its side and 12 mm across, looking 33 degrees off the approach toward the
jaws, with image right along the closing axis. At the probe hover the block's top lies 105 mm along the
camera's axis, 0.26 mm per pixel at 640x360. The twin renders the camera at the MJCF's 1920x1080, and the
strategy scales a frame down to 640 columns before it measures. A measurement takes 9 to 18 ms on the
strategy's tick:

1. The camera's pose comes from the measured joints and robot.yaml (`RobotKinematics.camera`), its focal
   length from the field of view and the frame's height (1200 px at 1080 rows).
2. The block's colour is cut out by an HSV window. The arm's own jaws are masked out from the model, one
   convex hull per jaw body of its collision primitives: the twin's jaws are gold, the hue of `cuboid5`.
3. Blobs whose centroid lies within `max_shift_m` plus 15 mm of the aim count. The gold remnants the
   mask leaves lie 73 mm (probe hover) and 101 mm (straddle hover) away.
4. The block's known shape is fitted to each blob by the IoU of its silhouette: a box by its corners, the
   cylinder by its end circles. The visible side face pulls a plain centroid 6 mm toward the camera; the
   fit lands within 0.5 mm and 1.5 degrees on the simulation's render, every block at both hovers.
5. The blob is the block when the fitted block shows at least a fifth of itself outside the jaws and
   inside the frame, and the blob's area is 0.55 to 1.25 of that visible silhouette. One block reads 0.94
   to 1.01, two touching blocks of one colour 1.52 to 1.83.

The look then aims at the measured centre through `grasp`'s `_aim`, with the measured yaw as the block's
rotation, and drives to the new hover with the top-down hand, so the wrist rolls to the new heading under
its speed limit. It measures again only once every arm joint has stood within 0.5 degrees for 5 ticks:
`_settled` follows the TCP, which stays put while the wrist rolls. A look ends "centred" when the block
lies within `centre_tol_m` (3 mm) of the aim and its yaw within 3 degrees of the heading's, or after
`rounds` (2) moves with the last aim standing. A look never counts an attempt and never ends a run.

After a move the next measurement must find the block within half the move plus `centre_tol_m` of the new
aim. A frame turned by 180 degrees, a mirrored stream, a mount turned by more than about 40 degrees or a
frame from another camera sends the arm the wrong way, and the residual grows. The aim then reverts to
where the look began and the arm goes back there. This move guard does not catch a wrong focal length:
20 % too long or 30 % too short, the blob reads the wrong size and the look is skipped; within that range
the fit is biased by up to 6 mm, so set `focal_px` from the real camera's intrinsics.

| Why a look ends without centring | Note |
| --- | --- |
| no frame from the camera within `frame_wait_s` (1 s) | centrowanie pominiete: brak klatki z kamery wrist |
| frames older than `max_frame_age_s` (0.2 s) | centrowanie pominiete: klatka z kamery wrist za stara (N.N s) |
| no blob near the aim | centrowanie pominiete: brak bloku w kamerze |
| a blob of the wrong size: two touching blocks of one colour, or in the twin a yellow block more than 3 cm toward the camera, merged with the gold jaws | centrowanie pominiete: plama w kamerze nie pasuje rozmiarem (xN.NN) |
| two blobs of the block's size | centrowanie pominiete: dwa bloki w kamerze |
| a fit below `min_fit_iou` | centrowanie pominiete: ksztalt nie pasuje (IoU 0.NN) |
| the block farther than `max_shift_m` (40 mm) from the aim | centrowanie pominiete: blok w kamerze N mm od celu - nie ufam |
| the calibrated tip pixel more than a fifth of the width from the model's | centrowanie pominiete: tip_pixel daleko od modelu (N px) |
| an aim the arm cannot take top-down | centrowanie pominiete: punkt (...) m: chwytak nie ustawi sie pionowo (blad N st.), and `grasp`'s other reach texts |
| the arm does not stand still at the hover within 3 s | centrowanie pominiete: ramie nie stoi nad blokiem (N mm od zawisu) |
| the move guard fired and the arm went back | centrowanie cofniete: przesuniecie nie zgadza sie z obrazem (N mm po korekcie M mm) |
| the move guard fired and the old aim is out of reach | centrowanie pominiete: przesuniecie nie zgadza sie z obrazem, cofniecie niemozliwe: ... |

After any of these the probe goes on as in `grasp`, from the last aim the look stood on.

The colour windows default to the set's colours in `rl.objects`: hue 4.6 degrees for the cube, 45 for
`cuboid5`, 140 for `cuboid85` and 285 for the cylinder, 20 degrees to each side, with saturation from
0.35 and value from 0.25. The twin's table (hue 33, saturation 0.2) stays out. On the stand, sample a
pixel of the block in the panel and set `hue_deg`, `hue_window_deg`, `min_saturation` and `min_value`
per block when the lighting differs. A real camera needs three more settings. `focal_px` sets the focal
length when its field of view is not the MJCF's 48.46 degrees. `tip_pixel_x` and `tip_pixel_y` give the
pixel where the closed tips' midpoint appears in the frame as received; the model puts it at (902.6,
799.2) of 1920x1080, (300.5, 266.1) of 640x360. A tip seen 10 px to the right of the model's point, in a
frame 640 px wide, moves every measurement 10 px left: 2.6 mm toward the fixed jaw at the probe hover.
That corrects a turned mount to first order: 1 degree of mount error is 21 px at a focal length of
1200 px. `rotate_180` turns the frame of a camera mounted upside down.

OpenCV is needed only with a camera named: `start` refuses a run without it, naming
`opencv-python-headless`; the twin brings it. `wrist_camera: ""` runs plain `grasp`. In the twin the
camera is named `wrist` in sim and on the stand alike. On a simulated arm the twin renders it; on a real
arm it serves the workspace camera record named `wrist`, the physical USB camera on the mount, and never
triangulates from it. Until the twin's calibration wave skips arm cameras, disable the `wrist` record
during the wave. test_wrist_in_twin.py lifts a cube seen 15 mm off and a cuboid turned 35 degrees, seen
without a yaw, from the twin's rendered frames.

### Grasp library

With `grasp_source: library` the grasps come from a library per block instead of the block's geometry.
GraspGenX (NVlabs) computes them offline on the block meshes, and the lab caches its best grasps in the
block's frame (`rl/objects.py`: the centre, x the long side, z up) as pad points: where the strategy puts
the fingertips' midpoint, with the approach and the closing direction. The libraries bundled with the lab
are under `soarm_lab/grasps/`; `library_dir` names another folder. A kind without a library, or a library
made for other block dimensions, refuses the run when the strategy is built. Only `cube.json` ships today,
and it is a hand-written example until a GPU run replaces it.

When the pose is taken the library is composed with the pose's rotation. A cylinder's grasps are first
rolled about its axis until they point down, since a turn about its own axis is the same grasp. Every
grasp is folded over the block's yaw symmetry (4 turns for the cube, 2 for the others), because the
detector cannot tell those yaws apart, and each comes with the jaws both ways round. Then the filters:

| Dropped when | Note's word |
| --- | --- |
| the approach leans more than 8 degrees from vertical (the straddle descends vertically) | pochylenie |
| the score lies below `min_score` | wynik |
| the jaw cannot open to the width plus twice `open_margin_m` (88 mm) or close to it minus `squeeze_m` (9 mm) | szczeka |
| the pad point lies above the block's top or below its bottom | glebokosc |
| the pad point lies outside the block | poza blokiem |

Grasps the hand would execute alike (the same heading, offset, depth and width, from several folds or
library entries) are tried once, the best-scored. The rest rank by score in steps of 0.05, then by the
smaller turn of the wrist from its current heading. `_aim` tries them in that order: each sets the
heading, the jaw point (the block centre plus the grasp's offset, its depth below the measured top) and
the openings (the grasp's width plus twice `open_margin_m`, minus `squeeze_m`), and the first whose five
points the arm reaches top-down wins. The probe still goes to the block centre, and the hand stays
top-down whatever the library's own tilt. A grasp is skipped before the reach check when its squeeze
would not read as held ("scisk") or when its shallower fingertip would sit less than 3 mm below the top
("zbyt plytko"): the moving jaw is a hinge, and its tip trails the fixed one along the approach by
`RobotKinematics.tip_swing`, 0.3 mm at the cube's 41 mm opening and up to 23 mm wide open. After a tip
stall the retarget moves by half the chosen grasp's width, which is the short side in geometry mode.

A pose without rotation (the LLM detector alone) makes the aim fall back to the geometry: a library
composed at a guessed yaw would turn the wrist for nothing. The approach's note says which grasp the run
took: "(biblioteka #0, wynik 0.92)" or "(geometria: brak obrotu)"; `GraspStatus` records the source, the
candidates, the rejections and skips, the grasp's index, score, depth and offset.

| How the aim ends in library mode | Stop reason |
| --- | --- |
| every composed grasp fails a filter | biblioteka chwytow 'cube': zaden z N chwytow nie pasuje (pochylenie 8, ...) |
| every candidate fails a jaw rule | biblioteka chwytow 'cube': zaden chwyt nie pasuje do szczeki (zbyt plytko 4, ...) |
| no candidate is reachable top-down | `grasp`'s reach texts |

The commands around the library:

    uv run soarm-lab grasps meshes --out workspace/grasps/meshes   # the blocks as OBJ, GraspGenX's input
    uv run soarm-lab grasps convert cube workspace/grasps/raw/cube.yml   # GraspGenX's output -> a library
    uv run soarm-lab grasps show cube                              # the generator and the grasps

GraspGenX itself runs in its own clone on a CUDA GPU, never in this workspace;
[tools/graspgenx/README.md](../../tools/graspgenx/README.md) has the gripper description, the run and the
frames, and [soarm_lab/grasps/README.md](soarm_lab/grasps/README.md) the file format.
test_grasp_in_twin.py lifts a cube turned by 20 degrees with the bundled library.

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

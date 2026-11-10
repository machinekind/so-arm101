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

See the [policies README](soarm_lab/policies/README.md#train-a-policy) for training, experiments and bundled policy provenance.

## Use a measured station

Export the setup from the twin's environment, then evaluate on it from this workspace:

```bash
twin export ~/stations/desk1
uv run soarm-lab eval lift-v3 --station ~/stations/desk1
```

The export contains `scene.xml`, mesh `assets/` and `station.json` with the setup's
measurements. Set `station=$HOME/stations/desk1` when training to use the measured
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

- `policy`: runs a policy on the perceived object selected by `object_name`.
- `ik`: moves the tool centre point toward a goal without training, keeping the gripper fixed.
- `touch`: closes the gripper, moves it above an object from `obs.objects` and down until the arm
  stalls on it; built for coarse, late poses such as the LLM detector's.
- `touch_jev`: `touch` plus a decision step, in which TypeSafe's Jev model says after each descent
  whether the gripper touched the object and where to aim after a miss; see
  [Jev in the touch loop](#jev-in-the-touch-loop).
- `grasp`: grips and lifts a block from above, monitors the hold and recovers after a drop;
  see [Grasp a block](#grasp-a-block).
- `grasp_wrist`: refines the grasp aim and yaw from the wrist camera at the hover, then centres the
  open jaw on the block before it closes and retries a bad close; see [Wrist camera](#wrist-camera).
- `grasp_jev`: judges the close and lift, then applies the chosen grasp step;
  see [Jev in the grasp loop](#jev-in-the-grasp-loop).
- `pick_place`: picks a block and places it on another block or at a pose, then checks the
  result in the cameras; see [Pick and place](#pick-and-place).
- `build`: builds from a sentence, planned by the claude or codex CLI, or from a fixed plan: one
  `pick_place` run per step, a check after each, and a new plan when a step fails; see
  [Build from a plan](#build-from-a-plan).
- `calibrate_approach`: the arm approaches the corners of a printed board from four sides and at three
  heights, and the wrist camera measures where the jaws really ended up. `pick_place` and `build` use
  the result; see [Calibrate the approach](#calibrate-the-approach).
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
  block: cube
  grasp_source: geometry   # or library
  open_margin_m: 0.008
  squeeze_m: 0.008
  grip_strength: 8         # the gripper servo's P while the run has the arm; 0 keeps the stand's own
  grasp_depth: 0.5
  straddle_push_m: 0.004   # how far the straddle pushes past a light touch
  close_on_stall_m: 0.003  # a stall with both tips this far below the top closes there; 0: off
  jaw_gap_offset_m: 0.0    # how much wider the model's jaw gap is than the real one (this stand's gripper)
  jaw_file: ""             # e.g. a file in the stand folder: learns jaw_gap_offset_m from verified lifts
  lift_m: 0.04
  sag_m: 0.02              # a still arm this far below a lift or hover point counts as there
  stall_s: 2.0             # a still arm short of its point this long ends the run
  carry_s: 10.0
  wait_s: 20.0
  time_limit_s: 120.0
grasp_jev:
  block: cube
  model: jev-1.14.0
  min_confidence: 0.8
  send_frames: false
  wrist_camera: wrist
grasp_wrist:
  block: cuboid
  wrist_camera: wrist
  hue_deg: -1.0             # a colour word in the label wins; this is for labels without one
  label_hues: ""             # e.g. "light blue: 192, orange: 20" adds to or replaces the built-in hues
  centre_tol_m: 0.003
  look_before_straddle: true
  close_rounds: 3            # 0: close without the look between the jaw tips
  close_tol_m: 0.0025        # across the closing line; along it the block only has to sit between the pads
  correction_gain: 1.0
  residual_file: ""          # e.g. a file in the stand folder
pick_place:
  label: red cube
  block: cube
  place_on: blue cube        # empty: the pose place_x_m, place_y_m, place_z_m, place_yaw_deg
  supports: ""               # at a pose: the blocks it rests on, e.g. "blue cube, green cube"
  place_on_block: cube
  judge: jev                 # or none
  wrist_camera: wrist        # empty: no wrist look
  max_tilt_deg: 30           # 0 keeps the hand straight down
  place_retries: 2           # picks of a block that fell off the target or out of the jaw
  approach_file: ""          # the stand's approach table, e.g. ~/stands/desk/approach.json
  approach_direction: ""     # out, in, ccw or cw; empty: the table's preferred side
  pick_shift: always         # auto: only with a table set; off: never
  table_check_m: 0.006       # the pick's own correction may differ from the table this much...
  table_check_runs: 2        # ...before this many picks in a row stop the table's use
build:
  task: "a tower of the three cubes, blue at the bottom"
  plan_json: ""              # a fixed plan: no model is asked
  planner_cli: codex         # or claude
  judge: none                # and every other pick_place setting, for each step
  wrist_camera: wrist
charuco:                     # the twin's detector: start it in the Detectors tab
  square_m: 0.045            # as measured on the print
  marker_m: 0.035
calibrate_approach:
  square_m: 0.045            # the same two values as `charuco:`
  marker_m: 0.035
  table_file: ~/stands/desk/approach.json
  wrist_camera: wrist
  rotate_180: false          # as in grasp_wrist
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
[grasp_wrist settings](soarm_lab/strategies/wrist.py),
[pick_place settings](soarm_lab/strategies/pick_place.py),
[build settings](soarm_lab/strategies/build.py),
[calibrate_approach settings](soarm_lab/strategies/calibrate_approach.py),
[the twin's charuco detector](https://github.com/machinekind/twin/blob/main/docs/DETECTORS.md#charuco-detector) and
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
writes one log line per call, and puts a summary in the panel's note and stop reason. Each tick's
`Action.decisions` carries the records that came back since the previous tick, so a recorded episode
(`twin episode`) keeps them.

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

`grasp_jev` asks Jev after each close and lift. It requires `TYPESAFE_API_KEY` and
uses the same client and retry rules as `touch_jev`. Grasp's motion guards and drop recovery
remain active. Each attempt is judged again after recovery.

The task sent to Jev says whether a probe preceded the straddle.
With `probe: true`, Jev also judges each probe. Touch questions decide whether to continue or
correct the aim by `correction_m`. A claimed touch without usable probe geometry stops for review.
At close and lift, `held` records whether the jaws are empty, hold the block or hold an edge.
`next` chooses lift, wider regrasp, rotated regrasp or give-up.
The state supplies jaw and lift buckets, pose
sources, heading origin, earlier attempts and the geometry's `available_next` choices.

`min_confidence` sets the verdict threshold. `min_correction_confidence` applies to corrections
and next steps. Uncertain answers or unavailable steps stop for review. Regrasps consume
`max_attempts`. Lift and hold clocks start after the answer. `answer_timeout_s` limits each
call, and `time_limit_s` includes all model waits and retries.

A wider regrasp adds twice `open_margin_m` on the moving jaw's side, keeping the fixed tip
in place. A rotated regrasp turns the heading only when it was guessed and rotation changes
the grasp. Held blocks are lowered and released before retrying.
The open hand then moves `open_margin_m` toward the fixed jaw along the closing axis at release
height. It rises only after the fixed tip clears the block.
A drop during judging or lowering cancels the question and starts grasp's recovery.

With `send_frames: true`, only `wrist_camera` is sent. Frames are reduced to Jev's side limit
or omitted when too small, missing or stale. The state records frame references, ages, scaling
and omission reasons. `GraspJevStrategy(..., blob=fn)` accepts a colour-occupancy callback
for the region between the jaws. Until one is supplied, that sighting remains unknown.

`judging.decisions` keeps records for replies, failures and cancellations. Each tick's
`Action.decisions` carries the new ones once, for the episode recording. Stop-time cancellations
stay only in `judging.decisions`, because the run's log is closed by then.
Each record includes the decision point and question version.

See [grasp_jev settings](soarm_lab/strategies/grasp_jev.py) for the full interface.

## Grasp a block

`grasp` hovers over a block, opens the jaw, straddles the block, closes and lifts it.
The jaw rule verifies the hold after closing and at lift height. It then monitors the block
for `carry_s`. Success keeps the block held unless `release: true` opens the jaw.

`block` selects `cube`, `cuboid` or `cylinder`. Dimensions and yaw symmetry come
from [`blocks.block`](soarm_lab/blocks/__init__.py). `label` selects the detector's key in
`obs.objects` and defaults to `block`. Each grasp latches its pose and closes across the
short side. A pose without rotation keeps the current heading. All planned points must
be reachable with the hand down.

By default, the grasp depth counts from the catalog top height above the table.
`hover_m` sets the hover height above the probe bottom computed from the block dimensions.

`probe: true` first measures the top with a closed-jaw descent, as `touch` does.
The straddle then uses that measured top. With the probe on, `touch_z: 0` computes the
probe bottom from the block dimensions. An explicit `touch_z` overrides it.
Without the probe, `close_jaw`, `touch_z` and `need_contact` have no effect.

`open_margin_m` adds clearance on each side and `squeeze_m` sets how much tighter to close.
`grip_strength` sets how hard the jaw squeezes for as long as the run has the arm. It is the gripper
servo's P gain, 8 by default, and the squeeze grows with both P and `squeeze_m`. The twin puts the stand's
own grip back when the run ends, and 0 keeps the stand's own throughout. `grasp_wrist`, `grasp_jev`,
`pick_place` and `build` have the same setting.
`grasp_depth` sets tip depth as a fraction of block height. `lift_m` sets lift distance and
`hold_s` sets the verified hold duration before carry.

`max_tilt_deg` above zero lets a grasp out of reach with the hand straight down tilt the hand in
5° steps, up to that angle. It is 0 for `grasp`, `grasp_wrist` and `grasp_jev`, so they keep the
hand straight down unless it is raised. The hand tilts about its closing axis and leans away from
the base, so the jaws still close level across the block. The arm can tilt the hand only within its
own vertical plane, so the jaws then close square to the arm, and the block's narrow side must lie
near square. So a tilted grasp needs the block's yaw: from its pose, or from the wrist look of
`grasp_wrist`, which then checks a block whose pose has no yaw before the jaw goes down. Without
either, the hand stays straight down and the stop reason says "dlon pochylona potrzebuje obrotu
bloku". `GraspStatus.tilt_deg` reports the tilt, and the panel shows it. With `probe: true` the
hand stays straight down.

The straddle stops on a stall once the arm stands more than `straddle_push_m` above where it should
be. If both jaw tips are then at least `close_on_stall_m` below the block's top, or the hand is within
5 mm of the grasp point, the fingers are already beside the block: the jaw closes where the arm stands,
and the usual close, look and lift checks decide whether it holds. Otherwise a tip stands on the block.
A tip stall retries half the short side away. Retries alternate sides within `max_attempts`.
With the probe on, each retry probes again. Without the probe, an empty close after a sideways
retry tries the other side while attempts remain. Any other empty close or motion timeout
ends the run. Touch's lag guard, `time_limit_s` and the panel's Stop also apply.

Servos without an integral term settle short under load, and friction then holds a joint where it
hangs until its command comes back down to it. So on the way up and at the hover an arm that stands
still at most `sag_m` below its point, and mostly straight below it, counts as there. A descent expects
a joint that hung short at its start to stay put until its command reaches it, so that slack is not
read as a stall. An arm that stands still `stall_s` short of any point the strategy waits for ends the
run; the note names the phase, how far the hand stands from the point and the joints that lag, in
degrees, for example "w gore znad bloku: ramie stoi 29 mm pod celem (wrist_flex 12.7 st., elbow_flex
10.3 st., shoulder_lift 2.4 st. za malo) - koniec". The same holds for `touch`, `grasp_wrist`, `grasp_jev` and `pick_place`.

The twin's simulated blocks have stiff contacts, so the probe stalls on their top as on the real
arm and `need_contact` can stay on. With `need_contact: false`, reaching the probe bottom uses the
nominal top. Real jaw opening and pad reach still need calibration.

### Recovery

From the close until release, each lift, check and carry tick compares the jaw with
`obs.objects`. Squeeze keeps the hold valid. Creep restarts the hold check. A camera pose
away from squeezing jaws is recorded for a judge. Consecutive empty readings declare a drop.

After a drop, the arm rises to at least hover height and opens the jaw. Recovery needs
a plain "kamery" or "ostatnie widziane" pose captured after the settling delay, no older than
`max_pose_age_s` and low enough to be resting on the table. Simulation poses at rest height
also count. Annotated last-seen poses and captures from before the drop cannot guide recovery.

A sighting within `recover_m` triggers a regrasp through `retarget`.
By default, the arm opens at the hover and straddles below the catalog top height.
With `probe: true`, it first measures the top again. Tip retries and regrasps share `max_attempts`.
A farther sighting, unreachable pose, exhausted budget or no sighting within `wait_s` stops
for a person at the safe height with the jaw open.

The LLM detector's delayed answers can supply plain last-seen poses. With one camera, its
interval is too long for the tracker's settle gate near the hand. Recovery there needs a
second camera or the seeded detector.

`carry_s: 0` finishes at the verified lift. `release: true` opens after carry, and the run
succeeds once the jaw is open. A receiver must support the block. Subclasses can end
monitoring through `release()` and return its action from `carrying`.

See [grasp settings and hooks](soarm_lab/strategies/grasp.py) and the
[grasp plan](../../docs/grasp-plan.md) for later work.

### Wrist camera

`grasp_wrist` looks at the open-jaw hover before each straddle. It fits the block's known
shape from `blocks.block` to a colour blob, centres the jaws and turns them across the short
side. With `probe: true`, it also looks before the probe. `look_before_straddle` then controls
the second look, which corrects movement caused by the probe. Without the probe, the look
before the straddle always runs. Other settings extend `grasp`.

`wrist_camera` selects an entry in robot.yaml `cameras` and `obs.frames`. Empty runs plain
`grasp`. A named camera requires OpenCV, which the twin supplies. In simulation, the twin
renders the arm camera. On hardware, it serves the workspace camera with the same name.
Arm cameras are excluded from triangulation. Until the calibration wave skips them too,
disable the physical `wrist` record during the wave.

A colour word in the block's label sets the hue the look searches: red (356°), yellow (50°), green
(143°) and light blue (190°), as this wrist camera sees them. `label_hues` adds words or replaces
these hues per stand. A `hue_deg` within `hue_window_deg` of the label's colour fine-tunes it; one of
another colour is not used for that label, so a stand-wide `hue_deg` cannot hide a block of another
colour. For a label without a colour word (bare wood among them), set `hue_deg`, `hue_window_deg`,
`min_saturation` and `min_value` for the block and lighting. The default `hue_deg: -1` samples the
first usable frame inside the expected visible silhouette at the aim. It selects the hue window with the most pixels above `min_saturation` and `min_value`.
The window spans ±`hue_window_deg`, or ±20° by default, and must cover at least 30% of that region.
The status reports `hue_deg` and `hue_share`. Bare wood falls below `min_saturation`, so the look
is skipped and the grasp continues from the detector's aim.

`focal_px` overrides focal length from field of view.
Set both `tip_pixel_x` and `tip_pixel_y` to the closed-tip midpoint in
the received frame to correct mount alignment. `rotate_180` handles an upside-down camera.

Each look waits for all arm joints to settle before measuring. `max_frame_age_s` and
`frame_wait_s` limit stale or missing frames. The model masks the jaws before fitting.
Blob area, visible share and `min_fit_iou` reject unsuitable or ambiguous silhouettes.
`max_shift_m` limits corrections. `centre_tol_m` sets position tolerance and `rounds`
limits moves. A final measurement checks each move. An inconsistent result returns to
the original aim when reachable.

A look that cannot settle or obtain a usable measurement is skipped. The straddle (or probe)
then continues from the settled aim. Looks consume no grasp attempts. A grasp with the hand tilted
looks too, and is stricter: when the measured block does not fit a tilted grasp, or the look gives
no yaw for a block whose pose has none, the run ends before the jaw goes down. Focal length, mount
alignment and colour thresholds still need calibration on the stand.

#### Between the jaw tips

At the bottom of the straddle, with the open jaw around the block, the camera looks again before the
jaw closes. It fits the block in the frame and measures how far the block lies from the jaw tips'
midpoint, along the closing line and across it. The block and the tips come from the same frame and
the same measured joints, so most of an error in where the arm thinks it is cancels out. A wrong
`focal_px` only changes how far each correction goes. The tip pixel still matters: it sets where the
tips are in the frame.

The jaw closes when the block lies within `close_tol_m` of the tips' midpoint across the closing line
and sits between the open pads along it: the moving jaw then pushes it against the fixed one. That
room along the line comes from the opening, the block's width and the fingertip radius: about 7 mm on
each side at the default `open_margin_m`. Otherwise the jaw first slides along the closing line, so
neither tip drags the block, and the arm rises with the jaw open, moves the aim by `correction_gain`
times the part of the offset outside those bounds (along the line no farther in an attempt than the
pads allow), straddles again without the hover look and looks again, up to `close_rounds` times per
attempt. A part of the offset, along or across, that a correction moved but the next look still sees
within 1 mm of before is the camera's error, not the block's: it no longer counts in that attempt.
After the last round the jaw closes if the block lies within the middle third of the gripped face
across the closing line, and the close and lift checks judge the grip. A block still farther off
counts as a failed attempt; the last attempt closes anyway. `close_rounds: 0` closes without this
look.

With a camera, a failed close is retried instead of ending the run: an empty close, or a jaw that
stops more than 6 mm narrower than the block's width (an edge or corner grip; the jaw opens first). A
jaw that holds wider than the block is a full grip. The arm rises with the jaw open and comes back to
the hover, where the hover look finds the block again. This repeats up to `max_attempts`, and the last
attempt lifts what it holds. Plain `grasp` keeps its old behaviour.

A real gripper may read its jaw wider than the model puts it at the same reading. `jaw_gap_offset_m`
is that difference for the stand, 0 in simulation: every grasp-based strategy then opens, closes,
releases and checks the jaw as it really is, and finds the jaw tips where they really are. To measure
it, grip a block of known width and lift it: the success note shows the held gap ("szczeka 39 mm"), and
the offset is that gap minus the block's width. With `jaw_file` set (for example a file in the stand
folder) the strategy learns it on its own: each verified lift adds the held gap minus the block's width
to a running mean over the last 10 lifts, and the next run uses that mean instead of the setting. A
release opens past the jaw as it holds, not only past the block's width, so a gripper that reads wide
still lets go.

`residual_file` keeps, per stand, a running mean over the last 10 runs of how far the camera moved the
aim from the detector's first pose, written after each lift. The next run adds it to the first aim.
A missing file starts empty; an unreadable one is ignored and rewritten.

The status lists every look between the tips in `wrist.close_looks` (attempt, round, offsets, verdict)
and the last one in `wrist.final_offset_m`. Panel notes show each look ("kamera w szczekach: ..."), and
the success note ends with the attempts and the final offset.

Limits: the camera looks at the block from the side, so a height error between the jaw and the block,
for example from joint backlash, reads as an offset across the jaw, about half as large. In the twin
with 6 mm of hidden backlash at the hand, the jaws end about 2 mm from the block's centre.
A `correction_gain` above 1 overshot there. More rounds or a slightly larger `close_tol_m` avoid
retries the backlash cannot fix.

See [wrist settings](soarm_lab/strategies/wrist.py) for the full interface.

### Grasp library

`grasp_source: library` uses cached block grasps. GraspGenX generates them offline from
meshes supplied by `blocks.block`. Each file stores pad points, approach and closing axes
in the block frame. `library_dir` selects another folder and `min_score` filters scores.
A missing library or dimensions that differ from the catalog refuse the run.
Only a hand-written cube example is bundled.

The strategy composes grasps with the observed rotation and block symmetries. A pose
without orientation uses geometry. Library candidates are checked for tilt, score, jaw
travel, depth and position. Duplicates are tried once. Score bins and wrist turn determine
attempt order, followed by jaw and reach checks.

The chosen width sets opening and squeeze. Depth and offset place the fingertip midpoint
below the nominal top by default. With `probe: true`, depth counts from the measured top
and the probe targets the block centre. Tip swing must leave enough bite for the shallower
tip. The hand stays down. Tip retries move half the chosen width.
Status and panel notes identify the source, candidate and rejection reasons.

    uv run soarm-lab grasps meshes --out workspace/grasps/meshes
    uv run soarm-lab grasps convert cube workspace/grasps/raw/cube.yml
    uv run soarm-lab grasps show cube

See [generation setup](../../tools/graspgenx/README.md) and the
[library format](soarm_lab/grasps/README.md). GraspGenX runs in its own CUDA environment.

## Pick and place

`pick_place` runs one step of a building plan: pick block `label` and put it on block `place_on`,
or at a pose when `place_on` is empty: the held block's centre at `place_x_m`, `place_y_m`, its
bottom `place_z_m` above the table and its long side at `place_yaw_deg`, resting on the blocks named
in `supports` (comma-separated) or, with none, on the table. The planner's `resolve` gives such a
pose for each relation, a bridge `across` two blocks included. `block` and `place_on_block` name the
picked block and the target block in the catalog. The pick is `grasp` with the wrist look of `grasp_wrist` (off with
an empty `wrist_camera`) and, with `judge: jev`, the checks of `grasp_jev`. `judge: none` needs no
`TYPESAFE_API_KEY`. A drop goes through grasp's recovery, then the run goes on to the place.
`carry_s` and `release` have no effect.

Before the pick, the target needs a usable pose and a whole place plan must fit. After the lift the
held block goes up to the carry height, `carry_clearance_m` above the target and the catalog's
tallest block, lower where the arm cannot reach that, but 1 cm at least. With `align_yaw` the hand
turns the block to the target's yaw, by the smallest turn the two blocks' symmetry allows. A target
pose without yaw counts as yaw 0. Over a block target the run latches the newest camera pose of the
target seen on the way, captured at most `fresh_pose_s` before the hand started over it, and waits up
to `target_wait_s` for one. The hand may hide the target at the hover. A target that moved gets a
new hover, twice at most.

The descent is fast down to 1.5 cm above the release height, or starts at the carry height when that
is lower, and settles. The rest runs at `place_speed_mps` and stops on a stall or with `place_gap_m`
between the blocks, plus 3 mm for a block that slid down in the grip on the way. The jaw opens
`release_margin_m` past the block on each side, where the hand stands. The hand then leaves the
block without a touch:

1. It moves `release_ease_m` (2 mm, 3 mm at most) toward the fixed jaw, so the fixed pad leaves the
   block's face. The moving jaw opened to the other side, so it keeps its room.
2. It rises straight up at `retreat_speed_mps` (2 cm/s), `retreat_m` and at least until the open jaw's
   lowest point stands `retreat_clear_m` (2 cm) over the block's top. The check uses the jaw's points
   at the measured joints, so an arm that settles short rises further, up to `sag_m`, where it
   reaches. A rise that stands still short of its point ends the run, as in `grasp`.
3. Only then does it move to the start position, never lower. While any part of the open jaw could
   reach over the block, it moves at `retreat_speed_mps` and keeps the hand's orientation as long as
   the arm can, so the jaw does not swing down; beyond that it moves at `speed_mps`.

Where the arm cannot rise that high, a jaw 4 mm clear of the top still leaves; below that, as for a
third cube on two, the jaw opens fully so that its moving side clears the block, and the way to the
start position must lead away from the fixed jaw. Plans prefer the first exit. With neither, the run
ends `unreachable` before the pick. At the start position the cameras see the result. Poses captured
after the release must put the block within `place_tol_m` of the target's centre, at the target's
top plus half its height. A target not seen at the hover, an unreachable place or a target that
keeps moving puts the block back where it was picked. The carry back stays at least 1 cm above the
target the hand hovered over.

A block that fell is picked and placed again inside the run, at most `place_retries` times (2 by
default, 0 turns it off):

- after the release the check finds it off the target, or the target or a support moved and it no
  longer sits on them. It needs fresh poses, captured after the release, of the block lying flat and
  of the target or every support, within `target_wait_s`. The hand first moves over the block at a
  height where the jaw clears every block the cameras see by `retreat_clear_m`, then picks it as at
  the start;
- it dropped out of the jaw on the way: grasp's recovery picks it again, as before, but now as one of
  these retries. With none left the run ends `dropped` once the hand is up.

Each new pick gets the whole `max_attempts`. The panel note says "... - ponawiam (1 z 2)" and then
"| ponowienie 1 z 2" after each note; the stop reason ends with "; ponowienia: 1 z 2", and
`place.retries` and `place.retried` count them. When they run out, the run ends with the last
failure's outcome.

Where the place is out of reach with the hand straight down, the pick takes the least tilt up to
`max_tilt_deg` (30° by default) at which both the pick and the whole place fit, and the hand keeps
that tilt to the end, so the held block stays level. The tilted hand keeps its heading against the
arm as the arm turns, so the place heading follows from the pick: with `align_yaw` the target must
lie within a few degrees of square to the arm (about 30° at a 10° tilt, 10° at 30°), or the place
is unreachable. The release, the jaw clearances and the rise count with the tilt. A tilted pick
needs the picked block's yaw from its pose or the wrist look; otherwise, or when the look finds the
block too far from square to the arm, the run ends `unreachable` with the block where it lay. The
stop reason of a tilted place ends with the tilt, and `grasp.tilt_deg` reports it.

The picked block may rest on another block. Its bottom is its pose height minus half its height;
above 7 mm it counts as stacked. The grasp then counts from that bottom, keeps the open jaw 6 mm
above the block under it, and lowers the hover and the lift where the arm cannot reach them. While
the jaw closes, the hand keeps the fingertips centred on the block, so it does not slide across the
one under it. Every carry clears the stack it came from, and a put-back returns the block onto it.
The blocks under the picked one, found from their poses before the pick, are checked again once the
hand is away. A shift over `place_tol_m` sets `place.support_moved` and is added to the stop reason,
and a put-back then follows that block. A regrasp after a drop counts from where the block landed,
on the table or on a block.

### With an approach table

The fixed cameras and the arm are both a little off: the cameras by up to a centimetre where their
calibration is off, the arm by a few millimetres of gear play and sag that depend on the side the hand
comes from. A block placed where the cameras see the target lands off by both. `calibrate_approach`
measures how far, per spot, height and side, into an approach table (see
[Calibrate the approach](#calibrate-the-approach)). With `approach_file` set, `pick_place` uses it:

- the hand always comes into its last hover from one side: the last 3 cm (`approach_lead_m`) run
  straight along `approach_direction`, or the table's preferred side when that is empty. At the pick
  that is the hover the jaw opens over; after a wrist look or a correction moves the aim, the hand goes
  back and runs the leg again before it goes down. At the place it is the hover over the target;
- the place's hover and release move by the table's correction at that spot, height and side. The pick's
  first aim moves by it too, and so does a pick again after a drop or a fall. The look between the jaw tips may then move the aim again, but it measures
  through the arm's own play, so the place counts the held block across the jaws where the table's aim
  put it (`place.hand_shift_m` says how far that is from the look's aim);
- what the run compares stays in camera poses: the latch on the target, the jaw clearances and the
  check after the release. `place.target_xy` is still where the cameras see the target.

Where the table has no answer (outside its corners by more than 3 cm, at another height, with the hand
tilted, or when the leg's start is out of reach), the place takes the pick's own correction instead: the
aim the wrist camera centred the jaws on, minus where the cameras saw the block. With the wrist camera
and no table, `pick_shift: always` (the default) does the same everywhere; `auto` does it only with a
table set, `off` never. The pick's correction cannot tell a camera error from a block pushed after the
wrist look: it carries such a push to the place too. The pick's own correction also carries the look's
error, so it differs from a good table by a few millimetres. When the two differ by more than
`table_check_m` (6 mm), the panel says "chwyt rozni sie od tablicy o 7 mm (1 raz z rzedu) - zostaje
tablica" and the run keeps the table: one push of the block does that too. When `table_check_runs` (2)
picks in a row with the same table differ by more, as after a fixed camera was moved, the run stops
trusting the table and says "tablica nie zgadza sie z chwytem 2 razy z rzedu". The count lasts while the
twin runs and starts again with a rewritten table. A put-back takes no correction. `place.correction_m`, `place.correction_from` (`table`, `pick` or empty) and
`place.table_miss` say what the run did; the panel adds "poprawka z tablicy 7 mm (ccw)" or "poprawka z
chwytu 10 mm - cel poza tablica" to the carry note. A table file that cannot be used leaves
`place.table_problem` and the run goes on without it.

`PlaceStatus.outcome` names how the run ended:

| outcome | meaning | block |
| --- | --- | --- |
| `placed` | verified on the target (the only success) | on the target |
| `not_found` | no usable pose of `label` before the pick, or none fresh after the release | where it was / unknown |
| `target_not_seen` | no usable pose of the target or a support before the pick, or none fresh at the hover | where it was / put back |
| `pick_failed` | the grasp failed, or Jev stopped it | on the table |
| `dropped` | dropped and not picked again, or no retry left | on the table |
| `fell_off` | seen after the release, but not on the target or at the pose, or leaning at a pose, and no retry left or possible | near the target |
| `target_moved` | the target or a support moved by the check (and no retry left or possible, unless the block still sits on them), the target kept moving at the hover, or the supports moved apart | near the target / put back |
| `unreachable` | the block or the place cannot be reached with the hand down, straight or tilted, or the open hand could not leave the placed block untouched | where it was / put back |
| `stopped` | stopped by a person, the time limit or a motion guard | anywhere |

At a pose the hand turns the block to `place_yaw_deg` with `align_yaw` (`false` keeps the hand's
heading), by the smallest turn its symmetry allows among the headings whose open jaw stays 5 mm away
from every other object the cameras report, sized from the catalog by the label's shape word. A cube
set beside another so has its jaw close parallel to the gap, with no finger in it. With no such heading the
place is `unreachable` before the pick. Each support needs a usable pose before the pick, and the
carry clears the pose's height. Over the supports the run latches each one's newest camera pose as it
does a target block's and moves the pose with them: across by their mean shift, up by the highest
rise. Supports that moved more than `place_tol_m` against each other send the block back as
`target_moved`. After the release the block must lie within `place_tol_m` of the pose, at its height
and level within 5°, and each support seen must stand within `place_tol_m` of where it was latched.

The stop reason in the panel says the same in Polish. `place.put_back` tells whether the block
went back, `place.opened_wide` whether the jaw opened fully to leave it. Known gaps: the twin's cube
detector maps blocks on the table only, so it cannot verify a stacked block or a bridge yet. A pose is
only as good as the supports' camera poses it came from, and the planner lets a bridge rest as little
as 1 cm on each block, so supports seen 1 cm off can drop it. The jaw check at a pose covers the release only, and a label
without a shape word counts as a 3 cm square. With the hand down
the arm reaches about 9 cm high, so off a stack it picks only the top block of two. Straight down, a
third cube on two fits only around 20 cm from the base, where the hand can leave away from the fixed
jaw toward the start position. With the hand tilted, a third cube fits out to about 30 cm and a
fourth from about 18 to 26 cm, on stacks square to the arm. Moves do not go around other blocks, so a tall stack
within about 10 cm of the pick or in the way from the start position can be hit. A block that slid
down in the grip by more than those 3 mm is pressed onto its target by the rest. A block dropped on
the way that lands high on another block, such as a cube on a cube, is not picked again: recovery
waits for a person. One the check after the release finds there, lying flat, is.

[`reach`](soarm_lab/reach.py) helps a planner choose where to build: `can_place(kind, x, y, bottom_z)`
says whether a block fits with its bottom at that height, and with what tilt, and `reach_map` gives
the least tilt over a table grid. Both follow the place plan in outline and are a little cautious:
they count only a rise that clears the placed block, not the fully open jaw.

See [pick_place settings](soarm_lab/strategies/pick_place.py) for the full interface.

## Build from a plan

`build` turns a sentence (`task`) into steps and runs them. The local claude or codex CLI, logged in
with a subscription, gets the blocks the cameras see and answers with steps: pick this block, put it
on that one, beside it, across two blocks as a bridge, or at a table spot. The call runs beside the
run, so the arm holds still meanwhile and
the panel shows the wait. The plan is walked through the scene before anything moves; a plan that
cannot run goes back to the model once with its problems, and a second bad one ends the run. A
question from the model ends the run with the question; "already done" ends it as a success.

`plan_json` gives the steps instead and asks no model, for a demo without a network. It takes the
model's JSON or only its list of steps, with unused fields left out:

```json
[{"pick": "green cube", "place": {"relation": "on", "targets": ["blue cube"]}},
 {"pick": "red cube", "place": {"relation": "on", "targets": ["green cube"]}}]
```

A bridge: green beside blue, then the cuboid across both.

```json
[{"pick": "green cube", "place": {"relation": "next_to", "targets": ["blue cube"], "side": "left",
                                  "gap_m": 0.015}},
 {"pick": "yellow cuboid", "place": {"relation": "across", "targets": ["blue cube", "green cube"]}}]
```

Each step is one `pick_place` run inside this run, with this run's settings (judge, wrist camera,
speeds, tilt, approach table) and its own `step_time_limit_s`. "on" names the block to put it on; the other
relations become a pose (`place_x_m`, `place_y_m`, `place_z_m`, `place_yaw_deg`), and a bridge's two
blocks are its `supports`. Before each step the arm turns back to the pose it
started the run in, so the step starts as a fresh `pick_place` run would. After each step the run
waits `settle_s` and for fresh poses, then checks that the step and the steps under it still hold.
What follows depends on the step:

| step ended | then |
| --- | --- |
| `placed`, check passes | the next step |
| `not_found`, `target_not_seen`, `target_moved` | once the blocks are seen again, the same step once more, unless its `pick_place` already picked the block again |
| `unreachable`, `dropped`, `fell_off`, `pick_failed`, or the check fails | a new plan from the scene seen now |
| `stopped` | the run ends |

A block that falls is picked again inside the step, up to `place_retries` times, so a step ends
`fell_off`, `target_moved` or `dropped` only after those tries, and its stop reason counts them. The
build does not repeat such a step as it stands: it replans, so the model can choose another place.
The time a step may take, `step_time_limit_s`, covers its retries.

A new plan comes from the model, told what ran and what failed, or, with a fixed plan, from the
fixed plan itself: every step again from the first one that no longer holds, with blocks lying on a
block it needs set aside on a free table spot first; a fixed plan stops when the same step is
unreachable twice in a row. At most `max_replans` new plans, `max_steps` steps per plan and
`max_stack` blocks per stack. The panel shows each step as "krok 2/3: red cube -> green cube
(placed)". Every model call is kept as a decision with its prompt and answer.

Known gaps: plans count reach with the hand straight down, so they refuse some places a tilted hand
would reach. Moves do not go around other blocks.

## Calibrate the approach

TL;DR: print a board, tape it on the table, run `calibrate_approach` for about 25 minutes, then set
`approach_file` for `pick_place` and `build`. Blocks then land within a few millimetres of where the
target really is, even with a fixed camera a centimetre off and gear play in the arm.

The routine sees the board's corners through the stand's own cameras and detector, the same way it
sees blocks, so each corner carries the same camera error a block would. The arm then approaches a
corner the way `pick_place` approaches a place: up to the carry height, a straight 3 cm leg from one
side, then down to where a held cube is released on the first, second or third tower level. Once the
arm stands still, the wrist camera measures the corner against the jaw tips. The camera is fixed to the
jaws, so no joint reading goes into the measurement. The difference is the correction for that spot,
height and side. It holds the fixed camera's error, sag and gear play together.

### On the stand

1. Print the board from the twin: `uv run twin board --layout 3x5 --square-mm 45 --out board.png`,
   at 100 % scale. Measure one square and one marker with a ruler; the defaults are 45 mm and 35 mm.
2. Tape the board flat about 20 cm in front of the arm, long side across the arm, where the towers go.
   Clear all blocks off the table.
3. In `configs/local.yaml` set the measured sizes under both `charuco:` and `calibrate_approach:`, and
   `table_file`, `wrist_camera` and `rotate_180` under `calibrate_approach:` (see the example above).
   Under `pick_place:` and `build:` set `approach_file` to the same file.
4. Reinstall soarm-lab in the twin's environment and start the twin. Turn the arm aside, so `kamera1`
   sees the whole board.
5. In the Detectors tab, start `charuco` (the twin brings it). Eight points `charuco c0` to `charuco c7`
   appear on the board.
6. In the Strategies tab, start `calibrate_approach` and leave the table alone. It takes about 25 min:
   tilted looks for the wrist camera's lens (about 2 min), wrist turns that measure the camera's mount
   (about 2 min), 4 sides x 3 heights x 2 repeats per corner (about 18 min) and a self-check (about
   3 min). It stops early if the corners do not fit the board's size ("rogi tablicy nie pasuja do jej
   wymiarow"): fix the sizes in both places.
7. Read the end note. Good: a self-check median of 1.5 mm or less and a side spread under 1 mm.
   `uv run soarm-lab approach-table ~/stands/desk/approach.json` shows the details, and a suggested
   `focal_px` and tip pixel for `grasp_wrist`.
8. Optional: move the board to another area and run again. The samples add up in the same file.
9. Run it again after moving or recalibrating a fixed camera, or after re-zeroing the servos. When runs
   keep saying "tablica nie zgadza sie z chwytem 2 razy z rzedu", check first that no fixed camera moved
   and that the blocks are not pushed during the pick; then run it again. A single "chwyt rozni sie od
   tablicy" needs nothing.

### The table

The file is JSON, written whole each time, with the date, the board, the protocol, the wrist camera's
calibration and mount, every sample, and how much the two repeats of each side differ. The side with
the smallest difference becomes the preferred one. A run that stops writes `<table_file>.partial.json`,
which `pick_place` does not use. `max_table_age_days` makes `pick_place` ignore an old table.

Known gaps: the board covers about 4.5 x 13.5 cm; places further than 3 cm outside its corners take the
pick's correction. Corners are seen at table height, blocks a few centimetres higher, which leaves up to
about 1 mm of a camera tilt error. Each corner is measured at the heading the wrist camera sees it best
from, while a place turns the hand to the target's yaw; in the simulation the landing error after the leg
changed little with the heading or with the way the hand came, and the table holds no heading. The
table's height correction is stored but not used: a release can
still start a few millimetres higher or lower than planned. In the simulation the radial sides (`out`,
`in`) repeat worse than the tangential ones, as the leg moves only some of the joints; the preferred
side came out `ccw` or `cw` on every simulated stand.

## Simulation campaign

`jobs/overnight.sh` runs seeded scenarios in the twin's simulation, keeps one JSON file per run and
writes `report.md`: per scenario the runs, the successes, how many attempts and wrist-camera corrections
they took, the final offset's median and 90th percentile, why runs failed and any simulation artefacts.
No arm moves and no model is asked.

- approach: simulated stands with a ChArUco board, 6 mm of gear play and the fixed camera's pose tilted
  so it sees 10 mm off; each runs the whole calibration, then the table's correction is checked against
  the simulation's truth at points it did not measure, approached as `pick_place` approaches a place:
  after a stop at a pick spot, by a straight carry that turns the hand to a heading drawn over the whole
  turn. The report splits the error by direction, level and heading.
- grasp: `grasp_wrist` with the detector 1.5 cm off; the same with 6 mm of hidden joint backlash; the
  same with the cube nudged 5 mm after the hover look.
- pick_place: a cube on a cube, with and without the wrist camera, and with every camera 5, 10 and
  15 mm off (5 mm also with `pick_shift: auto`); on simulated stands with 6 mm of play and the camera
  10 mm off, with the stand's approach table (camera tilted, and shifted), with the pick's correction
  only and with none; a cuboid across two cubes; a cube next to a cube. Each run onto a block keeps the
  simulation's truth on the corrections in use under `in_use`: at the pick, how far the table's and the
  pick's own correction lay from the one needed; at the release, how far the TCP and the block in the
  hand were from where the plan needed them.
- build: a tower of three cubes from a fixed plan and from a scripted planner; from the fixed plan with
  the wrist camera, also with every camera 5 mm off, and on the simulated stands with and without the
  approach table; both plans again with the middle cube knocked off mid-build; a bridge.

The pick_place and build scenarios step the physics every 1 ms, not the twin's own 5 ms, so a block let
go a few millimetres above another lands without sinking into it. They check at every physics step that
no block sinks more than 0.7 mm into another block or the table; a run where one does counts as failed.

```bash
RUNS=50 OUT=workspace/campaign/tonight bash jobs/overnight.sh
uv run soarm-lab campaign report workspace/campaign/tonight
```

The scenarios are the tests marked `campaign`. `SOARM_CAMPAIGN_RUNS` sets the seeds per scenario and
`SOARM_CAMPAIGN_DIR` the folder for the runs. Without them the test suite runs the grasp and the
pick_place scenarios at their default size, and skips the approach and build campaigns, the stand runs
and the runs that only measure. The stand runs share a stand by tens; each stand is calibrated once, in
about 3 minutes, and its table kept under `tables/` in the campaign folder.

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

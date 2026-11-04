# Grasp plan: from `touch` to pickup with the LLM detector and wrist camera

Written 2026-10-31. The main and backup paths are due for decision in #109 on 2026-11-02.
The demo is #80 on 2026-11-11.

## Goal

The arm picks a named block, such as "the red cube", lifts it and recovers if it falls or is
removed. The Classic World set has a 2.5 cm unit and contains a cube, two cuboids and a cylinder.
The LLM detector finds the block. Cameras refine its pose. Success follows #95: the jaw servo
stalls on the block and the cameras see it lift.

## What exists

- `touch` probes with a closed gripper until the arm stalls. It accepts poses that are seconds
  old and 1–3 cm off. `touch_jev` asks Jev after each descent and retries.
- `Observation.objects[label]` carries position, `source` ("kamery", "w dloni",
  "ostatnie widziane") and `age_s`. Rotation is identity because the LLM detector returns a pixel.
- The detector contract accepts `mask` and `size_m`. Given both, the twin fits a box to the
  fixed-camera masks and recovers yaw. No detector supplies them yet.
- The jaw rule detects contact when the jaw stops wider than commanded without creeping closed.
  The Feetech backend reads position. The twin's tracker uses the same rule for "w dloni".
- The wrist camera uses the official mount. Only `dream` reads it.

## Architecture

The twin records observations, the strategy commands motion and Jev judges ambiguous cases.
All run in the twin's process. so-arm101 imports only `machinekind-twin-api`.

| Component | Repository | Status |
|---|---|---|
| Register the wrist camera and exclude it from triangulation | twin | Needs configuration |
| Refine an LLM seed into a colour mask, attach block dimensions and track between LLM calls | twin detector plugin | New |
| Fit boxes to masks and track yaw, pose and `source` | twin | Exists |
| Store dimensions and colour ranges per block | soarm-lab data | New, #94 |
| Probe, rise, open, straddle, close, verify, lift and verify again | soarm-lab `grasp` | New subclass of `touch` |
| Centre and estimate yaw from the wrist camera at hover | soarm-lab | New, needs `kinematics` |
| Compare the jaw rule with tracker `source` each carrying tick | soarm-lab `grasp` | New recovery loop |
| Judge grasp state from state buckets and a wrist frame, and record decisions | soarm-lab | Jev client exists; questions are new |
| Cache grasps from an offline GraspGenX run per block mesh | soarm-lab data | New |
| Fine-tune lift with biased, delayed and latched poses | soarm-lab `rl` | Needs three changes |
| Supervise safety, judge runs (#176) and log episodes (#151) | twin | Supervisor exists; judge and log are planned |

No API change is needed. Decision records reach the twin's log through `Action.decisions`.

## Recovery

Check each tick while carrying, until the strategy commands release.

| Jaw rule | Twin's `source` | Meaning | Action |
|---|---|---|---|
| holding | "w dloni" | The block is held | Continue |
| empty | "kamery", on the table | The block dropped and its position is known | Rise, re-aim and regrasp |
| empty | "ostatnie widziane", for a few seconds | The block is hidden or falling | Hold at a safe height and wait |
| empty | "ostatnie widziane", longer, or far away | The block was removed | Stop and hold for a person |
| holding | "kamery" elsewhere | There is a second block or a tracker error | Ask Jev with the wrist frame |

Never descend to a stale pose or chase a removed block. Recoveries consume the attempt budget.
At handover, the strategy's responsibility ends when it commands release.

## Steps

Steps follow their dependencies. Estimates are working days for one person.

1. **Block set** takes 0.5 day plus measuring in soarm-lab. Store dimensions, mass and colour
   ranges per block. This closes #94.
2. **Wrist camera** takes 0.5 day in the twin. Configure frames in `obs.frames` and exclude the
   camera from triangulation. Steps 5 and 7 need it.
3. **Seeded detector refinement** takes 2–3 days in the twin. Extend the LLM detector or wrap any
   seed detector. Grow a colour mask from its pixel and select the nearest matching blob within
   the pixel error. Attach `size_m` from the label and track between LLM calls. This supplies yaw
   and fresh poses. Calibrate colour ranges on the stand.
4. **`grasp` strategy** takes 3–4 days in soarm-lab. Subclass `TouchStrategy` and add phases after
   `after_touch`. Verify the jaw after closing and lifting. Use sideways `retarget` when a tip
   lands on the block, within an attempt budget. Start with the cube and cylinder.
5. **Wrist centring and yaw** takes 3–4 days after step 4 has its phases. Before descent, centre
   the colour blob on a calibrated fingertip pixel. Learn scale with a 1 cm move and pass rotation
   to `ik_step`. This enables cuboid grasps.
6. **Recovery** takes 1–2 days after step 4. Implement the table above inside `grasp`.
7. **Jev grasp judge** takes 2 days after step 4. Add `grasp_state` and `grasp_questions` beside
   the touch equivalents. Use the wrist frame, `JudgeRunner` and existing records. This feeds #160.
8. **GraspGenX library** takes 3–4 days in parallel. Describe the SO-101 gripper and run once per
   block mesh offline. Cache top grasps in the block frame and compose them with the twin's pose
   at runtime. This closes #105 without a point cloud and covers cuboids and the cylinder.
9. **Lift policy backup** takes 3–5 days after step 1 and PR 25. Add pose bias and delay per
   episode, latch poses in `PolicyStrategy`, and judge lifts from the jaw rule and forward
   kinematics. Match the real 1.3 s jaw closing time. Plan one GPU run after CPU tests.
   This is the backup path for #109.
10. **Stand work** continues throughout. Apply servo offsets and calibration from twin PRs 60
    and 61, calibrate colours, and score every block with the #94 protocol.

## Timeline

Seven working days remain before the demo. Steps 1, 2 and 4 fit the first week and give a
cube grasp with recovery using the jaw rule alone. Step 3 runs in parallel in the twin to
provide fresh poses and yaw. Steps 5, 6 and 7 land in the second week. Steps 8 and 9 reach the demo only if
the earlier work stays on schedule.

Nothing in this plan will have run on the stand by the #109 decision on 2026-11-02, so that
decision cannot rely on a 90% result. Choose the engineered path as main and the lift policy
as backup, consistent with the tracker's LLM path.

## Risks

- An LLM pixel may miss the block. The grower selects a nearby matching colour blob.
- Two blocks of one colour may merge. The size check rejects blobs too large for one block.
- A tilted approach may push the block. Centre at hover and descend vertically, as #95 requires.
- Jaw thresholds depend on servo offsets. Calibrate first.
- Jev does not guarantee production image support. Record its verdict while keeping grasping
  independent of it.

## Tracker

- #94 defines the Classic World block set.
- #160 records `touch` and `touch_jev` as the current state.
- Create work packages for steps 3–9 and scope #105 to the offline library.

# Grasp plan: from `touch` to pick-up with the LLM detector and the wrist camera

Written 2026-10-31. Decision on the main and backup path: #109, 2026-11-02. Demo: #80, 2026-11-11.

## Goal

The arm picks a named block ("the red cube") from the table, lifts it, and recovers when the
block falls out of the jaws or is taken out. The block comes from the Classic World set
(2.5 cm unit: cube, two cuboids, cylinder). The object is found by the LLM detector; the
cameras make its pose precise. Success is the definition of #95: the jaw servo stalls on the
object and the cameras see it lift.

## What exists

- `touch` descends the closed gripper onto the object until the arm stalls on it, from a pose
  that is seconds old and 1-3 cm off. `touch_jev` asks Jev after each descent and retries.
- `Observation.objects[label]` already carries position, `source` ("kamery", "w dloni",
  "ostatnie widziane") and `age_s`. Rotation is identity: the LLM detector gives a pixel only.
- The detector contract accepts `mask` and `size_m`. With both, the twin fits a box of that
  size to the masks of the fixed cameras and the pose gets yaw. Nobody fills them yet.
- The jaw rule (jaw stops wider than commanded, does not creep) is the only contact signal.
  The Feetech backend reads positions, not load. The twin's tracker uses the same rule for
  "w dloni".
- The wrist camera is on the official mount, read today only by the `dream` strategy.

## Architecture

Everything runs in the twin's process. The twin sees and records, the strategy decides and
moves, Jev settles the ambiguous cases. so-arm101 imports only `machinekind-twin-api`.

| Block | Repo | New or exists |
|---|---|---|
| Wrist camera registered as a camera that is not used for triangulation | twin | new, config |
| Seeded refinement: LLM pixel -> colour mask -> `size_m` from the block set, tracked at frame rate between LLM calls | twin, detector plugin | new |
| Box fit from masks, yaw, object tracker, `source` | twin | exists |
| Block set: dimensions and colour ranges per block | soarm-lab data | new (#94) |
| `grasp` strategy: touch's probe, then rise, open, straddle, close, verify, lift, verify | soarm-lab | new, subclass of `touch` |
| Wrist-camera centring and yaw at the hover | soarm-lab | new, needs `kinematics` |
| Recovery: every tick while carrying, the jaw rule against the twin's `source` | soarm-lab, inside `grasp` | new |
| Jev grasp judge: state buckets plus the wrist frame, decision records | soarm-lab | client exists, questions new |
| Grasp library per block from GraspGenX, run offline on the block meshes | soarm-lab data | new |
| Lift policy fine-tune with a biased, delayed, latched pose | soarm-lab `rl` | exists, three changes |
| Safety supervisor, run judge (#176), episode log (#151) | twin | supervisor exists, the rest planned |

No API change. Decision records reach the twin's log through `Action.decisions` once the pin
moves to api 0.1.3.

## Recovery

While the block is in the jaws, until the strategy opens them on purpose, check each tick:

| Jaw rule | Twin's `source` | Meaning | Action |
|---|---|---|---|
| holding | "w dloni" | fine | carry on |
| empty | "kamery", on the table | dropped, position known | rise, re-aim, regrasp |
| empty | "ostatnie widziane", for a few seconds | hidden by the hand or falling | hold at a safe height, wait |
| empty | "ostatnie widziane", longer, or far away | taken out | stop, hold for a person |
| holding | "kamery" elsewhere | a second block or a tracker error | ask Jev with the wrist frame |

Rules: never descend to a stale pose; a snatch is not a miss, so no chase; recoveries count
against the attempt budget. At the handover the strategy's responsibility ends when it commands
the release.

## Steps

Ordered by dependency. Effort in working days for one person.

1. **Block set** (soarm-lab, 0.5 day plus measuring). Dimensions, mass and colour ranges per
   block, as data. Closes #94.
2. **Wrist camera in the twin** (twin, 0.5 day). Config entry, frames in `obs.frames`, not in
   triangulation. Needed by steps 5 and 7.
3. **Seeded refinement in the detector** (twin, 2-3 days). An option on the LLM detector or a
   wrapper around any seed detector: grow a colour mask from the pixel, snap to the nearest
   matching blob within the pixel error, attach `size_m` from the label, follow the blob between
   LLM calls. Gives yaw and a fresh pose. Calibrate colour ranges on the stand.
4. **`grasp` strategy** (soarm-lab, 3-4 days). Subclass `TouchStrategy`, new phases after
   `after_touch`, jaw-rule verification after close and after lift, `retarget` sideways when a
   jaw tip lands on the block, attempt budget. Cube and cylinder first.
5. **Wrist-camera centring and yaw** (soarm-lab, 3-4 days, starts once step 4 has its phases).
   Centre phase before the descent: colour blob offset from a calibrated fingertip pixel, scale
   learned by moving 1 cm, rotation passed into `ik_step`. Makes the cuboids feasible.
6. **Recovery loop** (soarm-lab, 1-2 days after step 4). The table above, inside `grasp`.
7. **Jev grasp judge** (soarm-lab, 2 days after step 4). `grasp_state` and `grasp_questions`
   beside the touch ones, the wrist frame only, the same `JudgeRunner` and records. Feeds #160.
8. **GraspGenX library** (soarm-lab, 3-4 days, parallel). SO-101 gripper description, one offline
   run per block mesh, top grasps cached in the block frame, composed with the twin's pose at run
   time. Closes #105 without a point cloud. Value shows on cuboids and the cylinder.
9. **Lift policy backup** (soarm-lab `rl`, 3-5 days after step 1 and PR 25). Per-episode pose
   bias and delay in the randomisation, pose latching in `PolicyStrategy`, lift verdict from the
   jaw rule and forward kinematics. One GPU run, planned together, after CPU tests. Match the
   real 1.3 s jaw closing time. Backup path for #109.
10. **Stand days** (throughout). Servo zero offsets and calibration from twin PRs 60 and 61,
    colour calibration, the #94 protocol scored on every block.

## Timeline

Seven working days remain before the demo. Steps 1, 2 and 4 fit the first week and give a
cube grasp with recovery by the jaw rule alone. Step 3 in parallel in the twin gives fresh poses
and yaw. Steps 5, 6 and 7 land in the second week. Steps 8 and 9 are parallel tracks that
reach the demo only if nothing above slips.

The #109 decision on 2026-11-02 cannot rest on a 90% result: nothing above will have run on the
stand by then. It can rest on this plan, with the engineered path as main and the lift policy
as backup, which matches what the tracker already says about the LLM path.

## Risks

- The LLM's pixel lands off the block. The grower snaps to colour, not to the pixel.
- Two blocks of one colour merge in the mask. The size check rejects a blob too large for one.
- The tilted approach pushes the block before the straddle. Centre at the hover, descend
  vertically, as #95 asks.
- Jaw thresholds shift with servo zero offsets. Calibrate first.
- Jev images are not production-guaranteed. The judge is a record, never the thing the grasp
  depends on.

## Tracker

- #94 object set: the Classic World blocks.
- #160: `touch` and `touch_jev` as the current state.
- New work packages for steps 3 to 9; #105 re-scoped to the offline library.

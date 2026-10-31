# Grasp libraries

One library per block kind of `rl/objects.py`, `<kind>.json`, read by the `grasp` strategy with
`grasp_source: library` (`soarm_lab/grasp_library.py`). Shipped here:

- `cube.json`: a hand-written example (generator `hand`), the conversion of
  `tools/graspgenx/examples/cube.yml`. A GraspGenX run replaces it.

The other kinds have no library yet; `grasp_source: library` refuses them.

## The file

```json
{
 "format": "soarm-lab-grasps",
 "version": 1,
 "kind": "cube",
 "half_size": [0.0125, 0.0125, 0.0125],
 "frame": "block frame of rl/objects.py (centre, x the long side, z up); ...",
 "generator": {"name": "graspgenx", "version": "1.0.0", "commit": "<sha>", "gripper": "so101",
               "checkpoint": "release", "date": "YYYY-MM-DD", "note": "", "dropped": "3 of 50"},
 "grasps": [
  {"position": [0.0, 0.0, 0.0], "approach": [0.0, 0.0, -1.0], "closing": [0.0, 1.0, 0.0],
   "width_m": 0.025, "score": 0.92}
 ]
}
```

`soarm-lab grasps convert` writes it with one value per line (JSON indent 1), floats rounded to 6
decimals, the grasps sorted by score, best first.

- The block frame is that of `rl/objects.py`: the block's centre, x along its longest side, z up as it
  lies on the table, so its top is at z = `half_size[2]`. Lengths are in metres.
- `position` is the pad point: the centre of the gripper description's open sweep box
  (`sweep_volume.offset` in `tools/graspgenx/so101/config.json`), 27.4 mm behind the fixed fingertip. The
  strategy puts the fingertips' midpoint there.
- `approach` and `closing` are unit vectors in the block frame: the gripperframe site's x and z, so the
  approach points along the fingers and the closing direction toward the moving jaw.
- `width_m` is the block's width along `closing`, what the jaw closes on; `score` is the generator's.
- `generator` records what made the file: `name` (`graspgenx`, or `hand` for a hand-written file), the
  GraspGenX version, commit and checkpoint, the gripper, the date, a note and how many of GraspGenX's
  grasps the converter dropped because their pad point lay outside the block.

The loader refuses a file whose `half_size` differs from the table's: after the blocks are measured
(`MEASURED = True`) every library must be regenerated, or the example re-converted. It also refuses axes
that are not unit vectors or not perpendicular (beyond 1e-5) and pad points more than 1 mm outside the
block. [tools/graspgenx/README.md](../../../../tools/graspgenx/README.md) has the GraspGenX run that makes
a library, the frames and when to regenerate.

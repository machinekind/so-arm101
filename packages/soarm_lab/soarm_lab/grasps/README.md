# Grasp libraries

`grasp_source: library` loads one `<block>.json` file per block. The bundled `cube.json`
is the hand-written conversion of `tools/graspgenx/examples/cube.yml`, with generator
`hand`. Other blocks have no library yet and refuse library mode.

## Format

```json
{
 "format": "soarm-lab-grasps",
 "version": 1,
 "kind": "cube",
 "half_size": [0.0145, 0.0145, 0.0145],
 "frame": "block frame of soarm_lab/blocks (centre, x the long side, z up); ...",
 "generator": {"name": "graspgenx", "version": "1.0.0", "commit": "<sha>", "gripper": "so101",
               "checkpoint": "release", "date": "YYYY-MM-DD", "note": "", "dropped": "3 of 50"},
 "grasps": [
  {"position": [0.0, 0.0, 0.0], "approach": [0.0, 0.0, -1.0], "closing": [0.0, 1.0, 0.0],
   "width_m": 0.029, "score": 0.92}
 ]
}
```

The converter writes indent 1 JSON with six-decimal floats, sorted by descending score.

- The body frame is centred on the block, with x along its long side and z up in the resting
  pose. Lengths are in metres. Scene poses rotate this frame.
- `position` is the pad point at `sweep_volume.offset`, the centre of the gripper's open
  sweep box. The strategy places the fingertip midpoint there.
- `approach` and `closing` are unit vectors in the body frame, corresponding to TCP x and z.
- `width_m` is block width along closing. `score` is the generator score.
- `generator` records name, version, commit, gripper, checkpoint, date, note and dropped count.

The loader checks `half_size` against `blocks.block`. Regenerate libraries after measuring
blocks or changing catalog dimensions. Axes must be unit and perpendicular within 1e-5.
Pad points may lie at most 1 mm outside the block.

See [generation and frame conversion](../../../../tools/graspgenx/README.md).

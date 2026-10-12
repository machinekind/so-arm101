# Mounts

One folder per mount. Each holds `transform.yaml` and, later, the CAD of the part.

`transform.yaml` places a child frame in a parent frame:

- `parent`, `child`: frame names. MJCF names where they exist.
- `pos`: metres, `[x, y, z]`.
- `quat`: unit quaternion, `[w, x, y, z]` (the MuJoCo order).
- `cad`: path of the CAD file, or `TODO`.

Values marked TODO are not measured yet.

| Folder | Mount |
|---|---|
| `table/` | arm base on the table top |
| `inverted/` | arm base hanging upside down |
| `wrist_cam/` | camera on the wrist |
| `top_cam/` | camera above the table |

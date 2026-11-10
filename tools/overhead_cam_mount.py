"""Builds the overhead camera mount fixture from TheRobotStudio's STL files.

    uv run python tools/overhead_cam_mount.py <folder with the upstream STLs>

The folder holds `arm_base.stl`, `cam_mount_bottom.stl`, `cam_mount_middle.stl` and `cam_mount_top.stl`
from SO-ARM100 `Optional/Overhead_Cam_Mount_32x32_UVC_Module/stl` (see THIRD_PARTY_NOTICES for the commit).
Writes the three pole parts in metres and `fixture.yaml` into the robot folder's
`fixtures/overhead_cam_32`, and prints how the placement was found.

Upstream files are in millimetres with +y up. The fixture's frame has x forward (where the camera
head points, the arm's forward), y left and z up, its origin on the pole's axis under the foot. Each
part's frame sits on the pole's axis at the part's lower end.
"""

from __future__ import annotations

import struct
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROBOT = ROOT / "packages" / "soarm" / "soarm" / "robot"
OUT = ROBOT / "fixtures" / "overhead_cam_32"
#: Upstream (x, y, z) -> fixture axes (x, -z, y): +y up becomes +z up.
TO_FIXTURE = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
#: Length of the tabs that join the arm plate and the pole's foot [mm].
TAB = 10.0
#: The camera module: board above the plate, lens below it around the hole [mm].
BOARD_ABOVE, LENS_BELOW, LENS_RADIUS = 6.0, 18.0, 8.0
#: The lens's centre of projection below the plate's lower face [mm]; a nominal value.
PUPIL_BELOW = 6.0
RGBA = [0.25, 0.25, 0.25, 1.0]
#: A binary STL triangle: normal, three vertices, attribute bytes.
STL = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])


def read_stl(path: Path) -> np.ndarray:
    """(n, 3, 3) triangles of a binary STL."""
    data = path.read_bytes()
    n = struct.unpack("<I", data[80:84])[0]
    return np.frombuffer(data[84:84 + 50 * n], dtype=STL)["v"].astype(float)


def write_stl(path: Path, tris: np.ndarray, header: str) -> None:
    n = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tris), dtype=STL)
    rec["n"], rec["v"] = n, tris
    head = header.encode("ascii")[:80].ljust(80, b" ")
    path.write_bytes(head + struct.pack("<I", len(tris)) + rec.tobytes())


def round_holes(tris: np.ndarray, up: int, radius: tuple[float, float] = (2.0, 3.0),
                cell: float = 0.2) -> np.ndarray:
    """Centres (the two horizontal coordinates, mm) of the round through-holes seen from above.

    The mesh's silhouette on a grid of `cell` mm; a hole is a patch of empty cells cut off from the
    border that is mostly the largest circle inside it, of a radius in `radius` mm."""
    flat = [i for i in range(3) if i != up]
    xy = tris[:, :, flat]
    lo = xy.reshape(-1, 2).min(axis=0) - 2 * cell
    shape = np.ceil((xy.reshape(-1, 2).max(axis=0) + 2 * cell - lo) / cell).astype(int)
    solid = np.zeros(shape, bool)
    for a, b, c in (xy - lo) / cell:
        i0, j0 = np.floor(np.minimum(np.minimum(a, b), c)).astype(int)
        i1, j1 = np.ceil(np.maximum(np.maximum(a, b), c)).astype(int)
        gi, gj = np.meshgrid(np.arange(i0, i1 + 1) + 0.5, np.arange(j0, j1 + 1) + 0.5, indexing="ij")
        d = (b[0] - a[0]) * (c[1] - a[1]) - (c[0] - a[0]) * (b[1] - a[1])
        if abs(d) < 1e-12:
            continue
        u = ((b[0] - gi) * (c[1] - gj) - (c[0] - gi) * (b[1] - gj)) / d
        v = ((c[0] - gi) * (a[1] - gj) - (a[0] - gi) * (c[1] - gj)) / d
        solid[i0:i1 + 1, j0:j1 + 1] |= (u >= 0) & (v >= 0) & (u + v <= 1)
    # Label the empty cells: each takes the smallest index of its 4-connected patch.
    label = np.where(solid, solid.size, np.arange(solid.size).reshape(shape))
    while True:
        new = label.copy()
        for axis in (0, 1):
            for step in (1, -1):
                new = np.minimum(new, np.where(solid, solid.size, np.roll(label, step, axis=axis)))
        new[solid] = solid.size
        if np.array_equal(new, label):
            break
        label = new
    out = []
    for k in np.unique(label[~solid]):
        cells = np.argwhere(label == k)
        lo_c, hi_c = cells.min(axis=0), cells.max(axis=0)
        if k == 0 or (hi_c - lo_c).max() * cell > 8.0:
            continue                                     # the outside, or too big for a screw hole
        # The largest circle inside the patch: its centre and radius, whatever the patch's other corners.
        box = solid[lo_c[0] - 1:hi_c[0] + 2, lo_c[1] - 1:hi_c[1] + 2]
        walls = np.argwhere(box) + lo_c - 1
        dist = np.linalg.norm(cells[:, None] - walls[None], axis=2).min(axis=1)
        r = (dist.max() - 0.5) * cell
        if radius[0] < r < radius[1] and len(cells) * cell**2 < 1.6 * np.pi * r**2:
            out.append(lo + (cells[dist >= dist.max() - 0.5].mean(axis=0) + 0.5) * cell)
    return np.array(sorted(out, key=lambda c: (round(c[0]), c[1])))


def so101_base_footprint() -> np.ndarray:
    """The SO-101 base mesh in the robot's base frame [mm], placed as so101.xml places it."""
    geom = next(g for g in ET.parse(ROBOT / "so101.xml").iter("geom") if g.get("mesh") == "base_so101_v2")
    q = np.array(geom.get("quat", "1 0 0 0").split(), float)
    w, x, y, z = q / np.linalg.norm(q)
    R = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                  [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                  [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    pos = np.array(geom.get("pos", "0 0 0").split(), float)
    return (read_stl(ROBOT / "assets" / "base_so101_v2.stl") @ R.T + pos) * 1000.0


def part(tris: np.ndarray, axis_xz: np.ndarray, bottom: float) -> np.ndarray:
    """Upstream triangles [mm] -> the part's frame [m]: on the pole's axis at height `bottom`."""
    return (tris - np.array([axis_xz[0], bottom, axis_xz[1]])) @ TO_FIXTURE.T / 1000.0


def box(pts: np.ndarray, floor: float | None = None) -> dict[str, np.ndarray]:
    """The points' bounding box as a collider; `floor` lowers its bottom to that height."""
    lo, hi = pts.reshape(-1, 3).min(axis=0), pts.reshape(-1, 3).max(axis=0)
    if floor is not None:
        lo[2] = floor
    return {"pos": (lo + hi) / 2, "size": (hi - lo) / 2}


def flow(value, digits: int = 5) -> str:
    """A YAML flow value, numbers rounded to `digits` decimals (5: 0.01 mm)."""
    if isinstance(value, dict):
        return "{" + ", ".join(f"{k}: {flow(v, digits)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple, np.ndarray)):
        return "[" + ", ".join(flow(v, digits) for v in value) + "]"
    return repr(round(float(value), digits) + 0.0)


def main(src: Path) -> None:
    plate, bottom = read_stl(src / "arm_base.stl"), read_stl(src / "cam_mount_bottom.stl")
    middle, top = read_stl(src / "cam_mount_middle.stl"), read_stl(src / "cam_mount_top.stl")
    pv, bv = plate.reshape(-1, 3), bottom.reshape(-1, 3)

    # The SO-101 base sits on the arm plate by its four screw holes; base (x, y) = plate (x, -z) - shift.
    base_holes = round_holes(so101_base_footprint(), up=2)
    plate_holes = round_holes(plate, up=1) * np.array([1.0, -1.0])
    plate_holes = np.array(sorted(plate_holes, key=lambda c: (round(c[0]), c[1])))
    assert len(base_holes) == len(plate_holes) == 4, (base_holes, plate_holes)
    shift = (plate_holes - base_holes).mean(axis=0)
    fit = float(np.abs(plate_holes - base_holes - shift).max())
    base_low = so101_base_footprint()[:, :, 2].min()

    # The plate's back strip and the foot are equally wide; their tabs interlock end to end.
    foot_w = bv[:, 0].max() - bv[:, 0].min()
    strip = pv[:, 0].min()
    assert abs((pv[pv[:, 2] > pv[:, 2].max() - TAB, 0].max() - strip) - foot_w) < 0.05
    foot_h = bv[(bv[:, 0] < 1.0), 1].max()                  # the foot plate's top face
    pole = bv[bv[:, 1] > foot_h + 0.5]
    axis = (pole.min(axis=0) + pole.max(axis=0))[[0, 2]] / 2
    # Base origin in the foot's file frame, for the arm with the pole on its left: the plate's -z end
    # joins the foot's +z end. With the pole on its right, the plate's +z end joins the foot's -z end.
    dx = bv[:, 0].min() - strip
    dy = bv[:, 1].min() - pv[:, 1].min()
    left_dz = (bv[:, 2].max() - TAB) - pv[:, 2].min()
    right_dz = (bv[:, 2].min() + TAB) - pv[:, 2].max()
    origin_x = shift[0] + dx
    origin_y = pv[:, 1].max() + dy - base_low                 # the base's bottom on the plate's top face
    placements = {}
    for side, dz in (("left", left_dz), ("right", right_dz)):
        origin_z = -shift[1] + dz
        # The fixture's origin in the base frame; base axes equal the fixture's (x, -z, y).
        xyz = [axis[0] - origin_x, -(axis[1] - origin_z), 0.0 - origin_y]
        placements[side] = [round(float(v) / 1000.0, 4) for v in xyz]   # 0.1 mm

    # Pole parts. The bottom's tenon starts at its shoulder; the middle and the top share one frame.
    bottom_m = part(bottom, axis, 0.0)
    bv_m = bottom_m.reshape(-1, 3)
    # The pole's long faces have no vertices: the first height in the part's upper half is the shoulder.
    shoulder = min(v for v in np.unique(np.round(bv[:, 1], 3)) if v > bv[:, 1].max() / 2)
    mid_low, top_low = middle[:, :, 1].min(), top[:, :, 1].min()
    middle_m, top_m = part(middle, np.zeros(2), mid_low), part(top, np.zeros(2), top_low)
    mid_shoulder = round(top_low - mid_low, 2)
    head = top_m[:, :, 0].max(axis=1) > 0.0128                # the plate past the pole's side
    pole_top = top_m[top_m[:, :, 0].max(axis=1) <= 0.0128]

    # The camera plate: its lower face's normal, and the lens hole's centre on that face.
    n = np.cross(top[:, 1] - top[:, 0], top[:, 2] - top[:, 0])
    n /= np.linalg.norm(n, axis=1, keepdims=True)
    lower = top[(n[:, 1] < -0.8) & (top[:, :, 1].min(axis=1) > 350.0)]
    nb = np.cross(lower[:, 1] - lower[:, 0], lower[:, 2] - lower[:, 0]).sum(axis=0) * [1.0, 1.0, 0.0]
    nb /= np.linalg.norm(nb)                                  # the plate is symmetric about z = 0
    tilt = float(np.degrees(np.arccos(-nb[1])))
    along = np.cross(nb, [0.0, 0.0, 1.0])
    along *= np.sign(along[0])
    walls = top[(np.abs(n @ nb) < 0.2) & (np.abs(top[:, :, 2]).max(axis=1) < 12.5)].reshape(-1, 3)
    walls = walls[(walls @ nb > (lower[0, 0] @ nb) - 3.5) & (walls[:, 0] > 15.0)]
    s = walls @ along
    hole = (s.min() + s.max()) / 2 * along + float(lower[0, 0] @ nb) * nb
    hole[2] = 0.0
    pupil = hole + PUPIL_BELOW * nb
    z_cam = TO_FIXTURE @ nb
    x_cam = np.array([0.0, -1.0, 0.0])                        # the image's right: the arm's right
    T_cam = np.eye(4)
    T_cam[:3, :3] = np.column_stack([x_cam, np.cross(z_cam, x_cam), z_cam])
    T_cam[:3, 3] = TO_FIXTURE @ (pupil - np.array([0.0, top_low, 0.0])) / 1000.0
    # The head's collider: the plate with the board above it and the lens below the hole.
    plate_pts = top[head].reshape(-1, 3)
    lens = [hole + d * nb + r * u for d in (0.0, LENS_BELOW) for r in (-LENS_RADIUS, LENS_RADIUS)
            for u in (along, np.array([0.0, 0.0, 1.0]))]
    module = np.concatenate([plate_pts - BOARD_ABOVE * nb, np.array(lens)])
    head_box = box(np.concatenate([top_m[head].reshape(-1, 3), part(module, np.zeros(2), top_low)]))

    OUT.mkdir(parents=True, exist_ok=True)
    for name, tris in (("cam_mount_bottom", bottom_m), ("cam_mount_middle", middle_m),
                       ("cam_mount_top", top_m)):
        note = f"SO-ARM100 {name}.stl modified by Machinekind: metres, part frame"
        write_stl(OUT / f"{name}.stl", tris, note)
    yaml = f"""\
# TheRobotStudio's overhead camera mount for a 32x32 mm UVC camera module (SO-ARM100, Apache-2.0):
# a pole of two or three parts on a foot that joins the arm plate the SO-101 stands on. See README.md.
# Written by tools/overhead_cam_mount.py from the upstream STL files.
name: overhead_cam_32
title: Statyw kamery z gory (modul 32x32)
# Frame: x forward (where the camera head points), y left, z up [m]; origin on the pole's axis under the foot.
parts:
  bottom:
    mesh: cam_mount_bottom.stl
    rgba: {flow(RGBA)}
    top: [0.0, 0.0, {shoulder / 1000:.4f}]
    colliders:
      foot: {flow(box(bv_m[bv_m[:, 2] <= foot_h / 1000 + 1e-6]))}
      pole: {flow(box(bv_m[bv_m[:, 2] > foot_h / 1000 + 1e-6], floor=foot_h / 1000))}
  middle:
    mesh: cam_mount_middle.stl
    rgba: {flow(RGBA)}
    top: [0.0, 0.0, {mid_shoulder / 1000:.4f}]
    colliders:
      pole: {flow(box(middle_m))}
  top:
    mesh: cam_mount_top.stl
    rgba: {flow(RGBA)}
    colliders:
      pole: {flow(box(pole_top))}
      # The camera plate with the module on it: the board above, the lens below.
      head: {flow(head_box)}
    # The module's lens, looking down and {tilt:.0f} deg forward. T_cam2part in OpenCV axes,
    # the top of the image forward.
    cameras:
      camera:
{chr(10).join(f"        - {flow([*row[:3], round(row[3], 4)], digits=9)}" for row in T_cam)}
# The full design; a build without the middle part is [bottom, top].
stack: [bottom, middle, top]
# Where the fixture stands in the SO-101's base frame when the arm stands on the design's arm plate:
# the pole on the arm's left or right.
placements:
  left: {{xyz: {flow(placements["left"])}}}
  right: {{xyz: {flow(placements["right"])}}}
"""
    (OUT / "fixture.yaml").write_text(yaml, encoding="utf-8")
    print(f"screw holes: base {base_holes.round(2).tolist()}, plate {plate_holes.round(2).tolist()}, "
          f"shift {shift.round(2).tolist()} mm (worst {fit:.2f} mm)")
    print(f"pole axis in the foot's file frame {axis.round(3).tolist()} mm; base origin x {origin_x:.2f}, "
          f"{origin_y:.2f} mm above the table")
    print(f"placements {placements}; shoulders: bottom {shoulder} mm, middle {mid_shoulder} mm; "
          f"plate tilt {tilt:.2f} deg")
    print(f"heights: bottom+top {(shoulder + top[:, :, 1].max() - top_low) / 1000:.4f} m, "
          f"bottom+middle+top {(shoulder + mid_shoulder + top[:, :, 1].max() - top_low) / 1000:.4f} m")


if __name__ == "__main__":
    main(Path(sys.argv[1]))

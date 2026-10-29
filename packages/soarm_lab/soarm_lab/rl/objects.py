"""The objects the arm grasps: Classic World wooden blocks, their geometry, mass and what the policy sees.

    spec = object_spec("cuboid85")            # half sizes, mass, collision parts, yaw symmetry
    box = scene_object("cuboid85")            # a `station.Box` for the scene
    desc = descriptor("cuboid85")             # the policy's view of the kind: one-hot + half sizes

A shape-conditioned policy (`TaskConfig.shape_conditioned`) observes the block's pose AND its kind and
dimensions (`descriptor`), so one policy can grasp several blocks. On the desk the detector names the
kind; the dimensions come from `BLOCKS`, not from the image.

Every block lies on the table with its longest side along its body x and its bottom face at z = 0;
`ObjectSpec.half_size` is the bounding box in that pose. The cylinder lies on its side, its axis along
x. `bottom_height` gives the height of the LOWEST point above the table for any rotation, from the
part geometry: an object's centre minus the resting half height says "lifted" for a cuboid stood on
its end, the lowest point does not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

from ..station import Box, Part

# ---------------------------------------------------------------- the blocks
#: Whether `BLOCKS` holds measurements. While False every number below is a PLACEHOLDER: unmeasured,
#: a 2.5 cm unit assumed, the mass from painted maple at ~700 kg/m^3 times the volume.
MEASURED = False

#: Per kind: the shape, the bounding box as the block lies on the table (x the longest side) [mm] and
#: the mass [g]. For the cylinder the box is length x diameter x diameter. Unmeasured, 2.5 cm unit
#: assumed, maple ~700 kg/m^3; set `MEASURED = True` once these are Marcin's measurements.
BLOCKS: dict[str, tuple[str, tuple[float, float, float], float]] = {
    "cube": ("box", (25.0, 25.0, 25.0), 10.9),               # 15.6 cm^3
    "cuboid5": ("box", (50.0, 25.0, 25.0), 21.9),            # 31.3 cm^3
    "cuboid85": ("box", (85.0, 25.0, 25.0), 37.2),           # 53.1 cm^3
    "cylinder": ("cylinder", (50.0, 25.0, 25.0), 17.2),      # 24.5 cm^3, axis along x
}

#: The kinds of the first milestone, in the order of the descriptor's one-hot. Appending a kind
#: changes `DESCRIPTOR_DIM` and with it every shape-conditioned policy's input.
KINDS = ("cube", "cuboid5", "cuboid85", "cylinder")

#: The rotation of a cylinder part whose axis (MuJoCo's z) lies along the body's x: +90 deg about y.
_AXIS_X = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)

#: The yaw symmetry the detector cannot see [rad]: a cube looks the same every 90 deg, a cuboid with a
#: square cross-section and a cylinder on its side every 180 deg.
_YAW_PERIOD = {"cube": math.pi / 2, "cuboid5": math.pi, "cuboid85": math.pi, "cylinder": math.pi}

#: A colour per kind (painted blocks). No blue: the pickplace target pad is blue.
COLOURS = {"cube": (0.85, 0.25, 0.2, 1.0), "cuboid5": (0.95, 0.75, 0.15, 1.0),
           "cuboid85": (0.2, 0.65, 0.35, 1.0), "cylinder": (0.6, 0.3, 0.7, 1.0)}

#: The scale of the half sizes in the descriptor [m]: a 5 cm half size is 1.
DESCRIPTOR_SCALE = 0.05
#: The descriptor's length: the one-hot kind, then the three half sizes.
DESCRIPTOR_DIM = len(KINDS) + 3


@dataclass(frozen=True)
class ObjectSpec:
    """One object as the simulation builds it."""

    kind: str
    #: Half extents of the bounding box as it lies on the table [m]; `half_size[2]` is the height of
    #: the body's origin above the table top at rest.
    half_size: tuple[float, float, float]
    #: [kg]
    mass: float
    #: The collision parts in the body frame (`station.Part`).
    parts: tuple[Part, ...]
    #: The yaw symmetry the detector cannot see [rad] (`task.fold_yaw`).
    yaw_period: float


@cache
def object_spec(kind: str) -> ObjectSpec:
    """The spec of `kind`, from `BLOCKS`."""
    if kind not in KINDS:
        raise ValueError(f"unknown object kind {kind!r}; known: {', '.join(KINDS)}")
    shape, size_mm, mass_g = BLOCKS[kind]
    half = tuple(float(x) / 2000.0 for x in size_mm)
    if shape == "box":
        parts = (Part("box", half),)
    elif shape == "cylinder":
        if half[1] != half[2]:
            raise ValueError(f"{kind}: a cylinder's box is length x diameter x diameter, got {size_mm} mm")
        parts = (Part("cylinder", (half[1], half[0], 0.0), quat=_AXIS_X),)
    else:
        raise ValueError(f"{kind}: unknown shape {shape!r}")
    return ObjectSpec(kind, half, mass_g / 1000.0, parts, _YAW_PERIOD[kind])


def scene_object(kind: str, name: str = "cube", xy: tuple[float, float] = (0.2, 0.0)) -> Box:
    """The block `kind` as a free body of the scene, lying at `xy` (base frame) on the table."""
    spec = object_spec(kind)
    return Box(name, spec.half_size, xy, rgba=COLOURS[kind], mass=spec.mass, parts=spec.parts)


def descriptor(kind: str) -> np.ndarray:
    """What a shape-conditioned policy sees of the object: (DESCRIPTOR_DIM,) float32.

    The kind as a one-hot over `KINDS`, then the half sizes divided by `DESCRIPTOR_SCALE`.
    """
    out = np.zeros(DESCRIPTOR_DIM, np.float32)
    out[KINDS.index(kind)] = 1.0
    out[len(KINDS):] = np.asarray(object_spec(kind).half_size) / DESCRIPTOR_SCALE
    return out


# ---------------------------------------------------------------- the lowest point
def _quat_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@cache
def _supports(kind: str) -> tuple[tuple[Part, tuple[tuple[float, ...], ...] | None], ...]:
    """Each part with its rotation in the body frame as plain floats, None for no rotation."""
    out = []
    for part in object_spec(kind).parts:
        R = _quat_mat(part.quat)
        out.append((part, None if np.array_equal(R, np.eye(3)) else tuple(tuple(map(float, row)) for row in R)))
    return tuple(out)


def bottom_height(xp, kind: str, pos, rot):
    """(N,) the height of the object's lowest point above the table [m].

    `pos` (N, 3) and `rot` (N, 3, 3): the body's position and rotation in the base frame (the table
    top at z = 0). `xp` is numpy or jax.numpy, as in `task`. Per part, the support along -z: for
    a box sum(|n| * half sides), for a cylinder radius * sqrt(n_x^2 + n_y^2) + half length * |n_z|,
    with n the table's normal in the part's frame. Only scalars and elementwise operations, so the
    dtype of `pos` is kept (float32 in JAX).
    """
    z = [rot[..., 2, i] for i in range(3)]                  # the table's normal in the body frame
    lowest = None
    for part, R in _supports(kind):
        n = z if R is None else [z[0] * R[0][j] + z[1] * R[1][j] + z[2] * R[2][j] for j in range(3)]
        centre = z[0] * part.pos[0] + z[1] * part.pos[1] + z[2] * part.pos[2]
        s = part.size
        if part.kind == "box":
            support = abs(n[0]) * s[0] + abs(n[1]) * s[1] + abs(n[2]) * s[2]
        elif part.kind == "cylinder":
            support = s[0] * (n[0] * n[0] + n[1] * n[1]) ** 0.5 + s[1] * abs(n[2])
        else:
            raise ValueError(f"{kind}: no support function for a {part.kind!r} part")
        value = centre - support
        lowest = value if lowest is None else xp.minimum(lowest, value)
    return pos[..., 2] + lowest

"""Block geometry, mass, and yaw symmetry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache

import numpy as np

from ..station import Box, Part
from .colour import WOOD, Rgba

#: Rotate the cylinder's local z axis onto the body's x axis.
_AXIS_X = (math.sqrt(0.5), 0.0, math.sqrt(0.5), 0.0)


@dataclass(frozen=True)
class BlockSpec:
    """Define geometry in the body frame independently of scene position and yaw."""

    name: str
    #: Bounding-box half extents along the body axes [m].
    half_size: tuple[float, float, float]
    #: Mass [kg].
    mass: float
    #: Collision geometry uses the body frame.
    parts: tuple[Part, ...]
    #: Smallest yaw rotation that leaves the appearance unchanged [rad].
    yaw_period: float


def box_block(name: str, size_mm: tuple[float, float, float], mass_g: float, *,
              yaw_period: float) -> BlockSpec:
    """Build a box from body-axis side lengths [mm] and mass [g]."""
    half = (size_mm[0] / 2000.0, size_mm[1] / 2000.0, size_mm[2] / 2000.0)
    return BlockSpec(name, half, mass_g / 1000.0, (Part("box", half),), yaw_period)


def cylinder_block(name: str, length_mm: float, diameter_mm: float, mass_g: float, *,
                   yaw_period: float) -> BlockSpec:
    """Build a cylinder along body x from length and diameter [mm] and mass [g]."""
    half_length, radius = length_mm / 2000.0, diameter_mm / 2000.0
    part = Part("cylinder", (radius, half_length, 0.0), quat=_AXIS_X)
    return BlockSpec(name, (half_length, radius, radius), mass_g / 1000.0, (part,), yaw_period)


def scene_object(spec: BlockSpec, name: str, xy: tuple[float, float], rgba: Rgba = WOOD) -> Box:
    """Create a free body at base-frame `xy` [m], with body z vertical and its bottom on the table."""
    return Box(name, spec.half_size, xy, rgba=rgba, mass=spec.mass, parts=spec.parts)


def _rotation(quat: tuple[float, float, float, float]) -> np.ndarray:
    w, x, y, z = np.asarray(quat, float) / np.linalg.norm(quat)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


@cache
def _part_rotations(spec: BlockSpec) -> tuple[tuple[Part, tuple[tuple[float, ...], ...] | None], ...]:
    """Pair each part with its body-frame rotation as floats, using None for the identity."""
    out = []
    for part in spec.parts:
        R = _rotation(part.quat)
        plain = tuple(tuple(map(float, row)) for row in R)
        out.append((part, None if np.array_equal(R, np.eye(3)) else plain))
    return tuple(out)


def bottom_height(xp, spec: BlockSpec, pos, rot):
    """Return the lowest point's height above the table [m] as an (N,) array.

    `pos` (N, 3) and `rot` (N, 3, 3) give the body's base-frame pose with the table at z = 0.
    Use numpy or jax.numpy as `xp`.
    """
    z = [rot[..., 2, i] for i in range(3)]                  # Table normal in the body frame.
    lowest = None
    for part, R in _part_rotations(spec):
        n = z if R is None else [z[0] * R[0][j] + z[1] * R[1][j] + z[2] * R[2][j] for j in range(3)]
        centre = z[0] * part.pos[0] + z[1] * part.pos[1] + z[2] * part.pos[2]
        s = part.size
        if part.kind == "box":
            reach = abs(n[0]) * s[0] + abs(n[1]) * s[1] + abs(n[2]) * s[2]
        elif part.kind == "cylinder":
            reach = s[0] * (n[0] * n[0] + n[1] * n[1]) ** 0.5 + s[1] * abs(n[2])
        else:
            raise ValueError(f"{spec.name}: no height for a {part.kind!r} part")
        value = centre - reach
        lowest = value if lowest is None else xp.minimum(lowest, value)
    return pos[..., 2] + lowest

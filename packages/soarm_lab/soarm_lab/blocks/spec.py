"""Block geometry, mass, and yaw symmetry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import cache
from typing import NamedTuple

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


@cache
def _axis_map(quat: tuple[float, float, float, float]) -> tuple[int, int, int]:
    """Map part axes to body axes. Reject rotations that scaling along body axes would shear."""
    A = np.abs(_rotation(quat))
    if not np.allclose(A, np.round(A), atol=1e-9):
        raise ValueError(f"a part rotated by {quat} is not axis-aligned: a body scale would shear it")
    return tuple(int(i) for i in np.argmax(A, axis=0))


def part_scale(part: Part, scale):
    """Convert body-axis scale factors (..., 3) to the frame of an axis-aligned `part`."""
    return scale[..., list(_axis_map(tuple(part.quat)))]


def _part_size(part: Part, ps) -> list:
    """Return scaled MuJoCo part dimensions from part-frame multipliers `ps` (..., 3)."""
    s = part.size
    if part.kind == "box":
        return [s[0] * ps[..., 0], s[1] * ps[..., 1], s[2] * ps[..., 2]]
    if part.kind == "cylinder":
        return [s[0] * ps[..., 0], s[1] * ps[..., 2], s[2] * ps[..., 2]]
    raise ValueError(f"no size rule for a {part.kind!r} part")


def check_scalable(spec: BlockSpec) -> None:
    """Require one centred, axis-aligned part with a supported size rule. Raise `ValueError` otherwise."""
    if len(spec.parts) != 1 or any(spec.parts[0].pos):
        raise ValueError(f"{spec.name}: a size spread needs a one-part block centred on its body")
    _part_size(spec.parts[0], part_scale(spec.parts[0], np.ones(3)))


def symmetric_scale(xp, spec: BlockSpec, s_x, s_y, s_z):
    """Return body-axis scale factors (..., 3) that preserve the block's symmetry.

    Cylinders share radial scales. A yaw period below pi shares the body x and y scales.
    """
    check_scalable(spec)
    s = [s_x, s_y, s_z]
    part = spec.parts[0]
    if part.kind == "cylinder":
        first, second = sorted(_axis_map(tuple(part.quat))[:2])
        s[second] = s[first]
    if spec.yaw_period < math.pi - 1e-9:
        s[1] = s[0]
    return xp.stack(s, -1)


def scale_from_mm(xp, spec: BlockSpec, d_x, d_y, d_z):
    """Return body-axis scale factors (..., 3) from side length changes [mm].

    A side of length L scales by 1 + d / L.
    `symmetric_scale` preserves radial and yaw symmetry.
    """
    sides = [2000.0 * h for h in spec.half_size]
    s = [1.0 + d / side for d, side in zip((d_x, d_y, d_z), sides, strict=True)]
    return symmetric_scale(xp, spec, *s)


class Geometry(NamedTuple):
    """Store geometry and inertia for each world's scaled block."""

    #: (N, 3) `geom_size` [m].
    size: np.ndarray
    #: (N,) `geom_rbound` [m].
    rbound: np.ndarray
    #: Half extents for `geom_aabb` and the body's bounding volume, shape (N, 3) [m].
    aabb: np.ndarray
    #: (N, 3) `body_inertia`, diagonal in the part's frame [kg m^2].
    inertia: np.ndarray


def scaled_geometry(spec: BlockSpec, scale, mass) -> Geometry:
    """Compute collision geometry and inertia from body-axis `scale` (N, 3) and `mass` (N,) [kg].

    The block must pass `check_scalable`. Cylinders require equal radial scales.
    """
    check_scalable(spec)
    part = spec.parts[0]
    scale = np.atleast_2d(np.asarray(scale, float))
    mass = np.broadcast_to(np.asarray(mass, float), scale.shape[:1])
    ps = part_scale(part, scale)
    if part.kind == "cylinder" and np.abs(ps[:, 0] - ps[:, 1]).max() > 1e-9:
        raise ValueError(f"{spec.name}: a cylinder's radial scales must be equal, got {ps[:, :2].tolist()}")
    size = np.stack(_part_size(part, ps), 1)
    if part.kind == "box":
        a, b, c = size[:, 0], size[:, 1], size[:, 2]
        rbound = np.sqrt(a * a + b * b + c * c)
        aabb = size
        inertia = mass[:, None] / 3.0 * np.stack([b * b + c * c, a * a + c * c, a * a + b * b], 1)
    else:
        r, h = size[:, 0], size[:, 1]
        rbound = np.sqrt(r * r + h * h)
        aabb = np.stack([r, r, h], 1)
        radial = mass * (3.0 * r * r + 4.0 * h * h) / 12.0
        inertia = np.stack([radial, radial, mass * r * r / 2.0], 1)
    return Geometry(size, rbound, aabb, inertia)


def bottom_height(xp, spec: BlockSpec, pos, rot, scale=None):
    """Return the lowest point's height above the table [m] as an (N,) array.

    `pos` (N, 3) and `rot` (N, 3, 3) give the body's base-frame pose with the table at z = 0.
    `scale` (N, 3) gives dimensionless multipliers along the body axes. None keeps the nominal size.
    Use numpy or jax.numpy as `xp`.
    """
    z = [rot[..., 2, i] for i in range(3)]                  # Table normal in the body frame.
    lowest = None
    for part, R in _part_rotations(spec):
        n = z if R is None else [z[0] * R[0][j] + z[1] * R[1][j] + z[2] * R[2][j] for j in range(3)]
        if scale is None:
            p, s = part.pos, part.size
        else:
            p = [part.pos[i] * scale[..., i] for i in range(3)]
            s = _part_size(part, part_scale(part, scale))
        centre = z[0] * p[0] + z[1] * p[1] + z[2] * p[2]
        if part.kind == "box":
            reach = abs(n[0]) * s[0] + abs(n[1]) * s[1] + abs(n[2]) * s[2]
        elif part.kind == "cylinder":
            reach = s[0] * (n[0] * n[0] + n[1] * n[1]) ** 0.5 + s[1] * abs(n[2])
        else:
            raise ValueError(f"{spec.name}: no height for a {part.kind!r} part")
        value = centre - reach
        lowest = value if lowest is None else xp.minimum(lowest, value)
    return pos[..., 2] + lowest


def can_stand(spec: BlockSpec) -> bool:
    """Return whether the block is longer along body x than body z."""
    return spec.half_size[0] > spec.half_size[2]


def end_yaw_period(spec: BlockSpec) -> float:
    """Return the yaw symmetry period [rad] with body x vertical.

    Zero means all yaw angles look alike. A square box end repeats every quarter turn.
    Other box ends repeat every half turn. Multiple or off-centre parts use a full turn.
    """
    if len(spec.parts) != 1 or any(spec.parts[0].pos):
        return 2.0 * math.pi
    part = spec.parts[0]
    if part.kind == "cylinder" and _axis_map(tuple(part.quat))[2] == 0:
        return 0.0
    if part.kind == "box" and math.isclose(spec.half_size[1], spec.half_size[2]):
        return math.pi / 2
    return math.pi


def rest_height(xp, spec: BlockSpec, scale, on_end=None):
    """Return resting heights of body origins (...) [m] for body-axis scales `scale` (..., 3).

    `on_end` (...) selects body x up. Other blocks have body z up. None selects body z up for all blocks.
    """
    lying = spec.half_size[2] * scale[..., 2]
    return lying if on_end is None else xp.where(on_end, spec.half_size[0] * scale[..., 0], lying)


def quat_on_end(xp, quat):
    """Compose `quat` (..., 4) with a -90 degree turn about body y.

    Quaternions use (w, x, y, z). A yaw-only input gives body x up.
    """
    w, x, y, z = quat[..., 0], quat[..., 1], quat[..., 2], quat[..., 3]
    return math.sqrt(0.5) * xp.stack([w + y, x + z, y - w, z - x], -1)


def rotation_on_end(xp, yaw):
    """Return rotations (..., 3, 3) with body x up, turned by `yaw` (...) [rad] about the vertical."""
    c, s = xp.cos(yaw), xp.sin(yaw)
    z, o = xp.zeros_like(c), xp.ones_like(c)
    rows = [xp.stack([z, -s, -c], -1), xp.stack([z, c, -s], -1), xp.stack([o, z, z], -1)]
    return xp.stack(rows, -2)


def yaw_on_end(xp, rot):
    """Read yaw (...) [rad] from body y in rotations (..., 3, 3) produced by `rotation_on_end`."""
    return xp.arctan2(-rot[..., 0, 1], rot[..., 1, 1])


def is_on_end(xp, rot):
    """Return a mask (...) of rotations (..., 3, 3) with the body x axis within 45 degrees of vertical."""
    return xp.abs(rot[..., 2, 0]) >= math.sqrt(0.5)

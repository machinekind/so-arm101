"""Look up block specifications across catalogs."""

from __future__ import annotations

import math

from . import classic_world
from .colour import WOOD, random_rgba
from .spec import (
    BlockSpec,
    Geometry,
    bottom_height,
    box_block,
    can_stand,
    check_scalable,
    cylinder_block,
    end_yaw_period,
    is_on_end,
    part_scale,
    quat_on_end,
    rest_height,
    rotation_on_end,
    scale_from_mm,
    scaled_geometry,
    scene_object,
    symmetric_scale,
    yaw_on_end,
)

__all__ = ["WOOD", "BlockSpec", "Geometry", "block", "bottom_height", "box_block", "can_stand",
           "check_scalable", "cylinder_block", "end_yaw_period", "is_on_end", "part_scale", "quat_on_end",
           "random_rgba", "rest_height", "rotation_on_end", "scale_from_mm", "scaled_geometry",
           "scene_object", "symmetric_scale", "yaw_on_end"]

#: The default cube preserves the bundled policies' geometry.
_CUBE_3CM = box_block("cube3cm", (30.0, 30.0, 30.0), 30.0, yaw_period=math.pi / 2)

_CATALOGS = ({_CUBE_3CM.name: _CUBE_3CM}, classic_world.BLOCKS)


def block(name: str) -> BlockSpec:
    """Return the named block specification or raise `ValueError`."""
    for catalog in _CATALOGS:
        if name in catalog:
            return catalog[name]
    known = sorted(k for catalog in _CATALOGS for k in catalog)
    raise ValueError(f"unknown block {name!r}; known: {', '.join(known)}")

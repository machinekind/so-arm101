"""Measured geometry for the Classic World wooden block catalog."""

from __future__ import annotations

import math

from .spec import box_block, cylinder_block

#: Estimate masses from wood density until the blocks are weighed [g/mm^3].
DENSITY = 0.7e-3

BLOCKS = {b.name: b for b in (
    box_block("cube", (29.0, 29.0, 29.0), DENSITY * 29.0 ** 3, yaw_period=math.pi / 2),
    box_block("cuboid", (60.0, 29.0, 14.0), DENSITY * 60.0 * 29.0 * 14.0, yaw_period=math.pi),
    cylinder_block("cylinder", 60.0, 29.0, DENSITY * math.pi * 14.5 ** 2 * 60.0, yaw_period=math.pi),
)}

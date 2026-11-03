"""The planner's prompt: a short system text and the message with the world state and the task."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from .relations import DEFAULT_LIMITS, Limits
from .schema import Step
from .world import World

#: Logged with every plan so answers can be traced to the prompt that produced them.
PROMPT_VERSION = "blocks-plan-2"


def system_text(limits: Limits = DEFAULT_LIMITS) -> str:
    """What the model needs to know about the arm, the frame and the relations."""
    return f"""\
You plan moves for a robot arm that builds with wooden blocks on a table.

The arm moves one block at a time: it picks a block from above with a two-finger gripper and puts it \
down. It cannot hold two blocks, push blocks or turn them over; blocks keep lying as the catalog \
describes them.

Frame: the robot's base, in metres. x points away from the robot, y to the robot's left, z up from the \
table. Sides are seen from the robot: front = +x, back = -x, left = +y, right = -y. yaw_deg is the \
heading of a block's x axis (its long side), counter-clockwise from +x.

The world state lists each object under its label: the label exactly as the cameras name it, \
identical blocks numbered ("red cube", "red cube 2"). Each has its kind, size_mm (x, y, z as it lies), \
the centre x_m, y_m, z_m, yaw_deg (null when unknown), resting_on ("table", "gripper", the label below \
it, a list of labels when it bridges blocks, or null when unknown), and the source and age_s of its pose. \
reach is the table area the arm can put blocks in: distance from the base and bearing from +x. \
build_area gives, for each structure height in blocks, an x_m, y_m box where the arm can set blocks \
down at every level up to that height (null when there is none); taller structures get a smaller box.

Each step picks one block and places it by a relation. Code computes the exact pose from the cameras \
when the step runs, so give coordinates only for "at":
- on: centred on top of targets[0], turned like it.
- across: a bridge resting on targets[0] and targets[1], its long side from one to the other. Their \
tops must be level and the block long enough to rest on both.
- next_to: on the table beside targets[0] toward side, gap_m apart (null = {limits.gap_m:g} m), turned \
like it.
- at: on the table at x_m, y_m with heading yaw_deg.
Fields a relation does not use are null.

Rules:
- Use labels exactly as in the world state, numbers included; never invent blocks.
- Build each structure inside the build_area box for its height: every block of it, pillars included. \
Loose blocks may lie anywhere in reach.
- Pick only blocks with nothing on top. Place on or across blocks only when nothing is on them.
- Nothing stays on a cylinder. Stacks are at most {limits.max_stack} blocks high.
- Order the steps so every support stands before something goes on it.
- If the world state already satisfies the task, answer done = true with no steps.
- If the task is unclear or impossible with these blocks, ask a short question with no steps.
- Otherwise question is an empty string.
- Write why and question in the language of the task.
"""


SYSTEM = system_text()


@dataclass(frozen=True)
class StepRecord:
    """A step that ran, and what came of it."""

    step: Step
    ok: bool
    outcome: str

    def to_dict(self) -> dict[str, Any]:
        return {**self.step.to_dict(), "ok": self.ok, "outcome": self.outcome}


def build_prompt(task: str, world: World | Mapping[str, Any], history: Sequence[StepRecord] = (),
                 failed: StepRecord | None = None) -> str:
    """The user message: world state, steps done so far, the failed step when replanning, and the task."""
    state = world.to_json() if isinstance(world, World) else dict(world)
    parts = ["World state:", world_json(state)]
    if history:
        parts += ["", "Steps already run:", *(json.dumps(r.to_dict(), ensure_ascii=False) for r in history)]
    if failed is not None:
        parts += ["", "This step failed:", json.dumps(failed.to_dict(), ensure_ascii=False),
                  "Plan what is left of the task from the world state above."]
    parts += ["", "Task:", task.strip()]
    return "\n".join(parts) + "\n"


def world_json(state: Mapping[str, Any]) -> str:
    """Valid JSON with one object per line, short enough for a prompt."""
    lines = []
    for key, value in state.items():
        name = json.dumps(key)
        if isinstance(value, list) and value:
            items = ",\n  ".join(json.dumps(v, ensure_ascii=False) for v in value)
            lines.append(f"{name}: [\n  {items}\n]")
        elif isinstance(value, dict) and value and all(isinstance(v, dict) for v in value.values()):
            items = ",\n  ".join(f"{json.dumps(k, ensure_ascii=False)}: {json.dumps(v, ensure_ascii=False)}"
                                  for k, v in value.items())
            lines.append(f"{name}: {{\n  {items}\n}}")
        else:
            lines.append(f"{name}: {json.dumps(value, ensure_ascii=False)}")
    return "{\n" + ",\n".join(lines) + "\n}"

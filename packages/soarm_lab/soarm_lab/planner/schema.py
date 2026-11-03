"""The plan a language model returns: its JSON Schema and the dataclasses `parse_plan` builds.

The schema suits strict structured output (OpenAI) and `claude --json-schema`: every object is closed
and lists all its properties as required; a value that may be absent is typed with "null".
Sides are seen from the robot: front = +x (away from the robot), back = -x, left = +y, right = -y.
"""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .world import label_key

ON, ACROSS, NEXT_TO, AT = RELATIONS = ("on", "across", "next_to", "at")
SIDES = ("left", "right", "front", "back")
#: Unit vector of each side in the base frame.
SIDE_VECTORS = {"left": (0.0, 1.0), "right": (0.0, -1.0), "front": (1.0, 0.0), "back": (-1.0, 0.0)}
#: How many target labels each relation takes.
TARGET_COUNT = {ON: 1, ACROSS: 2, NEXT_TO: 1, AT: 0}

_NUMBER_OR_NULL = ["number", "null"]

PLACE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["relation", "targets", "side", "gap_m", "x_m", "y_m", "yaw_deg"],
    "properties": {
        "relation": {
            "type": "string",
            "enum": list(RELATIONS),
            "description": "on: centred on top of targets[0], turned like it. "
                           "across: a bridge resting on targets[0] and targets[1], its long side from one "
                           "to the other. next_to: on the table beside targets[0] toward side, gap_m apart, "
                           "turned like it. at: on the table at x_m, y_m, heading yaw_deg.",
        },
        "targets": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Labels exactly as in the world state, numbers included: one for on and "
                           "next_to, two for across, none for at.",
        },
        "side": {
            "type": ["string", "null"],
            "enum": [*SIDES, None],
            "description": "next_to only, else null. Seen from the robot: front = +x (away from the robot), "
                           "back = -x (toward it), left = +y, right = -y.",
        },
        "gap_m": {
            "type": _NUMBER_OR_NULL,
            "description": "next_to only, else null: free space between the two blocks [m]; null = default.",
        },
        "x_m": {"type": _NUMBER_OR_NULL, "description": "at only, else null: x of the block's centre [m]."},
        "y_m": {"type": _NUMBER_OR_NULL, "description": "at only, else null: y of the block's centre [m]."},
        "yaw_deg": {
            "type": _NUMBER_OR_NULL,
            "description": "at only, else null: heading of the block's long side, degrees counter-clockwise "
                           "from +x; null = 0.",
        },
    },
}

STEP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["pick", "place", "why"],
    "properties": {
        "pick": {"type": "string",
                 "description": "Label of the block to move, exactly as in the world state, numbers "
                                "included."},
        "place": PLACE_SCHEMA,
        "why": {"type": "string", "description": "One short sentence: what this step is for."},
    },
}

PLAN_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["steps", "done", "question"],
    "properties": {
        "steps": {
            "type": "array",
            "items": STEP_SCHEMA,
            "description": "Moves in order, one block each. Empty when done or asking a question.",
        },
        "done": {
            "type": "boolean",
            "description": "True when the world state already satisfies the task; steps is then empty.",
        },
        "question": {
            "type": "string",
            "description": "A short question for the operator when the task is unclear or impossible; steps "
                           "is then empty. Otherwise an empty string.",
        },
    },
}


def plan_schema() -> dict[str, Any]:
    """A copy of the plan schema, safe to change."""
    return copy.deepcopy(PLAN_SCHEMA)


def schema_json() -> str:
    """The plan schema as one line of JSON, e.g. for `claude --json-schema`."""
    return json.dumps(PLAN_SCHEMA, separators=(",", ":"))


class PlanError(ValueError):
    """The model's answer does not fit the plan schema or a relation's arguments."""


@dataclass(frozen=True)
class Place:
    """Where the picked block goes, as a relation to other blocks or a table position."""

    relation: str
    targets: tuple[str, ...] = ()
    side: str | None = None
    gap_m: float | None = None
    x_m: float | None = None
    y_m: float | None = None
    yaw_deg: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {"relation": self.relation, "targets": list(self.targets), "side": self.side,
                "gap_m": self.gap_m, "x_m": self.x_m, "y_m": self.y_m, "yaw_deg": self.yaw_deg}


@dataclass(frozen=True)
class Step:
    """Pick one block and place it."""

    pick: str
    place: Place
    why: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"pick": self.pick, "place": self.place.to_dict(), "why": self.why}


@dataclass(frozen=True)
class Plan:
    steps: tuple[Step, ...]
    done: bool
    question: str

    def to_dict(self) -> dict[str, Any]:
        return {"steps": [s.to_dict() for s in self.steps], "done": self.done, "question": self.question}


def place_problem(place: Place) -> str | None:
    """Why the relation's arguments do not fit it, or None when they do."""
    if place.relation not in RELATIONS:
        return f"unknown relation {place.relation!r}; use one of {', '.join(RELATIONS)}"
    count = TARGET_COUNT[place.relation]
    if len(place.targets) != count:
        return f"{place.relation} takes {count} target(s), got {len(place.targets)}"
    if len(set(place.targets)) != len(place.targets):
        return f"{place.relation} needs different targets"
    if place.relation == NEXT_TO:
        if place.side not in SIDES:
            return f"next_to needs side, one of {', '.join(SIDES)}"
        if place.gap_m is not None and not place.gap_m >= 0.0:
            return "next_to needs a gap_m of zero or more"
    if place.relation == AT and (place.x_m is None or place.y_m is None):
        return "at needs x_m and y_m"
    return None


def parse_plan(data: Mapping[str, Any]) -> Plan:
    """Build a plan from the model's JSON answer, strictly; raise `PlanError` on any mismatch.

    Labels are normalised as the twin keys objects. Fields a relation does not use become None.
    """
    top = _object(data, PLAN_SCHEMA, "plan")
    steps = top["steps"]
    if not isinstance(steps, list):
        raise PlanError("plan.steps must be an array")
    done, question = top["done"], top["question"]
    if not isinstance(done, bool):
        raise PlanError("plan.done must be true or false")
    if not isinstance(question, str):
        raise PlanError("plan.question must be a string")
    parsed = tuple(_step(item, f"steps[{i}]") for i, item in enumerate(steps))
    if done and parsed:
        raise PlanError("plan.done is true but steps are given")
    if question.strip() and parsed:
        raise PlanError("plan asks a question and gives steps; do one of them")
    return Plan(parsed, done, question.strip())


def _step(data: Any, where: str) -> Step:
    obj = _object(data, STEP_SCHEMA, where)
    pick, why = obj["pick"], obj["why"]
    if not isinstance(pick, str) or not label_key(pick):
        raise PlanError(f"{where}.pick must be a non-empty label")
    if not isinstance(why, str):
        raise PlanError(f"{where}.why must be a string")
    return Step(label_key(pick), _place(obj["place"], f"{where}.place"), why.strip())


def _place(data: Any, where: str) -> Place:
    obj = _object(data, PLACE_SCHEMA, where)
    relation, targets, side = obj["relation"], obj["targets"], obj["side"]
    if relation not in RELATIONS:
        raise PlanError(f"{where}.relation must be one of {', '.join(RELATIONS)}")
    if not isinstance(targets, list) or not all(isinstance(t, str) and label_key(t) for t in targets):
        raise PlanError(f"{where}.targets must be an array of labels")
    if side is not None and side not in SIDES:
        raise PlanError(f"{where}.side must be one of {', '.join(SIDES)} or null")
    numbers = {key: _number(obj[key], f"{where}.{key}") for key in ("gap_m", "x_m", "y_m", "yaw_deg")}
    if relation != NEXT_TO:
        side, numbers["gap_m"] = None, None
    if relation != AT:
        numbers["x_m"] = numbers["y_m"] = numbers["yaw_deg"] = None
    place = Place(relation, tuple(label_key(t) for t in targets), side, **numbers)
    problem = place_problem(place)
    if problem:
        raise PlanError(f"{where}: {problem}")
    return place


def _object(data: Any, schema: Mapping[str, Any], where: str) -> Mapping[str, Any]:
    if not isinstance(data, Mapping):
        raise PlanError(f"{where} must be an object")
    keys, wanted = set(data), set(schema["required"])
    if keys != wanted:
        missing, extra = sorted(wanted - keys), sorted(map(str, keys - wanted))
        parts = ([f"missing {', '.join(missing)}"] if missing else []) + (
            [f"unknown {', '.join(extra)}"] if extra else [])
        raise PlanError(f"{where}: " + "; ".join(parts))
    return data


def _number(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise PlanError(f"{where} must be a finite number or null")
    return float(value)

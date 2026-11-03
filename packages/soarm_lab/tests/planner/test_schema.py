"""The plan schema is strict enough for structured output, and `parse_plan` checks answers against it."""

from __future__ import annotations

import copy
import json
from collections.abc import Iterator
from typing import Any

import pytest

from soarm_lab.planner.schema import (
    PLAN_SCHEMA,
    Place,
    PlanError,
    Step,
    parse_plan,
    place_problem,
    plan_schema,
    schema_json,
)

#: Keywords both OpenAI strict mode and `claude --json-schema` accept.
KEYWORDS = {"type", "properties", "required", "additionalProperties", "items", "enum", "description"}


def walk(node: Any, path: str = "$") -> Iterator[tuple[str, dict[str, Any]]]:
    if isinstance(node, dict):
        yield path, node
        for key, value in node.get("properties", {}).items():
            yield from walk(value, f"{path}.{key}")
        if "items" in node:
            yield from walk(node["items"], f"{path}[]")


def test_every_object_is_closed_and_fully_required() -> None:
    objects = 0
    for path, node in walk(PLAN_SCHEMA):
        assert set(node) <= KEYWORDS, path
        types = node["type"] if isinstance(node["type"], list) else [node["type"]]
        if "object" in types:
            objects += 1
            assert node["additionalProperties"] is False, path
            assert sorted(node["required"]) == sorted(node["properties"]), path
        if "enum" in node:
            allowed = {"string": str, "null": type(None)}
            assert all(isinstance(v, tuple(allowed[t] for t in types)) for v in node["enum"]), path
    assert objects == 3                                   # plan, step, place
    assert json.loads(schema_json()) == PLAN_SCHEMA
    assert plan_schema() == PLAN_SCHEMA and plan_schema() is not PLAN_SCHEMA


def test_the_side_description_defines_the_frame() -> None:
    side = PLAN_SCHEMA["properties"]["steps"]["items"]["properties"]["place"]["properties"]["side"]
    assert "front = +x" in side["description"] and "left = +y" in side["description"]


def test_labels_keep_their_numbers() -> None:
    step = PLAN_SCHEMA["properties"]["steps"]["items"]["properties"]
    assert "numbers included" in step["pick"]["description"]
    assert "numbers included" in step["place"]["properties"]["targets"]["description"]
    plan = parse_plan({"steps": [{"pick": "Red Cube 3", "place": place("on", ["red  cube 2"]), "why": ""}],
                       "done": False, "question": ""})
    assert plan.steps[0].pick == "red cube 3" and plan.steps[0].place.targets == ("red cube 2",)


def place(relation: str, targets: list[str], **values: Any) -> dict[str, Any]:
    out = {"relation": relation, "targets": targets, "side": None, "gap_m": None, "x_m": None, "y_m": None,
           "yaw_deg": None}
    out.update(values)
    return out


BRIDGE = {
    "steps": [
        {"pick": "Green  Cube", "place": place("next_to", ["red cube"], side="right", gap_m=0.03),
         "why": "first pillar"},
        {"pick": "yellow cuboid", "place": place("across", ["red cube", "green cube"]), "why": "the deck"},
        {"pick": "blue cube", "place": place("on", ["yellow cuboid"], side="left", gap_m=0.1), "why": "top"},
        {"pick": "white cube", "place": place("at", [], x_m=0.2, y_m=0.1, yaw_deg=30), "why": "aside"},
    ],
    "done": False,
    "question": "",
}


def test_a_bridge_plan_parses() -> None:
    plan = parse_plan(BRIDGE)
    assert [s.pick for s in plan.steps] == ["green cube", "yellow cuboid", "blue cube", "white cube"]
    assert plan.steps[0].place == Place("next_to", ("red cube",), "right", 0.03)
    assert plan.steps[1].place == Place("across", ("red cube", "green cube"))
    assert plan.steps[2].place == Place("on", ("yellow cuboid",))    # unused fields are dropped
    assert plan.steps[3].place == Place("at", (), x_m=0.2, y_m=0.1, yaw_deg=30.0)
    assert not plan.done and plan.question == ""
    assert parse_plan(plan.to_dict()) == plan
    assert parse_plan(json.loads(json.dumps(plan.to_dict()))) == plan


def test_done_and_question_answers_parse() -> None:
    assert parse_plan({"steps": [], "done": True, "question": ""}).done
    assert parse_plan({"steps": [], "done": False, "question": " Which cube? "}).question == "Which cube?"


def broken(change) -> dict[str, Any]:
    data = copy.deepcopy(BRIDGE)
    change(data)
    return data


@pytest.mark.parametrize(("data", "message"), [
    ([], "plan must be an object"),
    (broken(lambda d: d.pop("done")), "missing done"),
    (broken(lambda d: d.update(extra=1)), "unknown extra"),
    (broken(lambda d: d.update(steps={})), "steps must be an array"),
    (broken(lambda d: d.update(done="no")), "done must be true or false"),
    (broken(lambda d: d.update(question=None)), "question must be a string"),
    (broken(lambda d: d.update(done=True)), "done is true but steps"),
    (broken(lambda d: d.update(question="which?")), "asks a question and gives steps"),
    (broken(lambda d: d["steps"][0].update(pick="  ")), "pick must be a non-empty label"),
    (broken(lambda d: d["steps"][0].update(why=3)), "why must be a string"),
    (broken(lambda d: d["steps"][0]["place"].pop("yaw_deg")), "missing yaw_deg"),
    (broken(lambda d: d["steps"][0]["place"].update(relation="under")), "relation must be one of"),
    (broken(lambda d: d["steps"][0]["place"].update(targets="red cube")), "targets must be an array"),
    (broken(lambda d: d["steps"][0]["place"].update(side="up")), "side must be one of"),
    (broken(lambda d: d["steps"][0]["place"].update(side=None)), "next_to needs side"),
    (broken(lambda d: d["steps"][0]["place"].update(gap_m=-0.01)), "gap_m of zero or more"),
    (broken(lambda d: d["steps"][0]["place"].update(gap_m=True)), "gap_m must be a finite number"),
    (broken(lambda d: d["steps"][0]["place"].update(gap_m=float("inf"))), "gap_m must be a finite number"),
    (broken(lambda d: d["steps"][1]["place"].update(targets=["red cube"])), "across takes 2 target"),
    (broken(lambda d: d["steps"][1]["place"].update(targets=["red cube", "Red Cube"])), "different targets"),
    (broken(lambda d: d["steps"][2]["place"].update(targets=[])), "on takes 1 target"),
    (broken(lambda d: d["steps"][3]["place"].update(y_m=None)), "at needs x_m and y_m"),
    (broken(lambda d: d["steps"][3]["place"].update(targets=["red cube"])), "at takes 0 target"),
])
def test_a_broken_answer_is_refused(data: Any, message: str) -> None:
    with pytest.raises(PlanError, match=message):
        parse_plan(data)


def test_place_problem_checks_hand_built_steps() -> None:
    assert place_problem(Place("on", ("red cube",))) is None
    assert "unknown relation" in str(place_problem(Place("under", ("red cube",))))
    assert "next_to needs side" in str(place_problem(Step("a", Place("next_to", ("b",))).place))

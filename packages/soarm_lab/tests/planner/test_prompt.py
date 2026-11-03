"""The system text and the message with world state, history, the failed step and the task."""

from __future__ import annotations

import json

import numpy as np
from machinekind_twin_api.strategy import ObjectPose

from soarm_lab.planner.prompt import PROMPT_VERSION, SYSTEM, StepRecord, build_prompt, system_text, world_json
from soarm_lab.planner.relations import Limits
from soarm_lab.planner.schema import RELATIONS, Place, Step
from soarm_lab.planner.world import world_state

WORLD = world_state({"red cube": ObjectPose(np.array([0.2, 0.03, 0.0145]), np.eye(3), "kamery", 0.1),
                     "red cube 2": ObjectPose(np.array([0.17, 0.1, 0.0145]), np.eye(3), "kamery", 0.1),
                     "yellow cuboid": ObjectPose(np.array([0.2, -0.1, 0.007]), np.eye(3), "kamery", 0.1)})
STEP = Step("green cube", Place("next_to", ("red cube",), "right", 0.03), "first pillar")


def test_the_system_text_names_relations_frame_and_answers() -> None:
    assert PROMPT_VERSION
    for word in (*RELATIONS, "front = +x", "left = +y", "labels exactly", "numbers included", '"red cube 2"',
                 "build_area", "inside the build_area box", "done = true", "question"):
        assert word in SYSTEM, word
    assert "null = 0.01 m" in SYSTEM and "at most 3 blocks high" in SYSTEM
    assert "at most 4 blocks high" in system_text(Limits(max_stack=4))


def test_the_world_json_is_valid_with_one_object_per_line() -> None:
    state = WORLD.to_json()
    text = world_json(state)
    assert json.loads(text) == state
    lines = text.splitlines()
    for label in ("red cube", "red cube 2", "yellow cuboid"):
        assert sum(line.startswith(f'  "{label}": {{"kind"') for line in lines) == 1, label
    assert sum(line.startswith('  {"blocks_high"') for line in lines) == 3
    assert json.loads(world_json({"objects": {}, "reach": {}})) == {"objects": {}, "reach": {}}


def test_a_first_prompt_has_the_world_and_the_task_last() -> None:
    text = build_prompt("  Build a bridge.  ", WORLD)
    assert text.startswith("World state:\n{") and text.endswith("Task:\nBuild a bridge.\n")
    assert "Steps already run" not in text and "failed" not in text
    assert build_prompt("x", WORLD.to_json()) == build_prompt("x", WORLD)


def test_a_replan_prompt_has_the_history_and_the_failure() -> None:
    done = StepRecord(STEP, True, "'green cube' rests on 'table' where planned")
    failed = StepRecord(Step("yellow cuboid", Place("across", ("red cube", "green cube"))), False,
                        "'yellow cuboid' does not rest on 'green cube'")
    text = build_prompt("Build a bridge.", WORLD, history=[done], failed=failed)
    lines = text.splitlines()
    record = json.loads(lines[lines.index("Steps already run:") + 1])
    assert record["pick"] == "green cube" and record["ok"] is True and record["place"]["side"] == "right"
    record = json.loads(lines[lines.index("This step failed:") + 1])
    assert record["ok"] is False and record["outcome"] == "'yellow cuboid' does not rest on 'green cube'"
    assert text.index("This step failed:") < text.index("Task:")

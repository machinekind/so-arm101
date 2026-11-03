"""Plan block building with a language model: CLI runner, world state, plan schema, relations and prompt.

`cli` asks the local claude or codex CLI for a JSON answer. `world` turns `Observation.objects` into
the state the model reads, with the build area, and suggests free spots (`best_spot`). `placing`
says from the arm model where a block can be set down at a given height. `schema` is the JSON Schema of
the plan the model returns and `parse_plan`. `relations` turns a step's relation into the pose where the
held block goes and checks afterwards that it holds. `prompt` builds the model's messages.
"""

from .placing import HandDownReach, PlaceCheck, arm_reach
from .prompt import PROMPT_VERSION, SYSTEM, StepRecord, build_prompt, system_text
from .relations import DEFAULT_LIMITS, Limits, RelationError, TargetPose, holds, resolve
from .schema import PLAN_SCHEMA, Place, Plan, PlanError, Step, parse_plan, plan_schema, schema_json
from .world import Box, Reach, World, WorldObject, best_spot, world_state

__all__ = ["DEFAULT_LIMITS", "PLAN_SCHEMA", "PROMPT_VERSION", "SYSTEM", "Box", "HandDownReach", "Limits",
           "Place", "PlaceCheck", "Plan", "PlanError", "Reach", "RelationError", "Step", "StepRecord",
           "TargetPose", "World", "WorldObject", "arm_reach", "best_spot", "build_prompt", "holds",
           "parse_plan", "plan_schema", "resolve", "schema_json", "system_text", "world_state"]

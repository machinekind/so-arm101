"""Policy evaluation on the CPU: in plain MuJoCo, not in the engine the policy was trained in.

A policy trained in MuJoCo Warp (float32, GPU) runs here in MuJoCo on the CPU (float64), with or
without randomization. It is the cheapest transfer test: a policy that does not survive a change of
physics engine will surely not survive the move to the real arm.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..station import Station
from .env import ArmEnv
from .policy import Policy
from .randomize import Dynamics, Randomization


def evaluate(policy: Policy, episodes: int = 50, *, randomization: Randomization | None = None,
             station: Station | None = None, seed: int = 10_000, on_frame=None,
             kind: str | None = None) -> dict[str, Any]:
    """Success rates and median end distance or height of `policy` over `episodes` CPU episodes.

    The object is the one of the policy's task (`object_kind` in its `meta.json`); `kind` puts another
    of `objects.KINDS` on the table instead (a shape-conditioned policy then sees that kind's
    descriptor). The result's `kind` names the object, None for reach.

    `randomization=None` is the nominal arm: the dynamics from `station.dynamics` (identification)
    without spread, so "CPU without randomization" tells about the measured arm, not the Menagerie
    model (which is what it uses when no `station` is given). In the result, `randomized` says whether
    the dynamics were really sampled, and `centre` around what.
    """
    spec = (station or Station.load()).spec()
    if policy.meta.robot != spec.name:
        raise RuntimeError(f"polityka uczona na {policy.meta.robot}, stanowisko ma {spec.name}")
    task = policy.task
    if kind is not None:
        if task.cube is None:
            raise ValueError(f"{task.name} has no object to replace with {kind!r}")
        task = replace(task, object_kind=kind)
    rand = randomization if randomization is not None else \
        Randomization.nominal(Dynamics.from_dict(station.dynamics) if station is not None else None)
    env = ArmEnv(task, station=station, randomization=rand, render_mode="rgb_array" if on_frame else None)
    succ, final, ever, place = [], [], [], []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            hit = False
            for _ in range(task.episode_steps):
                obs, _, term, trunc, info = env.step(policy.act(obs))
                hit |= info["success"]
                if on_frame is not None:
                    on_frame(env.render())
                if term or trunc:
                    break
            succ.append(float(info["success"]))
            ever.append(float(hit))
            final.append(info["distance"] if task.name == "reach" else info["height"])
            if "dist_target" in info:
                place.append(info["dist_target"])
    finally:
        env.close()
    out = {"episodes": episodes, "success": float(np.mean(succ)), "ever_success": float(np.mean(ever)),
           "randomized": rand.randomized, "centre": rand.centre.source,
           "kind": task.object_kind if task.cube else None}
    key = "final_distance_mm" if task.name == "reach" else "final_height_mm"
    out[key] = float(np.median(final) * 1000)
    if place:                                          # pickplace: cube to target (xy) at the episode's end
        out["final_place_mm"] = float(np.median(place) * 1000)
    return out

"""The touch strategy on a live twin with the cube ONLY from the cameras.

Frames of two simulated cameras -> the twin's example cube detector (`machinekind-twin-cube`) -> the twin's
`Perception` (a cube fitted to the silhouettes with the arm masked out, then tracked near the hand) ->
`ObjectPose` in the observation -> `TouchStrategy` in the twin's `StrategyRunner` -> the safety supervisor
-> physics. The simulation's contact sensors between the cube and the jaws tell whether the hand touched it.

The chain runs in SIMULATED time, without threads, as in test_lift_from_cameras.py: the twin's loop at
50 Hz, the detector every 0.1 s and the strategy at 20 Hz on one clock.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.twin, pytest.mark.render]
pytest.importorskip("mujoco")
pytest.importorskip("machinekind_twin")

from machinekind_twin import scene as sc  # noqa: E402
from machinekind_twin.clock import VirtualClock  # noqa: E402
from machinekind_twin.detection import Perception  # noqa: E402
from machinekind_twin.detectors import load_detector  # noqa: E402
from machinekind_twin.kinematics import inverse, pose  # noqa: E402
from machinekind_twin.runtime import Twin  # noqa: E402
from machinekind_twin.strategy import StrategyRunner  # noqa: E402
from machinekind_twin.ui.watch import arm_mask  # noqa: E402
from machinekind_twin.workspace import CameraRecord, Workspace  # noqa: E402

from soarm_lab.strategies.touch import TouchStrategy  # noqa: E402

HALF = 0.015


def look(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


def cube_position(twin: Twin) -> np.ndarray:
    with twin.lock:
        s = twin.scene
        Ti = inverse(s.T_base2world)
        return Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3]


def contact(twin: Twin) -> float:
    """The most contacts between the cube and either jaw right now (each sensor counts contacts)."""
    with twin.lock:
        s = twin.scene
        return max(float(s.data.sensordata[s.model.sensor(f"cube_jaw{k}").adr[0]]) for k in range(2))


def test_touch_finds_a_cube_seen_only_by_cameras(twin_robots):
    ws = Workspace(robot="so101")
    K = [[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]]
    for name, eye in (("a", [0.55, -0.45, 0.45]), ("b", [0.5, 0.5, 0.5])):
        T = look(eye, [0.2, 0, 0]).tolist()
        ws.add_camera(CameraRecord(name, "sim", 640, 480, K=K, sim_pose=T, T_cam2base=T,
                                   calibration={"trusted": True}))
    strategy = TouchStrategy()
    twin = Twin(ws, clock=VirtualClock(t0=0.0))
    try:
        twin.configure(objects=[sc.Box("cube", (HALF, HALF, HALF), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                       mass=0.03)], grasp_sensors=["cube"])
        twin.connect("sim", threaded=False)
        start = cube_position(twin)

        clock = {"t": 0.0}
        perception = Perception(
            twin, ws, clock=lambda: clock["t"],
            frame=lambda c: (twin.render(c), clock["t"]),
            occluders=lambda times: twin.render_with(
                lambda sc_: {c: arm_mask(sc_, c, dilate=5) for c in times}))
        detector = perception.start_detector(load_detector("cube"), {"size_m": 2 * HALF, "interval_s": 0.1},
                                             threaded=False)

        runner = StrategyRunner(twin, strategy, objects=perception.objects)
        runner.start(threaded=False)
        # A clock tick every 10 ms: physics every 20 ms, vision every 100 ms, the strategy every 50 ms.
        for k in range(1, int(20.0 / 0.01) + 1):
            clock["t"] = k * 0.01
            if k % 2 == 0:
                twin.step(0.02)
            if k % 10 == 0:
                detector.step_once()
            if k % 5 == 0 and not runner.step_once(clock["t"]):
                break
        at_end = contact(twin)
        for _ in range(25):
            twin.step(0.02)
        settled = contact(twin)
        push = float(np.linalg.norm((cube_position(twin) - start)[:2]))
        perception.stop_all("koniec testu")
        where = (f"powod {strategy.status.stopped_because!r}, zrodlo {strategy.status.pose_source!r}, "
                 f"kontakt {at_end:.0f} i {settled:.0f}, przesuniecie {push * 1000:.1f} mm")

        assert runner.status.stopped_because == "zadanie wykonane", where
        assert strategy.status.pose_source == "kamery", where
        assert at_end > 0.5 and settled > 0.5 and push < 0.01, where
    finally:
        twin.close()

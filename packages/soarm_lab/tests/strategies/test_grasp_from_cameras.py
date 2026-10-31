"""The grasp strategy on a live twin with the block ONLY from the cameras, pulled out of the jaws mid-carry.

Frames of two simulated cameras -> the twin's example cube detector (`machinekind-twin-cube`, every 0.3 s)
-> the twin's `Perception` (a box fitted to the silhouettes with the arm masked out, then tracked through
the gripper's hold) -> `ObjectPose` in the observation -> `GraspStrategy` in the twin's `StrategyRunner` ->
the safety supervisor -> physics, in simulated time and without threads, as in test_touch_from_cameras.py.

While the strategy carries the cube, the test moves it out of the jaws onto the table in front of the tips
(`teleport`), as a person pulling it out would, right after a frame. The jaw closes on nothing, the twin's
tracker lowers its last pose under the hand ("ostatnie widziane (upuszczona)"), the next frames show the
cube where it lies ("kamery"), and the strategy grasps it again there. At a 0.1 s interval the twin takes
the next frame's pose on the tick the jaw stops squeezing and the lowered pose never shows; the detector's
own default is 0.33 s. The simulation's truth judges the run: the cube lifted, both jaws on it.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.twin, pytest.mark.render]
mujoco = pytest.importorskip("mujoco")
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

from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.rl.objects import COLOURS, object_spec  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, GraspStrategy  # noqa: E402

KIN = RobotKinematics(load_robot())
#: Where the cube lies at the start, and where the test puts it mid-carry: 7 cm from where it was
#: picked, in front of the tips, where neither camera looks through the arm [m, base frame].
START_XY = (0.2, 0.0)
PULLED_XY = (0.26, 0.04)
#: The block's contact as test_grasp_in_twin.py makes it: as stiff as the 5 ms step allows.
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
#: The detector's interval [s], and in the test's 10 ms clock ticks.
VISION_S = 0.3
VISION_TICKS = 30


def look(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


def truth(twin: Twin, name: str) -> np.ndarray:
    """The block's position in the base frame, as the simulation has it."""
    with twin.lock:
        s = twin.scene
        Ti = inverse(s.T_base2world)
        return Ti[:3, :3] @ s.data.xpos[s.model.body(name).id] + Ti[:3, 3]


def contact(twin: Twin, name: str) -> list[float]:
    """The contacts between the block and each jaw right now (fixed, moving)."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def teleport(twin: Twin, name: str, xy: tuple[float, float], z: float) -> None:
    """Puts the block `name` upright and at rest at `xy` and height `z` [m, base frame]."""
    with twin.lock:
        s = twin.scene
        j = s.model.body(name).jntadr[0]
        adr, dof = s.model.jnt_qposadr[j], s.model.jnt_dofadr[j]
        p = s.T_base2world @ np.array([xy[0], xy[1], z, 1.0])
        s.data.qpos[adr:adr + 3] = p[:3]
        s.data.qpos[adr + 3:adr + 7] = (1.0, 0.0, 0.0, 0.0)
        s.data.qvel[dof:dof + 6] = 0.0
        mujoco.mj_forward(s.model, s.data)


def test_a_cube_pulled_out_mid_carry_is_grasped_again_where_it_lies(twin_robots):
    ws = Workspace(robot="so101")
    K = [[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]]
    for name, eye in (("a", [0.55, -0.45, 0.45]), ("b", [0.5, 0.5, 0.5])):
        T = look(eye, [0.2, 0, 0]).tolist()
        ws.add_camera(CameraRecord(name, "sim", 640, 480, K=K, sim_pose=T, T_cam2base=T,
                                   calibration={"trusted": True}))
    spec = object_spec("cube")
    strategy = GraspStrategy(GraspSettings(carry_s=3.0, wait_s=10.0))
    twin = Twin(ws, clock=VirtualClock(t0=0.0))
    try:
        cube = sc.Box("cube", spec.half_size, START_XY, rgba=COLOURS["cube"], mass=spec.mass)
        twin.configure(objects=[cube], grasp_sensors=["cube"])
        twin.connect("sim", threaded=False)
        with twin.lock:
            model = twin.scene.model
            g = model.geom("cube").id
            model.geom_solref[g] = STIFF_SOLREF
            model.geom_solimp[g][:2] = STIFF_SOLIMP
            model.geom_priority[g] = 1

        clock = {"t": 0.0}
        perception = Perception(
            twin, ws, clock=lambda: clock["t"],
            frame=lambda c: (twin.render(c), clock["t"]),
            occluders=lambda times: twin.render_with(
                lambda sc_: {c: arm_mask(sc_, c, dilate=5) for c in times}))
        detector = perception.start_detector(load_detector("cube"),
                                             {"size_m": 2 * spec.half_size[0], "interval_s": VISION_S},
                                             threaded=False)
        #: (time, the strategy's phase before the tick, the cube's source) on every tick.
        sources: list[tuple[float, str, str]] = []

        def objects():
            got = perception.objects()
            cube = got.get("cube")
            sources.append((clock["t"], strategy.status.phase, cube.source if cube is not None else ""))
            return got

        runner = StrategyRunner(twin, strategy, objects=objects)
        gripper = KIN.spec.gripper
        assert gripper is not None
        g_index = KIN.spec.joints.index(gripper)
        opened: list[float] = []

        def drive(seconds: float, until=None) -> None:
            # A clock tick every 10 ms: physics every 20 ms, vision every 300 ms, the strategy every 50 ms.
            start = round(clock["t"] / 0.01)
            for k in range(start + 1, start + round(seconds / 0.01) + 1):
                clock["t"] = k * 0.01
                if k % 2 == 0:
                    twin.step(0.02)
                    if strategy.started and strategy.status.phase == "recover":
                        opened.append(float(KIN.to_q(dict(twin.status.measured))[g_index]))
                if k % VISION_TICKS == 0:
                    detector.step_once()
                if k % 5 == 0 and not runner.step_once(clock["t"]):
                    break
                if until is not None and until():
                    break

        runner.start(threaded=False)
        drive(60.0, until=lambda: strategy.status.phase == "carry")
        assert strategy.status.phase == "carry", (strategy.status, strategy.grasp)
        # Pulled right after the cameras' frames: their next frames come 0.3 s later, so the twin first has
        # only the hand to go by.
        while round(clock["t"] / 0.01) % VISION_TICKS:
            drive(0.01)
        pulled_t = clock["t"]
        teleport(twin, "cube", PULLED_XY, spec.half_size[2])
        drive(90.0)
        for _ in range(25):
            twin.step(0.02)
        perception.stop_all("koniec testu")
        where = (f"{strategy.status.stopped_because!r}, {strategy.grasp}, "
                 f"zrodla {[s for s in dict.fromkeys(s for _, _, s in sources)]}")

        assert runner.status.stopped_because == "zadanie wykonane", where
        assert strategy.grasp.drops == 1 and strategy.grasp.recoveries == 1, where
        assert strategy.status.attempt == 2, where
        assert strategy.grasp.block_xy is not None
        assert np.linalg.norm(np.subtract(strategy.grasp.block_xy, PULLED_XY)) < 0.01, where
        # The twin says "w dloni" while carrying, lowers the cube under the hand when the jaw closes on
        # nothing, sees it on the table before the regrasp, and says "w dloni" again after it.
        before = [s for t, _, s in sources if t <= pulled_t]
        after = [(s, phase) for t, phase, s in sources if t > pulled_t]
        names = [s for s, _ in after]
        regrasp = next(k for k, (_, phase) in enumerate(after) if phase == "approach")
        assert before[-1] == "w dloni", where
        dropped = names.index("ostatnie widziane (upuszczona)")
        assert "w dloni" not in names[dropped:regrasp], where
        assert "kamery" in names[dropped:regrasp] and names[-1] == "w dloni", where
        # The recovery opened the jaw past the tracker's `open` gate.
        assert opened and max(opened) > KIN.spec.jaw.open, where
        assert truth(twin, "cube")[2] > 0.04
        assert all(c > 0.5 for c in contact(twin, "cube"))
        assert twin.owner is None
    finally:
        twin.close()

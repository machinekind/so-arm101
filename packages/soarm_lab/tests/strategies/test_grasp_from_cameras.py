"""Test recovery after pulling a carried cube from the jaws in the twin.

Two simulated cameras feed the cube detector and tracker. The strategy runs through the
twin's runner and safety supervisor on a virtual clock. During carry, the test moves the
cube onto the table after a frame. The tracker reports a drop guess, then a camera sighting,
and the strategy grasps the cube again. Simulation height and jaw contacts verify the hold.
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

from soarm_lab.blocks import block  # noqa: E402
from soarm_lab.kinematics import RobotKinematics  # noqa: E402
from soarm_lab.robot import load_robot  # noqa: E402
from soarm_lab.strategies.grasp import GraspSettings, GraspStrategy  # noqa: E402

KIN = RobotKinematics(load_robot())
#: Initial and mid-carry landing positions visible to both cameras [m, base frame].
START_XY = (0.2, 0.0)
PULLED_XY = (0.26, 0.04)
#: Stiff block contact shared with the twin grasp tests.
STIFF_SOLREF = (0.01, 1.0)
STIFF_SOLIMP = (0.95, 0.99)
#: Detector interval [s] and number of virtual-clock ticks per detection.
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
    """Return the simulated block position [m, base frame]."""
    with twin.lock:
        s = twin.scene
        Ti = inverse(s.T_base2world)
        return Ti[:3, :3] @ s.data.xpos[s.model.body(name).id] + Ti[:3, 3]


def contact(twin: Twin, name: str) -> list[float]:
    """Return contacts with the fixed and moving jaws."""
    with twin.lock:
        s = twin.scene
        return [float(s.data.sensordata[s.model.sensor(f"{name}_jaw{k}").adr[0]]) for k in range(2)]


def teleport(twin: Twin, name: str, xy: tuple[float, float], z: float) -> None:
    """Place the block upright and at rest at `xy` and height `z` [m, base frame]."""
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
    spec = block("cube")
    strategy = GraspStrategy(GraspSettings(carry_s=3.0, wait_s=10.0))
    twin = Twin(ws, clock=VirtualClock(t0=0.0))
    try:
        cube = sc.Box("cube", spec.half_size, START_XY, rgba=(0.85, 0.25, 0.2, 1.0), mass=spec.mass)
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
        #: Record time [s], phase before the tick and pose source.
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
            # Schedule physics, vision and strategy ticks on a shared virtual clock.
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
        # Drop just after a frame so the tracker must use the jaw before the next detection.
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
        # The tracker must reacquire the dropped cube before reporting another hold.
        before = [s for t, _, s in sources if t <= pulled_t]
        after = [(s, phase) for t, phase, s in sources if t > pulled_t]
        names = [s for s, _ in after]
        regrasp = next(k for k, (_, phase) in enumerate(after) if phase == "approach")
        assert before[-1] == "w dloni", where
        dropped = names.index("ostatnie widziane (upuszczona)")
        assert "w dloni" not in names[dropped:regrasp], where
        assert "kamery" in names[dropped:regrasp] and names[-1] == "w dloni", where
        # Open enough for the tracker to accept nearby detections.
        assert opened and max(opened) > KIN.spec.jaw.open, where
        assert truth(twin, "cube")[2] > 0.04
        assert all(c > 0.5 for c in contact(twin, "cube"))
        assert twin.owner is None
    finally:
        twin.close()

"""Caly lancuch sim-2-real naraz: polityka lift na zywym blizniaku, kostka WYLACZNIE z kamer.

Polityka jako strategia laboratorium: kadry symulowanych kamer -> dopasowanie sylwetki (z maska
ramienia) -> sledzenie kostki w dloni (percepcja blizniaka) -> `ObjectPose` w obserwacji ->
`PolicyStrategy` w `StrategyRunner` blizniaka -> nadzor bezpieczenstwa -> fizyka. Sprawdzamy na
prawdzie z fizyki, czy kostka zostala podniesiona. Test jedzie na bazowej `lift-v3`.

Caly lancuch idzie w czasie SYMULOWANYM, bez watkow: petla blizniaka 50 Hz (`Twin.step`), wizja
co 0,1 s, strategia 20 Hz (`StrategyRunner.step_once`) na jednym zegarze.

Dostawca kostki bierze rozkaz chwytaka z `status.command` blizniaka (jak panel, `_vision_cube`),
a nie z `runner.q_cmd`: `StrategyRunner` pyta o obiekty juz przed `start` strategii.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.twin, pytest.mark.render]
mujoco = pytest.importorskip("mujoco")
pytest.importorskip("lerobot_mp")

from lerobot_mp.twin import scene as sc  # noqa: E402
from lerobot_mp.twin.clock import VirtualClock  # noqa: E402
from lerobot_mp.twin.kinematics import RobotKinematics, inverse, pose  # noqa: E402
from lerobot_mp.twin.perception import CubeDetector, CubeTracker, TableMapper  # noqa: E402
from lerobot_mp.twin.runtime import Twin  # noqa: E402
from lerobot_mp.twin.strategy import ObjectPose, StrategyRunner  # noqa: E402
from lerobot_mp.twin.ui.watch import arm_mask  # noqa: E402
from lerobot_mp.twin.workspace import CameraRecord, Workspace  # noqa: E402

from soarm_lab.rl import task as tk  # noqa: E402
from soarm_lab.strategies.policy import PolicySettings, PolicyStrategy  # noqa: E402


def look(eye, target):
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(z, [0, 0, 1.0])
    x /= np.linalg.norm(x)
    return pose(np.column_stack([x, np.cross(z, x), z]), eye)


def test_lift_policy_lifts_a_cube_seen_only_by_cameras(twin_robots):
    ws = Workspace()
    K = [[560.0, 0, 322.0], [0, 560.0, 236.0], [0, 0, 1]]
    for name, eye in (("a", [0.55, -0.45, 0.45]), ("b", [0.5, 0.5, 0.5])):
        T = look(eye, [0.2, 0, 0]).tolist()
        ws.add_camera(CameraRecord(name, "sim", 640, 480, K=K, sim_pose=T, T_cam2base=T,
                                   calibration={"trusted": True}))
    strategy = PolicyStrategy(PolicySettings(policy="lift-v3"))
    task = strategy.task
    h = task.cube_half
    twin = Twin(ws, clock=VirtualClock(t0=0.0))
    try:
        twin.configure(objects=[sc.Box("cube", (h, h, h), (0.2, 0.0), rgba=(0.85, 0.25, 0.2, 1.0),
                                       mass=task.cube_mass)], grasp_sensors=["cube"])
        twin.connect("sim", threaded=False)
        mapper = TableMapper.from_workspace(ws)
        det = CubeDetector()
        kin = RobotKinematics(ws.spec())
        grip = kin.spec.gripper
        g = kin.spec.joints.index(grip)
        pos, quat = tk.sample_cubes(task, np.random.default_rng(5), 1)
        with twin.lock:
            s = twin.scene
            m, d = s.model, s.data
            b = m.body("cube").id
            a = m.jnt_qposadr[m.body_jntadr[b]]
            T = s.T_base2world
            qb, qw = np.zeros(4), np.zeros(4)
            mujoco.mju_mat2Quat(qb, T[:3, :3].ravel())
            mujoco.mju_mulQuat(qw, qb, quat[0])
            d.qpos[a:a + 3] = T[:3, :3] @ pos[0] + T[:3, 3]
            d.qpos[a + 3:a + 7] = qw
            mujoco.mj_forward(m, d)

        tracker = CubeTracker()
        last = {"det": None}
        clock = {"t": 0.0}

        def vision():
            frames = {c: twin.render(c) for c in ("a", "b")}
            occ = twin.render_with(lambda sc_: {c: arm_mask(sc_, c, dilate=5) for c in ("a", "b")})
            last["det"] = det.detect_frames(frames, mapper, occ)

        def objects():
            joints = dict(twin.status.measured) or twin.joints()
            q = kin.to_q(joints)
            sent = (twin.status.command or {}).get(grip)
            cmd = kin.to_q({**joints, grip: float(sent)})[g] if sent is not None else q[g]
            closed = strategy.limits.lo[g] if strategy.started else kin.to_q({grip: 0.0})[g]
            got = tracker.update(last["det"], kin.tcp(joints), q[g], cmd, closed, clock["t"])
            return {} if got is None else {"cube": ObjectPose(got[0], got[1], tracker.source)}

        runner = StrategyRunner(twin, strategy, objects=objects)
        runner.start(threaded=False)
        # Zegar co 10 ms: fizyka co 20 ms, wizja co 100 ms, strategia co 50 ms.
        for k in range(1, int(30.0 / 0.01) + 1):
            clock["t"] = k * 0.01
            if k % 2 == 0:
                twin.step(0.02)
            if k % 10 == 0:
                vision()
            if k % 5 == 0 and not runner.step_once(clock["t"]):
                break
        with twin.lock:
            s = twin.scene
            Ti = inverse(s.T_base2world)
            height = (Ti[:3, :3] @ s.data.xpos[s.model.body("cube").id] + Ti[:3, 3])[2] - h
        where = (f"takt {strategy.status.step}, powod {strategy.status.stopped_because!r}, kostka ze zrodla "
                 f"{tracker.source!r}, {height * 100:.1f} cm nad blatem")
        # Koniec po sukcesie (kostka nad blatem przez `end_on_success` taktow) albo limit epizodu.
        assert strategy.status.stopped_because in ("zadanie wykonane", "koniec epizodu"), where
        assert height > task.lift_height, where
    finally:
        twin.close()

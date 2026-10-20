"""`soarm-lab dream`: Dream-1 on the SO-101 without the twin, in the lab's simulation or on the real arm.

    soarm-lab dream models                              # the checkpoints the account may run
    soarm-lab dream sim --prompt "pick up the cube"      # the bundled station, rendered cameras
    soarm-lab dream arm --port /dev/tty.usbmodem1 --prompt "pick up the cube" \
        --camera scene=0 --camera wrist=1                # the real arm, OpenCV cameras

Both runs tick `strategies.dream.DreamStrategy` at its rate (20 Hz) on the contract types of
`machinekind_twin_api.strategy` (`RunInfo`, `Observation`, `Frame`), as the twin's `StrategyRunner` would,
and send its targets to the arm. The twin's safety supervisor is not here, so the loop adds its own guards:

* the command ramps from the measured pose toward each target at most `--speed` x `robot.yaml`'s `max_vel`
  (the strategy caps its targets the same way, and at `max_step_deg` per tick);
* targets stay within `robot.yaml`'s limits intersected with the servos' own (`FeetechArm.joint_limits`); a
  joint that starts outside is led back by the ramp, never jumped (as `soarm-panel` sends);
* a servo fault (overload, overheating, a dead link) ends the run;
* Ctrl-C, `q` + Enter, the end of `--seconds`, or the strategy's own end: the arm holds where it is measured,
  torque on, and the port closes (`FeetechArm.disconnect` keeps torque on).

The strategy holds the arm while a camera frame is missing or older than 1 s, and says so. A line typed on
stdin while it runs is a new prompt: the arm holds and a fresh episode starts with it.

The simulation is the lab's `Scene` on a station (`--station`, default `sim`) with a cube on the table. Its
scene camera is a station camera, its wrist camera the arm's own `robot/wrist_cam`. `--record out.mp4` writes
both views at the tick rate.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from machinekind_twin_api.strategy import Frame, Observation, RunInfo

from .strategies.dream import DEFAULT_CHECKPOINT, DreamSettings, DreamStrategy, open_client

logger = logging.getLogger(__name__)

#: How long the arm run waits for every camera's first frame [s].
FIRST_FRAME_TIMEOUT_S = 5.0
#: The arm's wrist camera in the SO-101's MJCF, under the scene's prefix.
SIM_WRIST_CAMERA = "robot/wrist_cam"
#: The size a simulated camera without a station record is rendered at (width, height): 16:9, as Dream reads.
SIM_RENDER_SIZE = (640, 360)
#: The camera names the runs hand the strategy: `--camera scene=...`, `--camera wrist=...`.
SCENE, WRIST = "scene", "wrist"
KEYS_HELP = """\
  type a new prompt and Enter: the arm holds, then a fresh episode starts with it
  q and Enter, or Ctrl-C: stop; the arm holds where it is, torque on"""


class Rig(Protocol):
    """What the loop drives: the simulated scene or the real arm, with their cameras."""

    def read(self) -> dict[str, float]: ...

    def send(self, command: Mapping[str, float]) -> None: ...

    def frames(self) -> dict[str, Frame]: ...

    def advance(self, seconds: float) -> None: ...

    def faults(self) -> list[str]: ...

    def hold(self) -> None: ...

    def close(self) -> None: ...


# ------------------------------------------------------------------ models


def list_models() -> int:
    """Prints the checkpoints the account may run (`DreamClient.get_models()`)."""
    client = open_client()
    models = client.get_models()
    if not models:
        print("No models for this key.")
    for model in models:
        servable = "servable" if model.servable else "not servable yet"
        cameras = ", ".join(model.cameras or []) or "-"
        print(f"  {model.id:44s} {model.owner:7s} body {model.embodiment or '-':10s} {servable:16s} "
              f"cameras {cameras}")
    print(f"Pass one as --checkpoint (default {DEFAULT_CHECKPOINT}).")
    return 0


# ------------------------------------------------------------- simulation


class SimRig:
    """The lab's scene on a station, with a cube on the table; renders a scene and a wrist camera."""

    def __init__(self, station: str, scene_camera: str | None, wrist_camera: str | None):
        import mujoco

        from .sim import Box, SceneConfig, build
        from .station import Station

        st = Station.load(station)
        self.spec = st.spec()
        self.scene = build(SceneConfig(st, objects=[Box("cube", (0.015, 0.015, 0.015), (0.22, 0.0))]))
        self.scene.set_joints(dict(st.home or self.spec.home))
        own = [view.name for view in st.cameras]
        if not own:
            raise ValueError(f"{st.folder}: the station has no camera to see the scene with")
        model = self.scene.model
        present = [model.camera(i).name for i in range(model.ncam)]
        self.cameras = {SCENE: scene_camera or own[0]}
        wrist = wrist_camera if wrist_camera is not None else (SIM_WRIST_CAMERA if SIM_WRIST_CAMERA in present
                                                               else "")
        if wrist:
            self.cameras[WRIST] = wrist
        for name in self.cameras.values():
            if name not in present:
                raise ValueError(f"no camera {name!r} in the scene; it has {', '.join(present)}")
        self._own = set(own)
        width, height = SIM_RENDER_SIZE
        self._renderer = mujoco.Renderer(model, height=height, width=width)

    def _render(self, camera: str) -> np.ndarray:
        if camera in self._own:
            return self.scene.render(camera)
        self._renderer.update_scene(self.scene.data, camera=camera)
        return self._renderer.render().copy()

    def read(self) -> dict[str, float]:
        return self.scene.joints()

    def send(self, command: Mapping[str, float]) -> None:
        self.scene.command(dict(command))

    def frames(self) -> dict[str, Frame]:
        out = {}
        for name, camera in self.cameras.items():
            t0 = time.monotonic()
            image = self._render(camera)
            out[name] = Frame(image=image, age_s=time.monotonic() - t0)
        return out

    def advance(self, seconds: float) -> None:
        self.scene.step(seconds)

    def faults(self) -> list[str]:
        return []

    def hold(self) -> None:
        self.scene.command(self.scene.joints())

    def close(self) -> None:
        self._renderer.close()
        self.scene.close()


# -------------------------------------------------------------------- arm


class CameraReader:
    """One OpenCV camera on a thread of its own: the newest frame, RGB, and when it was captured."""

    def __init__(self, name: str, source: str, size: tuple[int, int], rotate: int = 0):
        import cv2

        self.name = name
        self._cv2 = cv2
        #: A webcam mounted upside down or on its side: the frame is turned clockwise by this many degrees.
        self._rotate = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                        270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotate]
        device: int | str = int(source) if source.isdigit() else source
        self._cap = cv2.VideoCapture(device)
        if not self._cap.isOpened():
            raise RuntimeError(f"camera {name}: cannot open {source!r} (an index such as 0, or a device "
                               "path); on macOS the terminal needs camera access (System Settings > Privacy)")
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
        self._lock = threading.Lock()
        self._latest: tuple[np.ndarray, float] | None = None
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name=f"camera-{name}", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        while not self._stop.is_set():
            ok, bgr = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            rgb = self._cv2.cvtColor(bgr, self._cv2.COLOR_BGR2RGB)
            if self._rotate is not None:
                rgb = self._cv2.rotate(rgb, self._rotate)
            with self._lock:
                self._latest = (rgb, time.monotonic())

    def frame(self) -> Frame | None:
        with self._lock:
            latest = self._latest
        if latest is None:
            return None
        return Frame(image=latest[0], age_s=time.monotonic() - latest[1])

    def close(self) -> None:
        self._stop.set()
        self._thread.join(1.0)
        self._cap.release()


def detect_port() -> str:
    """The one USB-serial device, as `soarm-panel` finds it."""
    from serial.tools.list_ports import comports

    devices = sorted(str(port.device) for port in comports() if getattr(port, "vid", None) is not None)
    if len(devices) != 1:
        found = ", ".join(devices) or "none"
        raise ValueError(f"pass --port: USB-serial devices found: {found}")
    return devices[0]


class ArmRig:
    """The real SO-101 through the `feetech` backend, and OpenCV cameras."""

    def __init__(self, port: str, baudrate: int, cameras: Mapping[str, str], size: tuple[int, int],
                 rotate: Mapping[str, int] | None = None):
        from machinekind_twin_api.backend import BackendContext

        from soarm.feetech import FeetechArm

        from .robot import load_robot

        self.spec = load_robot()
        spec = self.spec
        ctx = BackendContext(robot=spec.name, joints=tuple(spec.joints), gripper=spec.gripper,
                             limits=dict(spec.limits), max_vel=dict(spec.max_vel), port=port,
                             settings={"baudrate": baudrate})
        self.readers: dict[str, CameraReader] = {}
        self.arm = FeetechArm(ctx)
        self.arm.connect()
        try:
            servo = self.arm.joint_limits()
            self.limits: dict[str, tuple[float, float]] = {}
            for joint in spec.joints:
                low, high = spec.limits[joint]
                if joint in servo:
                    low, high = max(low, servo[joint][0]), min(high, servo[joint][1])
                self.limits[joint] = (low, high)
            for name, source in cameras.items():
                self.readers[name] = CameraReader(name, source, size, (rotate or {}).get(name, 0))
            deadline = time.monotonic() + FIRST_FRAME_TIMEOUT_S
            while any(r.frame() is None for r in self.readers.values()):
                if time.monotonic() > deadline:
                    silent = [n for n, r in self.readers.items() if r.frame() is None]
                    raise RuntimeError(f"no frame from camera {', '.join(silent)} "
                                       f"in {FIRST_FRAME_TIMEOUT_S:.0f} s")
                time.sleep(0.05)
        except BaseException:
            self.close()
            raise

    def read(self) -> dict[str, float]:
        return self.arm.read_joints()

    def send(self, command: Mapping[str, float]) -> None:
        # The backend clamps to its limits: a joint standing outside its range would jump to the limit at
        # servo speed. Widened to where the command stands, the ramp leads it back (as soarm-panel does).
        widened = {}
        for joint, value in command.items():
            low, high = self.limits[joint]
            widened[joint] = (min(low, value), max(high, value))
        self.arm.set_joint_limits(widened)
        self.arm.send_joints(dict(command))

    def frames(self) -> dict[str, Frame]:
        out = {}
        for name, reader in self.readers.items():
            frame = reader.frame()
            if frame is not None:
                out[name] = frame
        return out

    def advance(self, seconds: float) -> None:
        """Reality advances on its own."""

    def faults(self) -> list[str]:
        return self.arm.faults()

    def hold(self) -> None:
        """Holds the arm where it is measured; torque stays on."""
        if self.arm.is_connected:
            self.send(self.arm.read_joints())

    def close(self) -> None:
        for reader in self.readers.values():
            reader.close()
        self.arm.disconnect()


# ------------------------------------------------------------------- loop


class _Recorder:
    """`--record`: the views side by side, one video frame per tick."""

    def __init__(self, path: Path, hz: float):
        import cv2

        self._cv2, self._path, self._hz = cv2, path, hz
        self._writer: Any = None
        self._size = (0, 0)

    def write(self, frames: Mapping[str, Frame]) -> None:
        cv2 = self._cv2
        images = [frames[name].image for name in (SCENE, WRIST) if name in frames]
        if not images:
            return
        height = images[0].shape[0]
        images = [im if im.shape[0] == height else
                  cv2.resize(im, (round(im.shape[1] * height / im.shape[0]), height)) for im in images]
        picture = cv2.cvtColor(np.ascontiguousarray(np.hstack(images)), cv2.COLOR_RGB2BGR)
        size = (picture.shape[1], picture.shape[0])
        if self._writer is None:
            self._size = size
            self._writer = cv2.VideoWriter(str(self._path), cv2.VideoWriter_fourcc(*"mp4v"), self._hz, size)
        if size != self._size:
            picture = cv2.resize(picture, self._size)
        self._writer.write(picture)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.release()
            print(f"Recorded {self._path}")


def _prompts(strategy: DreamStrategy, quit_event: threading.Event) -> None:
    """Reads stdin: `q` ends the run, any other line is a new prompt."""
    for line in sys.stdin:
        text = line.strip()
        if text.lower() in ("q", "quit", "exit"):
            quit_event.set()
            return
        if text:
            strategy.set_prompt(text)
            print(f"New prompt {text!r}: the arm holds, then a fresh episode starts.", flush=True)


def _ramp(command: dict[str, float], target: Mapping[str, float],
          cap: Mapping[str, float]) -> dict[str, float]:
    """`command` moved toward `target` by at most `cap` per joint; joints without a target stay."""
    out = dict(command)
    for joint, value in target.items():
        if joint in out and math.isfinite(value):
            step = min(max(value - out[joint], -cap[joint]), cap[joint])
            out[joint] = out[joint] + step
    return out


def run(strategy: DreamStrategy, rig: Rig, info: RunInfo, seconds: float | None = None,
        record: Path | None = None) -> int:
    """Ticks `strategy` on `rig` until it ends, `seconds` pass, `q`, or Ctrl-C; then holds the arm."""
    hz = strategy.hz
    period = 1.0 / hz
    cap = {joint: info.max_vel[joint] / hz for joint in info.joints}
    recorder = _Recorder(record, hz) if record is not None else None
    quit_event = threading.Event()
    reason = "stopped"
    started = False
    try:
        measured = rig.read()
        command = dict(measured)
        print(f"Opening {strategy.config.checkpoint} (the first time, dream's IK takes up to a minute to "
              "prepare)", flush=True)
        first = Observation(t=0.0, step=0, joints=measured, command=command, frames=rig.frames())
        strategy.start(info, first)
        started = True
        print(f"Prompt: {strategy.prompt!r}\n{KEYS_HELP}", flush=True)
        threading.Thread(target=_prompts, args=(strategy, quit_event), name="prompts", daemon=True).start()
        t0 = time.monotonic()
        deadline = t0
        chunks, note_shown, note_at = 0, "", -math.inf
        step = 0
        while True:
            now = time.monotonic()
            t = now - t0
            if quit_event.is_set():
                reason = "q"
                break
            if seconds is not None and t >= seconds:
                reason = f"{seconds:g} s passed"
                break
            faults = rig.faults()
            if faults:
                reason = "servo fault: " + "; ".join(faults)
                break
            measured = rig.read()
            frames = rig.frames()
            obs = Observation(t=t, step=step, joints=measured, command=command, frames=frames)
            action = strategy.act(obs)
            command = _ramp(command, action.target, cap)
            rig.send(command)
            rig.advance(period)
            if recorder is not None:
                recorder.write(frames)
            st = strategy.status
            if st.chunks != chunks:
                chunks = st.chunks
                print(f"[{t:6.1f} s] chunk {st.chunks}: {st.rows} rows, {st.rtt_ms:.0f} ms, "
                      f"skipped {st.skipped}, {st.mode or '-'}", flush=True)
            elif (action.note and not action.note.startswith("chunk ") and action.note != note_shown
                  and t - note_at >= 2.0):
                print(f"[{t:6.1f} s] {action.note}", flush=True)
                note_shown, note_at = action.note, t
            if action.done:
                reason = action.note or "the strategy ended the run"
                break
            step += 1
            deadline += period
            delay = deadline - time.monotonic()
            if delay < -period:
                deadline = time.monotonic()
            elif delay > 0:
                time.sleep(delay)
    except KeyboardInterrupt:
        reason = "Ctrl-C"
    finally:
        try:
            rig.hold()
        finally:
            # The recording first: the stop ends the model session over the network and may be interrupted.
            if recorder is not None:
                recorder.close()
            if started:
                strategy.stop(reason)
            rig.close()
    print(f"Stopped: {reason}. The arm holds where it is.")
    return 0


# --------------------------------------------------------------- commands


def _settings(a: argparse.Namespace, wrist: bool) -> DreamSettings:
    defaults = DreamSettings()
    return DreamSettings(prompt=a.prompt, checkpoint=a.checkpoint or defaults.checkpoint, scene_camera=SCENE,
                         wrist_camera=WRIST if wrist else "", fill_frame=a.fill,
                         max_step_deg=a.max_step_deg or defaults.max_step_deg)


def _run_info(spec: Any, limits: Mapping[str, tuple[float, float]], speed: float, simulated: bool) -> RunInfo:
    from soarm import ROBOT_DIR

    return RunInfo(robot=spec.name, joints=tuple(spec.joints), gripper=spec.gripper, limits=dict(limits),
                   max_vel={joint: speed * float(v) for joint, v in spec.max_vel.items()},
                   folder=Path(ROBOT_DIR), simulated=simulated)


def run_sim(a: argparse.Namespace) -> int:
    rig = SimRig(a.station, a.scene_camera, a.wrist_camera)
    print(f"Simulation on station {a.station!r}: scene camera {rig.cameras[SCENE]!r}, "
          f"wrist camera {rig.cameras.get(WRIST, 'none')!r}")
    strategy = DreamStrategy(_settings(a, WRIST in rig.cameras))
    info = _run_info(rig.spec, rig.spec.limits, a.speed, simulated=True)
    return run(strategy, rig, info, a.seconds, a.record)


def _cameras(specs: list[str]) -> dict[str, str]:
    out = {}
    for item in specs:
        name, sep, source = item.partition("=")
        if not sep or name not in (SCENE, WRIST) or not source:
            raise ValueError(f"--camera {item!r}: expected scene=SOURCE or wrist=SOURCE "
                             "(an OpenCV index or a device path)")
        out[name] = source
    if SCENE not in out:
        raise ValueError("--camera scene=SOURCE is required: the camera that sees the table")
    return out


def _rotations(specs: list[str]) -> dict[str, int]:
    out = {}
    for item in specs:
        name, sep, degrees = item.partition("=")
        if not sep or name not in (SCENE, WRIST) or degrees not in ("0", "90", "180", "270"):
            raise ValueError(f"--rotate {item!r}: expected scene=DEG or wrist=DEG, DEG one of 0, 90, 180 "
                             "and 270")
        out[name] = int(degrees)
    return out


def _size(text: str) -> tuple[int, int]:
    width, sep, height = text.lower().partition("x")
    if not sep or not width.isdigit() or not height.isdigit():
        raise ValueError(f"--camera-size {text!r}: expected WIDTHxHEIGHT, e.g. 640x360")
    return int(width), int(height)


def run_arm(a: argparse.Namespace) -> int:
    cameras = _cameras(a.camera)
    if not 0 < a.speed <= 1:
        raise ValueError(f"--speed {a.speed}: a fraction of robot.yaml's max_vel, above 0 and at most 1")
    port = a.port or detect_port()
    rig = ArmRig(port, a.baudrate, cameras, _size(a.camera_size), _rotations(a.rotate))
    for name, reader in rig.readers.items():
        frame = reader.frame()
        shape = "no frame" if frame is None else f"{frame.image.shape[1]}x{frame.image.shape[0]}"
        print(f"Camera {name}: {cameras[name]} ({shape})")
    print(f"Arm on {port}, speed {a.speed:g} x max_vel; measured: "
          + ", ".join(f"{j} {v:.1f}" for j, v in rig.read().items()))
    for fault in rig.faults():
        print(f"Warning: {fault}")
    strategy = DreamStrategy(_settings(a, WRIST in cameras))
    info = _run_info(rig.spec, rig.limits, a.speed, simulated=False)
    return run(strategy, rig, info, a.seconds, a.record)


COMMANDS = {"models": lambda a: list_models(), "sim": run_sim, "arm": run_arm}


def main(a: argparse.Namespace) -> int:
    """`soarm-lab dream <models|sim|arm>`; a failure prints as one line instead of a traceback."""
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    try:
        # dream's IK solver (jaxls) logs every compile step through loguru.
        from loguru import logger as loguru_logger

        loguru_logger.disable("jaxls")
    except ImportError:
        pass
    try:
        return int(COMMANDS[a.dream_cmd](a))
    except (RuntimeError, ValueError, OSError) as exc:
        print(f"soarm-lab dream: {exc}", file=sys.stderr)
        return 2

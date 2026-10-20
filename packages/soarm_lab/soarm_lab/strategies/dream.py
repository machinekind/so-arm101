"""Dream-1, Dream Labs' robot model, as a strategy for the twin: `dream` under `machinekind_twin.strategies`.

The twin finds `DreamStrategy` through the entry point, builds `DreamSettings` from its config file's `dream:`
section or the panel's form, and runs it through its `StrategyRunner`, as it runs `policy` and `ik`.
`soarm-lab dream` (`soarm_lab.dream_run`) runs the same strategy without the twin, in simulation or on the
arm. Nothing here imports the twin: the contract types come from `machinekind_twin_api.strategy`. The Dream
client (`dream`, from the `dreamlabs` library, the `dream` extra) is imported in `start`, never at module
level, so the module loads with numpy alone.

    # configs/local.yaml of the twin: prefills the panel's form
    dream:
      prompt: pick up the cube
      checkpoint: dreamlabs/dream-1-beta
      scene_camera: a              # the twin's camera that sees the table
      wrist_camera: ""             # none: its slot gets `fill_frame`

How the SO-101 goes to Dream-1
------------------------------
Dream-1 is a bimanual model trained on YAM arms. Dream's client runs it on another arm "zero-shot" through
its own IK: `policy({"checkpoint": ..., "ik": {"arm": "so101_faas", "head": "eef"}, "embodiment": ...})`.
`so101_faas` is dream's model of the XLeRobot, two SO-101s. The client then names the robot
`other_embodiment`, so the model reads none of the arm's state (it gets zeros) and acts on the cameras and
the prompt. It answers each hand's tool pose, composed on the hand pose the client computes from
`state.<side>_arm`, and the client solves those poses back to joints (`q_r_chunk`, `q_l_chunk`).

* **One arm.** The client builds both sides unconditionally, so this one arm is the right side and a dummy
  left side goes with every tick: `state.left_arm` at dream's left rest pose (`LEFT_REST_RAD`), the left
  gripper open, and a `video.wrist_l` frame (`fill_frame`: the scene frame again, or black). `q_l_chunk`
  in the reply is ignored.
* **The loop's units.** `state.<side>_arm` is the five SO-101 joints in radians in the frame of dream's URDF
  (`so101_faas_<side>.urdf`); dream holds its sixth, a dummy, at 0, and the client adds it. Its joints 1..5
  have the same origins, rotations and axes as the MJCF in `soarm/robot/so101.xml` (checked joint by joint,
  and by forward kinematics: each joint moves the tool the same way in both). The MJCF's zero is tick 2048,
  LeRobot's middle pose, which is also the radians 0 dream asks for. So the loop's radians are
  `deg2rad(degrees)` of this repository, sign +1 on every joint, no offset. `joint_units` in dream's spec
  (LeRobot -100..100 of the XLeRobot's own calibration) is the checkpoint's side, which the client converts.
* **The gripper.** `state.<side>_grip` is 0 closed .. 1 open: this repository's 0..100 / 100. The client
  makes `state.<side>_gripper` from it; sending it ourselves is refused.
* **The embodiment.** `embodiment_config()` is what `robots.adapter.zero_shot_embodiment(6, gripper=True)` in
  the dreamlabs library builds: the cameras `scene`, `wrist_l`, `wrist_r`. The client sends `scene` in the
  model's `overhead` slot. It is read locally for checks and never sent.
* **Frames.** Every tick each camera frame is stretched to 320x176, JPEG quality 85, base64
  (`encode_frame`), as the Dream client documents it. The client keeps 9 of them, 0.2 s apart.

Chunks and pacing
-----------------
`get_action` takes a round trip (hundreds of ms), so it runs on a thread of its own and `act` never waits
for it. A chunk is `[48, 6]`: five joints in radians, then the grip; one row per tick at `hz` (Dream's
`action_hz`). The client says when to ask again (`refill_due`: async, the next chunk lands as the arm has run
`execution_horizon` rows; sync, once the chunk is used up). A chunk that lands is installed past the rows
the arm ran during its round trip, as dream's own station does (`robots.control`). The client blends each
new chunk into the motion the arm is on, anchored on the joints this strategy commanded last
(`commanded`). An empty buffer holds the last target.

Guards
------
* **speed cap**: every target moves at most `max_step_deg` per tick (the gripper `max_grip_step`) and at
  most `run.max_vel / hz`, so the start ramps from the measured pose and no row jumps;
* **limits**: targets are clipped to `robot.yaml`'s limits intersected with `run.limits`;
* **stop on lag**: as in `policy`, the run ends on the measured pose when an arm joint lags its target by
  more than `max_tracking_deg`;
* **stale cameras**: a missing frame, or one older than `stale_frame_s`, holds the arm;
* **no model**: the run ends when no chunk came for `max_wait_s` (a model still starting, a dead link),
  and on an error the client marks as not retryable (a refused key, a malformed request).

A prompt change mid-run (`set_prompt`) waits for no request in flight, then ends the client's episode
(`reset_episode`), drops the buffered rows and starts a fresh episode with the new prompt.

Every policy the Dream client opens also uploads each observation tick (state and frames) to Dream Labs'
recorder, in the background.
"""

from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import logging
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from machinekind_twin_api.strategy import Action, Observation, RunInfo, Strategy

from ..robot import load_robot

logger = logging.getLogger(__name__)

#: The public Dream-1 checkpoint.
DEFAULT_CHECKPOINT = "dreamlabs/dream-1-beta"
#: Dream's IK model of the SO-101 (`dream_ik.specs`): the XLeRobot, two SO-101s.
IK_ARM = "so101_faas"
#: The frame Dream-1 reads, (width, height), and its JPEG quality, as the Dream client documents them.
FRAME_SIZE = (320, 176)
JPEG_QUALITY = 85
#: The arm joints the loop sends, base to wrist roll: dream's joint1..joint5 (it holds joint6 at 0).
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
GRIPPER = "gripper"
#: The dummy left arm [rad, dream's URDF frame]: dream's rest pose of the left SO-101 of `so101_faas`.
LEFT_REST_RAD = (0.3409, -1.7104, 1.5285, 1.1724, -1.4119)
#: The camera roles of the embodiment: the scene camera, then a wrist camera per arm.
SCENE, WRIST_L, WRIST_R = "scene", "wrist_l", "wrist_r"
#: Errors of the Dream client worth calling again after a pause, by class name, beside any with `retryable`.
RETRY_ERRORS = ("DreamTimeout", "DreamProtocolError", "DreamSessionExpired")
#: How long `stop` waits for a request in flight before it closes the client [s].
STOP_WAIT_S = 2.0
#: What the strategy says when the Dream library is missing.
INSTALL_HINT = "install the dream extra (Python 3.11 to 3.13): uv sync --extra dream"


@dataclass
class DreamSettings:
    #: The task, in plain language.
    prompt: str = "pick up the cube"
    #: The model to run: an id `soarm-lab dream models` lists.
    checkpoint: str = DEFAULT_CHECKPOINT
    #: The camera that sees the scene, by the twin's camera name; empty: the first camera of the observation.
    scene_camera: str = ""
    #: The camera on the arm's wrist; empty: none, and its slot gets `fill_frame`.
    wrist_camera: str = ""
    #: What fills a camera slot no camera of ours has (the dummy left wrist always): "scene" or "black".
    fill_frame: str = "scene"
    #: Ticks per second, one row of a chunk each (Dream's `action_hz`).
    hz: float = 20.0
    #: Rows of each chunk the arm runs before the next one takes over (Dream's `execution_horizon`).
    execution_horizon: int = 30
    #: The most an arm joint's target moves per tick [deg]; 4 deg at 20 Hz is 80 deg/s.
    max_step_deg: float = 4.0
    #: The most the gripper's target moves per tick [0..100 units].
    max_grip_step: float = 10.0
    #: The run ends, holding the measured pose, when an arm joint lags its target by more than this [deg].
    max_tracking_deg: float = 25.0
    #: A camera frame older than this holds the arm [s].
    stale_frame_s: float = 1.0
    #: The run ends when no chunk came for this long [s]: a model still starting, a dead link.
    max_wait_s: float = 120.0


@dataclass
class DreamStatus:
    """What the strategy knows about its run, for logs, the runner's status line and tests."""

    #: Ticks done.
    step: int = 0
    #: Chunks received.
    chunks: int = 0
    #: Rows of the last chunk, and how many of them were skipped (run during its round trip).
    rows: int = 0
    skipped: int = 0
    #: Rows of the current chunk still to run.
    rows_left: int = 0
    #: The last request's round trip [ms].
    rtt_ms: float = float("nan")
    #: The client's pacing, "async" or "sync".
    mode: str = ""
    #: The prompt the model runs on.
    prompt: str = ""
    #: The last tick's speed cap held a joint back.
    capped: bool = False
    note: str = ""
    stopped_because: str = ""


def embodiment_config() -> dict[str, Any]:
    """The zero-shot embodiment of an SO-101 pair, as `robots.adapter.zero_shot_embodiment(6, gripper=True)`.

    Each `<side>_arm` is six wide, dream's spec of the arm with its dummy sixth joint; the client sends the
    gripper drives `<side>_gripper`. The cameras are `scene`, `wrist_l` and `wrist_r`.
    """
    width = 6
    eef = [("right_eef_xyz", 3), ("right_eef_rot6", 6), ("right_grip", 1),
           ("left_eef_xyz", 3), ("left_eef_rot6", 6), ("left_grip", 1)]
    drives = [("right_gripper", 1), ("left_gripper", 1)]
    state = [("left_arm", width), ("right_arm", width), *eef, *drives]
    action = [*eef, ("left_arm", width), ("right_arm", width), *drives]
    video = {role: f"observation.images.{role}" for role in (SCENE, WRIST_L, WRIST_R)}
    return {"state": _spans(state), "action": _spans(action), "video": video}


def _spans(parts: list[tuple[str, int]]) -> dict[str, list[int]]:
    """Each part to its `[start, end)` span, laid end to end from 0."""
    out, start = {}, 0
    for name, width in parts:
        out[name] = [start, start + width]
        start += width
    return out


def policy_config(settings: DreamSettings) -> dict[str, Any]:
    """The config `DreamClient.policy()` takes for this arm."""
    return {
        "checkpoint": settings.checkpoint,
        "ik": {"arm": IK_ARM, "head": "eef"},
        "embodiment": embodiment_config(),
        "action_hz": float(settings.hz),
        "execution_horizon": int(settings.execution_horizon),
    }


def _optional(name: str) -> Any:
    """The module `name`, or None when it cannot be imported."""
    try:
        return importlib.import_module(name)
    except ImportError:
        return None


def encode_frame(rgb: np.ndarray, size: tuple[int, int] = FRAME_SIZE, quality: int = JPEG_QUALITY) -> str:
    """An RGB frame as Dream-1 reads it: stretched to `size` (width, height), JPEG at `quality`, base64.

    OpenCV when it is installed (`INTER_AREA`, as Dream's own stations), else Pillow. No `data:` prefix.
    """
    image = np.ascontiguousarray(np.asarray(rgb, dtype=np.uint8)[..., :3])
    cv2 = _optional("cv2")
    if cv2 is not None:
        if (image.shape[1], image.shape[0]) != size:
            image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
        ok, buffer = cv2.imencode(".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                                  [cv2.IMWRITE_JPEG_QUALITY, int(quality)])
        if not ok:
            raise RuntimeError(f"dream: a frame of shape {image.shape} could not be JPEG-encoded")
        data = bytes(buffer.tobytes())
    else:
        pil = _optional("PIL.Image")
        if pil is None:
            raise RuntimeError(f"dream: encoding camera frames needs OpenCV or Pillow; {INSTALL_HINT}")
        picture = pil.fromarray(image)
        if picture.size != size:
            picture = picture.resize(size, pil.Resampling.BOX)
        out = io.BytesIO()
        picture.save(out, format="JPEG", quality=int(quality))
        data = out.getvalue()
    return base64.b64encode(data).decode("ascii")


def open_client() -> Any:
    """A `dream.DreamClient`; the key comes from `DREAM_API_KEY` or `~/.dreamlabs/credentials`.

    Raises RuntimeError naming the install command when the Dream library or its IK is missing.
    """
    if importlib.util.find_spec("dream") is None:
        raise RuntimeError(f"dream: the Dream Labs client is not installed; {INSTALL_HINT}")
    if importlib.util.find_spec("dream_ik") is None:
        raise RuntimeError("dream: the SO-101 needs dream's IK (`dream_ik`), which only the `dreamlabs` "
                           f"library ships, not dreamlabs-client alone; {INSTALL_HINT}")
    return importlib.import_module("dream").DreamClient()


class DreamStrategy(Strategy[DreamSettings]):
    """Dream-1 driving one SO-101 through Dream's client and its IK: frames and a prompt in, joints out."""

    settings = DreamSettings

    def __init__(self, settings: DreamSettings | None = None, *,
                 client_factory: Callable[[], Any] | None = None,
                 encode: Callable[[np.ndarray], str] | None = None):
        """`client_factory`: makes the client instead of `open_client` (tests); `encode`: instead of
        `encode_frame`."""
        self.config = settings if settings is not None else DreamSettings()
        c = self.config
        positive = ("hz", "max_step_deg", "max_grip_step", "max_tracking_deg", "stale_frame_s", "max_wait_s")
        for name in positive:
            if not getattr(c, name) > 0:
                raise ValueError(f"dream: {name} must be positive, is {getattr(c, name)}")
        if int(c.execution_horizon) < 1:
            raise ValueError(f"dream: execution_horizon must be at least 1, is {c.execution_horizon}")
        if c.fill_frame not in ("scene", "black"):
            raise ValueError(f"dream: fill_frame must be 'scene' or 'black', is {c.fill_frame!r}")
        if not c.prompt.strip():
            raise ValueError("dream: the prompt is empty")
        self.hz = float(c.hz)
        #: The run ends itself (`max_wait_s`, an error) or when the twin stops it.
        self.time_limit_s = None
        self.status = DreamStatus(prompt=c.prompt)
        self._client_factory = client_factory or open_client
        self._encode = encode or encode_frame
        self._client: Any = None
        self._prompt = c.prompt
        #: Guards what the request thread and `set_prompt` hand over: `_done` and `_pending_prompt`.
        self._lock = threading.Lock()
        self._pending_prompt: str | None = None
        self._thread: threading.Thread | None = None
        self._done: tuple[str, Any, float] | None = None
        #: The joints the strategy drives, in its order (ARM_JOINTS, then the gripper), and their limits.
        self._names = (*ARM_JOINTS, GRIPPER)
        self._lo = np.full(6, -np.inf)
        self._hi = np.full(6, np.inf)
        self._cap = np.zeros(6)
        #: The last target [deg, gripper 0..100], and the row the arm drives toward.
        self._q = np.zeros(6)
        self._goal = np.zeros(6)
        #: The rows still to run [deg, gripper 0..100], and how many the installed chunk had.
        self._buffer: deque[np.ndarray] = deque()
        self._installed = 0
        #: Rows run since the request in flight took its observation.
        self._rows_since_request = 0
        self._retry_at = 0.0
        self._last_chunk_at = 0.0
        self._fatal: str | None = None
        self._black: str | None = None

    @property
    def started(self) -> bool:
        return self._client is not None

    # ------------------------------------------------------------------- start

    def start(self, run: RunInfo, obs: Observation) -> None:
        """Opens the Dream client and its policy, then starts from where the arm is. Raises to refuse the run.

        Opening the policy asks the platform for the checkpoint and prepares dream's IK: a few seconds, up to
        about a minute the first time on a machine.
        """
        missing = [name for name in self._names if name not in run.joints]
        if missing:
            raise RuntimeError(f"dream: runs an SO-101 ({', '.join(self._names)}); the robot {run.robot!r} "
                               f"lacks {', '.join(missing)}")
        spec = load_robot(run.folder)
        for k, name in enumerate(self._names):
            lo, hi = spec.limits.get(name, (-np.inf, np.inf))
            if name in run.limits:
                a, b = run.limits[name]
                if max(lo, min(a, b)) <= min(hi, max(a, b)):
                    lo, hi = max(lo, min(a, b)), min(hi, max(a, b))
            self._lo[k], self._hi[k] = lo, hi
            vel = run.max_vel.get(name, spec.max_vel.get(name, np.inf))
            step = self.config.max_grip_step if name == GRIPPER else self.config.max_step_deg
            self._cap[k] = min(step, float(vel) / self.hz) if vel > 0 else step

        client = self._client_factory()
        client.policy(policy_config(self.config))
        self._client = client
        self._q = self._vector(obs.joints)
        self._goal = self._q.copy()
        self._buffer.clear()
        self._installed = 0
        self._rows_since_request = 0
        self._retry_at = 0.0
        self._fatal = None
        self._last_chunk_at = time.monotonic()
        self.status = DreamStatus(prompt=self._prompt)
        logger.info("Dream: %s on %r, IK %s", self.config.checkpoint, self._prompt, IK_ARM)

    def _vector(self, joints: Mapping[str, float]) -> np.ndarray:
        """The strategy's joints from a mapping [deg, gripper 0..100]; a missing one keeps the last target."""
        return np.array([float(joints.get(name, self._q[k])) for k, name in enumerate(self._names)])

    # ------------------------------------------------------------------ prompt

    @property
    def prompt(self) -> str:
        return self._prompt

    def set_prompt(self, prompt: str) -> None:
        """A new task, from any thread.

        At the next tick with no request in flight the client's episode ends, the buffered rows are dropped,
        and a fresh episode starts with this prompt. The arm holds meanwhile.
        """
        text = prompt.strip()
        if not text:
            raise ValueError("dream: the prompt is empty")
        with self._lock:
            self._pending_prompt = text

    def _apply_prompt(self, now: float) -> None:
        with self._lock:
            pending = self._pending_prompt
            if pending is None or self._thread is not None:
                return
            self._pending_prompt = None
        self._client.reset_episode()
        self._buffer.clear()
        self._installed = 0
        self._goal = self._q.copy()
        self._prompt = pending
        self.status.prompt = pending
        self._last_chunk_at = now
        self._retry_at = 0.0
        logger.info("Dream: new prompt %r, a fresh episode", pending)

    # --------------------------------------------------------------------- tick

    def act(self, obs: Observation) -> Action:
        """One tick: hand the client this tick, run the chunk's next row, ask for the next chunk when due."""
        if self._client is None:
            raise RuntimeError("dream: act() before start()")
        st = self.status
        now = time.monotonic()
        st.step += 1
        measured = self._vector(obs.joints)
        lag = float(np.abs(measured[:5] - self._q[:5]).max())
        if lag > self.config.max_tracking_deg:
            return self._halt(f"the arm lags its target by {lag:.0f} deg: a collision or a block",
                              hold=measured)
        self._collect(now)
        if self._fatal is not None:
            return self._halt(self._fatal)
        self._apply_prompt(now)

        frames, why = self._frames(obs)
        if frames is None:
            st.note = f"{why}: holding"
            return self._tick(now, advance=False)
        self._client.set_observation(frames, self._observation(measured), timestamp=now)

        if self._thread is None and now >= self._retry_at:
            remaining = len(self._buffer)
            if self._client.refill_due(self._installed - remaining, remaining):
                self._request()
        return self._tick(now, advance=True)

    def _tick(self, now: float, advance: bool) -> Action:
        """Takes this tick's row (when `advance`), moves the target toward it under the speed cap."""
        st = self.status
        if advance and self._buffer:
            self._goal = self._buffer.popleft()
            self._rows_since_request += 1
        if not self._buffer and now - self._last_chunk_at > self.config.max_wait_s:
            return self._halt(f"no chunk from the model for {self.config.max_wait_s:.0f} s"
                              + (f" ({st.note})" if st.note else ""))
        goal = np.clip(self._goal, self._lo, self._hi)
        step = np.clip(goal - self._q, -self._cap, self._cap)
        st.capped = bool(np.any(np.abs(goal - self._q) > self._cap + 1e-9))
        self._q = self._q + step
        st.rows_left = len(self._buffer)
        if advance:
            st.note = self._note(now)
        return Action(target=self._target(self._q), note=st.note)

    def _target(self, q: np.ndarray) -> dict[str, float]:
        return {name: float(q[k]) for k, name in enumerate(self._names)}

    def _note(self, now: float) -> str:
        st = self.status
        if self._retry_at > now:
            return st.note
        if st.chunks == 0:
            return f"waiting for the first chunk ({now - self._last_chunk_at:.0f} s)"
        if not self._buffer:
            return f"chunk {st.chunks} used up, waiting for the next"
        return (f"chunk {st.chunks}: {len(self._buffer)} rows left, {st.rtt_ms:.0f} ms"
                + (", speed-capped" if st.capped else ""))

    # ----------------------------------------------------------- observation

    def _frames(self, obs: Observation) -> tuple[dict[str, str] | None, str]:
        """The tick's frames as the client takes them, or (None, why) when a camera is missing or stale."""
        c = self.config
        names = list(obs.frames)
        scene_name = c.scene_camera or next((n for n in names if n != c.wrist_camera), "")
        if not scene_name:
            return None, "no camera frames"
        wanted = [scene_name] + ([c.wrist_camera] if c.wrist_camera else [])
        images: dict[str, np.ndarray] = {}
        for name in wanted:
            frame = obs.frames.get(name)
            if frame is None:
                return None, f"camera {name!r} has no frame"
            if frame.age_s > c.stale_frame_s:
                return None, f"camera {name!r} frame is {frame.age_s:.1f} s old"
            images[name] = frame.image
        scene = self._encode(images[scene_name])
        fill = scene if c.fill_frame == "scene" else self._black_frame()
        wrist = self._encode(images[c.wrist_camera]) if c.wrist_camera else fill
        return {f"video.{SCENE}": scene, f"video.{WRIST_L}": fill, f"video.{WRIST_R}": wrist}, ""

    def _black_frame(self) -> str:
        if self._black is None:
            self._black = self._encode(np.zeros((FRAME_SIZE[1], FRAME_SIZE[0], 3), np.uint8))
        return self._black

    def _observation(self, measured: np.ndarray) -> dict[str, Any]:
        """The state the client takes: this arm on the right, the dummy left arm at rest, its gripper open."""
        return {
            "prompt": self._prompt,
            "state.right_arm": np.radians(measured[:5]).astype(np.float32),
            "state.right_grip": np.array([np.clip(measured[5] / 100.0, 0.0, 1.0)], np.float32),
            "state.left_arm": np.array(LEFT_REST_RAD, np.float32),
            "state.left_grip": np.array([1.0], np.float32),
        }

    # ---------------------------------------------------------------- chunks

    def _request(self) -> None:
        """Asks for the next chunk on a thread of its own, anchored on the joints commanded last."""
        commanded = {"q_r_chunk": np.radians(self._q[:5])}
        self._rows_since_request = 0
        client = self._client

        def ask() -> None:
            t0 = time.monotonic()
            try:
                result: tuple[str, Any, float] = ("ok", client.get_action(commanded=commanded),
                                                  time.monotonic() - t0)
            except Exception as exc:  # noqa: BLE001 - sorted on the control thread (`_failed`)
                result = ("error", exc, time.monotonic() - t0)
            with self._lock:
                self._done = result

        self._thread = threading.Thread(target=ask, name="dream-get-action", daemon=True)
        self._thread.start()

    def wait_request(self, timeout: float | None = None) -> None:
        """Waits for the request in flight, if any (tests, shutdown); the next tick collects its answer."""
        thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _collect(self, now: float) -> None:
        """Installs the chunk a finished request brought, or sorts its error."""
        with self._lock:
            done, self._done = self._done, None
        if done is None:
            return
        self._thread = None
        kind, value, rtt_s = done
        st = self.status
        st.rtt_ms = rtt_s * 1e3
        st.mode = str(getattr(self._client, "mode", "") or "")
        if kind == "error":
            self._failed(value, now)
            return
        try:
            rows = self._rows(value)
        except ValueError as exc:
            self._fatal = f"the model's chunk is unusable: {exc}"
            return
        skip = self._rows_since_request
        if skip >= len(rows):
            logger.warning("Dream: a chunk of %d rows landed after %d rows; dropped", len(rows), skip)
            return
        self._buffer = deque(rows[skip:])
        self._installed = len(self._buffer)
        self._last_chunk_at = now
        self._retry_at = 0.0
        st.chunks += 1
        st.rows, st.skipped = len(rows), skip
        logger.debug("Dream: chunk %d, %d rows, %d skipped, %.0f ms", st.chunks, len(rows), skip, st.rtt_ms)

    def _rows(self, reply: Any) -> list[np.ndarray]:
        """`q_r_chunk` as rows of targets [deg, gripper 0..100]: five joints [rad], then the grip 0..1."""
        if not isinstance(reply, Mapping) or "q_r_chunk" not in reply:
            keys = sorted(reply) if isinstance(reply, Mapping) else type(reply).__name__
            raise ValueError(f"no q_r_chunk in the reply ({keys})")
        chunk = np.asarray(reply["q_r_chunk"], dtype=float)
        if chunk.ndim != 2 or chunk.shape[0] == 0 or chunk.shape[1] != 6:
            raise ValueError(f"q_r_chunk has shape {chunk.shape}, expected (rows, 6)")
        if not np.all(np.isfinite(chunk)):
            raise ValueError("q_r_chunk holds a number that is not finite")
        rows = np.empty_like(chunk)
        rows[:, :5] = np.degrees(chunk[:, :5])
        rows[:, 5] = 100.0 * np.clip(chunk[:, 5], 0.0, 1.0)
        return list(rows)

    def _failed(self, exc: BaseException, now: float) -> None:
        """A failed request: call again after the wait the error asks for, or end the run."""
        name = type(exc).__name__
        if not (bool(getattr(exc, "retryable", False)) or name in RETRY_ERRORS):
            self._fatal = f"Dream refused the request: {name}: {exc}"
            return
        wait = getattr(exc, "retry_after_s", None)
        wait = float(wait) if isinstance(wait, (int, float)) and wait > 0 else 1.0
        self._retry_at = now + wait
        reason = getattr(exc, "reason", None) or name
        self.status.note = f"model: {reason}, asking again in {wait:.0f} s"
        logger.warning("Dream: %s: %s; asking again in %.0f s", name, exc, wait)

    # ---------------------------------------------------------------------- end

    def _halt(self, reason: str, hold: np.ndarray | None = None) -> Action:
        """The run is over: hold `hold` (the last target by default) and say why."""
        self.status.stopped_because = reason
        self.status.note = reason
        logger.info("Dream stopped: %s", reason)
        target = self._q if hold is None else hold
        return Action(target=self._target(target), done=True, note=reason)

    def stop(self, reason: str) -> None:
        """The run ended (the twin, the runner, our own `done`): waits briefly for a request, then closes."""
        if not self.status.stopped_because:
            self.status.stopped_because = reason or "stopped"
        client, self._client = self._client, None
        self.wait_request(STOP_WAIT_S)
        if client is not None:
            # The close ends the session over HTTPS and may take seconds (its own timeout is 15 s): done on a
            # daemon thread and waited for briefly, so Ctrl-C returns the terminal without a second Ctrl-C.
            closer = threading.Thread(target=self._close_client, args=(client,), name="dream-close",
                                      daemon=True)
            closer.start()
            closer.join(STOP_WAIT_S)

    @staticmethod
    def _close_client(client: Any) -> None:
        try:
            client.close()
        except Exception:  # noqa: BLE001 - a stop must not fail on the session release
            logger.exception("Dream: closing the client failed")

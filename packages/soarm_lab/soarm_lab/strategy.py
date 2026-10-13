"""An RL policy of the lab as a strategy of the twin: `policy` under the `lerobot_mp.strategies` entry points.

The twin finds `PolicyStrategy` through the entry point in this package's `pyproject.toml`, builds
`PolicySettings` from its config file's `policy:` section or the panel's form, creates the strategy as
`PolicyStrategy(settings)` and runs it through its `StrategyRunner`: the runner claims the arm, builds
an observation every tick and sends the strategy's joint targets through the safety supervisor, in
simulation and on hardware alike. Nothing here imports the twin: the observation and the run info are
read by their attributes (`obs.joints`, `obs.command`, `obs.goal`, `obs.objects`, `obs.t`;
`run.robot`, `run.folder`, `run.limits`, `run.simulated`), and the answer is this module's `PolicyAction`,
an object with the attributes of the twin's `Action`.

    # configs/local.yaml of the twin: prefills the panel's form
    policy:
      policy: lift-v3              # a bundled policy, one trained here (workspace/policies) or a path

The loop does exactly what the training environment does - the observation from `task.observe`, the
action through `task.apply_action` - only the angles come from the servos, not from physics. It is the
twin's former `rl/runner.py` (`PolicyRunner`) with the twin's side taken out: claiming the arm, the
thread, STOP and Home now belong to the `StrategyRunner`. The guards the training environment does not
have and a real arm needs stay here:

* **stop on lag**: when the measured pose falls behind the target by more than `max_tracking_deg`
  (a blocked joint, a collision, a servo that cannot keep up) the run ends and the arm holds where it
  MEASURES - the twin's own tracking guard (STOP) comes on top, for every owner;
* **time limit**: a run on the arm lasts as many ticks as a training episode (`episode_limit`);
* **end on success**: `end_on_success` success ticks in a row end the run (lift: the cube above the
  table). Lift success counts only from a fresh cube pose (cameras, "in the hand", or simulation
  truth), never from "last seen", and only while the jaw holds (still, commanded tighter, not
  creeping closed during the streak);
* **reach goal in the training area**: a goal outside it (under the table, behind the base) is
  projected onto the area training goals were drawn from;
* **pickplace**: the place target is the goal projected onto the table (z = 0) and onto the training
  ring of targets. Success counts only for a cube SEEN lying (cameras, not "in the hand", not "last
  seen") on the target, released (the jaw does not hold it, the gripper command is open), still
  during the streak, and only after it was carried (the same latch as in training).

Where the inputs come from: the goal is `obs.goal` (the panel's goal); without one the settings'
`goal_x`, `goal_y`, `goal_z` hold it. The cube is `obs.objects["cube"]`: a pose in the base frame and
its `source` ("kamery", "w dloni", "ostatnie widziane", "symulacja"); an empty source counts as
simulation truth on a simulated arm and as stale on a real one.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .kinematics import RobotKinematics
from .rl import task as tk
from .rl.policy import Policy, find_policy
from .robot import load_robot

logger = logging.getLogger(__name__)

#: Cube pose sources (the twin's `CubeTracker.source`) lift success may be counted from: a fresh
#: camera detection, the cube carried in the jaws, or simulation truth.
FRESH_CUBE_SOURCES = ("kamery", "w dloni", "symulacja")
#: Sources pickplace success may be counted from: the cube SEEN lying (not "w dloni": that pose follows
#: the jaws) or simulation truth.
RESTING_CUBE_SOURCES = ("kamery", "symulacja")
#: pickplace: during a success streak the cube moves no more than this [m].
PLACE_STILL = 0.005


class NoCube(RuntimeError):
    """A task with a cube, and the observation does not say where it is."""


@dataclass
class PolicySettings:
    #: A bundled policy (reach-v3, lift-v3), one trained here (`soarm-lab policies`), or a path to policy.pt.
    policy: str = "reach-v3"
    #: The run ends, holding the measured pose, when an arm joint lags its target by more than this [deg].
    max_tracking_deg: float = 25.0
    #: End the run after a training episode's number of ticks.
    episode_limit: bool = True
    #: How long the arm waits for the first cube pose [s] (lift, pickplace).
    wait_for_cube_s: float = 3.0
    #: The goal when the observation has none [m, base frame]: reach drives the TCP there, pickplace
    #: places the cube under it.
    goal_x: float = 0.22
    goal_y: float = 0.0
    goal_z: float = 0.12


@dataclass
class PolicyAction:
    """The twin's `Action` by its attributes."""

    target: dict[str, float] = field(default_factory=dict)
    done: bool = False
    success: bool | None = None
    error: float | None = None
    note: str = ""


@dataclass
class PolicyStatus:
    """What the strategy knows about its run, for logs and tests (the twin keeps its own runner status)."""

    #: Policy ticks done (a tick waiting for the cube does not count).
    step: int = 0
    distance: float = float("nan")
    success: bool = False
    stopped_because: str = ""
    last_action: list[float] = field(default_factory=list)
    #: How many ticks in a row the task is done.
    success_streak: int = 0
    #: reach: the goal was clipped to the training area. pickplace: the place target was moved onto the
    #: training ring of targets.
    goal_clamped: bool = False
    #: pickplace: the cube was carried in this run (latch).
    carried: bool = False


def project_goal(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """Najblizszy punkt obszaru, z ktorego `sample_goals` losowal cele reach.

    Promien od osi podstawy w `goal_radius`, wysokosc w `goal_height`, x > 0,05 m.
    Polityka poza nim nigdy nie byla - cel pod blatem prowadzil szczeki w blat.
    """
    g = np.asarray(goal, float).copy()
    g[2] = np.clip(g[2], *task.goal_height)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.goal_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    # x > 0,05 przy promieniu r to kat od osi x najwyzej arccos(0,05 / r); margines 1 mm.
    b_max = float(np.arccos(min(1.0, 0.051 / r_new)))
    bearing = float(np.clip(bearing, -b_max, b_max))
    g[0], g[1] = r_new * np.cos(bearing), r_new * np.sin(bearing)
    return g


def project_target(task: tk.TaskConfig, goal: np.ndarray) -> np.ndarray:
    """Cel odlozenia pickplace z punktu `goal` (uklad podstawy): na blat (z = 0) i na pierscien celow.

    Promien od osi podstawy w `target_radius`, kierunek w `target_bearing` - tam
    `sample_targets` losowal cele w treningu; poza nim polityka nigdy nie odkladala.
    """
    g = np.asarray(goal, float)
    r = float(np.hypot(g[0], g[1]))
    r_new = float(np.clip(r, *task.target_radius))
    bearing = float(np.arctan2(g[1], g[0])) if r > 1e-9 else 0.0
    bearing = float(np.clip(bearing, *task.target_bearing))
    return np.array([r_new * np.cos(bearing), r_new * np.sin(bearing), 0.0])


class PolicyStrategy:
    """An RL policy of soarm-lab (reach, lift, pickplace) driving the arm, as in training."""

    settings = PolicySettings

    def __init__(self, settings: PolicySettings | None = None, *, policy: Policy | None = None):
        """`policy`: an already loaded policy instead of `settings.policy` (tests, notebooks)."""
        self.config = settings if settings is not None else PolicySettings()
        if self.config.max_tracking_deg <= 0:
            raise ValueError(f"policy: max_tracking_deg must be positive, is {self.config.max_tracking_deg}")
        self.policy = policy if policy is not None else Policy.load(find_policy(self.config.policy))
        self.task = self.policy.task
        #: The policy's own rate: the twin's runner ticks at it.
        self.hz = float(self.task.control_hz)
        #: The run ends itself (episode limit, success); the twin needs no time limit of its own.
        self.time_limit_s = None
        self.status = PolicyStatus()
        end = getattr(self.task, "end_on_success", None)
        #: Po ilu taktach sukcesu z rzedu epizod sie konczy (0 = nigdy). Z zadania (K8);
        #: polityki zapisane przed tym polem - lift 10, reach 0 (cel przeciagany na zywo).
        self.end_on_success = int(end) if end is not None else (10 if self.task.cube else 0)
        self.max_tracking = np.radians(self.config.max_tracking_deg)
        self.episode_limit = self.config.episode_limit
        #: Ile sekund ramie stoi, czekajac na pierwsza pozycje kostki z kamer.
        self.wait_for_cube = self.config.wait_for_cube_s
        self._default_goal = np.array([self.config.goal_x, self.config.goal_y, self.config.goal_z])
        self._goal = self._default_goal.copy()
        # Known from `start`: the arm's kinematics, the training limits and the hardware's.
        self.kin: RobotKinematics | None = None
        self.limits: tk.Limits | None = None
        self.g, self.na = 0, 0
        self.q_cmd = np.zeros(self.task.act_dim)
        self.prev_action = np.zeros(self.task.act_dim)
        self._hw_lo = self._hw_hi = np.zeros(self.task.act_dim)
        self.simulated = False
        self._cube: np.ndarray | None = None
        self._cube_src: str | None = None
        self._grip_prev: float | None = None
        self._streak_grip = 0.0
        #: pickplace: kostka byla niesiona (jak `task.carry_latch` w treningu) i gdzie byla na poczatku serii.
        self._carried = False
        self._streak_cube: np.ndarray | None = None

    # -------------------------------------------------------------------- goal

    @property
    def goal(self) -> np.ndarray:
        return self._goal

    @goal.setter
    def goal(self, value) -> None:
        g = np.asarray(value, float)
        if self.task.name == "reach":
            p = project_goal(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p - g) > 1e-4)
            g = p
        elif self.task.target:
            # Cel odlozenia lezy na blacie - z = 0 to nie przyciecie, tylko definicja.
            p = project_target(self.task, g)
            self.status.goal_clamped = bool(np.linalg.norm(p[:2] - g[:2]) > 1e-4)
            g = p
        self._goal = g

    @property
    def target(self) -> np.ndarray:
        """pickplace: cel odlozenia (uklad podstawy, z = 0) - `goal` rzutowany przez `project_target`."""
        return project_target(self.task, self._goal)

    # ------------------------------------------------------------------- start

    def start(self, run: Any, obs: Any) -> None:
        """Checks the policy against the arm and starts from where the arm is. Raises to refuse the run."""
        robot = getattr(self.policy.meta, "robot", "so101")
        if robot != run.robot:
            raise RuntimeError(f"polityka uczona na {robot}, stanowisko ma {run.robot}")
        spec = load_robot(run.folder)
        if self.policy.meta.act_dim != len(spec.joints):
            raise RuntimeError(f"polityka ma {self.policy.meta.act_dim} wyjsc, a ramie {spec.name} "
                               f"{len(spec.joints)} stawow")
        self.kin = RobotKinematics(spec)
        self.limits = tk.Limits.of(self.kin)
        #: Indeks chwytaka i liczba stawow ramienia (chwytak ostatni, `Limits.of`).
        self.g, self.na = self.limits.grip, self.limits.n_arm
        #: Szczeka "trzyma" (sukces lift): jak w `CubeTracker` blizniaka - stoi (< `grip_still` miedzy
        #: taktami), szerzej niz rozkaz o `grip_block_margin` i niz zamknieta o `grip_open_margin`.
        #: Jednostki MJCF chwytaka (`spec.jaw`; SO-101: rad).
        jaw = spec.jaw
        self.grip_still, self.grip_block_margin, self.grip_open_margin = jaw.still, jaw.margin, jaw.margin
        #: W trakcie serii sukcesu szczeka nie domyka sie o wiecej niz tyle (SO-101: 0,03 rad ~ 3 jednostki).
        self.grip_creep = jaw.creep
        self.simulated = bool(run.simulated)

        measured = dict(obs.joints)
        self._hw_lo, self._hw_hi = self._hardware_limits(getattr(run, "limits", None) or {})
        self.q_cmd = np.clip(self.kin.to_q(measured), self._hw_lo, self._hw_hi)
        grip = self._squeezing_grip_cmd(measured, getattr(obs, "command", None) or {})
        if grip is not None:
            self.q_cmd[self.g] = grip
        self.prev_action = np.zeros(self.task.act_dim)
        self.status = PolicyStatus(goal_clamped=self.status.goal_clamped)
        self._cube, self._cube_src = None, None
        self._grip_prev = None
        self._carried, self._streak_cube = False, None
        self._take_goal(obs)

    def _squeezing_grip_cmd(self, measured: Mapping[str, float],
                            command: Mapping[str, float]) -> float | None:
        """Kat rozkazu chwytaka [rad] z `obs.command`, gdy chwytak SCISKA (rozkaz ciasniej niz pomiar).

        Start od zmierzonego kata szczeki to zerowa sila: ponowne Uruchom z kostka w szczekach
        (po STOP, Dom, Zatrzymaj) puszczalo chwyt pierwsza akcja, a `_jaw_holding` nigdy nie
        widzial szczeki "szerzej niz rozkaz" - sukces lift nie mogl sie policzyc. W treningu
        q_cmd trzymanej kostki tez jest ciasniejszy od pomiaru. Inne stawy - jak dotad, od pomiaru.
        """
        name = self.kin.spec.gripper
        cmd = command.get(name)
        if cmd is None or name not in measured:
            return None
        g = self.g
        q_cmd = float(self.kin.to_q({**measured, name: float(cmd)})[g])
        if float(self.kin.to_q(measured)[g]) - q_cmd <= self.grip_block_margin:
            return None
        return float(np.clip(q_cmd, self._hw_lo[g], self._hw_hi[g]))

    def _hardware_limits(self, hw: Mapping[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray]:
        """Limity treningu przeciete z limitami, ktore naprawde obowiazuja na ramieniu (`run.limits`).

        Normalizacja obserwacji zostaje na limitach treningu - zmiana skali
        bylaby dla polityki inna obserwacja. Przycinamy tylko cel: inaczej
        q_cmd stal do 7 st. za limitem serwa (ramie nr 2, wrist_flex +88).
        """
        lo, hi = self.limits.lo.copy(), self.limits.hi.copy()
        for k, name in enumerate(self.kin.spec.joints):
            if name in hw:
                a, b = self.kin.to_q({name: hw[name][0]})[k], self.kin.to_q({name: hw[name][1]})[k]
                lo[k], hi[k] = max(lo[k], min(a, b)), min(hi[k], max(a, b))
        bad = lo > hi
        lo[bad], hi[bad] = self.limits.lo[bad], self.limits.hi[bad]
        return lo, hi

    def _take_goal(self, obs: Any) -> None:
        goal = getattr(obs, "goal", None)
        self.goal = self._default_goal if goal is None else goal

    # -------------------------------------------------------------- observation

    def observation(self, measured: Mapping[str, float],
                    objects: Mapping[str, Any] | None = None) -> np.ndarray:
        """The policy's observation vector, as `task.observe` builds it in training."""
        q = self.kin.to_q(measured)
        tcp = self.kin.tcp(measured)[:3, 3]
        cube_pos, cube_rot = np.zeros(3), np.eye(3)
        target = None
        if self.task.cube:
            got = (objects or {}).get(self.task.cube)
            if got is None:
                raise NoCube(f"zadanie {self.task.name} potrzebuje polozenia kostki - kamery jej nie widza")
            cube_pos, cube_rot = np.asarray(got.position, float), np.asarray(got.rotation, float)
            self._cube_src = str(getattr(got, "source", "") or "") or None
            self._cube = cube_pos.copy()
            if self.policy.meta.randomization.get("fold_yaw", False):
                # W treningu polityka widziala obrot zlozony do +-45 st. (jak z detektora);
                # poza "w dloni" i prawda z symulacji tego nie maja - skladamy tak samo.
                cube_rot = tk.fold_yaw(np, cube_rot)
        if self.task.target:
            target = self.target[None]
        obs = tk.observe(np, self.task, self.limits, q[None], self.q_cmd[None], tcp[None],
                         self.prev_action[None], goal=self.goal[None], cube_pos=cube_pos[None],
                         cube_rot=cube_rot[None], target=target)
        return obs[0].astype(np.float32)

    # --------------------------------------------------------------------- tick

    def act(self, obs: Any) -> PolicyAction:
        """One policy tick: the next joint targets, or the end of the run."""
        if self.kin is None:
            raise RuntimeError("policy: act() przed start()")
        st = self.status
        measured = dict(obs.joints)
        self._take_goal(obs)
        q = self.kin.to_q(measured)
        lag = np.abs(q[:self.na] - self.q_cmd[:self.na]).max()
        if lag > self.max_tracking:
            # Ramie staje tam, gdzie JEST: ostatni cel lezy wtedy ~25 st. za przeszkoda.
            return self._halt(f"ramie nie nadaza za celem ({np.degrees(lag):.0f} st.) - kolizja albo blokada",
                              target=self._hold_pose(measured, q))
        holding = self._jaw_holding(q[self.g])
        try:
            vec = self.observation(measured, getattr(obs, "objects", None))
        except NoCube as exc:
            # Na starcie percepcja moze jeszcze nie miec pierwszej detekcji - ramie
            # stoi i czekamy; zgubiona kostka W TRAKCIE epizodu zatrzymuje polityke.
            if st.step == 0 and float(obs.t) < self.wait_for_cube:
                st.stopped_because = "czekam na kostke z kamer"
                return PolicyAction(note=st.stopped_because)
            return self._halt(str(exc))
        action = self.policy.act(vec)
        q_cmd = tk.apply_action(np, self.task, self.limits, self.q_cmd[None], action[None])[0]
        self.q_cmd = np.clip(q_cmd, self._hw_lo, self._hw_hi)
        target = self.kin.from_q(self.q_cmd)
        self.prev_action = action
        st.step += 1
        st.stopped_because = ""
        st.last_action = action.tolist()
        if self.task.name == "reach":
            st.distance = float(np.linalg.norm(self.goal - self.kin.tcp(measured)[:3, 3]))
            st.success = st.distance < self.task.success_dist
        elif self.task.target:
            st.success = self._placed(holding)
            st.carried = self._carried
        else:
            # Na ramieniu nie ma czujnikow kontaktu szczek - sukcesem jest kostka
            # (z kamer albo "w dloni") wyzej nad blatem niz `lift_height`, i to tylko,
            # gdy szczeka ja TRZYMA (stoi, ciasniej rozkazana niz jest). Zmierzone
            # (blizniak, 1 z 18 sukcesow): seria 1 s przy 10,9 cm, a kostka obracala
            # sie w szczekach, szczeka domykala sie 31,8 -> 0 przez 3,5 s i kostka spadla
            # 4 s po "zadanie wykonane".
            # Szczeka "stoi" z taktu na takt nawet przy powolnym domykaniu (9 jednostek/s
            # to 0,005 rad na takt) - dlatego tez: w trakcie serii nie domyka sie o wiecej
            # niz `grip_creep` od jej poczatku.
            if holding and st.success_streak > 0 and q[self.g] < self._streak_grip - self.grip_creep:
                holding = False
            st.success = bool(self._cube is not None and self._cube_fresh() and holding
                              and self._cube[2] - self.task.cube_half > self.task.lift_height)
            if st.success and st.success_streak == 0:
                self._streak_grip = float(q[self.g])
        st.success_streak = st.success_streak + 1 if st.success else 0
        if self.end_on_success > 0 and st.success_streak >= self.end_on_success:
            return self._halt("zadanie wykonane", target=target, success=True)
        if self.episode_limit and st.step >= self.task.episode_steps:
            return self._halt("koniec epizodu", target=target, success=st.success)
        return PolicyAction(target=target, success=st.success, error=self._error(), note=self._note())

    def _placed(self, holding: bool) -> bool:
        """pickplace: sukces tego taktu (i aktualizacja zatrzasku "niesiona", `status.distance`).

        Kostka widziana lezaca (`RESTING_CUBE_SOURCES`) na celu (xy blizej niz `place_radius`,
        spod nizej niz `place_height`), szczeka jej nie trzyma, rozkaz chwytaka otwarty,
        wczesniej byla niesiona, a od poczatku serii nie przesunela sie o `PLACE_STILL`.
        Zatrzask: prawda symulacji albo "w dloni" z kamer wyzej nad blatem niz `lift_height`.
        """
        cube, src, task = self._cube, self._cube_src, self.task
        if cube is None:
            return False
        if src is None and self.simulated:
            src = "symulacja"                               # dostawca bez zrodla w sim = prawda sceny
        height = float(cube[2]) - task.cube_half
        tgt = self.target
        d_t = float(np.hypot(*(cube[:2] - tgt[:2])))
        self.status.distance = d_t
        carried_before = self._carried
        if src in ("symulacja", "w dloni") and height > task.lift_height:
            self._carried = True
        grip_open = bool(tk.grip_is_open(np, self.limits, self.q_cmd[None])[0])
        ok = (src in RESTING_CUBE_SOURCES and carried_before and d_t < task.place_radius
              and height < task.place_height and not holding and grip_open)
        if ok and self.status.success_streak > 0 and self._streak_cube is not None \
                and np.linalg.norm(cube - self._streak_cube) >= PLACE_STILL:
            ok = False                                      # kostka jeszcze sie toczy / przesuwa
        if ok and (self.status.success_streak == 0 or self._streak_cube is None):
            self._streak_cube = np.asarray(cube, float).copy()
        return ok

    def _jaw_holding(self, grip_q: float) -> bool:
        """Czy szczeka trzyma: stoi od poprzedniego taktu, jest szerzej niz rozkaz i niz pusta, zamknieta.

        Te same progi, co `CubeTracker` blizniaka ("w dloni"). Rozkaz z POPRZEDNIEGO taktu - za nim
        serwo wlasnie jedzie. Szczeka, ktora sie domyka (kostka wypada, obraca sie), nie stoi.
        """
        prev, self._grip_prev = self._grip_prev, float(grip_q)
        still = prev is not None and abs(grip_q - prev) < self.grip_still
        squeezing = (grip_q - self.q_cmd[self.g] > self.grip_block_margin
                     and grip_q > self.limits.lo[self.g] + self.grip_open_margin)
        return bool(still and squeezing)

    def _hold_pose(self, measured: Mapping[str, float], q: np.ndarray) -> dict[str, float]:
        """The last target of a run stopped on lag: the arm where it is, as the twin's `hold_measured`.

        A gripper that squeezes (measured wider than commanded) is left out of the target and keeps its
        command: commanding the measured angle of a blocked jaw is zero force, and a carried cube falls.
        """
        pose = {name: float(measured[name]) for name in self.kin.spec.joints if name in measured}
        grip = self.kin.spec.gripper
        if grip in pose and q[self.g] - self.q_cmd[self.g] > self.grip_block_margin:
            del pose[grip]
        return pose

    def _cube_fresh(self) -> bool:
        """Czy poza kostki jest swieza - tylko z takiej wolno liczyc sukces lift.

        Tracker blizniaka trzyma ostatnia poze do 6 s jako "ostatnie widziane". Zmierzone w
        blizniaku (kamera a): kostka wypadla ze szczek po 0,5 s niesienia, a runner
        liczyl sukces ze starej pozy w powietrzu i konczyl "zadanie wykonane" z kostka
        na blacie. Bez zrodla: symulacja (dostawca = prawda sceny) tak, prawdziwe ramie nie.
        """
        if self._cube_src is None:
            return self.simulated
        return self._cube_src in FRESH_CUBE_SOURCES

    # ---------------------------------------------------------------------- end

    def _halt(self, reason: str, target: Mapping[str, float] | None = None,
              success: bool = False) -> PolicyAction:
        """The run is over: the last target (or the measured pose) and the reason for the twin's panel."""
        self.status.stopped_because = reason
        logger.info("Polityka zatrzymana: %s", reason)
        return PolicyAction(target=dict(target or {}), done=True, success=success, error=self._error(),
                            note=reason)

    def stop(self, reason: str) -> None:
        """The twin ended the run (Stop, STOP, Home, the arm taken, or our own `done`)."""
        # After our own `done` the twin reports "zadanie wykonane" or "strategia skonczyla"; ours says more.
        if reason in ("zadanie wykonane", "strategia skonczyla") and self.status.stopped_because:
            return
        self.status.stopped_because = reason or "zatrzymana"

    def _error(self) -> float | None:
        d = self.status.distance
        return float(d) if np.isfinite(d) else None

    def _note(self) -> str:
        st = self.status
        if self.task.name == "reach":
            txt = f"do celu {st.distance * 1000:.0f} mm" + (" (w celu)" if st.success else "")
            return txt + (" - cel przyciety do obszaru z treningu" if st.goal_clamped else "")
        if self.task.target:
            txt = f"kostka do celu {st.distance * 1000:.0f} mm" + (" (niesiona)" if st.carried else "")
            return txt + (" - cel przesuniety na obszar z treningu" if st.goal_clamped else "")
        if self._cube is None:
            return ""
        height = 100 * (float(self._cube[2]) - self.task.cube_half)
        return f"kostka {height:.1f} cm nad blatem ({self._cube_src or 'bez zrodla'})"

"""Zadania RL blizniaka: co polityka widzi, co robi i za co dostaje nagrode.

Jedna definicja dla trzech miejsc, w ktorych polityka zyje:

* srodowisko Gymnasium na CPU (`env.TwinEnv`, numpy, jedno srodowisko),
* srodowisko wsadowe na GPU (`batch.BatchEnv`, torch, tysiace swiatow MuJoCo Warp),
* petla na prawdziwym ramieniu (`strategy.PolicyStrategy` w `StrategyRunner` blizniaka, numpy, katy z serw).

Funkcje ponizej biora tablice z wymiarem wsadu `(N, ...)` i modul `xp` - numpy
albo torch - wiec wszystkie trzy licza DOKLADNIE te same wzory. Dwie kopie
"prawie tej samej" obserwacji to klasyczne ciche zrodlo sim-2-real: polityka
uczy sie na jednej, a jezdzi na drugiej.

Wszystkie polozenia sa w ukladzie PODSTAWY ramienia (blat na z = 0), katy
stawow w radianach MJCF, w kolejnosci `RobotSpec.joints` (chwytak ostatni).

Akcja ma `n_joints` liczb w [-1, 1] (6 dla SO-101: piec stawow ramienia i chwytak):

* stawy ramienia - PRZYROST celu o najwyzej `arm_step` rad na takt, liczony od
  poprzedniego celu (jak `runtime.Twin.set_target`), przyciety do tych samych
  limitow, ktore stosuje `SafetySupervisor` na prawdziwym ramieniu;
* chwytak - cel BEZWZGLEDNY (-1 zamkniety, +1 otwarty), ograniczony co do
  predkosci tak, jak ogranicza go nadzor (`grip_step` zakresu na takt).

Zadania: `reach` (TCP do punktu), `lift` (chwyt i podniesienie kostki) i `pickplace`
(kostka przeniesiona na cel na blacie i puszczona). W pickplace sukcesem jest tylko
kostka NIESIONA (zatrzask `carry_latch`: chwycona i uniesiona ponad `lift_height`),
potem lezaca na celu, puszczona i nieruchoma - dopchniecie jej po blacie sie nie liczy.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from ..kinematics import RobotKinematics
from ..robot import RobotSpec

TASKS = ("reach", "lift", "pickplace")


@dataclass(frozen=True)
class TaskConfig:
    name: str = "reach"
    #: Czestotliwosc polityki i krok fizyki [s]. 20 Hz przy 5 ms to 10 krokow na akcje.
    control_hz: float = 20.0
    timestep: float = 0.005
    episode_steps: int = 100
    #: Najwiekszy przyrost celu stawu ramienia na takt [rad] (0,05 przy 20 Hz = 57 st/s).
    arm_step: float = 0.05
    #: Najwiekszy ruch celu chwytaka na takt, jako czesc zakresu (0,15 przy 20 Hz =
    #: 300 jednostek/s - tyle, ile przepuszcza nadzor bezpieczenstwa).
    grip_step: float = 0.15
    #: reach: sukces, gdy TCP jest blizej celu niz tyle [m].
    success_dist: float = 0.02
    #: reach: skad losowany jest cel - promien od osi podstawy i wysokosc nad blatem [m].
    goal_radius: tuple[float, float] = (0.13, 0.32)
    goal_height: tuple[float, float] = (0.03, 0.22)
    #: lift: kostka (polowa boku [m], masa [kg]) i gdzie lezy na blacie.
    cube_half: float = 0.015
    cube_mass: float = 0.03
    cube_radius: tuple[float, float] = (0.15, 0.25)
    cube_bearing: tuple[float, float] = (-0.8, 0.8)
    #: lift: sukces, gdy spod kostki jest wyzej nad blatem niz tyle [m].
    #: pickplace: kostka chwycona wyzej niz tyle jest "niesiona" (`carry_latch`).
    lift_height: float = 0.06
    #: Start ramienia: poza domowa plus/minus tyle stopni na kazdym stawie ramienia.
    start_noise_deg: float = 8.0
    #: Kara za szarpanie: waga ||a - a_poprzednie||^2.
    action_rate: float = 0.02
    #: Ile taktow sukcesu Z RZEDU konczy epizod (0 = nigdy). Tak samo zatrzymuje sie
    #: runner na ramieniu ("zadanie wykonane"). Bez tego lift-v2 po chwycie jechal
    #: dalej: podstawa -105 st., lokiec -92, nadgarstek +95 i -150 (wszystkie limity),
    #: kostka 37 cm nad blatem zamiast 6 - i trzymal tak do konca epizodu (8,5 s).
    end_on_success: int = 0
    #: Kara za cel stawu ramienia przy limicie: waga * sum((1 - zapas/limit_margin)^2)
    #: po stawach blizej limitu niz `limit_margin` [rad]. Nagroda nie mowila nic o
    #: postawie, wiec nic nie trzymalo polityki z dala od limitow.
    limit_penalty: float = 0.0
    limit_margin: float = 0.15
    #: Kara za ruch ramienia, gdy zadanie jest juz wykonane: waga * ||a_ramie||^2.
    #: Seria `end_on_success` ma byc trzymaniem kostki, a nie wymachem z nia.
    hold_still: float = 0.0
    #: Kara za obrot nadgarstka od pozy domowej: waga * ((q_roll - dom) / pi)^2. W reach
    #: nagroda liczy tylko polozenie TCP, a obrot nadgarstka go nie zmienia - reach-v2
    #: krecila nim do limitu +150 st. w kazdym epizodzie (643 z 2000 taktow przy limicie).
    roll_penalty: float = 0.0
    #: Liczba stawow ramienia razem z chwytakiem (chwytak ostatni) - wymiar akcji.
    #: Polityki zapisane przed tym polem to SO-101, czyli 6.
    n_joints: int = 6
    #: pickplace: kostka "na celu", gdy jej srodek jest blizej srodka celu (w xy) niz
    #: `place_radius` [m], a spod kostki nizej nad blatem niz `place_height` [m].
    place_radius: float = 0.03
    place_height: float = 0.01
    #: pickplace: cel co najmniej tyle [m] (w xy) od kostki na starcie.
    target_min_sep: float = 0.10
    #: pickplace: kostka "lezy", gdy jej predkosc jest mniejsza niz tyle [m/s].
    settle_speed: float = 0.02
    #: pickplace: skad losowany jest cel na blacie - jak kostka (`make_task(spec=)` bierze
    #: pierscien kostki ze `spec.desk`).
    target_radius: tuple[float, float] = (0.15, 0.25)
    target_bearing: tuple[float, float] = (-0.8, 0.8)

    @property
    def obs_dim(self) -> int:
        return obs_dim(self.name, self.n_joints)

    @property
    def act_dim(self) -> int:
        return self.n_joints

    @property
    def substeps(self) -> int:
        return int(round(1.0 / self.control_hz / self.timestep))

    @property
    def cube(self) -> str | None:
        return "cube" if self.name in ("lift", "pickplace") else None

    @property
    def target(self) -> str | None:
        """Nazwa ciala celu odlozenia (podkladka mocap w scenie) albo None."""
        return "target" if self.name == "pickplace" else None


def obs_dim(name: str, n: int) -> int:
    """q(n) + cel stawow(n) + TCP(3) + [cel(3) + cel-TCP(3)] + [kostka(3) + kostka-TCP(3) + obrot(6)]
    + [pickplace: cel odlozenia(3) + cel-kostka(3)] + akcja(n)."""
    return 2 * n + 3 + (6 if name == "reach" else 12) + (6 if name == "pickplace" else 0) + n


#: Wymiary obserwacji SO-101 (6 stawow) - uklad polityk bazowych.
OBS_DIMS = {name: obs_dim(name, 6) for name in TASKS}


def make_task(name: str = "reach", spec: RobotSpec | None = None, **overrides) -> TaskConfig:
    """Zadanie `name`; ze `spec` - w skali tego ramienia (liczba stawow, pierscienie celow i kostek)."""
    if name not in TASKS:
        raise ValueError(f"nieznane zadanie {name!r}; dostepne: {', '.join(TASKS)}")
    base = TaskConfig(name=name, episode_steps=100 if name == "reach" else 200)
    if name == "reach":
        base = replace(base, roll_penalty=0.5)
    if name == "lift":
        base = replace(base, end_on_success=10, limit_penalty=0.5, hold_still=0.5)
    if name == "pickplace":
        base = replace(base, episode_steps=300, end_on_success=10, limit_penalty=0.5, hold_still=0.5)
    if spec is not None:
        d = spec.desk
        base = replace(base, n_joints=len(spec.joints), goal_radius=d.goal_radius, goal_height=d.goal_height,
                       cube_radius=d.cube_radius, cube_bearing=d.cube_bearing,
                       target_radius=d.cube_radius, target_bearing=d.cube_bearing)
    return replace(base, **overrides)


# --------------------------------------------------------------- limity
@dataclass(frozen=True)
class Limits:
    """Zakres celow stawow [rad]: przeciecie limitow nadzoru z zakresami MJCF.

    Limity nadzoru to `limits` z `robot.yaml` - te same, ktore blizniak bierze domyslnie
    (`robot_config` bez pliku konfiguracji).
    """

    lo: np.ndarray
    hi: np.ndarray
    home: np.ndarray
    #: Liczba stawow ramienia (bez chwytaka) - to one ida przyrostem `arm_step`.
    n_arm: int
    #: Indeks chwytaka (ostatni staw).
    grip: int
    #: Indeks stawu obrotu narzedzia (`spec.roll_joint`) - kara `roll_penalty`; None = brak.
    roll: int | None

    @staticmethod
    def of(kin: RobotKinematics) -> Limits:
        spec = kin.spec
        if spec.gripper != spec.joints[-1]:
            raise ValueError("zadania RL zakladaja chwytak jako ostatni staw")
        lo = kin.to_q({n: spec.limits[n][0] for n in spec.joints})
        hi = kin.to_q({n: spec.limits[n][1] for n in spec.joints})
        lo, hi = np.maximum(lo, kin.lo), np.minimum(hi, kin.hi)
        grip = len(spec.joints) - 1
        roll = spec.joints.index(spec.roll_joint) if spec.roll_joint else None
        return Limits(lo, hi, np.clip(kin.to_q(spec.home), lo, hi), n_arm=grip, grip=grip, roll=roll)


# -------------------------------------------------------- numpy / torch
def _cat(xp, parts):
    return xp.concatenate(parts, axis=-1) if xp is np else xp.cat(parts, dim=-1)


def _clip(xp, x, lo, hi):
    return np.clip(x, lo, hi) if xp is np else xp.clamp(x, lo, hi)


def _norm(x):
    return ((x * x).sum(-1)) ** 0.5


def _as(xp, a, like):
    """Stala numpy jako tablica tego samego rodzaju i urzadzenia co `like`."""
    return np.asarray(a, like.dtype) if xp is np else xp.as_tensor(a, dtype=like.dtype, device=like.device)


# ---------------------------------------------------------------- akcja
def apply_action(xp, task: TaskConfig, limits: Limits, q_cmd, action):
    """Nowy cel stawow (N, n_joints) z poprzedniego celu i akcji w [-1, 1]."""
    na = limits.n_arm
    lo, hi = _as(xp, limits.lo, q_cmd), _as(xp, limits.hi, q_cmd)
    a = _clip(xp, action, -1.0, 1.0)
    arm = _clip(xp, q_cmd[..., :na] + a[..., :na] * task.arm_step, lo[:na], hi[:na])
    span = hi[na:] - lo[na:]
    want = lo[na:] + (a[..., na:] + 1.0) * 0.5 * span
    step = task.grip_step * span
    grip = q_cmd[..., na:] + _clip(xp, want - q_cmd[..., na:], -step, step)
    return _cat(xp, [arm, grip])


# ------------------------------------------------------------ obserwacja
def normalize_q(xp, limits: Limits, q):
    lo, hi = _as(xp, limits.lo, q), _as(xp, limits.hi, q)
    return 2.0 * (q - lo) / (hi - lo) - 1.0


def observe(xp, task: TaskConfig, limits: Limits, q, q_cmd, tcp, prev_action, goal=None,
            cube_pos=None, cube_rot=None, target=None):
    """Wektor obserwacji (N, obs_dim). `cube_rot` to macierze obrotu (N, 3, 3).

    `target` (N, 3) - pickplace: cel odlozenia na blacie (z = 0) w ukladzie podstawy.
    """
    parts = [normalize_q(xp, limits, q), normalize_q(xp, limits, q_cmd), tcp]
    if task.name == "reach":
        parts += [goal, goal - tcp]
    else:
        rot6 = _cat(xp, [cube_rot[..., :, 0], cube_rot[..., :, 1]])
        parts += [cube_pos, cube_pos - tcp, rot6]
        if task.name == "pickplace":
            parts += [target, target - cube_pos]
    parts.append(prev_action)
    return _cat(xp, parts)


def fold_yaw(xp, R):
    """Obrot wokol z zlozony do +-45 st. - tak widzi kostke detektor (symetria 90 st.)."""
    yaw = xp.arctan2(R[..., 1, 0], R[..., 0, 0]) if xp is np else xp.atan2(R[..., 1, 0], R[..., 0, 0])
    yaw = (yaw + np.pi / 4) % (np.pi / 2) - np.pi / 4
    c, s = xp.cos(yaw), xp.sin(yaw)
    z, o = xp.zeros_like(c), xp.ones_like(c)
    rows = [xp.stack([c, -s, z], -1), xp.stack([s, c, z], -1), xp.stack([z, z, o], -1)]
    return xp.stack(rows, -2)


# ---------------------------------------------------------------- nagroda
def limit_cost(xp, task: TaskConfig, limits: Limits, q_cmd):
    """(N,) kara za cele stawow ramienia blizej limitu niz `limit_margin` (0 w srodku zakresu)."""
    na = limits.n_arm
    lo, hi = _as(xp, limits.lo[:na], q_cmd), _as(xp, limits.hi[:na], q_cmd)
    room = xp.minimum(q_cmd[..., :na] - lo, hi - q_cmd[..., :na])
    near = _clip(xp, 1.0 - room / task.limit_margin, 0.0, 1.0)
    return task.limit_penalty * (near * near).sum(-1)


def _float(xp, b, like):
    return b.to(like.dtype) if xp is not np else b.astype(like.dtype)


def grasped(xp, jaw_contacts):
    """(N,) czy kostka dotyka obu szczek naraz (`jaw_contacts` (N, 2))."""
    return (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)


def carry_latch(xp, task: TaskConfig, carried, jaw_contacts, cube_pos):
    """(N,) zatrzask "kostka byla niesiona": chwycona obiema szczekami wyzej nad blatem niz `lift_height`.

    Raz ustawiony trzyma do resetu (wywolujacy zeruje go na starcie epizodu). Liczony
    PRZED nagroda i do niej podawany - bez niego kostka dopchnieta po blacie na cel
    bylaby "odlozona", a `reward` zostaje funkcja bez stanu.
    """
    height = cube_pos[..., 2] - task.cube_half
    return carried | (grasped(xp, jaw_contacts) & (height > task.lift_height))


def grip_is_open(xp, limits: Limits, q_cmd):
    """(N,) czy rozkaz chwytaka jest w otwartej polowie zakresu (puszczenie kostki)."""
    g = limits.grip
    lo, hi = float(limits.lo[g]), float(limits.hi[g])
    return q_cmd[..., g] > lo + 0.5 * (hi - lo)


def reward(xp, task: TaskConfig, tcp, action, prev_action, goal=None, cube_pos=None, jaw_contacts=None,
           q_cmd=None, limits: Limits | None = None, target=None, cube_vel=None, carried=None,
           grip_open=None):
    """(nagroda, sukces, porazka) dla kazdego swiata.

    `jaw_contacts` (N, 2): czy kostka dotyka szczeki stalej i ruchomej.
    Porazka = kostka spadla ze stolu; epizod sie wtedy konczy.
    `q_cmd` (N, n_joints) i `limits` - do kary za limity (`limit_penalty`); bez nich jej nie ma.

    pickplace: `target` (N, 3) cel na blacie, `cube_vel` (N, 3) predkosc kostki [m/s],
    `carried` (N,) zatrzask z `carry_latch` (policzony przed nagroda), `grip_open` (N,)
    rozkaz chwytaka otwarty (domyslnie z `q_cmd` i `limits`, patrz `grip_is_open`).
    """
    cost = task.action_rate * ((action - prev_action) ** 2).sum(-1)
    if task.limit_penalty > 0 and q_cmd is not None and limits is not None:
        cost = cost + limit_cost(xp, task, limits, q_cmd)
    if task.roll_penalty > 0 and q_cmd is not None and limits is not None and limits.roll is not None:
        k = limits.roll
        roll = (q_cmd[..., k] - float(limits.home[k])) / np.pi
        cost = cost + task.roll_penalty * roll * roll
    if task.name == "reach":
        d = _norm(goal - tcp)
        # Dwie skale: zgrubna ciagnie z daleka, dokladna nagradza ostatnie milimetry.
        r = 0.5 * (1.0 - xp.tanh(d / 0.10)) + 0.5 * (1.0 - xp.tanh(d / 0.01))
        success = d < task.success_dist
        failure = d < -1.0                                  # w reach nie ma porazki
        return r - cost, success, failure
    if task.name == "pickplace":
        return _reward_pickplace(xp, task, tcp, action, cost, cube_pos, jaw_contacts, limits, target,
                                 cube_vel, carried,
                                 grip_open if grip_open is not None else grip_is_open(xp, limits, q_cmd))
    d = _norm(cube_pos - tcp)
    grasped = (jaw_contacts[..., 0] > 0.5) & (jaw_contacts[..., 1] > 0.5)
    height = cube_pos[..., 2] - task.cube_half
    lifted = _clip(xp, height / task.lift_height, 0.0, 1.0)
    success = grasped & (height > task.lift_height)
    g = grasped.to(tcp.dtype) if xp is not np else grasped.astype(tcp.dtype)
    s = success.to(tcp.dtype) if xp is not np else success.astype(tcp.dtype)
    r = (1.0 - xp.tanh(d / 0.05)) + g + 4.0 * g * lifted + 2.0 * s
    if task.hold_still > 0:
        na = action.shape[-1] - 1 if limits is None else limits.n_arm
        cost = cost + task.hold_still * s * (_clip(xp, action[..., :na], -1.0, 1.0) ** 2).sum(-1)
    failure = height < -0.05
    return r - cost, success, failure


def _reward_pickplace(xp, task: TaskConfig, tcp, action, cost, cube_pos, jaw_contacts, limits, target,
                      cube_vel, carried, opened):
    """pickplace: podejdz, chwyc, unies, przenies nad cel, odloz, pusc.

    Sukces = byla niesiona (`carried`) & lezy na celu (`placed`) & szczeki jej nie
    trzymaja, a rozkaz chwytaka jest otwarty (`released`) & stoi (`still`). Nagroda
    za "na celu" tez tylko po niesieniu - inaczej dopychanie po blacie by sie oplacalo.
    """
    d = _norm(cube_pos - tcp)
    g = grasped(xp, jaw_contacts)
    height = cube_pos[..., 2] - task.cube_half
    lifted = _clip(xp, height / task.lift_height, 0.0, 1.0)
    d_t = _norm(cube_pos[..., :2] - target[..., :2])
    placed = (d_t < task.place_radius) & (height < task.place_height)
    released = ~g & opened
    still = _norm(cube_vel) < task.settle_speed
    success = carried & placed & released & still
    gf, cf = _float(xp, g, tcp), _float(xp, carried, tcp)
    pf, sf = _float(xp, placed, tcp), _float(xp, success, tcp)
    r = ((1.0 - xp.tanh(d / 0.05)) + gf + 2.0 * gf * lifted + 4.0 * cf * (1.0 - xp.tanh(d_t / 0.10))
         + 4.0 * cf * pf + 4.0 * sf)
    if task.hold_still > 0:
        na = action.shape[-1] - 1 if limits is None else limits.n_arm
        cost = cost + task.hold_still * sf * (_clip(xp, action[..., :na], -1.0, 1.0) ** 2).sum(-1)
    failure = height < -0.05
    return r - cost, success, failure


def success_streak(xp, task: TaskConfig, streak, success):
    """(nowa seria sukcesow z rzedu, czy konczy epizod) - to samo w env, batch i runnerze.

    Koniec epizodu po serii jest "ucieciem" (jak limit czasu), a nie stanem koncowym:
    PPO dolicza mu wartosc stanu. Gdyby liczyl zero, polityka nauczylaby sie NIE konczyc
    - kostka trzymana tuz ponizej `lift_height` daje ~6 na takt przez reszte epizodu,
    a zakonczenie tylko 8 x 10 taktow.
    """
    streak = xp.where(success, streak + 1, streak * 0)
    done = streak >= task.end_on_success if task.end_on_success > 0 else success & ~success
    return streak, done


# ---------------------------------------------------------------- starty
def sample_goals(kin: RobotKinematics, task: TaskConfig, rng: np.random.Generator, n: int) -> np.ndarray:
    """Cele `reach` (n, 3) w ukladzie podstawy - na pewno osiagalne.

    Losujemy w przestrzeni STAWOW i bierzemy TCP z kinematyki prostej, jak fala
    kalibracyjna: kazdy cel jest osiagalny z definicji, a filtr trzyma go nad
    stolem i przed ramieniem.
    """
    spec = kin.spec
    out = []
    ranges = {**{j: (np.degrees(lo), np.degrees(hi)) for j, lo, hi in zip(spec.joints, kin.lo, kin.hi)},
              **spec.wave_ranges}
    while len(out) < n:
        j = {name: float(rng.uniform(*ranges[name])) for name in spec.arm_joints}
        p = kin.tcp(j)[:3, 3]
        r = float(np.hypot(p[0], p[1]))
        if task.goal_radius[0] <= r <= task.goal_radius[1] and task.goal_height[0] <= p[2] <= task.goal_height[1] \
                and p[0] > 0.05:
            out.append(p)
    return np.array(out)


def sample_start(task: TaskConfig, limits: Limits, rng: np.random.Generator, n: int) -> np.ndarray:
    """Katy startowe (n, n_joints): poza domowa z szumem, chwytak otwarty w zadaniach z kostka."""
    na, g = limits.n_arm, limits.grip
    q = np.repeat(limits.home[None], n, axis=0)
    q[:, :na] += np.radians(rng.uniform(-task.start_noise_deg, task.start_noise_deg, (n, na)))
    if task.cube:
        span = limits.hi[g] - limits.lo[g]
        q[:, g] = limits.lo[g] + rng.uniform(0.6, 1.0, n) * span
    return np.clip(q, limits.lo, limits.hi)


def sample_cubes(task: TaskConfig, rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Kostki na blacie: polozenia (n, 3) w ukladzie podstawy i kwaterniony (n, 4) obrotu wokol z."""
    r = rng.uniform(*task.cube_radius, n)
    b = rng.uniform(*task.cube_bearing, n)
    pos = np.stack([r * np.cos(b), r * np.sin(b), np.full(n, task.cube_half)], axis=1)
    yaw = rng.uniform(-np.pi, np.pi, n)
    quat = np.stack([np.cos(yaw / 2), np.zeros(n), np.zeros(n), np.sin(yaw / 2)], axis=1)
    return pos, quat


#: Ile razy `sample_targets` losuje od nowa cel za blisko kostki, zanim wezmie zapasowy.
TARGET_TRIES = 64


def sample_targets(task: TaskConfig, rng: np.random.Generator, cube_pos: np.ndarray) -> np.ndarray:
    """Cele pickplace (n, 3) na blacie (z = 0), co najmniej `target_min_sep` (xy) od kostek `cube_pos` (n, 3).

    Losowanie biegunowe w `target_radius` / `target_bearing`, jak kostki; za bliskie
    wiersze losujemy od nowa (tylko je), najwyzej `TARGET_TRIES` razy. Wiersz, ktory
    wciaz nie pasuje, dostaje cel na zewnetrznym promieniu po drugiej stronie pierscienia.
    Blizniak torch: `batch.BatchEnv._sample_targets` - ta sama regula.
    """
    cube = np.asarray(cube_pos, float).reshape(-1, 3)
    n = len(cube)
    out = np.zeros((n, 3))
    bad = np.ones(n, bool)
    for _ in range(TARGET_TRIES):
        k = int(bad.sum())
        if k == 0:
            break
        r = rng.uniform(*task.target_radius, k)
        b = rng.uniform(*task.target_bearing, k)
        out[bad, 0], out[bad, 1] = r * np.cos(b), r * np.sin(b)
        bad = np.hypot(*(out[:, :2] - cube[:, :2]).T) <= task.target_min_sep
    if bad.any():
        cb = np.arctan2(cube[bad, 1], cube[bad, 0])
        lo, hi = task.target_bearing
        far = np.where(cb > 0.5 * (lo + hi), lo, hi)
        r = task.target_radius[1]
        out[bad, 0], out[bad, 1] = r * np.cos(far), r * np.sin(far)
    return out

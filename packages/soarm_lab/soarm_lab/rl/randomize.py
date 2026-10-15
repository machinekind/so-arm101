"""Domain randomization: how much the simulation may differ from the arm on the desk.

The ranges are multipliers around the model's NOMINAL values. The nominal need not be the MuJoCo
Menagerie model: `Station.dynamics` holds the servo identification the twin measured on the real arm,
and `Randomization.around` centres the ranges on it, so randomization covers the uncertainty of a
measurement instead of a guess. This is "training calibration": the policy trains on a spread around
the measured arm.

There is no promise that the real arm surely lies within this spread. Measured on synthetic
recordings with known dynamics (full excitation, kp 0.7-1.3, friction 0.6-1.4; the twin's
identification tests): after identification the true damping, armature and delay lie within the
ranges around the result, while kp and dry friction, which the motion does not determine, lie within
them only thanks to the wider `UNFITTED_RANGES`. An arm with kp 0.5 or friction x3 would be outside.

The twin identifies the dynamics. The result arrives here as data: the `dynamics` field of the
exported `station.json` (`Dynamics.from_dict(station.dynamics)`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

#: Multiplier ranges for the parameters the identification did NOT fit (`Dynamics.fitted`), when
#: randomization is centred on its result. The result leaves them at the model's value (1.0), and the
#: fitted damping, armature and delay compensate for their true value: with a true kp of 0.8 the
#: result had armature -13 %, so it is good only TOGETHER with the kp the arm really has. The default
#: kp range 0.85-1.15 around 1.0 does not contain the true kp of 0.7 / 0.75 / 0.8 / 1.2 / 1.3 from
#: the synthetic sweep: training would not see a single world with such dynamics. The twin's
#: identification measures the uncertainty of its result on exactly these ranges.
UNFITTED_RANGES: dict[str, tuple[float, float]] = {"kp": (0.6, 1.5), "frictionloss": (0.5, 2.0)}

#: The widest band of a fitted multiplier that `Randomization.around` carries over to a range
#: (1 - b, 1 + b). More makes no sense: a band of +-inf (e.g. armature x0.45 from a fit that
#: explained nothing) would give multipliers <= 0, and the centre of such a result is wrong anyway;
#: the twin does not save such a result as dynamics.
MAX_BAND = 0.6
#: The largest extra delay [ticks] taken from the delay band, for the same reason.
MAX_DELAY_BAND = 2.0
#: The multiplier parameters that `around` widens by their band.
MULTIPLIERS = ("kp", "damping", "armature", "frictionloss")


@dataclass
class Dynamics:
    """Identified arm dynamics: multipliers relative to the Menagerie model."""

    kp: float = 1.0                 # servo position gain
    damping: float = 1.0            # joint damping
    armature: float = 1.0           # rotor inertia
    frictionloss: float = 1.0       # dry friction
    #: Delay from a command to the servo's motion [policy ticks].
    delay: float = 0.0
    #: Where these numbers come from: "menagerie" (nominal) or a description of the measurement.
    source: str = "menagerie"
    #: The identification's fit error [deg], if there was one.
    fit_deg: float | None = None
    #: The parameters the identification REALLY fitted (the rest stayed at the model's values).
    #: Empty = nothing was measured (Menagerie, or dynamics written without this field).
    fitted: tuple[str, ...] = ()
    #: Uncertainty of the fitted parameters from the twin's identification (the result's band):
    #: multipliers relative (0.1 = +-10 %), the delay in policy TICKS (like `delay`); inf = the
    #: measurement says nothing about the parameter. Empty = no measurement (Menagerie, or dynamics
    #: written without this field). Kept here because `Randomization.around` widens the training
    #: ranges by it: with the default +-25 % a centre with a band of +-inf (armature x0.45) gives
    #: worlds of 0.34-0.59 while the truth is 1.0.
    band: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fitted"] = list(self.fitted)                 # JSON has no tuples
        # inf as "inf": json.dumps writes Infinity, which strict JSON parsers (a browser's) reject.
        out["band"] = {k: (float(v) if np.isfinite(v) else "inf") for k, v in self.band.items()}
        return out

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> Dynamics:
        data = dict(data or {})
        data["fitted"] = tuple(data.get("fitted") or ())
        data["band"] = {str(k): float(v) for k, v in (data.get("band") or {}).items()}
        return Dynamics(**data)

    def is_fitted(self, name: str) -> bool:
        """Whether `name` comes from the measurement (and not from the Menagerie model)."""
        return name in self.fitted


@dataclass
class Randomization:
    kp: tuple[float, float] = (0.85, 1.15)
    damping: tuple[float, float] = (0.75, 1.3)
    armature: tuple[float, float] = (0.75, 1.3)
    frictionloss: tuple[float, float] = (0.6, 1.5)
    cube_mass: tuple[float, float] = (0.6, 1.6)
    cube_friction: tuple[float, float] = (0.7, 1.3)
    #: Joint angle measurement noise [deg]; the servo encoder has 0.088 deg per tick.
    obs_noise_deg: float = 0.3
    #: Action delay: drawn from {min_delay .. max_delay} ticks per episode.
    max_delay: int = 1
    #: The cube as the twin's camera perception gives it, not as physics knows it: the position from
    #: {0 .. cube_delay} ticks ago, refreshed every {1 .. cube_period} ticks, with noise [m], and the
    #: rotation folded to +-45 deg (the cube has a 90 deg symmetry). Measured: a policy trained on the
    #: truth lifted 7 of 20 cubes with a 10 Hz / 150 ms perception.
    cube_delay: int = 3
    cube_period: int = 3
    cube_noise: float = 0.002
    fold_yaw: bool = True
    #: The centre of the ranges: the nominal from identification, by default the Menagerie model.
    centre: Dynamics = field(default_factory=Dynamics)
    #: The smallest action delay [ticks]; without randomization = max_delay = the measured one.
    min_delay: int = 0

    @staticmethod
    def none() -> Randomization:
        """No randomization, the Menagerie model, ideal perception: for tests and deterministic checks."""
        one = (1.0, 1.0)
        return Randomization(one, one, one, one, one, one, obs_noise_deg=0.0, max_delay=0, cube_delay=0,
                             cube_period=1, cube_noise=0.0, fold_yaw=False)

    @staticmethod
    def nominal(dyn: Dynamics | None = None) -> Randomization:
        """The arm as identification measured it: no spread, no noise, ideal perception.

        This is "CPU without randomization" (`cpu_nominal` in `meta.evals`, and `evaluate`'s
        default). It is not `none()`: after identification the Menagerie model would say nothing
        about the measured arm.
        """
        dyn = dyn or Dynamics()
        d = int(round(dyn.delay))
        return replace(Randomization.none(), centre=dyn, min_delay=d, max_delay=d)

    @staticmethod
    def around(dyn: Dynamics, spread: float = 1.0) -> Randomization:
        """Ranges around the measured dynamics; `spread` scales their width.

        `spread = 0` is training "without randomization": exactly the measured dynamics and delay,
        but the cube perception model (delay, refresh, noise, symmetry) STAYS: it is not an
        uncertainty of the model, it is what the cameras give on the desk. A lift policy trained
        with `none()` (the Menagerie model and the cube from physics) lifted 1 of 8 cubes from the
        cameras.

        After identification (`dyn.fitted` not empty) the parameters it did NOT fit (kp, dry
        friction) get the wide `UNFITTED_RANGES` instead of the defaults: their 1.0 is the model, not
        a measurement, and the rest of the result is good only together with their true value.

        Fitted parameters with a band (`dyn.band`) get a range of at least (1 - b, 1 + b) around the
        centre (b at most `MAX_BAND`), the delay up to `delay + b` ticks (at most `MAX_DELAY_BAND`
        more). A fixed +-25 % does not cover a +-35 % damping band that identification reports, and
        ignores a +-inf band entirely. `spread` scales the range AFTER the widening, so `spread = 0`
        still means "no spread".
        """
        base = Randomization()
        spread = max(0.0, float(spread))
        if dyn.fitted:
            base = replace(base, **{k: r for k, r in UNFITTED_RANGES.items() if k not in dyn.fitted})
        band = dyn.band or {}
        for k in MULTIPLIERS:
            b = band.get(k)
            if k in dyn.fitted and b is not None and not np.isnan(b):
                b = min(float(b), MAX_BAND)
                lo, hi = getattr(base, k)
                base = replace(base, **{k: (min(lo, 1.0 - b), max(hi, 1.0 + b))})

        def widen(r):
            return (1.0 - (1.0 - r[0]) * spread, 1.0 + (r[1] - 1.0) * spread)

        out = replace(base, kp=widen(base.kp), damping=widen(base.damping), armature=widen(base.armature),
                      frictionloss=widen(base.frictionloss), cube_mass=widen(base.cube_mass),
                      cube_friction=widen(base.cube_friction), centre=dyn)
        if spread > 0:
            hi = int(np.ceil(dyn.delay)) + 1
            b = band.get("delay")
            if "delay" in dyn.fitted and b is not None and not np.isnan(b):
                hi = max(hi, int(np.ceil(dyn.delay + spread * min(float(b), MAX_DELAY_BAND))))
            out.min_delay, out.max_delay = 0, hi
        else:
            out.min_delay = out.max_delay = int(round(dyn.delay))
        return out

    @property
    def randomized(self) -> bool:
        """Whether the dynamics are sampled (and not only set to the centre)."""
        ranges = (self.kp, self.damping, self.armature, self.frictionloss, self.cube_mass, self.cube_friction)
        return any(hi > lo for lo, hi in ranges) or self.max_delay > self.min_delay

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        """Draws for `n` worlds: multipliers, delays and the cube's refresh period [ticks].

        The arm's dynamics multipliers are already multiplied by the centre.
        """
        c = self.centre
        return {
            "kp": c.kp * rng.uniform(*self.kp, n),
            "damping": c.damping * rng.uniform(*self.damping, n),
            "armature": c.armature * rng.uniform(*self.armature, n),
            "frictionloss": c.frictionloss * rng.uniform(*self.frictionloss, n),
            "cube_mass": rng.uniform(*self.cube_mass, n),
            "cube_friction": rng.uniform(*self.cube_friction, n),
            "delay": rng.integers(min(self.min_delay, self.max_delay), self.max_delay + 1, n),
            "cube_delay": rng.integers(0, self.cube_delay + 1, n),
            "cube_period": rng.integers(1, max(1, self.cube_period) + 1, n),
        }

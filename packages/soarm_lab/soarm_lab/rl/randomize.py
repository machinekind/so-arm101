"""Randomizacja dziedziny: jak bardzo symulacja moze sie roznic od ramienia na biurku.

Zakresy to mnozniki wokol wartosci NOMINALNYCH modelu. Nominal nie musi byc
modelem z MuJoCo Menagerie: `Station.dynamics` trzyma wynik identyfikacji
serw zmierzony na prawdziwym ramieniu (`rl.sysid`), a `Randomization.around`
stawia zakresy wokol niego - wtedy randomizacja pokrywa niepewnosc pomiaru,
a nie zgadywanie. To jest "kalibracja treningu": polityka uczy sie na
rozrzucie wokol zmierzonego ramienia.

Nie ma tu obietnicy, ze prawdziwe ramie na pewno lezy w tym rozrzucie.
Zmierzone na nagraniach syntetycznych o znanej dynamice (pelne pobudzenie,
kp 0,7-1,3, tarcie 0,6-1,4 - patrz `tests/test_twin_sysid.py`): po
identyfikacji tlumienie, armatura i opoznienie prawdy leza w zakresach wokol
wyniku, a kp i tarcie suche - ktorych ruch nie wyznacza - tylko dzieki
szerszym zakresom `UNFITTED_RANGES`. Ramie z kp 0,5 albo tarciem x3 byloby
poza nimi.

`Dynamics` i `UNFITTED_RANGES` to kopia: identyfikacja (`rl.sysid`) zostaje w blizniaku
razem ze swoimi. Wynik identyfikacji przychodzi tu jako dane - `station.json`
eksportu, pole `dynamics` (`Dynamics.from_dict(station.dynamics)`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

#: Zakresy mnoznikow parametrow, ktorych identyfikacja NIE dopasowala (`Dynamics.fitted`),
#: gdy randomizacja stoi wokol jej wyniku. Wynik identyfikacji zostawia je na modelu
#: (1,0), a dopasowane tlumienie, armatura i opoznienie kompensuja ich prawdziwa wartosc:
#: przy prawdzie kp 0,8 wynik mial armature -13 % - czyli jest dobry tylko RAZEM z kp,
#: ktore naprawde ma ramie. Domyslne kp 0,85-1,15 wokol 1,0 nie zawieralo prawd kp
#: 0,7 / 0,75 / 0,8 / 1,2 / 1,3 z przegladu - trening nie widzial ani jednego swiata
#: z taka dynamika. `rl.sysid` mierzy niepewnosc wyniku wlasnie na tych zakresach.
UNFITTED_RANGES: dict[str, tuple[float, float]] = {"kp": (0.6, 1.5), "frictionloss": (0.5, 2.0)}

#: Najszersze pasmo dopasowanego mnoznika, jakie `Randomization.around` przenosi na zakres
#: (1 - b, 1 + b). Wiecej nie ma sensu: pasmo +-inf (s6 weryfikatora: armatura x0,45 z
#: dopasowania, ktore nic nie wyjasnilo) dawaloby mnozniki <= 0, a srodek takiego wyniku
#: i tak jest zly - takiego wyniku nie zapisuje sie jako dynamiki (`FitResult.useful`).
MAX_BAND = 0.6
#: Najwieksze dodatkowe opoznienie [takty] z pasma opoznienia - z tego samego powodu.
MAX_DELAY_BAND = 2.0
#: Parametry-mnozniki, ktore `around` rozszerza wedlug pasma.
MULTIPLIERS = ("kp", "damping", "armature", "frictionloss")


@dataclass
class Dynamics:
    """Zidentyfikowana dynamika ramienia - mnozniki wzgledem modelu Menagerie."""

    kp: float = 1.0                 # wzmocnienie pozycyjne serw
    damping: float = 1.0            # tlumienie w stawach
    armature: float = 1.0           # bezwladnosc wirnika
    frictionloss: float = 1.0       # tarcie suche
    #: Opoznienie od rozkazu do ruchu serwa [takty polityki].
    delay: float = 0.0
    #: Skad te liczby: "menagerie" (nominal) albo opis pomiaru.
    source: str = "menagerie"
    #: Blad dopasowania identyfikacji [st.], jesli byla.
    fit_deg: float | None = None
    #: Parametry, ktore identyfikacja NAPRAWDE dopasowala (reszta zostala z modelu).
    #: Puste = nic nie mierzono (Menagerie albo dynamika sprzed tego pola).
    fitted: tuple[str, ...] = ()
    #: Niepewnosc dopasowanych parametrow z identyfikacji (`sysid.FitResult.band`): mnozniki
    #: wzglednie (0,1 = +-10 %), opoznienie w TAKTACH polityki (jak `delay`); inf = pomiar
    #: nic o parametrze nie mowi. Puste = brak pomiaru (Menagerie albo stara dynamika).
    #: Trzymane tu, bo `Randomization.around` rozszerza wedlug niego zakresy treningu -
    #: wczesniej pasmo zylo tylko w opisie `source`, a srodek z pasmem +-inf dostawal
    #: zwykle +-25 % (armatura x0,45 -> swiaty 0,34-0,59 przy prawdzie 1,0).
    band: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["fitted"] = list(self.fitted)                 # JSON workspace'u nie ma krotek
        # inf jako "inf": json.dumps pisze Infinity, ktorego scisly JSON (przegladarka) nie czyta.
        out["band"] = {k: (float(v) if np.isfinite(v) else "inf") for k, v in self.band.items()}
        return out

    @staticmethod
    def from_dict(data: dict[str, Any] | None) -> Dynamics:
        data = dict(data or {})
        data["fitted"] = tuple(data.get("fitted") or ())
        data["band"] = {str(k): float(v) for k, v in (data.get("band") or {}).items()}
        return Dynamics(**data)

    def is_fitted(self, name: str) -> bool:
        """Czy `name` pochodzi z pomiaru (a nie z modelu Menagerie)."""
        return name in self.fitted


@dataclass
class Randomization:
    kp: tuple[float, float] = (0.85, 1.15)
    damping: tuple[float, float] = (0.75, 1.3)
    armature: tuple[float, float] = (0.75, 1.3)
    frictionloss: tuple[float, float] = (0.6, 1.5)
    cube_mass: tuple[float, float] = (0.6, 1.6)
    cube_friction: tuple[float, float] = (0.7, 1.3)
    #: Szum pomiaru katow [st.] - enkoder serwa ma 0,088 st. na tik.
    obs_noise_deg: float = 0.3
    #: Opoznienie akcji: losowane z {0 .. max_delay} taktow na epizod.
    max_delay: int = 1
    #: Percepcja kostki tak, jak ja daje `perception` z kamer, a nie fizyka: polozenie
    #: sprzed {0 .. cube_delay} taktow, odswiezane co {1 .. cube_period} taktow, z szumem
    #: [m] i z obrotem zlozonym do +-45 st. (kostka ma symetrie 90 st.). Zmierzone:
    #: polityka uczona na prawdzie z 10 Hz / 150 ms percepcji podnosila 7 z 20 kostek.
    cube_delay: int = 3
    cube_period: int = 3
    cube_noise: float = 0.002
    fold_yaw: bool = True
    #: Srodek zakresow - nominal z identyfikacji, domyslnie model Menagerie.
    centre: Dynamics = field(default_factory=Dynamics)
    #: Najmniejsze opoznienie akcji [takty]; bez randomizacji = max_delay = zmierzone.
    min_delay: int = 0

    @staticmethod
    def none() -> Randomization:
        """Bez randomizacji, model Menagerie, idealna percepcja - do testow i porownan deterministycznych."""
        one = (1.0, 1.0)
        return Randomization(one, one, one, one, one, one, obs_noise_deg=0.0, max_delay=0, cube_delay=0,
                             cube_period=1, cube_noise=0.0, fold_yaw=False)

    @staticmethod
    def nominal(dyn: Dynamics | None = None) -> Randomization:
        """Ramie takie, jak je zmierzyla identyfikacja - bez rozrzutu, bez szumu, idealna percepcja.

        To jest "CPU bez randomizacji" z panelu. Wczesniej bylo nim `none()`, czyli
        model Menagerie - po identyfikacji ta liczba nie mowila nic o zmierzonym ramieniu.
        """
        dyn = dyn or Dynamics()
        d = int(round(dyn.delay))
        return replace(Randomization.none(), centre=dyn, min_delay=d, max_delay=d)

    @staticmethod
    def around(dyn: Dynamics, spread: float = 1.0) -> Randomization:
        """Zakresy wokol zmierzonej dynamiki; `spread` skaluje ich szerokosc.

        `spread = 0` to trening "bez randomizacji": dokladnie zmierzona dynamika i jej
        opoznienie, ale model percepcji kostki (opoznienie, odswiezanie, szum, symetria)
        ZOSTAJE - to nie jest niepewnosc modelu, tylko to, co kamery daja na biurku.
        Wczesniej "bez randomizacji" bralo `none()`: model Menagerie zamiast identyfikacji
        i kostke z fizyki, na ktorej polityka lift podnosila z kamer 1 z 8.

        Po identyfikacji (`dyn.fitted` niepuste) parametry, ktorych NIE dopasowala (kp,
        tarcie suche), dostaja szerokie `UNFITTED_RANGES` zamiast domyslnych: ich 1,0 to
        model, a nie pomiar, a reszta wyniku jest dobra tylko razem z ich prawdziwa wartoscia.

        Dopasowane parametry z pasmem (`dyn.band`) dostaja zakres co najmniej (1 - b, 1 + b)
        wokol srodka (b najwyzej `MAX_BAND`), opoznienie - do `delay + b` taktow (najwyzej
        `MAX_DELAY_BAND` wiecej). Stale +-25 % nie pokrywalo pasma +-35 % tlumienia, ktore
        identyfikacja sama zglaszala (s17 weryfikatora), a pasma +-inf nie widzialo wcale.
        `spread` skaluje zakres PO rozszerzeniu, wiec `spread = 0` dalej znaczy "bez rozrzutu".
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
        """Czy dynamika jest losowana (a nie tylko ustawiona na srodek)."""
        ranges = (self.kp, self.damping, self.armature, self.frictionloss, self.cube_mass, self.cube_friction)
        return any(hi > lo for lo, hi in ranges) or self.max_delay > self.min_delay

    def sample(self, rng: np.random.Generator, n: int) -> dict[str, np.ndarray]:
        """Mnozniki dla `n` swiatow (juz przemnozone przez srodek)."""
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

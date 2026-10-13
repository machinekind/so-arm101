"""Polityki DOSTARCZANE z pakietem (`soarm_lab/policies`) na CPU - to one pojada na prawdziwym ramieniu.

Z testow blizniaka (`tests/test_twin_policies.py`); scena to stanowisko `sim` z pakietu.

Testy RL sprawdzaja srodowisko i nagrode; tu sprawdzamy zachowanie samych plikow
polityk, ktore operator dostaje w panelu. Przeglad zmierzyl na starej `lift-v2`:
kostka konczyla 13-22 cm nad blatem (zadanie: 6 cm), stawy ramienia przez
kilkadziesiat taktow przy limitach i wymachy po chwycie do 72 st. - a zaden test
tego nie widzial, bo sprawdzal tylko "czy podniosla". Progi ponizej przepuszczaja
`lift-v3` (zmierzone: kostka 8-14 cm, 6 taktow przy limicie po chwycie, ruch po
pierwszym sukcesie do 46 st.) i zatrzymuja `lift-v2` (git show
ee44e21:assets/policies/lift-v2/policy.pt): kostka srednio 16-18 cm, do 22 cm,
24-56 taktow przy limicie po chwycie, ruch po sukcesie do 72 st.

Epizod na CPU trwa ~0,07 s, wiec 50 epizodow to kilka sekund.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

pytest.importorskip("mujoco")
pytest.importorskip("torch")

from soarm_lab.rl.env import TwinEnv  # noqa: E402
from soarm_lab.rl.policy import Policy, bundled_dir  # noqa: E402
from soarm_lab.rl.randomize import Randomization  # noqa: E402

REACH = bundled_dir() / "reach-v3" / "policy.pt"
LIFT = bundled_dir() / "lift-v3" / "policy.pt"
#: "Przy limicie" = rozkaz stawu blizej niz 1 st. od limitu (`task.Limits`).
NEAR_LIMIT = np.radians(1.0)


@dataclass
class Episode:
    q_cmd: np.ndarray            # (T, 6) rozkazy [rad]
    success: list[bool]
    grasped: list[bool]
    info: dict
    end: float                   # reach: odleglosc TCP od celu [m]; lift: wysokosc kostki nad blatem [m]

    def near_limit(self, limits, joints=slice(0, 5)) -> np.ndarray:
        q = self.q_cmd[:, joints]
        return (np.abs(q - limits.lo[joints]) < NEAR_LIMIT) | (np.abs(q - limits.hi[joints]) < NEAR_LIMIT)

    def first(self, flags: list[bool]) -> int | None:
        return next((i for i, f in enumerate(flags) if f), None)


def run(policy: Policy, rand: Randomization, episodes: int, seed: int = 10_000) -> tuple[list[Episode], object]:
    task = policy.task
    env = TwinEnv(task, randomization=rand)
    out = []
    try:
        for ep in range(episodes):
            obs, _ = env.reset(seed=seed + ep)
            qs, succ, grasp = [], [], []
            for _ in range(task.episode_steps):
                obs, _, term, trunc, info = env.step(policy.act(obs))
                qs.append(env.q_cmd.copy())
                succ.append(info["success"])
                grasp.append(info.get("grasped", False))
                if term or trunc:
                    break
            end = info["distance"] if task.name == "reach" else info["height"]
            out.append(Episode(np.array(qs), succ, grasp, info, end))
        return out, env.limits
    finally:
        env.close()


def lift_report(policy: Policy, rand: Randomization, episodes: int) -> dict:
    """Liczby, na ktorych przeglad zlapal `lift-v2` - wspolne dla testu i sprawdzenia starej polityki."""
    eps, limits = run(policy, rand, episodes)
    heights = np.array([e.end for e in eps])
    limit_after_grasp, swing = 0, 0.0
    for e in eps:
        g, s = e.first(e.grasped), e.first(e.success)
        if g is not None:
            limit_after_grasp += int(e.near_limit(limits)[g:].any(axis=1).sum())
        if s is not None:
            after = e.q_cmd[s:, :5]
            swing = max(swing, float(np.degrees(after.max(0) - after.min(0)).max()))
    return {"finished": sum(e.info["finished"] and e.info["success"] for e in eps), "episodes": len(eps),
            "height_min": heights.min(), "height_mean": heights.mean(), "height_max": heights.max(),
            "limit_ticks_after_grasp": limit_after_grasp, "swing_after_success_deg": swing}


def check_lift(rep: dict, lift_height: float) -> None:
    # Konczy sama: seria sukcesow (`end_on_success`), nie limit czasu - i kostka jest w gorze.
    assert rep["finished"] == rep["episodes"], rep
    assert rep["height_min"] > lift_height, rep
    # Bez wymachiwania kostka pod sufit: lift-v2 srednio 16-18 cm, do 22 cm.
    assert rep["height_max"] < 0.16 and rep["height_mean"] < 0.135, rep
    # Stawy ramienia po chwycie nie siedza na limitach (lift-v2: 24 i 56 taktow).
    assert rep["limit_ticks_after_grasp"] <= 12, rep
    # Po pierwszym sukcesie ramie tylko trzyma i konczy (lift-v2: do 72 st. ruchu stawu).
    # Blizniak mial tu 55 st. Scena z eksportu rozni sie od jego sceny zaokragleniem XML (~1e-6),
    # a chwyt to kontakty: trajektorie zgodne do 1e-8 rozchodza sie przy chwycie i maksimum
    # z 30 epizodow trafia na inny epizod. Zmierzone na czterech blokach po 30 ziaren
    # (10/20/30/40 tys.): blizniak 44-54 st., laboratorium 26-59 st., p90 w obu 25,8 st.
    # 60 st. dalej odroznia lift-v2 (72 st.).
    assert rep["swing_after_success_deg"] < 60.0, rep


def test_both_base_policies_are_bundled():
    assert REACH.is_file() and LIFT.is_file()
    assert REACH.with_name("meta.json").is_file() and LIFT.with_name("meta.json").is_file()


def test_shipped_lift_lifts_ends_and_keeps_the_arm_calm():
    pol = Policy.load(LIFT)
    check_lift(lift_report(pol, Randomization(), 30), pol.task.lift_height)


def test_shipped_reach_ends_near_the_goal():
    pol = Policy.load(REACH)
    eps, limits = run(pol, Randomization.nominal(), 20)
    dist = np.array([e.end for e in eps])
    assert all(e.info["success"] for e in eps)
    assert dist.max() < 0.005 and np.median(dist) < 0.003, dist           # zmierzone: mediana 1,8 mm, max 2,9 mm
    # Zaden staw ramienia nie jedzie na limit - takze obrot nadgarstka. Obrot nie zmienia TCP,
    # wiec nagroda reach go nie widziala: reach-v2 krecila nim do +150 st. w kazdym epizodzie
    # (729 z 2000 taktow przy limicie). reach-v3 uczona z `roll_penalty`: mediana 13 st., max 48.
    near = sum(int(e.near_limit(limits).sum()) for e in eps)
    assert near == 0, near
    roll = np.degrees(np.abs(np.concatenate([e.q_cmd[:, 4] for e in eps]) - limits.home[4]))
    assert roll.max() < 90.0 and np.median(roll) < 30.0, (np.median(roll), roll.max())

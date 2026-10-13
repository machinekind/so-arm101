"""Wiersz polecen laboratorium: `soarm-lab <polecenie>` (trening, ewaluacja i polityki z `lerobot-twin`).

    soarm-lab train --task reach                     # polityka PPO na GPU (MuJoCo Warp) + ewaluacja na CPU
    soarm-lab train --task lift --station ~/desk1    # ... na stanowisku z `lerobot-twin export ~/desk1`
    soarm-lab train --task pickplace                 # ... kostka przeniesiona na cel na blacie
    soarm-lab eval lift-v3 --rand                    # ewaluacja polityki w zwyklym MuJoCo
    soarm-lab policies                               # polityki bazowe i uczone tutaj, z wynikami
    soarm-lab convert old/policy.pt                  # punkt kontrolny torcha -> policy.npz

`--station` to folder eksportu blizniaka (`scene.xml`, `assets/`, `station.json`) albo nazwa
stanowiska dostarczanego z pakietem (`sim`, domyslnie). `eval` i `policies` potrzebuja tylko numpy.
`train` to dawny trener w torchu: potrzebuje dodatkow `train` (`mujoco-warp`) i `torch` (torch z CUDA).
`convert` potrzebuje dodatku `torch`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _station(a: argparse.Namespace):
    from .station import Station

    return Station.load(a.station)


def _train_plan(dyn, no_rand: bool, spread: float):
    """(randomizacja treningu, [(etykieta, klucz w meta.evals, randomizacja ewaluacji)]).

    "Bez randomizacji" to zmierzona dynamika bez rozrzutu - nie model Menagerie
    (`Randomization.none()` gubil identyfikacje i model percepcji kostki, a log
    i tak pisal "dynamika: identyfikacja"). Ewaluacja "z randomizacja" jest tylko
    wtedy, gdy trening ja mial: wczesniej przy --no-rand druga ewaluacja nominalna
    trafiala do `cpu_rand` i panel pokazywal ja jako odpornosc na randomizacje.
    """
    from .rl.randomize import Randomization

    rand = Randomization.around(dyn, 0.0 if no_rand else spread)
    evals = [("bez randomizacji", "cpu_nominal", Randomization.nominal(dyn))]
    if rand.randomized:
        evals.append(("z randomizacja", "cpu_rand", rand))
    return rand, evals


def _describe(rand) -> str:
    c = rand.centre
    delay = f"{rand.min_delay}" if rand.min_delay == rand.max_delay else f"{rand.min_delay}-{rand.max_delay}"
    return (f"dynamika: {c.source} (kp x{c.kp:.2f}, tlumienie x{c.damping:.2f}, opoznienie {delay} takt.); "
            f"randomizacja: {'tak' if rand.randomized else 'brak'}; "
            f"percepcja kostki: {'jak z kamer' if rand.fold_yaw or rand.cube_delay else 'idealna'}")


def _train(a: argparse.Namespace) -> int:
    import json
    import time

    from .paths import policies_dir
    from .rl.evaluate import evaluate
    from .rl.policy import POLICY_FILE
    from .rl.ppo import PPOConfig, train
    from .rl.randomize import Dynamics

    st = _station(a)
    name = a.name or f"{a.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(a.out or policies_dir()) / name
    dyn = Dynamics.from_dict(st.dynamics)
    rand, evals = _train_plan(dyn, a.no_rand, a.spread)
    cfg = PPOConfig(num_envs=a.envs, iterations=a.iters, seed=a.seed)
    init = None
    if a.init:
        from .rl.policy import Policy, find_policy

        init = Policy.load(find_policy(a.init))
        a.task = init.task.name
        cfg.init_std = 0.25                            # douczanie: mniejsza eksploracja na starcie
        cfg.critic_warmup = 30                         # dla polityk bez zapisanego krytyka (patrz ppo)
        print(f"Start z polityki {a.init} ({init.task.name})")
    print(f"Trening {a.task} na stanowisku {st.folder}: {a.envs} swiatow x {a.iters} iteracji -> {out}")
    print(f"  {_describe(rand)}{'' if a.no_rand else f', rozrzut x{a.spread}'}")

    def show(p):
        if p.iteration == 1 or p.iteration % 10 == 0 or p.status != "uczenie":
            print(f"  it {p.iteration:4d}/{p.iterations}  {p.steps / 1e6:7.2f} M krokow  {p.fps / 1e3:5.0f} k/s  "
                  f"sukces {p.success:5.1%}  nagroda {p.reward:7.2f}  {p.elapsed:5.0f} s", flush=True)

    stop = None
    if a.stop_file:
        # Zatrzymanie z zewnatrz: plik - polityka z tego miejsca zostaje zapisana.
        import threading

        stop = threading.Event()
        flag = Path(a.stop_file)

        def watch():
            while not stop.is_set():
                if flag.exists():
                    stop.set()
                time.sleep(0.5)
        threading.Thread(target=watch, daemon=True).start()
    pol = train(a.task, cfg, station=st, randomization=rand, out_dir=out, on_progress=show, stop=stop,
                init=init, device=a.device)
    print("Ewaluacja na CPU (zwykle MuJoCo, inny silnik niz w treningu):")
    for label, key, r in evals:
        res = evaluate(pol, a.eval_episodes, randomization=r, station=st)
        pol.meta.evals[key] = res
        rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}
        print(f"  {label:17s} sukces {res['success']:5.1%}  ({json.dumps(rounded)})")
    print(f"Zapisano {pol.save(out / POLICY_FILE)}")
    return 0


def _eval(a: argparse.Namespace) -> int:
    from .rl.evaluate import evaluate
    from .rl.policy import Policy, find_policy
    from .rl.randomize import Dynamics, Randomization

    st = _station(a)
    path = find_policy(a.policy)
    pol = Policy.load(path)
    # Randomizacja wokol zmierzonej dynamiki - jak w treningu, nie wokol Menagerie.
    rand = Randomization.around(Dynamics.from_dict(st.dynamics)) if a.rand else None
    res = evaluate(pol, a.episodes, randomization=rand, station=st, seed=a.seed)
    print(f"{path}: {res}")
    return 0


def _policies(a: argparse.Namespace) -> int:
    from .paths import policies_dir
    from .rl.policy import list_policies, torch_checkpoints

    items = list_policies(a.dir or policies_dir())
    if not items:
        print("Brak polityk. Naucz pierwsza:  soarm-lab train --task reach")

    def pct(key):
        v = p["evals"].get(key, {}).get("success")
        return "  -  " if v is None else f"{v:5.1%}"

    for p in items:
        where = "bazowa" if p["bundled"] else "uczona"
        print(f"  {p['name']:32s} {p['task']:9s} {where:6s} GPU {p['success'] or 0:5.1%}  "
              f"CPU bez/z rand. {pct('cpu_nominal')} / {pct('cpu_rand')}  "
              f"{p['steps'] / 1e6:6.1f} M  {p['created']}")
    for pt in torch_checkpoints(a.dir or policies_dir()):
        print(f"  {pt.parent.name:32s} tylko policy.pt (torch): soarm-lab convert {pt}")
    return 0


def _convert(a: argparse.Namespace) -> int:
    from .rl.torch_policy import convert

    out = convert(a.checkpoint, a.out)
    print(f"Zapisano {out} i {out.with_name('meta.json')}")
    return 0


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="soarm-lab", description="Trening i ewaluacja polityk SO-101.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    station_help = ("stanowisko: folder z `lerobot-twin export` albo nazwa dostarczonego z pakietem "
                    "(domyslnie sim)")

    p = sub.add_parser("train", help="naucz polityke PPO na GPU (MuJoCo Warp)")
    p.add_argument("--task", choices=("reach", "lift", "pickplace"), default="reach")
    p.add_argument("--envs", type=int, default=4096, help="ile swiatow naraz")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="nazwa przebiegu (domyslnie zadanie + data)")
    p.add_argument("--out", default=None, help="katalog polityk (domyslnie workspace/policies)")
    p.add_argument("--station", default="sim", help=station_help)
    p.add_argument("--no-rand", action="store_true", help="bez randomizacji dziedziny")
    p.add_argument("--spread", type=float, default=1.0, help="szerokosc randomizacji wokol zmierzonej dynamiki")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--stop-file", default=None, help="przerwij trening, gdy ten plik sie pojawi")
    p.add_argument("--init", default=None, help="douczanie: start z tej polityki (nazwa albo policy.npz)")
    p.add_argument("--device", default="cuda:0", help="urzadzenie torcha i MuJoCo Warp (domyslnie cuda:0)")
    p.set_defaults(fn=_train)

    p = sub.add_parser("eval", help="ewaluacja polityki na CPU (zwykle MuJoCo)")
    p.add_argument("policy", help="nazwa polityki (np. lift-v3) albo sciezka do policy.npz")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--rand", action="store_true", help="z randomizacja dziedziny")
    p.add_argument("--seed", type=int, default=10_000, help="ziarno pierwszego epizodu")
    p.add_argument("--station", default="sim", help=station_help)
    p.set_defaults(fn=_eval)

    p = sub.add_parser("policies", help="polityki bazowe i uczone, z wynikami")
    p.add_argument("--dir", default=None, help="katalog polityk (domyslnie workspace/policies)")
    p.set_defaults(fn=_policies)

    p = sub.add_parser("convert", help="punkt kontrolny torcha (policy.pt) jako policy.npz (dodatek torch)")
    p.add_argument("checkpoint", help="policy.pt albo jego folder")
    p.add_argument("--out", default=None, help="plik policy.npz (domyslnie obok policy.pt; meta.json obok niego)")
    p.set_defaults(fn=_convert)
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

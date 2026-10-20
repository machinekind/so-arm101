"""The lab's command line: `soarm-lab <command>` (training, evaluation and policies, on twin stations).

    soarm-lab train --task reach                     # a PPO policy on the GPU (MuJoCo Warp) + CPU evaluation
    soarm-lab train --task lift --station ~/desk1    # ... on the station from `twin export ~/desk1`
    soarm-lab train --task pickplace                 # ... the cube carried onto a target on the table
    soarm-lab eval lift-v3 --rand                    # evaluate a policy in plain MuJoCo
    soarm-lab policies                               # bundled policies and those trained here, with results
    soarm-lab convert old/policy.pt                  # a torch checkpoint -> policy.npz
    soarm-lab dream models                           # Dream Labs: the checkpoints the account may run
    soarm-lab dream sim --prompt "pick up the cube"  # Dream-1 in the simulation
    soarm-lab dream arm --prompt "pick up the cube" --camera scene=0 --camera wrist=1   # on the arm

`--station` is a folder the twin exported (`scene.xml`, `assets/`, `station.json`) or the name of a
station bundled with the package (`sim`, the default). `eval` and `policies` need no torch. `train` is
the legacy torch trainer: it needs the extras `train` (`mujoco-warp`) and `torch` (torch with CUDA).
`convert` needs the `torch` extra. `dream` needs the `dream` extra and a Dream Labs API key (`dream_run`).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _station(a: argparse.Namespace):
    from .station import Station

    return Station.load(a.station)


def _train_plan(dyn, no_rand: bool, spread: float):
    """(training randomization, [(label, key in meta.evals, evaluation randomization)]).

    Training without randomization (`--no-rand`) is the measured dynamics with zero spread, not
    `Randomization.none()`: that is the Menagerie model with ideal perception, which would drop both the
    identified dynamics and the cube perception model. The randomized evaluation runs only when training
    was randomized: otherwise a second nominal evaluation would land in `cpu_rand`, and the panel would
    show it as robustness to randomization.
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
        cfg.init_std = 0.25                            # fine-tuning: less exploration at the start
        cfg.critic_warmup = 30                         # for policies saved without a critic (see ppo)
        print(f"Start z polityki {a.init} ({init.task.name})")
    print(f"Trening {a.task} na stanowisku {st.folder}: {a.envs} swiatow x {a.iters} iteracji -> {out}")
    print(f"  {_describe(rand)}{'' if a.no_rand else f', rozrzut x{a.spread}'}")

    def show(p):
        if p.iteration == 1 or p.iteration % 10 == 0 or p.status != "uczenie":
            print(f"  it {p.iteration:4d}/{p.iterations}  {p.steps / 1e6:7.2f} M krokow  {p.fps / 1e3:5.0f} k/s  "
                  f"sukces {p.success:5.1%}  nagroda {p.reward:7.2f}  {p.elapsed:5.0f} s", flush=True)

    stop = None
    if a.stop_file:
        # Stopping from outside: a file. The policy as it is at that point is saved.
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
    # Randomization around the station's measured dynamics, as in training, not around Menagerie.
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


def _dream(a: argparse.Namespace) -> int:
    from .dream_run import main as dream_main

    return dream_main(a)


def _dream_parser(sub) -> None:
    """`soarm-lab dream models|sim|arm`: Dream-1 on the SO-101 without the twin (`dream_run`)."""
    p = sub.add_parser("dream", help="Dream Labs' Dream-1 model on the SO-101 (the dream extra)",
                       description="Dream-1 on the SO-101 without the twin. Needs `uv sync --extra dream` and a "
                                   "Dream Labs API key: DREAM_API_KEY or ~/.dreamlabs/credentials.")
    dream_sub = p.add_subparsers(dest="dream_cmd", required=True)
    dream_sub.add_parser("models", help="the checkpoints the account may run").set_defaults(fn=_dream)

    def common(q: argparse.ArgumentParser, speed: float) -> None:
        q.add_argument("--prompt", required=True, help="the task, in plain language; a line typed while it runs "
                                                       "is a new prompt")
        q.add_argument("--checkpoint", default=None,
                       help="the model, an id `soarm-lab dream models` lists (default dreamlabs/dream-1-beta)")
        q.add_argument("--speed", type=float, default=speed,
                       help=f"fraction of robot.yaml's max_vel the command may move at (default {speed:g})")
        q.add_argument("--max-step-deg", type=float, default=None,
                       help="most a joint target moves per tick at 20 Hz [deg] (default 4)")
        q.add_argument("--fill", choices=("scene", "black"), default="scene",
                       help="what fills a camera slot no camera has: the dummy left wrist, and the right wrist "
                            "without --camera wrist=... (default scene: the scene frame again)")
        q.add_argument("--seconds", type=float, default=None, help="end the run after this long")
        q.add_argument("--record", type=Path, default=None, help="write the camera views to this .mp4")
        q.set_defaults(fn=_dream)

    q = dream_sub.add_parser("sim", help="Dream-1 in the lab's simulation, a cube on the table")
    common(q, 1.0)
    q.add_argument("--station", default="sim", help="a folder from `twin export` or a bundled station (sim)")
    q.add_argument("--scene-camera", default=None, help="the station camera in the scene slot (default: the first)")
    q.add_argument("--wrist-camera", default=None,
                   help="the camera in the wrist slot (default robot/wrist_cam, on the arm); '' for none")

    q = dream_sub.add_parser("arm", help="Dream-1 on the real arm (the feetech backend, OpenCV cameras)")
    common(q, 0.3)
    q.add_argument("--port", default=None, help="the arm's serial port (default: the one USB-serial device)")
    q.add_argument("--baudrate", type=int, default=1_000_000)
    q.add_argument("--camera", action="append", default=[], metavar="ROLE=SOURCE",
                   help="scene=0 (required) and wrist=1 (optional): an OpenCV camera index or device path")
    q.add_argument("--camera-size", default="640x360", help="the size asked of the cameras (default 640x360)")


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="soarm-lab", description="Trening i ewaluacja polityk SO-101.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    station_help = ("stanowisko: folder z `twin export` albo nazwa dostarczonego z pakietem "
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

    _dream_parser(sub)
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

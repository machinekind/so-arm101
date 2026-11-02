"""The lab's command line: `soarm-lab <command>` (training, evaluation and policies, on twin stations).

    soarm-lab train task=reach                      # train PPO and evaluate on CPU
    soarm-lab train task=lift station=$HOME/stations/desk1        # train on an exported station
    soarm-lab train experiment=lift_finetune         # fine-tune a bundled policy
    soarm-lab train --cfg job                        # show the composed configuration
    soarm-lab eval lift-v3 --rand                    # evaluate a policy in plain MuJoCo
    soarm-lab policies                               # bundled policies and those trained here, with results
    soarm-lab convert old/policy.pt                  # a torch checkpoint -> policy.npz
    soarm-lab dream models                           # Dream Labs: the checkpoints the account may run
    soarm-lab dream sim --prompt "pick up the cube"  # Dream-1 in the simulation
    soarm-lab dream arm --prompt "pick up the cube" --camera scene=0 --camera wrist=1   # on the arm

`station` for training and `--station` for evaluation select an exported station folder
(`scene.xml`, `assets/`, `station.json`) or a bundled station name (`sim` by default). `eval` and `policies` need neither jax nor torch.
`train` is PPO in JAX on MuJoCo Warp: it needs the `train` extra (jax, on Linux its CUDA build, optax,
mujoco-warp) and no torch. `convert` needs the `torch` extra. `dream` needs the `dream` extra and a Dream
Labs API key (`dream_run`).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _station(a: argparse.Namespace):
    from .station import Station

    return Station.load(a.station)


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
        print("Brak polityk. Naucz pierwsza:  soarm-lab train task=reach")

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
    q.add_argument("--rotate", action="append", default=[], metavar="ROLE=DEG",
                   help="turn a camera's frames clockwise, for one mounted upside down: scene=180, wrist=90")


def main(argv: list[str] | None = None) -> int:
    import sys

    args = list(sys.argv[1:] if argv is None else argv)
    if args and args[0] == "train":
        from .rl.experiment import run

        return run(args[1:])
    a = parser().parse_args(args)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="soarm-lab", description="Trening i ewaluacja polityk SO-101.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    station_help = ("stanowisko: folder z `twin export` albo nazwa dostarczonego z pakietem "
                    "(domyslnie sim)")

    p = sub.add_parser("train", help="naucz polityke PPO na GPU (MuJoCo Warp)")

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

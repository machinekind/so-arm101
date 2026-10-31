"""The lab's command line: `soarm-lab <command>` (training, evaluation and policies, on twin stations).

    soarm-lab train --task reach                     # a PPO policy on the GPU (MuJoCo Warp) + CPU evaluation
    soarm-lab train --task lift --station ~/desk1    # ... on the station from `twin export ~/desk1`
    soarm-lab train --task pickplace                 # ... the cube carried onto a target on the table
    soarm-lab train --init lift-v3 --kinds cube,cuboid5,cuboid85,cylinder   # one policy for four blocks
    soarm-lab eval lift-v3 --rand                    # evaluate a policy in plain MuJoCo
    soarm-lab eval <block policy> --kind cylinder    # ... on one of the blocks it was trained on
    soarm-lab policies                               # bundled policies and those trained here, with results
    soarm-lab convert old/policy.pt                  # a torch checkpoint -> policy.npz
    soarm-lab dream models                           # Dream Labs: the checkpoints the account may run
    soarm-lab dream sim --prompt "pick up the cube"  # Dream-1 in the simulation
    soarm-lab dream arm --prompt "pick up the cube" --camera scene=0 --camera wrist=1   # on the arm
    soarm-lab grasps meshes --out workspace/grasps/meshes   # the blocks as OBJ, for GraspGenX
    soarm-lab grasps convert cube workspace/grasps/raw/cube.yml   # GraspGenX's isaac_grasp YAML -> a library
    soarm-lab grasps show cube                       # the grasps of a library

`--station` is a folder the twin exported (`scene.xml`, `assets/`, `station.json`) or the name of a
station bundled with the package (`sim`, the default). `eval` and `policies` need neither jax nor torch.
`train` is PPO in JAX on MuJoCo Warp: it needs the `train` extra (jax, on Linux its CUDA build, optax,
mujoco-warp) and no torch. `convert` needs the `torch` extra. `dream` needs the `dream` extra and a Dream
Labs API key (`dream_run`).

`grasps` prepares the inputs of GraspGenX and reads its output into the grasp libraries the `grasp` strategy
uses (`grasp_library`, tools/graspgenx/README.md); GraspGenX itself never runs here.

`train --kinds` trains one shape-conditioned policy on several wooden blocks (`rl.objects`): the task
observes the block's descriptor, the environment is a `rl.mixed.MixedBatchEnv` with an equal share of
the worlds per kind, `--init lift-v3` widens the cube policy to the new inputs (`rl.ppo`), and the CPU
evaluation runs once per kind (`meta.evals["cpu_<kind>"]`, `["cpu_rand_<kind>"]`).
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


def _kind_arg(text: str) -> str:
    """An object kind (`--kind`), checked against the object table."""
    from .rl.objects import KINDS

    if text not in KINDS:
        raise argparse.ArgumentTypeError(f"unknown object kind {text!r}; known: {', '.join(KINDS)}")
    return text


def _kinds_arg(text: str) -> tuple[str, ...]:
    """Comma-separated object kinds (`--kinds cube,cylinder`), each known and none twice."""
    kinds = tuple(_kind_arg(k.strip()) for k in text.split(",") if k.strip())
    if not kinds:
        raise argparse.ArgumentTypeError("no object kind given")
    if len(set(kinds)) != len(kinds):
        raise argparse.ArgumentTypeError(f"an object kind repeats in {text!r}")
    return kinds


def _train_task(a: argparse.Namespace, st, init):
    """The task to train: `init`'s own, else `--task`'s, with `--kinds` and `--episode-steps` on top.

    `--kinds` makes it shape-conditioned. Its `object_kind` is the policy's own kind (`cpu_nominal`
    and `cpu_rand` evaluate it, the strategy uses it by default): the base task's when it is among
    the kinds, else the first of them.
    """
    from dataclasses import replace

    from .rl.task import make_task

    base = init.task if init is not None else make_task(a.task, spec=st.spec())
    over = {}
    if a.kinds:
        if base.cube is None:
            raise ValueError(f"--kinds needs a task with an object (lift, pickplace), not {base.name}")
        over["shape_conditioned"] = True
        if base.object_kind not in a.kinds:
            over["object_kind"] = a.kinds[0]
    if a.episode_steps:
        over["episode_steps"] = a.episode_steps
    return replace(base, **over)


def _kind_key(key: str, kind: str) -> str:
    """The per-kind key of an evaluation in `meta.evals`: `cpu_nominal` -> `cpu_<kind>`, `cpu_rand` ->
    `cpu_rand_<kind>`."""
    return f"cpu_{kind}" if key == "cpu_nominal" else f"{key}_{kind}"


def _kinds_summary(evals: dict) -> str:
    """For a policy evaluated per object kind: the mean and the worst kind's CPU success, else ''.

    The randomized evaluations when there are any, else the nominal ones.
    """
    from .rl.objects import KINDS

    for key, label in (("cpu_rand", "z rand."), ("cpu_nominal", "bez rand.")):
        got = {k: evals[_kind_key(key, k)].get("success") for k in KINDS if _kind_key(key, k) in evals}
        got = {k: v for k, v in got.items() if v is not None}
        if got:
            worst = min(got, key=got.__getitem__)
            mean = sum(got.values()) / len(got)
            return f"klocki {label} srednio {mean:5.1%}, najslabszy {worst} {got[worst]:5.1%}"
    return ""


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
    dyn = Dynamics.from_dict(st.dynamics)
    rand, evals = _train_plan(dyn, a.no_rand, a.spread)
    cfg = PPOConfig(num_envs=a.envs, iterations=a.iters, seed=a.seed)
    init, init_name = None, ""
    if a.init:
        from .rl.policy import Policy, find_policy

        path = find_policy(a.init)
        init, init_name = Policy.load(path), path.parent.name
        a.task = init.task.name
        cfg.init_std = 0.25                            # fine-tuning: less exploration at the start
        cfg.critic_warmup = 30                         # for policies saved without a critic (see ppo)
        print(f"Start z polityki {a.init} ({init.task.name})")
    task = _train_task(a, st, init)
    make_env = None
    if a.kinds:
        from functools import partial

        from .rl.mixed import MixedBatchEnv

        # One MixedBatchEnv even for one kind: the same environment and the same info["kind"].
        make_env = partial(MixedBatchEnv, kinds=a.kinds)
    name = a.name or f"{a.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(a.out or policies_dir()) / name
    blocks = f" (klocki: {', '.join(a.kinds)})" if a.kinds else ""
    print(f"Trening {a.task}{blocks} na stanowisku {st.folder}: {a.envs} swiatow x {a.iters} iteracji -> {out}")
    print(f"  {_describe(rand)}{'' if a.no_rand else f', rozrzut x{a.spread}'}")

    def show(p):
        if p.iteration == 1 or p.iteration % 10 == 0 or p.status != "uczenie":
            worst = ""
            if p.success_by_kind:
                kind = min(p.success_by_kind, key=p.success_by_kind.__getitem__)
                worst = f"  najslabszy {kind} {p.success_by_kind[kind]:5.1%}"
            print(f"  it {p.iteration:4d}/{p.iterations}  {p.steps / 1e6:7.2f} M krokow  {p.fps / 1e3:5.0f} k/s  "
                  f"sukces {p.success:5.1%}{worst}  nagroda {p.reward:7.2f}  {p.elapsed:5.0f} s", flush=True)

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
    pol = train(task, cfg, station=st, randomization=rand, out_dir=out, on_progress=show, stop=stop,
                init=init, device=a.device, make_env=make_env, init_name=init_name)

    def report(label, res):
        rounded = {k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()}
        print(f"  {label:27s} sukces {res['success']:5.1%}  ({json.dumps(rounded)})")

    print("Ewaluacja na CPU (zwykle MuJoCo, inny silnik niz w treningu):")
    for label, key, r in evals:                        # the policy's own kind
        pol.meta.evals[key] = res = evaluate(pol, a.eval_episodes, randomization=r, station=st)
        report(label, res)
    for kind in a.kinds or ():
        for label, key, r in evals:
            res = pol.meta.evals[key] if kind == task.object_kind else \
                evaluate(pol, a.eval_episodes, randomization=r, station=st, kind=kind)
            pol.meta.evals[_kind_key(key, kind)] = res
            report(f"{kind} {label}", res)
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
    res = evaluate(pol, a.episodes, randomization=rand, station=st, seed=a.seed, kind=a.kind)
    print(f"{path}{f' ({a.kind})' if a.kind else ''}: {res}")
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
        blocks = _kinds_summary(p["evals"])
        print(f"  {p['name']:32s} {p['task']:9s} {where:6s} GPU {p['success'] or 0:5.1%}  "
              f"CPU bez/z rand. {pct('cpu_nominal')} / {pct('cpu_rand')}  "
              f"{p['steps'] / 1e6:6.1f} M  {p['created']}{f'  {blocks}' if blocks else ''}")
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


#: Where `grasps` writes by default, and the SO-101's gripper description (`paths.data_path`).
GRASPS_REL = Path("workspace") / "grasps"
GRIPPER_CONFIG_REL = Path("tools") / "graspgenx" / "so101" / "config.json"


def _grasps_meshes(a: argparse.Namespace) -> int:
    from .grasp_library import write_block_obj
    from .paths import data_path
    from .rl.objects import KINDS

    out = Path(a.out) if a.out else data_path(GRASPS_REL / "meshes")
    for kind in a.kinds or KINDS:
        print(write_block_obj(kind, out / f"{kind}.obj"))
    return 0


def _grasps_convert(a: argparse.Namespace) -> int:
    import datetime

    import yaml

    from .grasp_library import convert_isaac_grasps, graspgenx_frame, read_gripper_config, save_library
    from .kinematics import RobotKinematics
    from .paths import data_path
    from .robot import load_robot

    source = Path(a.yaml)
    if not source.is_file():
        raise FileNotFoundError(f"GraspGenX output not found: {source}")
    gripper = Path(a.gripper) if a.gripper else data_path(GRIPPER_CONFIG_REL)
    if not gripper.is_file():
        raise FileNotFoundError(f"gripper config not found: {gripper}; pass --gripper")
    cfg = read_gripper_config(gripper)
    frame = graspgenx_frame(RobotKinematics(load_robot()), cfg.base_rotation)
    generator = {"name": a.generator, "version": "1.0.0" if a.generator == "graspgenx" else "",
                 "commit": a.commit, "gripper": "so101", "checkpoint": a.checkpoint,
                 "date": datetime.date.today().isoformat(), "note": a.note}
    data = yaml.safe_load(source.read_text(encoding="utf-8"))
    library = convert_isaac_grasps(data if isinstance(data, dict) else {}, a.kind, frame=frame,
                                   pad_point=cfg.pad_point, generator=generator)
    out = save_library(library, a.out or data_path(GRASPS_REL / f"{a.kind}.json"))
    dropped, total = library.generator["dropped"].split(" of ")
    why = f" ({dropped} of {total} dropped: pad point outside the block)" if dropped != "0" else ""
    print(f"{a.kind}: {len(library.grasps)} grasps -> {out}{why}")
    return 0


def _grasps_show(a: argparse.Namespace) -> int:
    from .grasp_library import library_path, load_library

    library = load_library(a.kind, a.library)
    print(f"{library_path(a.kind, a.library)}: {library.kind}, {len(library.grasps)} grasps")
    for key, value in library.generator.items():
        print(f"  {key}: {value or '-'}")

    def vector(v, scale: float, digits: int) -> str:
        return "(" + ", ".join(f"{x * scale + 0.0:.{digits}f}" for x in v) + ")"

    for i, g in enumerate(library.grasps):
        print(f"#{i} score {g.score:.2f} pad {vector(g.position, 1000.0, 1)} mm "
              f"approach {vector(g.approach, 1.0, 2)} closing {vector(g.closing, 1.0, 2)} "
              f"width {g.width_m * 1000:.1f} mm")
    return 0


def _grasps_parser(sub) -> None:
    """`soarm-lab grasps meshes|convert|show`: GraspGenX's inputs and its output as grasp libraries."""
    p = sub.add_parser("grasps", help="grasp libraries of the wooden blocks, computed offline by GraspGenX",
                       description="Prepares the inputs of GraspGenX and reads its output into the grasp "
                                   "libraries of the `grasp` strategy (tools/graspgenx/README.md). GraspGenX "
                                   "itself runs in its own clone on a CUDA GPU, never here.")
    grasps_sub = p.add_subparsers(dest="grasps_cmd", required=True)

    q = grasps_sub.add_parser("meshes", help="the blocks as OBJ meshes in their block frames: GraspGenX's "
                                             "input")
    q.add_argument("--out", default=None, help="the folder of <kind>.obj (default workspace/grasps/meshes)")
    q.add_argument("--kinds", type=_kinds_arg, default=None, metavar="KIND,KIND",
                   help="the blocks (default: every block of rl/objects.py)")
    q.set_defaults(fn=_grasps_meshes)

    q = grasps_sub.add_parser("convert", help="GraspGenX's isaac_grasp YAML of one block -> a grasp library")
    q.add_argument("kind", type=_kind_arg, help="the block the grasps were computed for (rl/objects.py)")
    q.add_argument("yaml", help="the file demo_object_mesh.py wrote (--output_file)")
    q.add_argument("--out", default=None, help="the library to write (default workspace/grasps/<kind>.json)")
    q.add_argument("--gripper", default=None,
                   help="the gripper description's config.json (default tools/graspgenx/so101/config.json)")
    q.add_argument("--generator", default="graspgenx", help="what made the grasps: graspgenx, or hand")
    q.add_argument("--commit", default="", help="the GraspGenX commit that ran")
    q.add_argument("--checkpoint", default="", help="the GraspGenX checkpoint that ran (release)")
    q.add_argument("--note", default="", help="a note stored with the library")
    q.set_defaults(fn=_grasps_convert)

    q = grasps_sub.add_parser("show", help="the generator and the grasps of a library")
    q.add_argument("kind", type=_kind_arg, help="the block (rl/objects.py)")
    q.add_argument("--library", default=None,
                   help="the folder of <kind>.json libraries (default: the ones bundled with soarm-lab)")
    q.set_defaults(fn=_grasps_show)


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
    p.add_argument("--device", default="cuda:0", help="device of JAX and MuJoCo Warp (default cuda:0)")
    p.add_argument("--kinds", type=_kinds_arg, default=None, metavar="KIND,KIND",
                   help="one shape-conditioned policy for these wooden blocks, an equal share of the worlds "
                        "each: cube, cuboid5, cuboid85, cylinder (rl/objects.py); default: the task's one cube")
    p.add_argument("--episode-steps", type=int, default=None,
                   help="episode length in policy ticks (default: the task's, lift 200)")
    p.set_defaults(fn=_train)

    p = sub.add_parser("eval", help="ewaluacja polityki na CPU (zwykle MuJoCo)")
    p.add_argument("policy", help="nazwa polityki (np. lift-v3) albo sciezka do policy.npz")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--rand", action="store_true", help="z randomizacja dziedziny")
    p.add_argument("--seed", type=int, default=10_000, help="ziarno pierwszego epizodu")
    p.add_argument("--station", default="sim", help=station_help)
    p.add_argument("--kind", type=_kind_arg, default=None,
                   help="the wooden block to evaluate on (rl/objects.py); default: the policy's own")
    p.set_defaults(fn=_eval)

    p = sub.add_parser("policies", help="polityki bazowe i uczone, z wynikami")
    p.add_argument("--dir", default=None, help="katalog polityk (domyslnie workspace/policies)")
    p.set_defaults(fn=_policies)

    p = sub.add_parser("convert", help="punkt kontrolny torcha (policy.pt) jako policy.npz (dodatek torch)")
    p.add_argument("checkpoint", help="policy.pt albo jego folder")
    p.add_argument("--out", default=None, help="plik policy.npz (domyslnie obok policy.pt; meta.json obok niego)")
    p.set_defaults(fn=_convert)

    _dream_parser(sub)
    _grasps_parser(sub)
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

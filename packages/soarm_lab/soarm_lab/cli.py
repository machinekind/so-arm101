"""Command-line training, evaluation, policy listing, and model execution.

    soarm-lab train task=reach
    soarm-lab train experiment=<name>
    soarm-lab train --cfg job
    soarm-lab eval <policy> --rand
    soarm-lab eval <policy> --variant <name>
    soarm-lab policies
    soarm-lab convert old/policy.pt
    soarm-lab dream models
    soarm-lab dream sim --prompt "pick up the cube"
    soarm-lab grasps meshes --out workspace/grasps/meshes
    soarm-lab grasps convert cube workspace/grasps/raw/cube.yml
    soarm-lab grasps show cube
    soarm-lab campaign report workspace/campaign/<run> --out report.md

`station` for training and `--station` for evaluation select an exported station folder
or a bundled station name. The default is `sim`.
`train` requires the `train` extra. `convert` requires the `torch` extra.
`dream` requires the `dream` extra and a Dream Labs API key.
Evaluation and policy listing use NumPy.
`grasps` writes block meshes and converts or lists libraries. GraspGenX runs separately.
See tools/graspgenx/README.md.
`campaign report` summarises the JSON files of a seeded simulation campaign (jobs/overnight.sh).
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _station(a: argparse.Namespace):
    from .station import Station

    return Station.load(a.station)


def _kind_arg(text: str) -> str:
    """Validate a block name against the catalog."""
    from .blocks.classic_world import BLOCKS

    if text not in BLOCKS:
        raise argparse.ArgumentTypeError(f"unknown block {text!r}; known: {', '.join(BLOCKS)}")
    return text


def _kinds_arg(text: str) -> tuple[str, ...]:
    """Validate comma-separated `--kinds` names, rejecting unknown blocks and duplicates."""
    kinds = tuple(_kind_arg(k.strip()) for k in text.split(",") if k.strip())
    if not kinds:
        raise argparse.ArgumentTypeError("no block given")
    if len(set(kinds)) != len(kinds):
        raise argparse.ArgumentTypeError(f"a block repeats in {text!r}")
    return kinds


def _variants_summary(evals: dict, variants: list[str]) -> str:
    """Summarize mean and worst CPU success, preferring randomized evaluations when available."""
    from .rl.policy import variant_key

    for key, label in (("cpu_rand", "z rand."), ("cpu_nominal", "bez rand.")):
        got = {v: evals[variant_key(key, v)].get("success") for v in variants if variant_key(key, v) in evals}
        got = {v: s for v, s in got.items() if s is not None}
        if got:
            worst = min(got, key=got.__getitem__)
            mean = sum(got.values()) / len(got)
            return f"warianty {label} srednio {mean:5.1%}, najslabszy {worst} {got[worst]:5.1%}"
    return ""


def _eval(a: argparse.Namespace) -> int:
    from .rl.evaluate import evaluate
    from .rl.policy import Policy, find_policy
    from .rl.randomize import Dynamics, Randomization

    st = _station(a)
    path = find_policy(a.policy)
    pol = Policy.load(path)
    # Randomization around the station's measured dynamics, as in training, not around Menagerie.
    rand = Randomization.around(Dynamics.from_dict(st.dynamics)) if a.rand else None
    res = evaluate(pol, a.episodes, randomization=rand, station=st, seed=a.seed, variant=a.variant)
    print(f"{path}{f' ({a.variant})' if a.variant else ''}: {res}")
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
        variants = _variants_summary(p["evals"], p["variants"])
        print(f"  {p['name']:32s} {p['task']:9s} {where:6s} GPU {p['success'] or 0:5.1%}  "
              f"CPU bez/z rand. {pct('cpu_nominal')} / {pct('cpu_rand')}  "
              f"{p['steps'] / 1e6:6.1f} M  {p['created']}{f'  {variants}' if variants else ''}")
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


#: Default paths for grasp output and the gripper configuration, resolved by `paths.data_path`.
GRASPS_REL = Path("workspace") / "grasps"
GRIPPER_CONFIG_REL = Path("tools") / "graspgenx" / "so101" / "config.json"


def _grasps_meshes(a: argparse.Namespace) -> int:
    from .blocks.classic_world import BLOCKS
    from .grasp_library import write_block_obj
    from .paths import data_path

    out = Path(a.out) if a.out else data_path(GRASPS_REL / "meshes")
    for kind in a.kinds or BLOCKS:
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
    """Build the `grasps meshes|convert|show` command parser."""
    p = sub.add_parser("grasps", help="grasp libraries of the wooden blocks, computed offline by GraspGenX",
                       description="Prepares the inputs of GraspGenX and reads its output into the grasp "
                                   "libraries of the `grasp` strategy (tools/graspgenx/README.md). GraspGenX "
                                   "itself runs in its own clone on a CUDA GPU, never here.")
    grasps_sub = p.add_subparsers(dest="grasps_cmd", required=True)

    q = grasps_sub.add_parser("meshes", help="the blocks as OBJ meshes in their block frames: GraspGenX's "
                                             "input")
    q.add_argument("--out", default=None, help="the folder of <kind>.obj (default workspace/grasps/meshes)")
    q.add_argument("--kinds", type=_kinds_arg, default=None, metavar="KIND,KIND",
                   help="the blocks (default: every block of blocks/classic_world.py)")
    q.set_defaults(fn=_grasps_meshes)

    q = grasps_sub.add_parser("convert", help="GraspGenX's isaac_grasp YAML of one block -> a grasp library")
    q.add_argument("kind", type=_kind_arg, help="the block the grasps were computed for (blocks/classic_world.py)")
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
    q.add_argument("kind", type=_kind_arg, help="the block (blocks/classic_world.py)")
    q.add_argument("--library", default=None,
                   help="the folder of <kind>.json libraries (default: the ones bundled with soarm-lab)")
    q.set_defaults(fn=_grasps_show)


def _campaign_report(a: argparse.Namespace) -> int:
    from .campaign import load, report

    folder = Path(a.folder)
    runs = load(folder)
    if not runs:
        raise FileNotFoundError(f"no campaign runs under {folder}")
    text = report(runs, folder, title=a.title)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")
        print(f"{len(runs)} runs -> {a.out}")
    else:
        print(text, end="")
    return 0


def _campaign_parser(sub) -> None:
    """Build the `campaign report` command parser."""
    p = sub.add_parser("campaign", help="seeded simulation campaigns: the report over their runs",
                       description="Reads the JSON files a seeded simulation campaign wrote (jobs/overnight.sh) "
                                   "and summarises each scenario as Markdown.")
    campaign_sub = p.add_subparsers(dest="campaign_cmd", required=True)
    q = campaign_sub.add_parser("report", help="runs, successes, attempts, offsets and failures per scenario")
    q.add_argument("folder", help="the campaign folder (SOARM_CAMPAIGN_DIR of the run)")
    q.add_argument("--out", default=None, help="write the report here instead of printing it")
    q.add_argument("--title", default="Simulation campaign", help="the report's heading")
    q.set_defaults(fn=_campaign_report)


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
    p.add_argument("--variant", default=None, help="evaluate on this variant of the policy; default: its own block")
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
    _campaign_parser(sub)
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

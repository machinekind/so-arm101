# soarm-lab

Everything learned or measured for the SO-101 demo. Distribution name `soarm-lab`, import name `soarm_lab`.

It imports the twin (`lerobot_mp`) and `soarm`. Neither of them imports this package.

The demo: two SO-101 arms, one mounted inverted, pick an object from a known set,
hand it from one arm to the other and place it on the quadruped Wojtek.

## Folders

| Folder | Holds |
|---|---|
| `rig/` | demo table, cameras, object set, protocol, handover, Wojtek |
| `perception/` | which object from the set, and where on the table |
| `training/rl/` | PPO, GPU batch env, tasks, policy format, randomization, curricula |
| `training/planner/` | geometric grasp, GraspGenX adapter |
| `training/vla/` | recording recipe, LeRobot config, fine-tune configs, Cosmos, automatic demos |
| `eval/` | `run_sim.py`, `run_real.py`, `results/`, `compare.md` |
| `policies/` | `meta.json` and Hugging Face links per policy, no weights |
| `soarm_lab/` | the Python package: `strategies.py` with the Strategy adapters |

Everything here is a stub. The RL code arrives from the twin with ticket #146.
`docs/MIGRATION.md` lists what moves where.

## Extras

- `twin`: the twin, `lerobot-mp`, from its GitHub repository (`main`).
- `train`: torch, MuJoCo and MuJoCo Warp for GPU training.

## Open point

`rig/`, `perception/`, `training/` and `eval/` sit next to the `soarm_lab/` package directory.
The wheel ships only `soarm_lab/`. Code that must be importable moves under `soarm_lab/`
or gets mapped into the wheel when it arrives.

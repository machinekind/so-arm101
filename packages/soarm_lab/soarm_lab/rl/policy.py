"""A policy as a file: the actor network, the observation normalizer and what to know about its training.

    pol = Policy.load("workspace/policies/reach-20260924-0130/policy.npz")
    action = pol.act(obs)                   # numpy (obs_dim,) or (N, obs_dim) -> [-1, 1]

`meta.json` next to the weights holds the configuration of the task the policy was trained on
(frequency, action step, limits). Running it with other values would move differently than in
training, and on the real arm that looks like "the policy does not work". That is why the strategy
and the evaluation build the task FROM THE FILE, not from the defaults.

Policies live in two places: the base ones bundled with the package (`soarm_lab/policies`,
`bundled_dir`) and the ones trained here (`DEFAULT_DIR`, see `soarm_lab.paths`). `find_policy`
finds the file by name or path.

Running a policy needs numpy only: the strategy, evaluation and `soarm-lab eval` never import torch.
The weights file is `policy.npz`, format 1, read with `np.load(allow_pickle=False)`; all arrays float32:

    format              ()                       1
    sizes               (L + 1,)                 obs_dim, *hidden, act_dim (int64)
    obs_mean, obs_var   (obs_dim,)               the observations' running mean and variance from training
    obs_count           ()                       how many observations they come from
    obs_clip            ()                       the normalized observation is clipped to +-obs_clip
    actor.{i}.weight    (sizes[i+1], sizes[i])   layer i: x @ weight.T + bias, i = 0 .. L-1
    actor.{i}.bias      (sizes[i+1],)
    critic.{i}.weight   the critic: sizes obs_dim, *hidden, 1; for fine-tuning only, optional
    critic.{i}.bias

    action = clip(actor(clip((obs - obs_mean) / sqrt(obs_var + 1e-8), -obs_clip, obs_clip)), -1, 1)

with ELU (alpha 1) after every layer but the last. The action is the mean of the Gaussian the
policy was trained with. The weights keep torch's `nn.Linear` layout (n_out, n_in): the same
matrix products as torch, so a single observation's action matches torch's to a few float32 ulps.
A torch checkpoint (`policy.pt`, the format before this one) converts with
`soarm-lab convert <policy.pt>`, which needs `soarm-lab[torch]` (`torch_policy.convert`).
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from ..paths import policies_dir
from . import task as tk

#: Policies trained here. In a checkout of the repository, relative to its root (`soarm_lab.paths`).
DEFAULT_DIR = policies_dir()
#: The base policies bundled with the package.
BUNDLED_DIR = Path(__file__).resolve().parents[1] / "policies"
#: The weights of a policy (numpy), next to its `meta.json`.
POLICY_FILE = "policy.npz"
#: A torch checkpoint of the format before `policy.npz`: `soarm-lab convert` turns it into one.
TORCH_FILE = "policy.pt"
#: The version of the `policy.npz` layout this module writes and reads.
FORMAT = 1
#: The normalizer: added to the variance under the square root, and the clip of the normalized observation.
VAR_EPS = 1e-8
OBS_CLIP = 5.0

#: Dense layers as (weight (n_out, n_in), bias (n_out,)), float32: y = x @ weight.T + bias.
Layers = list[tuple[np.ndarray, np.ndarray]]


@dataclass
class PolicyMeta:
    task: dict[str, Any]
    hidden: list[int]
    obs_dim: int
    act_dim: int
    created: str = field(default_factory=lambda: time.strftime("%Y-%m-%dT%H:%M:%S"))
    steps: int = 0
    iterations: int = 0
    success: float = float("nan")
    randomization: dict[str, Any] = field(default_factory=dict)
    #: Evaluation results by key: "gpu" (the trainer's own, `ppo`), "cpu_nominal" and "cpu_rand"
    #: (`soarm-lab train` evaluates on the CPU after training).
    evals: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: The arm the policy was trained on (`RobotSpec.name`). A `meta.json` without it is an SO-101's.
    robot: str = "so101"

    def task_config(self) -> tk.TaskConfig:
        # Fields missing from an older `meta.json` (e.g. `end_on_success`) take the task's value from
        # `make_task`, not the class default of zero: an older lift policy then stops after success on
        # the arm like a new one, and one fine-tuned from it gets the current penalties. Without
        # `n_joints` it is an SO-101 (6).
        data = {**asdict(tk.make_task(self.task.get("name", "reach"))), **self.task}
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = tuple(v)
        return tk.TaskConfig(**data)


class TorchCheckpoint(ValueError):
    """A `policy.pt` was given where a `policy.npz` is needed."""


def elu(x: np.ndarray) -> np.ndarray:
    """ELU with alpha 1, as torch's `nn.ELU`."""
    return np.where(x > 0, x, np.expm1(np.minimum(x, 0)))


def zero_layers(sizes: list[int]) -> Layers:
    return [(np.zeros((b, a), np.float32), np.zeros(b, np.float32)) for a, b in zip(sizes[:-1], sizes[1:])]


def run_mlp(layers: Layers, x: np.ndarray) -> np.ndarray:
    """The MLP: ELU after every layer but the last."""
    for i, (weight, bias) in enumerate(layers):
        x = x @ weight.T + bias
        if i < len(layers) - 1:
            x = elu(x)
    return x


class Policy:
    """The actor and its observation normalizer, in numpy. A new policy acts 0 for every observation."""

    def __init__(self, meta: PolicyMeta):
        self.meta = meta
        self.obs_mean = np.zeros(meta.obs_dim, np.float32)
        self.obs_var = np.ones(meta.obs_dim, np.float32)
        self.obs_count = np.float32(1e-4)
        self.obs_clip = np.float32(OBS_CLIP)
        self.actor: Layers = zero_layers(self.sizes)
        #: The critic's weights from training (for fine-tuning only; running does not need them). None
        #: for a policy saved without one: fine-tuning then warms up a new critic (`PPOConfig.critic_warmup`).
        self.critic: Layers | None = None

    @property
    def sizes(self) -> list[int]:
        """The actor's layer sizes: obs_dim, *hidden, act_dim."""
        return [int(self.meta.obs_dim), *(int(h) for h in self.meta.hidden), int(self.meta.act_dim)]

    def normalize(self, obs: np.ndarray) -> np.ndarray:
        x = np.asarray(obs, np.float32)
        x = (x - self.obs_mean) / np.sqrt(self.obs_var + np.float32(VAR_EPS))
        return np.clip(x, -self.obs_clip, self.obs_clip)

    def act(self, obs: np.ndarray) -> np.ndarray:
        """The deterministic action (the distribution's mean), clipped to [-1, 1]."""
        x = np.asarray(obs, np.float32)
        single = x.ndim == 1
        a = np.clip(run_mlp(self.actor, self.normalize(x[None] if single else x)), -1.0, 1.0)
        return a[0] if single else a

    @property
    def task(self) -> tk.TaskConfig:
        return self.meta.task_config()

    # ----------------------------------------------------------------- disk
    def arrays(self) -> dict[str, np.ndarray]:
        """The contents of `policy.npz` (see the module's docstring)."""
        out = {"format": np.asarray(FORMAT, np.int64), "sizes": np.asarray(self.sizes, np.int64),
               "obs_mean": np.asarray(self.obs_mean, np.float32), "obs_var": np.asarray(self.obs_var, np.float32),
               "obs_count": np.asarray(self.obs_count, np.float32),
               "obs_clip": np.asarray(self.obs_clip, np.float32)}
        for name, layers in (("actor", self.actor), ("critic", self.critic or [])):
            for i, (weight, bias) in enumerate(layers):
                out[f"{name}.{i}.weight"] = np.asarray(weight, np.float32)
                out[f"{name}.{i}.bias"] = np.asarray(bias, np.float32)
        return out

    def save(self, path: str | Path) -> Path:
        """Writes `path` (`policy.npz`) and `meta.json` next to it."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            np.savez(f, **self.arrays())
        path.with_name("meta.json").write_text(json.dumps(asdict(self.meta), indent=2, ensure_ascii=False,
                                                          default=float), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path) -> Policy:
        """A policy from its `policy.npz` (or its folder) and the `meta.json` next to it."""
        path = Path(path)
        if path.is_dir():
            path = path / POLICY_FILE
        if path.suffix == ".pt":
            raise TorchCheckpoint(f"{path} is a torch checkpoint; policies are policy.npz now. Convert it with "
                                  f"`soarm-lab convert {path}` (install soarm-lab[torch] for that)")
        meta = json.loads(path.with_name("meta.json").read_text(encoding="utf-8"))
        pol = Policy(PolicyMeta(**meta))
        with np.load(path, allow_pickle=False) as z:
            if int(z["format"]) != FORMAT:
                raise ValueError(f"{path}: policy format {int(z['format'])}, this soarm-lab reads {FORMAT}")
            sizes = [int(s) for s in z["sizes"]]
            if sizes != pol.sizes:
                raise ValueError(f"{path}: layers {sizes}, meta.json gives {pol.sizes}")
            pol.obs_mean, pol.obs_var = z["obs_mean"], z["obs_var"]
            pol.obs_count, pol.obs_clip = z["obs_count"][()], z["obs_clip"][()]
            n = len(sizes) - 1
            pol.actor = [(z[f"actor.{i}.weight"], z[f"actor.{i}.bias"]) for i in range(n)]
            if "critic.0.weight" in z.files:
                pol.critic = [(z[f"critic.{i}.weight"], z[f"critic.{i}.bias"]) for i in range(n)]
        return pol


def bundled_dir() -> Path:
    """The package's base policies (`reach-v3`, `lift-v3`): a start for fine-tuning on your own arm."""
    return BUNDLED_DIR


def is_bundled(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(bundled_dir().resolve())
        return True
    except ValueError:
        return False


def list_policies(root: str | Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    """The policies in `root` and the bundled base policies, newest first.

    A policy in `root` hides a base policy of the same name. A folder with only a torch checkpoint
    (`policy.pt`) is not listed: `torch_checkpoints` finds those.
    """
    out: dict[str, dict[str, Any]] = {}
    for base, bundled in ((bundled_dir(), True), (Path(root), False)):
        for meta_path in base.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            weights = meta_path.with_name(POLICY_FILE)
            if weights.is_file():
                out[meta_path.parent.name] = {
                    "path": str(weights), "name": meta_path.parent.name, "task": meta["task"]["name"],
                    "success": meta.get("success"), "created": meta.get("created", ""),
                    "steps": meta.get("steps", 0), "evals": meta.get("evals", {}), "bundled": bundled}
    return sorted(out.values(), key=lambda m: m["created"], reverse=True)


def torch_checkpoints(root: str | Path = DEFAULT_DIR) -> list[Path]:
    """Policies in `root` saved only as a torch checkpoint (`policy.pt`): they need `soarm-lab convert`."""
    return sorted(p for p in Path(root).glob(f"*/{TORCH_FILE}") if not p.with_name(POLICY_FILE).is_file())


def find_policy(name: str | Path, root: str | Path | None = None) -> Path:
    """The policy file (`policy.npz`) of `name`: a path to the file or its folder, or a policy's name.

    The order is that of `list_policies`: a policy in `root` (by default `paths.policies_dir()`)
    hides a base policy of the same name. A folder with only a `policy.pt` gives that file;
    `Policy.load` then says how to convert it.
    """
    path = Path(name).expanduser()
    if path.is_file():
        return path
    bases = [path, *(base / str(name) for base in (Path(root) if root is not None else policies_dir(),
                                                    bundled_dir()))]
    for folder in bases:
        for file in (POLICY_FILE, TORCH_FILE):
            if (folder / file).is_file():
                return folder / file
    names = [p["name"] for p in list_policies(root if root is not None else policies_dir())]
    known = ", ".join(names) or "zadnych"
    raise FileNotFoundError(f"nie ma polityki {str(name)!r} (plik, folder albo nazwa); znane: {known}")

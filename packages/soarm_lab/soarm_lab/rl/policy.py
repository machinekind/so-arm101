"""Polityka jako plik: siec aktora, normalizacja obserwacji i wszystko, co trzeba wiedziec o treningu.

    pol = Policy.load("workspace/policies/reach-20260924-0130/policy.pt")
    action = pol.act(obs)                   # numpy (obs_dim,) albo (N, obs_dim) -> [-1, 1]

W pliku jest konfiguracja zadania, na ktorym polityka sie uczyla (czestotliwosc,
krok akcji, limity) - uruchomienie jej z innymi daloby inne ruchy niz w
treningu, a to wyglada na prawdziwym ramieniu jak "polityka nie dziala".
Dlatego strategia i ewaluacja buduja zadanie Z PLIKU, a nie z domyslnych.

Polityki sa w dwoch miejscach: bazowe dostarczane z pakietem (`soarm_lab/policies`,
`bundled_dir`) i uczone na stanowisku (`DEFAULT_DIR`, patrz `soarm_lab.paths`).
`find_policy` znajduje plik po nazwie albo sciezce.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from ..paths import policies_dir
from . import task as tk

#: Polityki stanowiska. W klonie repozytorium wzgledem jego korzenia (`soarm_lab.paths`).
DEFAULT_DIR = policies_dir()
#: Polityki bazowe dostarczane z pakietem.
BUNDLED_DIR = Path(__file__).resolve().parents[1] / "policies"
POLICY_FILE = "policy.pt"


def mlp(sizes: list[int], out: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.ELU()]
    layers.append(nn.Linear(sizes[-1], out))
    return nn.Sequential(*layers)


class Normalizer(nn.Module):
    """Srednia i wariancja obserwacji liczone w trakcie treningu, potem zamrozone."""

    def __init__(self, dim: int, clip: float = 5.0):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.tensor(1e-4))
        self.clip = clip

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        b_mean, b_var, n = x.mean(0), x.var(0, unbiased=False), x.shape[0]
        delta = b_mean - self.mean
        tot = self.count + n
        self.mean += delta * n / tot
        self.var = (self.var * self.count + b_var * n + delta**2 * self.count * n / tot) / tot
        self.count = tot

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.clamp((x - self.mean) / torch.sqrt(self.var + 1e-8), -self.clip, self.clip)


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
    #: Wyniki ewaluacji: {"gpu": ..., "cpu": ..., "real": ...} - uzupelniane pozniej.
    evals: dict[str, Any] = field(default_factory=dict)
    notes: str = ""
    #: Ramie, na ktorym polityka sie uczyla (`RobotSpec.name`). Polityki sprzed tego pola - SO-101.
    robot: str = "so101"

    def task_config(self) -> tk.TaskConfig:
        # Pola dodane pozniej (np. `end_on_success`) biora wartosc zadania z `make_task`,
        # a nie zero z klasy: stara lift-v2 ma sie na ramieniu zatrzymac po sukcesie tak
        # samo jak nowa, a douczana z niej - dostac nowe kary. Bez `n_joints` - SO-101 (6).
        data = {**asdict(tk.make_task(self.task.get("name", "reach"))), **self.task}
        for k, v in data.items():
            if isinstance(v, list):
                data[k] = tuple(v)
        return tk.TaskConfig(**data)


class Policy(nn.Module):
    def __init__(self, meta: PolicyMeta):
        super().__init__()
        self.meta = meta
        self.norm = Normalizer(meta.obs_dim)
        self.actor = mlp([meta.obs_dim, *meta.hidden], meta.act_dim)
        #: Wagi krytyka z treningu (tylko do douczania; do jazdy niepotrzebne). Polityki
        #: zapisane przed jego dodaniem go nie maja - douczanie grzeje wtedy krytyka od zera.
        self.critic_state: dict | None = None

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(self.norm(obs))

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Akcja deterministyczna (srednia rozkladu), przycieta do [-1, 1]."""
        x = torch.as_tensor(np.asarray(obs, np.float32), device=self.norm.mean.device)
        single = x.ndim == 1
        a = self(x[None] if single else x).clamp(-1.0, 1.0).cpu().numpy()
        return a[0] if single else a

    @property
    def task(self) -> tk.TaskConfig:
        return self.meta.task_config()

    # ----------------------------------------------------------------- dysk
    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        blob = {"meta": asdict(self.meta), "state": self.state_dict()}
        if self.critic_state is not None:
            blob["critic"] = self.critic_state
        torch.save(blob, path)
        path.with_name("meta.json").write_text(json.dumps(asdict(self.meta), indent=2, ensure_ascii=False,
                                                          default=float), encoding="utf-8")
        return path

    @staticmethod
    def load(path: str | Path, device: str = "cpu") -> Policy:
        blob = torch.load(Path(path), map_location=device, weights_only=False)
        pol = Policy(PolicyMeta(**blob["meta"]))
        pol.load_state_dict(blob["state"])
        pol.critic_state = blob.get("critic")
        return pol.to(device).eval()


def bundled_dir() -> Path:
    """Polityki bazowe z pakietu (`reach-v3`, `lift-v3`) - start do douczania na swoim ramieniu."""
    return BUNDLED_DIR


def is_bundled(path: str | Path) -> bool:
    try:
        Path(path).resolve().relative_to(bundled_dir().resolve())
        return True
    except ValueError:
        return False


def list_policies(root: str | Path = DEFAULT_DIR) -> list[dict[str, Any]]:
    """Polityki stanowiska i bazowe z repozytorium, najnowsze pierwsze.

    Nazwa ze stanowiska przeslania bazowa o tej samej nazwie.
    """
    out: dict[str, dict[str, Any]] = {}
    for base, bundled in ((bundled_dir(), True), (Path(root), False)):
        for meta_path in base.glob("*/meta.json"):
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            pt = meta_path.with_name("policy.pt")
            if pt.is_file():
                out[meta_path.parent.name] = {
                    "path": str(pt), "name": meta_path.parent.name, "task": meta["task"]["name"],
                    "success": meta.get("success"), "created": meta.get("created", ""),
                    "steps": meta.get("steps", 0), "evals": meta.get("evals", {}), "bundled": bundled}
    return sorted(out.values(), key=lambda m: m["created"], reverse=True)


def find_policy(name: str | Path, root: str | Path | None = None) -> Path:
    """Plik `policy.pt` polityki `name`: sciezka do pliku albo folderu, nazwa ze stanowiska, nazwa bazowej.

    Kolejnosc jak w `list_policies`: polityka stanowiska (`root`, domyslnie `paths.policies_dir()`)
    przeslania bazowa o tej samej nazwie.
    """
    path = Path(name).expanduser()
    if path.is_file():
        return path
    if (path / POLICY_FILE).is_file():
        return path / POLICY_FILE
    for base in (Path(root) if root is not None else policies_dir(), bundled_dir()):
        candidate = base / str(name) / POLICY_FILE
        if candidate.is_file():
            return candidate
    names = [p["name"] for p in list_policies(root if root is not None else policies_dir())]
    known = ", ".join(names) or "zadnych"
    raise FileNotFoundError(f"nie ma polityki {str(name)!r} (plik, folder albo nazwa); znane: {known}")

"""torch for the modules that need it (`ppo`, `batch`, `torch_policy`), with an error naming the extra.

Running a policy needs numpy only (`policy.Policy`). torch is the optional extra `soarm-lab[torch]`:
the legacy PPO trainer and the converter of torch checkpoints. A JAX trainer will replace the torch
one; it is not written yet.
"""

from __future__ import annotations

#: The error without torch.
MISSING = ("torch is not installed. The legacy PPO trainer (soarm-lab train) and soarm-lab convert need it: "
           "install soarm-lab[torch] (pip install 'soarm-lab[torch]'). "
           "Running a policy (the strategy, soarm-lab eval) needs numpy only.")

try:
    import torch
except ImportError as exc:
    raise ImportError(MISSING) from exc

__all__ = ["torch"]

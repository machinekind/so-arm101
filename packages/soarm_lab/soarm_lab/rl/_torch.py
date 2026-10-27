"""torch for the converter of old torch checkpoints (`torch_policy`), with an error naming the extra.

Running a policy needs numpy only (`policy.Policy`); training needs JAX (`soarm-lab[train]`, `ppo`).
torch is the optional extra `soarm-lab[torch]` for one command: `soarm-lab convert`, which turns a
torch checkpoint (`policy.pt`, the format before `policy.npz`) into `policy.npz`.
"""

from __future__ import annotations

#: The error without torch.
MISSING = ("torch is not installed. soarm-lab convert (a torch checkpoint, policy.pt, to policy.npz) "
           "needs it: install soarm-lab[torch] (pip install 'soarm-lab[torch]'). "
           "Training needs soarm-lab[train], not torch; running a policy (the strategy, soarm-lab eval) "
           "needs numpy only.")

try:
    import torch
except ImportError as exc:
    raise ImportError(MISSING) from exc

__all__ = ["torch"]

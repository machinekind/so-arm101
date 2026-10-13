"""Uczenie ze wzmocnieniem SO-101: zadania, srodowiska CPU i GPU, PPO, polityki.

Przeniesione z blizniaka (`lerobot_mp.twin.rl`, bez `sysid`, ktore zostaje przy nim).
Import rejestruje srodowiska Gymnasium:

    SoArmLab/Reach-v0       dojazd TCP do punktu (cel losowany z osiagalnych)
    SoArmLab/Lift-v0        chwyt i podniesienie kostki 3 cm
    SoArmLab/PickPlace-v0   kostka przeniesiona na cel na blacie i puszczona
"""

from __future__ import annotations

import gymnasium as gym

for _id, _task in (("SoArmLab/Reach-v0", "reach"), ("SoArmLab/Lift-v0", "lift"),
                   ("SoArmLab/PickPlace-v0", "pickplace")):
    if _id not in gym.registry:
        gym.register(id=_id, entry_point="soarm_lab.rl.env:TwinEnv", kwargs={"task": _task})

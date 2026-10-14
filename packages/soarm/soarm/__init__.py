"""SO-101: the robot folder and its two hardware backends, `feetech` and `lerobot`, plus the COM<->TCP bridge.

`pyproject.toml` registers both halves with the digital twin (`lerobot_mp`):

* `lerobot_mp.robots`: `so101 = "soarm:ROBOT_DIR"`, the folder with `robot.yaml` and the model;
* `lerobot_mp.backends`: `feetech` and `lerobot`, the backend factories (`create`).

Nothing here imports the twin. The backends follow its plugin contracts, `lerobot_mp_api.backend`.
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "robot"

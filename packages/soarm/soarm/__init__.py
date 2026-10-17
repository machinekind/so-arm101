"""SO-101: the robot folder and its two hardware backends, `feetech` and `lerobot`, plus the COM<->TCP bridge.

`pyproject.toml` registers both halves with the digital twin (`machinekind_twin`):

* `machinekind_twin.robots`: `so101 = "soarm:ROBOT_DIR"`, the folder with `robot.yaml` and the model;
* `machinekind_twin.backends`: `feetech` and `lerobot`, the backend factories (`create`).

Nothing here imports the twin. The backends follow its plugin contracts, `machinekind_twin_api.backend`.
"""

from pathlib import Path

ROBOT_DIR = Path(__file__).parent / "robot"

"""soarm: the SO-101 robot. Robot folders, the robot.yaml loader and hardware backends.

The twin (`lerobot_mp`) loads this package as a plugin through entry points.
This package imports neither the twin nor `soarm_lab`.
"""

from .spec import RobotSpec, SpecError, data_dir, load, robot_dir, so101

__all__ = ["RobotSpec", "SpecError", "data_dir", "load", "robot_dir", "so101"]
__version__ = "0.1.0"

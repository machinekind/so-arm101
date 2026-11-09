"""Detectors: the lab's plugins for the twin's perception, one module per detector.

`charuco` finds the inner corners of the approach calibration's ChArUco board in the fixed cameras' frames,
so the twin places them with the same camera models it places blocks with. `packages/soarm_lab/pyproject.toml`
registers each detector under the `machinekind_twin.detectors` entry point group.
"""

from .charuco import CharucoDetector, CharucoSettings

__all__ = ["CharucoDetector", "CharucoSettings"]

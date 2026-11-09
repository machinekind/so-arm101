"""The ChArUco board's inner corners as objects: `charuco` in the `machinekind_twin.detectors` entry points.

Each inner corner the fixed cameras see becomes one detection labelled "<label_prefix> c<id>", e.g.
"charuco c3", at the corner's sub-pixel position and the board's thickness above the table. The twin turns
each into a base-frame point with its calibrated camera models, the same way it places blocks, so the
corners carry the same camera error as the blocks. `calibrate_approach` reads them from `obs.objects`.

    # configs/local.yaml of the twin
    charuco:
      square_m: 0.045        # measured on the print [m]
      marker_m: 0.035

A label never ends in a bare number: the twin reads a trailing number as the n-th object of a label.
OpenCV comes with the twin; the detector refuses to start without it.
"""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from machinekind_twin_api.detector import Detection, Detector, DetectorInfo, DetectRequest

from ..charuco import BoardSpec, charuco_detector, corner_label, detect_corners


@dataclass
class CharucoSettings:
    #: Squares along the board's short and long side.
    squares_x: int = 3
    squares_y: int = 5
    #: Square and marker sides, measured on the print [m].
    square_m: float = 0.045
    marker_m: float = 0.035
    #: OpenCV's predefined ArUco dictionary of the markers.
    dictionary: str = "DICT_6X6_250"
    #: The board's top above the table [m].
    thickness_m: float = 0.0
    #: Each corner's label is this prefix, a space and "c<id>".
    label_prefix: str = "charuco"
    #: Shortest time between two detections [s].
    interval_s: float = 0.5
    #: How far off a corner's pixel may be [px].
    pixel_error_px: float = 1.0


class CharucoDetector(Detector[CharucoSettings]):
    """Finds the ChArUco board's inner corners in the fixed cameras and labels each one "charuco c<id>"."""

    settings = CharucoSettings

    def __init__(self, settings: CharucoSettings | None = None):
        self.config = settings if settings is not None else CharucoSettings()
        cfg = self.config
        self.spec = BoardSpec(cfg.squares_x, cfg.squares_y, cfg.square_m, cfg.marker_m, cfg.dictionary,
                              cfg.thickness_m)
        self.spec.check()
        if not " ".join(cfg.label_prefix.split()):
            raise ValueError("charuco: label_prefix must not be empty")
        if cfg.label_prefix.split()[-1].isdigit():
            raise ValueError("charuco: label_prefix must not end in a number")
        if not cfg.interval_s > 0:
            raise ValueError(f"charuco: interval_s must be positive, is {cfg.interval_s}")
        if not cfg.pixel_error_px > 0:
            raise ValueError(f"charuco: pixel_error_px must be positive, is {cfg.pixel_error_px}")
        self.interval_s = float(cfg.interval_s)
        self.pixel_error_px = float(cfg.pixel_error_px)
        self._cv2: Any = None
        self._detector: Any = None

    def start(self, info: DetectorInfo) -> None:
        try:
            cv2 = importlib.import_module("cv2")
        except ImportError as exc:
            raise RuntimeError("charuco: the detector needs OpenCV with the aruco module: pip install "
                               "opencv-contrib-python (the twin brings it)") from exc
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("charuco: this OpenCV has no aruco module: pip install opencv-contrib-python")
        self._cv2 = cv2
        _, self._detector = charuco_detector(self.spec, cv2)

    def stop(self, reason: str) -> None:
        pass

    def detect(self, request: DetectRequest) -> Sequence[Detection]:
        if self._cv2 is None:
            raise RuntimeError("charuco: detect() before start()")
        out: list[Detection] = []
        for name in request.frames:
            frame = request.frames.get(name)
            if frame is None:
                continue
            ids, pixels = detect_corners(frame.image, self._detector, self._cv2)
            for corner, (u, v) in zip(ids, pixels, strict=True):
                out.append(Detection(label=corner_label(self.config.label_prefix, int(corner)), camera=name,
                                     pixel=(float(u), float(v)), height_m=self.spec.thickness_m))
        return out

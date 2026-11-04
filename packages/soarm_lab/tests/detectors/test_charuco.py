"""The `charuco` detector: one detection per inner corner, labels the twin keeps apart, nothing without a
board, and a clear refusal without OpenCV. Without the twin: the API's request and frame types only."""

from __future__ import annotations

import importlib
import re

import numpy as np
import pytest
from machinekind_twin_api import detector as api
from machinekind_twin_api.detector import DetectorInfo, DetectRequest
from machinekind_twin_api.strategy import Frame

from soarm_lab.charuco import BoardSpec
from soarm_lab.detectors.charuco import CharucoDetector, CharucoSettings

#: The twin's rule for a key that names the n-th object of a label: a space and a bare number at the end.
NUMBERED = re.compile(r"^(.*\S)\s+\d+$")
INFO = DetectorInfo(cameras=("kamera1",), robot="so101", simulated=True)


def _cv2():
    cv2 = pytest.importorskip("cv2")
    if not hasattr(cv2, "aruco"):
        pytest.skip("OpenCV without the aruco module")
    return cv2


def base_label(key: str) -> str:
    """The API's `base_label` where it has one, else the same rule."""
    own = getattr(api, "base_label", None)
    if own is not None:
        return own(key)
    key = " ".join(key.split()).lower()
    match = NUMBERED.match(key)
    return match.group(1) if match else key


def board_frame(cv2, spec: BoardSpec) -> np.ndarray:
    """The board printed at 3000 px/m on a grey page, as an RGB frame."""
    board = spec.image(cv2, 3000.0)
    page = np.full((board.shape[0] + 200, board.shape[1] + 300), 150, np.uint8)
    page[100:100 + board.shape[0], 150:150 + board.shape[1]] = board
    return np.repeat(page[:, :, None], 3, axis=2)


def test_each_inner_corner_is_one_detection_at_the_boards_height():
    cv2 = _cv2()
    detector = CharucoDetector(CharucoSettings(thickness_m=0.0005))
    detector.start(INFO)
    spec = detector.spec
    frame = board_frame(cv2, spec)
    found = detector.detect(DetectRequest(frames={"kamera1": Frame(frame, 0.0)}, step=0))
    assert sorted(d.label for d in found) == [f"charuco c{k}" for k in range(spec.n_corners)]
    for d in found:
        assert d.camera == "kamera1" and d.height_m == 0.0005 and d.mask is None and d.size_m is None
        k = int(d.label.split("c")[-1])
        u, v = spec.corner_xy([k])[0] * 3000.0 - 0.5 + np.array([150, 100])
        assert d.pixel == pytest.approx((u, v), abs=0.3)
    assert detector.interval_s == 0.5 and detector.pixel_error_px == 1.0


def test_the_labels_are_distinct_objects_for_the_twin():
    labels = [f"charuco c{k}" for k in range(12)]
    assert len({base_label(label) for label in labels}) == len(labels)
    assert base_label("charuco c3 2") == "charuco c3"


def test_frames_without_the_board_or_without_a_frame_give_nothing():
    _cv2()
    detector = CharucoDetector()
    detector.start(INFO)
    empty = np.full((480, 640, 3), 128, np.uint8)
    assert list(detector.detect(DetectRequest(frames={"kamera1": Frame(empty, 0.0)}, step=0))) == []

    class Missing(dict):
        def __getitem__(self, name):
            raise KeyError(name)

    frames = Missing(kamera1=None)
    assert list(detector.detect(DetectRequest(frames=frames, step=1))) == []


def test_without_opencv_the_detector_refuses_to_start_and_says_why(monkeypatch):
    real = importlib.import_module

    def no_cv2(name, *args, **kwargs):
        if name == "cv2":
            raise ImportError("no cv2")
        return real(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", no_cv2)
    with pytest.raises(RuntimeError, match="OpenCV"):
        CharucoDetector().start(INFO)


@pytest.mark.parametrize("change, words", [
    ({"square_m": 0.03, "marker_m": 0.035}, "marker_m"),
    ({"label_prefix": "  "}, "label_prefix"),
    ({"label_prefix": "board 2"}, "number"),
    ({"interval_s": 0.0}, "interval_s"),
    ({"pixel_error_px": -1.0}, "pixel_error_px"),
])
def test_bad_settings_say_what_is_wrong(change, words):
    with pytest.raises(ValueError, match=words):
        CharucoDetector(CharucoSettings(**change))

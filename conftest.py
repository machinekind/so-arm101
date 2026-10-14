"""Skips for the markers `twin` and `render` of both packages' tests (registered in pyproject.toml)."""

from __future__ import annotations

import functools
import importlib.util

import pytest


@functools.lru_cache(maxsize=1)
def render_problem() -> str:
    """Why MuJoCo cannot render offscreen here ("" = it can)."""
    try:
        import mujoco

        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size="0.1"/></worldbody></mujoco>')
        with mujoco.Renderer(model, height=8, width=8) as renderer:
            renderer.update_scene(mujoco.MjData(model))
            renderer.render()
    except Exception as exc:  # no OpenGL context: no display, no EGL/OSMesa
        return f"{type(exc).__name__}: {exc}"
    return ""


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Render tests skip where no OpenGL context can be made (e.g. CI without MUJOCO_GL=egl).
    Twin tests skip where the twin is not installed."""
    render = [item for item in items if item.get_closest_marker("render")]
    if render and render_problem():
        skip = pytest.mark.skip(reason=f"no offscreen rendering: {render_problem()}")
        for item in render:
            item.add_marker(skip)
    twin = [item for item in items if item.get_closest_marker("twin")]
    if twin and importlib.util.find_spec("lerobot_mp") is None:
        skip = pytest.mark.skip(reason="the twin (lerobot_mp) is not installed")
        for item in twin:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _twin_defaults(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Twin tests count on the twin's defaults; a file named by LEROBOT_MP_CONFIG would change them."""
    if request.node.get_closest_marker("twin"):
        monkeypatch.delenv("LEROBOT_MP_CONFIG", raising=False)

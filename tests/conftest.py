from __future__ import annotations

import functools

import pytest
import yaml

import soarm


@pytest.fixture(scope="session")
def robot_yaml() -> dict:
    """`robot.yaml` of the SO-101 as plain data."""
    return yaml.safe_load((soarm.ROBOT_DIR / "robot.yaml").read_text(encoding="utf-8"))


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
    """Render tests skip where no OpenGL context can be made (e.g. CI without MUJOCO_GL=egl)."""
    render = [item for item in items if "render" in item.keywords]
    if render and render_problem():
        skip = pytest.mark.skip(reason=f"no offscreen rendering: {render_problem()}")
        for item in render:
            item.add_marker(skip)

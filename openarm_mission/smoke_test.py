"""Command-line P1/P2 smoke test."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from openarm_mission.controller import BimanualCartesianController
from openarm_mission.model import OpenArmMission


def _ik_smoke(controller: BimanualCartesianController, trials: int = 20) -> tuple[int, int]:
    rng = np.random.default_rng(7)
    successes = 0
    total = 0
    for side in ("left", "right"):
        position, quaternion = controller.mission.tcp_pose(side)
        for _ in range(trials):
            target = position + rng.uniform(-0.012, 0.012, size=3)
            result = controller.solve_ik(side, target, quaternion)
            successes += int(result.converged)
            total += 1
    return successes, total


def _render(mission: OpenArmMission, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(mission.model, height=480, width=640)
    try:
        renderer.update_scene(mission.data, camera="mission_front_camera")
        image = renderer.render()
        Image.fromarray(image).save(path)
    finally:
        close = getattr(renderer, "close", None)
        if close is not None:
            close()
        else:
            gl_context = getattr(renderer, "_gl_context", None)
            if gl_context is not None:
                gl_context.free()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--render-path", type=Path)
    parser.add_argument("--physics-steps", type=int, default=300)
    args = parser.parse_args()

    mission = OpenArmMission()
    controller = BimanualCartesianController(mission)
    successes, total = _ik_smoke(controller)
    rate = successes / total
    if rate < 0.98:
        raise RuntimeError(f"IK smoke test failed: {successes}/{total} ({rate:.1%})")

    controller.step(args.physics_steps)
    if not np.all(np.isfinite(mission.data.qpos)):
        raise RuntimeError("Physics smoke test produced a non-finite state")

    print(f"model: nq={mission.model.nq}, nv={mission.model.nv}, nu={mission.model.nu}")
    print(f"IK: {successes}/{total} ({rate:.1%})")
    print(f"cup: {mission.cup_position().round(4).tolist()}")
    print(f"physics: {args.physics_steps} steps, state finite")

    if args.render_path is not None:
        _render(mission, args.render_path)
        print(f"render: {args.render_path}")


if __name__ == "__main__":
    main()

"""P6 inference-side: evaluate a trained policy on the OpenArm paper-cup relay.

Connects to a ``serve_policy.py`` policy server, queries it for 14-dim bimanual
action chunks, applies them with rolling replan (``ActionChunkBroker``) through
the OpenArm controller, and judges success by cup position (the relay task's
stage machine is script-driven and cannot advance for a learned policy, so we
do not rely on it for progression -- only ``task.update()`` for failure
detection). Bimanual interlock masks the left arm until the right arm has
placed the cup at the center handoff region, mirroring the P5 training data.

Run (after ``serve_policy.py`` is up on host:port):

    MUJOCO_GL=egl .venv/bin/python -m openarm_mission.policy_eval \
        --host 127.0.0.1 --port 8000 --episodes 1 --seed 7
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pathlib
from typing import Any

import imageio.v2 as imageio
import mujoco
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
from openpi_client.action_chunk_broker import ActionChunkBroker
from PIL import Image
from PIL import ImageDraw

from openarm_mission.config import ControllerConfig
from openarm_mission.config import FrictionMissionConfig
from openarm_mission.config import FrictionTaskConfig
from openarm_mission.controller import BimanualCartesianController
from openarm_mission.dataset import CAMERAS
from openarm_mission.dataset import TASK_INSTRUCTION
from openarm_mission.dataset import hide_collision_geomgroups
from openarm_mission.friction_task import FrictionRelayTask
from openarm_mission.model import OpenArmMission

ACTION_DIM = 14
RESIZE = 224
FPS = 20
# Number of steps a chunk is consumed before re-querying the policy. The model
# predicts action_horizon=10 step chunks; using the full chunk matches training.
REPLAN_STEPS = 10
HOLD_SECONDS = 0.5  # continuous stable hold required for success


@dataclasses.dataclass
class EpisodeResult:
    seed: int
    success: bool
    reason: str | None
    steps: int
    elapsed: float
    final_cup_xy: tuple[float, float]
    final_upright_deg: float


def _make_renderer(mission: OpenArmMission, height: int, width: int) -> mujoco.Renderer:
    renderer = mujoco.Renderer(mission.model, height=height, width=width)
    for flag in (
        mujoco.mjtRndFlag.mjRND_SHADOW,
        mujoco.mjtRndFlag.mjRND_REFLECTION,
        mujoco.mjtRndFlag.mjRND_SKYBOX,
        mujoco.mjtRndFlag.mjRND_FOG,
        mujoco.mjtRndFlag.mjRND_HAZE,
    ):
        renderer.scene.flags[flag] = 0
    return renderer


def _state(mission: OpenArmMission) -> np.ndarray:
    """16-dim bimanual state (left 7 joints + gripper, right same). Mirrors P5."""
    values: list[float] = []
    for side in ("left", "right"):
        arm = mission.arms[side]
        values.extend(np.asarray(mission.data.qpos[arm.qpos_indices], dtype=np.float64).tolist())
        values.append(float(np.mean(mission.data.qpos[arm.finger_qpos_indices])))
    return np.asarray(values, dtype=np.float32)


def _render_images(mission: OpenArmMission, renderer: mujoco.Renderer) -> dict[str, np.ndarray]:
    options = mujoco.MjvOption()
    hide_collision_geomgroups(options)
    images = {}
    for key, camera in CAMERAS.items():
        renderer.update_scene(mission.data, camera=camera, scene_option=options)
        images[key] = np.array(renderer.render(), dtype=np.uint8, copy=True)
    return images


def _load_inprocess_policy(policy_dir: str, config_name: str):
    """Load a trained policy in-process (no websocket server).

    This avoids the single-GPU conflict between a separate server process
    (whose torch import grabs the CUDA primary context and breaks the eval's
    EGL rendering) and the eval process. One process, one GPU: JAX (GPU, no
    prealloc) + mujoco EGL coexist.
    """
    from openpi.policies import policy_config as _policy_config
    from openpi.training import config as _config

    config = _config.get_config(config_name)
    return _policy_config.create_trained_policy(config, policy_dir)


def _build_obs(mission: OpenArmMission, controller: BimanualCartesianController, renderer: mujoco.Renderer) -> dict:
    """Build the observation dict expected by OpenArmInputs (post-repack keys)."""
    raw_images = _render_images(mission, renderer)
    images = {
        key: image_tools.convert_to_uint8(image_tools.resize_with_pad(img, RESIZE, RESIZE))
        for key, img in raw_images.items()
    }
    return {
        "images": images,
        "state": _state(mission),
        "prompt": TASK_INSTRUCTION,
    }


def _apply_interlock(
    action: np.ndarray,
    mission: OpenArmMission,
    task: FrictionRelayTask,
    *,
    enabled: bool,
) -> np.ndarray:
    """Mask the left arm until the cup is at the center handoff region.

    The P5 data never moves the left arm before the right arm finishes the
    handoff. We enforce the same structure so the policy (trained on that
    distribution) is not asked to act out of distribution.
    """
    action = np.asarray(action, dtype=np.float64).copy()
    if not enabled:
        return action
    cup = mission.cup_position()
    center = np.asarray(mission.config.handoff_center)
    dist = float(np.hypot(cup[0] - center[0], cup[1] - center[1]))
    if dist > task.config.handoff_radius:
        # Left arm idle: zero Cartesian delta, gripper open (-1).
        action[0:6] = 0.0
        action[6] = -1.0
    return action


def _cup_in_goal(mission: OpenArmMission, task: FrictionRelayTask) -> bool:
    cup = mission.cup_position()
    goal = np.asarray(mission.config.region_b_center)
    return float(np.hypot(cup[0] - goal[0], cup[1] - goal[1])) <= task.config.goal_radius


def _step_physics(mission: OpenArmMission, controller: BimanualCartesianController, task: FrictionRelayTask) -> None:
    substeps = max(1, round(1.0 / (FPS * mission.config.timestep)))
    for _ in range(substeps):
        controller.compute_ctrl()
        mujoco.mj_step(mission.model, mission.data)
        task.update()


def _compose_frame(images: dict[str, np.ndarray], step: int, task: FrictionRelayTask) -> np.ndarray:
    front = Image.fromarray(images["front"])
    left = Image.fromarray(images["left_wrist"])
    right = Image.fromarray(images["right_wrist"])
    header = 42
    canvas = Image.new("RGB", (front.width * 2, front.height * 2 + header), (17, 20, 25))
    canvas.paste(front, ((canvas.width - front.width) // 2, header))
    canvas.paste(left, (0, header + front.height))
    canvas.paste(right, (front.width, header + front.height))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 7),
        f"step={step:04d}  {task.stage.value}  upright={task.cup_upright_angle_deg():.1f}deg",
        fill=(235, 238, 242),
    )
    return np.asarray(canvas)


def eval_openarm(
    *,
    host: str,
    port: int,
    episodes: int,
    start_seed: int,
    max_steps: int,
    replan_steps: int,
    interlock: bool,
    video_out: pathlib.Path,
    record_npz: bool,
    policy_dir: str | None = None,
    policy_config_name: str = "pi05_openarm_paper_cup_relay_lora",
    # Must match the P5 dataset capture resolution (see dataset.py), otherwise
    # the model sees a different aspect ratio / FOV than during training.
    height: int = 120,
    width: int = 160,
) -> list[EpisodeResult]:
    mission = OpenArmMission(FrictionMissionConfig())
    controller = BimanualCartesianController(mission, ControllerConfig())
    task = FrictionRelayTask(mission, FrictionTaskConfig())
    renderer = _make_renderer(mission, height=height, width=width)
    # Force the EGL GL context to be created NOW, before the policy is loaded
    # (which imports jax/torch and can disrupt a lazily-created EGL context).
    options = mujoco.MjvOption()
    hide_collision_geomgroups(options)
    renderer.update_scene(mission.data, camera="mission_front_camera", scene_option=options)
    renderer.render()

    if policy_dir is not None:
        # In-process policy (avoids single-GPU server+eval conflict).
        policy = _load_inprocess_policy(policy_dir, policy_config_name)
        client = policy
    else:
        client = _websocket_client_policy.WebsocketClientPolicy(host, port)

    video_out.mkdir(parents=True, exist_ok=True)
    results: list[EpisodeResult] = []

    for offset in range(episodes):
        seed = start_seed + offset
        mission.reset()
        task.reset(seed)
        controller.reset_targets()
        if hasattr(client, "reset"):
            client.reset()

        broker = ActionChunkBroker(client, action_horizon=replan_steps)
        frames: list[np.ndarray] = []
        replay: list[dict[str, Any]] = []
        success = False
        reason: str | None = None
        hold_started_at: float | None = None

        for step in range(max_steps):
            obs = _build_obs(mission, controller, renderer)
            raw_images = {k: v.copy() for k, v in obs["images"].items()} if record_npz else {}
            try:
                action = broker.infer(obs)["actions"]
            except Exception as e:  # server error or disconnect
                reason = f"policy_error: {e}"
                break

            action = _apply_interlock(action, mission, task, enabled=interlock)
            controller.apply_action(action)
            _step_physics(mission, controller, task)

            frames.append(_compose_frame(_render_images(mission, renderer), step, task))
            if record_npz:
                replay.append(
                    {
                        "step": step,
                        "state": np.asarray(obs["state"], dtype=np.float32),
                        "action": np.asarray(action, dtype=np.float32),
                        "cup_pose": np.concatenate([mission.cup_position(), mission.cup_quaternion()]).astype(
                            np.float32
                        ),
                        "image_front": raw_images["front"],
                        "image_left_wrist": raw_images["left_wrist"],
                        "image_right_wrist": raw_images["right_wrist"],
                    }
                )

            if task.done:
                reason = "success" if task.success else f"task_failure: {task.summary().get('failure_reason')}"
                success = task.success
                break

            # Success by cup position + upright + stable hold.
            if _cup_in_goal(mission, task) and task.cup_upright_angle_deg() <= task.config.upright_limit_deg:
                if hold_started_at is None:
                    hold_started_at = task.elapsed
                elif task.elapsed - hold_started_at >= HOLD_SECONDS:
                    success = True
                    reason = "goal_reached"
                    break
            else:
                hold_started_at = None
        else:
            reason = "timeout"

        suffix = "success" if success else "failure"
        imageio.mimwrite(video_out / f"rollout_seed{seed:06d}_{suffix}.mp4", frames, fps=FPS)
        if record_npz and replay:
            np.savez_compressed(
                video_out / f"rollout_seed{seed:06d}_{suffix}.npz",
                **{
                    k: np.stack([r[k] for r in replay]) if k != "step" else np.asarray([r[k] for r in replay])
                    for k in replay[0]
                },
            )

        result = EpisodeResult(
            seed=seed,
            success=success,
            reason=reason,
            steps=len(frames),
            elapsed=task.elapsed,
            final_cup_xy=(float(mission.cup_position()[0]), float(mission.cup_position()[1])),
            final_upright_deg=float(task.cup_upright_angle_deg()),
        )
        results.append(result)
        print(
            f"episode seed={seed}: success={success} steps={result.steps} "
            f"elapsed={result.elapsed:.1f}s reason={reason} "
            f"final_xy=({result.final_cup_xy[0]:.3f},{result.final_cup_xy[1]:.3f}) "
            f"upright={result.final_upright_deg:.1f}deg"
        )

    succ = sum(r.success for r in results)
    print(f"=== {succ}/{len(results)} success ({succ / max(1, len(results)) * 100:.1f}%) ===")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained policy on the OpenArm paper-cup relay.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=600, help="Control steps per episode (20 Hz).")
    parser.add_argument("--replan-steps", type=int, default=REPLAN_STEPS)
    parser.add_argument(
        "--no-interlock", action="store_true", help="Disable bimanual interlock (let policy act freely)."
    )
    parser.add_argument("--video-out", type=pathlib.Path, default=pathlib.Path("data/openarm/videos"))
    parser.add_argument("--no-npz", action="store_true", help="Skip per-step npz recording.")
    parser.add_argument(
        "--policy-dir",
        type=str,
        default=None,
        help="If set, load the policy in-process from this checkpoint dir instead of using a websocket server (avoids single-GPU server+eval conflict).",
    )
    parser.add_argument(
        "--policy-config",
        type=str,
        default="pi05_openarm_paper_cup_relay_lora",
        help="Training config name for the in-process policy.",
    )
    parser.add_argument("--height", type=int, default=120, help="Render height; must match P5 capture (120).")
    parser.add_argument("--width", type=int, default=160, help="Render width; must match P5 capture (160).")
    args = parser.parse_args()

    results = eval_openarm(
        host=args.host,
        port=args.port,
        episodes=args.episodes,
        start_seed=args.start_seed,
        max_steps=args.max_steps,
        replan_steps=args.replan_steps,
        interlock=not args.no_interlock,
        video_out=args.video_out,
        record_npz=not args.no_npz,
        policy_dir=args.policy_dir,
        policy_config_name=args.policy_config,
        height=args.height,
        width=args.width,
    )
    summary = {
        "episodes": len(results),
        "successes": sum(r.success for r in results),
        "results": [dataclasses.asdict(r) for r in results],
    }
    (args.video_out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()

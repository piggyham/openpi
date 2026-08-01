"""P5 trajectory schema, recording, domain randomization, and validation."""

from __future__ import annotations

from dataclasses import asdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from PIL import Image

from openarm_mission.config import ControllerConfig
from openarm_mission.friction_expert import FrictionScriptedExpert

DATASET_VERSION = "openarm-p5-v1"
TASK_INSTRUCTION = (
    "Use the right arm to move the paper cup from the red area to the "
    "center, then use the left arm to move it to the blue area."
)
CAMERAS = {
    "front": "mission_front_camera",
    "left_wrist": "mission_left_wrist_camera",
    "right_wrist": "mission_right_wrist_camera",
}
STATE_NAMES = tuple(
    [f"left_joint_{index}_rad" for index in range(1, 8)]
    + ["left_gripper_opening_m"]
    + [f"right_joint_{index}_rad" for index in range(1, 8)]
    + ["right_gripper_opening_m"]
)
ACTION_NAMES = (
    "left_dx_m",
    "left_dy_m",
    "left_dz_m",
    "left_drx_rad",
    "left_dry_rad",
    "left_drz_rad",
    "left_gripper",
    "right_dx_m",
    "right_dy_m",
    "right_dz_m",
    "right_drx_rad",
    "right_dry_rad",
    "right_drz_rad",
    "right_gripper",
)


def episode_split(seed: int) -> str:
    """Return a deterministic 80/10/10 split based only on the seed."""
    bucket = int(seed) % 10
    if bucket == 0:
        return "test"
    if bucket == 1:
        return "validation"
    return "train"


def schema_dict(*, fps: int, height: int, width: int) -> dict[str, Any]:
    return {
        "version": DATASET_VERSION,
        "fps": fps,
        "task": TASK_INSTRUCTION,
        "state": {
            "dtype": "float32",
            "shape": [16],
            "names": list(STATE_NAMES),
            "units": ["rad"] * 7 + ["m"] + ["rad"] * 7 + ["m"],
            "order": "left arm, left gripper, right arm, right gripper",
        },
        "action": {
            "dtype": "float32",
            "shape": [14],
            "names": list(ACTION_NAMES),
            "units": (["m"] * 3 + ["rad"] * 3 + ["normalized"]) * 2,
            "order": "left Cartesian delta + gripper, then right",
            "gripper_sign": "-1=open, +1=closed",
            "alignment": "observation[t] -> action[t] -> target[t+1]",
        },
        "images": {
            key: {
                "camera": camera,
                "dtype": "uint8",
                "shape": [height, width, 3],
                "color_order": "RGB",
            }
            for key, camera in CAMERAS.items()
        },
        "timestamp": {
            "dtype": "float64",
            "unit": "seconds",
            "clock": "MuJoCo simulation time",
        },
    }


def _named_id(model: mujoco.MjModel, object_type, name: str) -> int:
    object_id = mujoco.mj_name2id(model, object_type, name)
    if object_id < 0:
        raise ValueError(f"MuJoCo object not found: {name}")
    return int(object_id)


def apply_visual_randomization(mission, seed: int) -> dict[str, Any]:
    """Randomize appearance and cameras without changing task semantics."""
    model = mission.model
    rng = np.random.default_rng(int(seed) + 95_000)

    light_id = _named_id(
        model,
        mujoco.mjtObj.mjOBJ_LIGHT,
        "mission_key_light",
    )
    light_position = np.array([0.5, 0.0, 1.4]) + rng.uniform(
        [-0.10, -0.12, -0.12],
        [0.10, 0.12, 0.12],
    )
    brightness = float(rng.uniform(0.70, 1.00))
    color_bias = rng.uniform(0.94, 1.06, size=3)
    model.light_pos[light_id] = light_position
    model.light_diffuse[light_id] = np.clip(
        brightness * color_bias,
        0.0,
        1.0,
    )

    material_scales: dict[str, float] = {}
    for name, limits in (
        ("mission_table_material", (0.75, 1.20)),
        ("mission_cup_material", (0.88, 1.05)),
        ("mission_handoff_material", (0.85, 1.10)),
    ):
        material_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_MATERIAL,
            name,
        )
        scale = float(rng.uniform(*limits))
        model.mat_rgba[material_id, :3] = np.clip(
            model.mat_rgba[material_id, :3] * scale,
            0.0,
            1.0,
        )
        material_scales[name] = round(scale, 6)

    camera_randomization: dict[str, dict[str, Any]] = {}
    for key, camera_name in CAMERAS.items():
        camera_id = _named_id(
            model,
            mujoco.mjtObj.mjOBJ_CAMERA,
            camera_name,
        )
        maximum_offset = 0.025 if key == "front" else 0.004
        offset = rng.uniform(-maximum_offset, maximum_offset, size=3)
        fovy_delta = float(rng.uniform(-3.0, 3.0))
        model.cam_pos[camera_id] += offset
        model.cam_fovy[camera_id] = np.clip(
            model.cam_fovy[camera_id] + fovy_delta,
            35.0,
            90.0,
        )
        camera_randomization[key] = {
            "position_offset_m": offset.round(6).tolist(),
            "fovy_deg": round(float(model.cam_fovy[camera_id]), 6),
        }

    mujoco.mj_forward(model, mission.data)
    return {
        "light_position_m": light_position.round(6).tolist(),
        "light_diffuse": model.light_diffuse[light_id].round(6).tolist(),
        "material_brightness_scales": material_scales,
        "cameras": camera_randomization,
    }


def _quaternion_inverse(quaternion: np.ndarray) -> np.ndarray:
    result = np.array(quaternion, dtype=np.float64, copy=True)
    result[1:] *= -1.0
    result /= np.dot(result, result)
    return result


def _quaternion_delta_to_rotation_vector(
    start: np.ndarray,
    end: np.ndarray,
) -> np.ndarray:
    delta = np.empty(4)
    mujoco.mju_mulQuat(delta, end, _quaternion_inverse(start))
    if delta[0] < 0.0:
        delta *= -1.0
    vector_norm = float(np.linalg.norm(delta[1:]))
    if vector_norm < 1e-12:
        return np.zeros(3)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(delta[0], -1.0, 1.0))
    return delta[1:] / vector_norm * angle


def _gripper_command(opening: float, config: ControllerConfig) -> float:
    span = config.gripper_closed_position - config.gripper_open_position
    if abs(span) < 1e-12:
        raise ValueError("Gripper command range cannot be zero")
    interpolation = (opening - config.gripper_open_position) / span
    return float(np.clip(2.0 * interpolation - 1.0, -1.0, 1.0))


class P5TrajectoryRecorder(FrictionScriptedExpert):
    """Record synchronized P4.5 expert observations and Cartesian actions."""

    def __init__(
        self,
        *,
        seed: int,
        output_dir: Path,
        dataset_fps: int = 20,
        runner_fps: int = 20,
        width: int = 160,
        height: int = 120,
        capture_images: bool = True,
    ):
        if dataset_fps <= 0 or runner_fps % dataset_fps:
            raise ValueError("runner_fps must be divisible by dataset_fps")
        self.dataset_fps = dataset_fps
        self._sample_stride = runner_fps // dataset_fps
        self._expert_frame = 0
        self._capture_images = capture_images
        self._image_width = width
        self._image_height = height
        self._samples: list[dict[str, Any]] = []
        self._data_renderer = None
        self._target_data = None
        super().__init__(
            seed=seed,
            output_dir=output_dir,
            fps=runner_fps,
            width=width,
            height=height,
            write_video=False,
        )
        self.visual_randomization = apply_visual_randomization(
            self.mission,
            seed,
        )
        self._target_data = mujoco.MjData(self.mission.model)
        if capture_images:
            self._data_renderer = mujoco.Renderer(
                self.mission.model,
                height=height,
                width=width,
            )
            for flag in (
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_SKYBOX,
                mujoco.mjtRndFlag.mjRND_FOG,
                mujoco.mjtRndFlag.mjRND_HAZE,
            ):
                self._data_renderer.scene.flags[flag] = 0
        self._render_options = mujoco.MjvOption()
        # Groups 3+ contain dense collision/debug meshes. They are not visual
        # observations and make software EGL rendering hundreds of times
        # slower than the intended appearance geoms in groups 0-2.
        self._render_options.geomgroup[3:] = 0
        self._capture_sample()

    def _state(self) -> np.ndarray:
        values: list[float] = []
        for side in ("left", "right"):
            arm = self.mission.arms[side]
            values.extend(
                np.asarray(
                    self.mission.data.qpos[arm.qpos_indices],
                    dtype=np.float64,
                ).tolist()
            )
            values.append(float(np.mean(self.mission.data.qpos[arm.finger_qpos_indices])))
        return np.asarray(values, dtype=np.float32)

    def _target_poses(self) -> tuple[np.ndarray, np.ndarray]:
        assert self._target_data is not None
        self._target_data.qpos[:] = self.mission.data.qpos
        self._target_data.qvel[:] = 0.0
        for side in ("left", "right"):
            arm = self.mission.arms[side]
            self._target_data.qpos[arm.qpos_indices] = self.controller.target_joint_qpos[side]
            self._target_data.qpos[arm.finger_qpos_indices] = self.controller.target_gripper[side]
        mujoco.mj_forward(self.mission.model, self._target_data)
        positions = []
        quaternions = []
        for side in ("left", "right"):
            position, quaternion = self.mission.tcp_pose(
                side,
                self._target_data,
            )
            positions.append(position)
            quaternions.append(quaternion)
        return np.asarray(positions), np.asarray(quaternions)

    def _render_images(self) -> dict[str, np.ndarray]:
        if not self._capture_images:
            return {
                key: np.zeros(
                    (self._image_height, self._image_width, 3),
                    dtype=np.uint8,
                )
                for key in CAMERAS
            }
        assert self._data_renderer is not None
        images = {}
        for key, camera in CAMERAS.items():
            self._data_renderer.update_scene(
                self.mission.data,
                camera=camera,
                scene_option=self._render_options,
            )
            images[key] = np.array(
                self._data_renderer.render(),
                dtype=np.uint8,
                copy=True,
            )
        return images

    def _capture_sample(self) -> None:
        positions, quaternions = self._target_poses()
        self._samples.append(
            {
                "timestamp": float(self.mission.data.time),
                "state": self._state(),
                "target_positions": positions,
                "target_quaternions": quaternions,
                "gripper_targets": np.asarray(
                    [
                        self.controller.target_gripper["left"],
                        self.controller.target_gripper["right"],
                    ],
                    dtype=np.float64,
                ),
                "cup_pose": np.concatenate(
                    [
                        self.mission.cup_position(),
                        self.mission.cup_quaternion(),
                    ]
                ).astype(np.float32),
                "stage": self.task.stage.value,
                "phase": self._phase_title,
                "images": self._render_images(),
            }
        )

    def _step_frame(self) -> None:
        super()._step_frame()
        self._expert_frame += 1
        if self._expert_frame % self._sample_stride == 0:
            self._capture_sample()

    def _reset_for_episode_attempt(self) -> None:
        super()._reset_for_episode_attempt()
        self._samples.clear()
        self._expert_frame = 0
        self._capture_sample()

    def _close_media(self) -> None:
        super()._close_media()
        if self._data_renderer is not None:
            close = getattr(self._data_renderer, "close", None)
            if close is not None:
                close()
            else:
                context = getattr(self._data_renderer, "_gl_context", None)
                if context is not None:
                    context.free()
            self._data_renderer = None

    def _actions(self) -> np.ndarray:
        count = len(self._samples)
        actions = np.zeros((count, 14), dtype=np.float32)
        config = self.controller.config
        for index in range(count):
            current = self._samples[index]
            following = self._samples[min(index + 1, count - 1)]
            for arm_index in range(2):
                offset = arm_index * 7
                translation = following["target_positions"][arm_index] - current["target_positions"][arm_index]
                rotation = _quaternion_delta_to_rotation_vector(
                    current["target_quaternions"][arm_index],
                    following["target_quaternions"][arm_index],
                )
                actions[index, offset : offset + 3] = translation
                actions[index, offset + 3 : offset + 6] = rotation
                actions[index, offset + 6] = _gripper_command(
                    following["gripper_targets"][arm_index],
                    config,
                )
        return actions

    def trajectory(self) -> dict[str, np.ndarray]:
        if not self._samples:
            raise RuntimeError("No trajectory samples were recorded")
        return {
            "timestamp": np.asarray(
                [sample["timestamp"] for sample in self._samples],
                dtype=np.float64,
            ),
            "state": np.asarray(
                [sample["state"] for sample in self._samples],
                dtype=np.float32,
            ),
            "action": self._actions(),
            "cup_pose": np.asarray(
                [sample["cup_pose"] for sample in self._samples],
                dtype=np.float32,
            ),
            "stage": np.asarray(
                [sample["stage"] for sample in self._samples],
                dtype="U32",
            ),
            "phase": np.asarray(
                [sample["phase"] for sample in self._samples],
                dtype="U48",
            ),
            **{
                f"image_{key}": np.asarray(
                    [sample["images"][key] for sample in self._samples],
                    dtype=np.uint8,
                )
                for key in CAMERAS
            },
        }


def validate_trajectory(
    trajectory: dict[str, np.ndarray],
    *,
    fps: int,
    controller_config: ControllerConfig | None = None,
) -> dict[str, Any]:
    """Validate shape, units, finite values, and sample alignment."""
    config = controller_config or ControllerConfig()
    timestamps = trajectory["timestamp"]
    state = trajectory["state"]
    action = trajectory["action"]
    count = len(timestamps)
    expected_dt = 1.0 / fps
    deltas = np.diff(timestamps)

    shapes_valid = (
        state.shape == (count, 16)
        and action.shape == (count, 14)
        and trajectory["cup_pose"].shape == (count, 7)
        and all(trajectory[f"image_{key}"].shape[0] == count for key in CAMERAS)
    )
    finite = bool(
        np.all(np.isfinite(timestamps))
        and np.all(np.isfinite(state))
        and np.all(np.isfinite(action))
        and np.all(np.isfinite(trajectory["cup_pose"]))
    )
    timestamp_error = float(np.max(np.abs(deltas - expected_dt))) if len(deltas) else 0.0
    max_translation = float(
        max(
            np.max(np.abs(action[:, :3])),
            np.max(np.abs(action[:, 7:10])),
        )
    )
    max_rotation_norm = float(
        max(
            np.max(np.linalg.norm(action[:, 3:6], axis=1)),
            np.max(np.linalg.norm(action[:, 10:13], axis=1)),
        )
    )
    grippers = action[:, [6, 13]]
    limits_valid = bool(
        max_translation <= config.max_translation_delta + 1e-5
        and max_rotation_norm <= config.max_rotation_delta + 1e-5
        and np.max(np.abs(grippers)) <= 1.0 + 1e-6
    )
    report = {
        "frames": count,
        "duration_seconds": round(float(timestamps[-1]), 6),
        "shapes_valid": shapes_valid,
        "finite": finite,
        "timestamp_monotonic": bool(np.all(deltas > 0.0)),
        "max_timestamp_alignment_error_seconds": round(
            timestamp_error,
            9,
        ),
        "max_translation_action_m": round(max_translation, 6),
        "max_rotation_action_rad": round(max_rotation_norm, 6),
        "action_limits_valid": limits_valid,
        "state_joint_abs_max_rad": round(
            float(
                np.max(
                    np.abs(
                        state[
                            :,
                            [*range(7), *range(8, 15)],
                        ]
                    )
                )
            ),
            6,
        ),
        "state_gripper_range_m": [
            round(float(np.min(state[:, [7, 15]])), 6),
            round(float(np.max(state[:, [7, 15]])), 6),
        ],
    }
    report["valid"] = bool(
        shapes_valid and finite and report["timestamp_monotonic"] and timestamp_error <= 1e-6 and limits_valid
    )
    return report


def save_episode(
    path: Path,
    trajectory: dict[str, np.ndarray],
    metadata: dict[str, Any],
) -> tuple[str, int]:
    """Atomically save one JPEG-packed raw episode and return its checksum."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    packed = {key: value for key, value in trajectory.items() if not key.startswith("image_")}
    for camera in CAMERAS:
        frames = trajectory[f"image_{camera}"]
        blobs: list[bytes] = []
        offsets = [0]
        for frame in frames:
            stream = BytesIO()
            Image.fromarray(frame).save(
                stream,
                format="JPEG",
                quality=85,
                subsampling=1,
            )
            blob = stream.getvalue()
            blobs.append(blob)
            offsets.append(offsets[-1] + len(blob))
        packed[f"image_{camera}_jpeg_data"] = np.frombuffer(
            b"".join(blobs),
            dtype=np.uint8,
        )
        packed[f"image_{camera}_jpeg_offsets"] = np.asarray(
            offsets,
            dtype=np.int64,
        )
    with temporary.open("wb") as stream:
        np.savez(stream, **packed)
    temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    metadata_path = path.with_suffix(".json")
    metadata_path.write_text(
        json.dumps(
            metadata,
            ensure_ascii=False,
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    return digest, path.stat().st_size


def load_episode(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {key: np.array(archive[key], copy=True) for key in archive.files if "_jpeg_" not in key}
        for camera in CAMERAS:
            data = archive[f"image_{camera}_jpeg_data"]
            offsets = archive[f"image_{camera}_jpeg_offsets"]
            frames = []
            for index in range(len(offsets) - 1):
                start = offsets[index]
                stop = offsets[index + 1]
                stream = BytesIO(data[int(start) : int(stop)].tobytes())
                frames.append(np.asarray(Image.open(stream).convert("RGB")))
            trajectory[f"image_{camera}"] = np.asarray(
                frames,
                dtype=np.uint8,
            )
    return trajectory


def record_episode(
    *,
    seed: int,
    output_dir: Path,
    fps: int = 20,
    width: int = 160,
    height: int = 120,
    capture_images: bool = True,
) -> dict[str, Any]:
    """Run, validate, and save one successful expert demonstration."""
    raw_dir = output_dir / "raw"
    episode_path = raw_dir / f"episode_seed{seed:06d}.npz"
    recorder = P5TrajectoryRecorder(
        seed=seed,
        output_dir=output_dir / "_runner",
        dataset_fps=fps,
        width=width,
        height=height,
        capture_images=capture_images,
    )
    summary = recorder.run_expert(write_summary=False)
    if not summary["expert_success"]:
        raise RuntimeError(f"Seed {seed} expert failed: {summary['expert_exception']}")
    trajectory = recorder.trajectory()
    validation = validate_trajectory(trajectory, fps=fps)
    if not validation["valid"]:
        raise RuntimeError(f"Seed {seed} trajectory validation failed: {validation}")
    metadata = {
        "version": DATASET_VERSION,
        "seed": seed,
        "split": episode_split(seed),
        "task": TASK_INSTRUCTION,
        "fps": fps,
        "image_size": [height, width],
        "capture_images": capture_images,
        "physics_randomization": summary["randomization"],
        "visual_randomization": recorder.visual_randomization,
        "validation": validation,
        "expert": {
            "success": summary["expert_success"],
            "simulation_seconds": summary["elapsed_seconds"],
            "recovery_count": summary["recovery_count"],
            "collision_count": summary["collision_count"],
            "final_xy_error_m": summary["final_xy_error_m"],
        },
        "friction_mission_config": asdict(recorder.mission.config),
        "friction_task_config": asdict(recorder.friction_task.config),
        "friction_expert_config": asdict(recorder.friction_expert_config),
    }
    digest, byte_count = save_episode(
        episode_path,
        trajectory,
        metadata,
    )
    return {
        "seed": seed,
        "split": episode_split(seed),
        "path": str(episode_path),
        "metadata_path": str(episode_path.with_suffix(".json")),
        "sha256": digest,
        "bytes": byte_count,
        "frames": validation["frames"],
        "duration_seconds": validation["duration_seconds"],
        "capture_images": capture_images,
        "validation": validation,
    }

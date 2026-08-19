"""Parquet loading and MuJoCo playback for the openarm_sim viewer.

Two playback modes:

- ``dynamic`` (default): the 16-dim joint targets are tracked by the existing
  torque-PD controller (``BimanualCartesianController.compute_ctrl``), i.e. the
  sim shows what the robot would physically do following the recording.
- ``kinematic``: ``states`` (the measured/obs pose) are written directly each
  control frame as the robot's ``actual`` pose; with an action track loaded,
  ``target_state()`` still reports the commanded pose, so the target/actual
  table shows the real robot's commanded-vs-measured pair (no integration).

Real-data episodes (``real_data.load_real_episode``) may carry **two** 16-dim
sequences: ``states`` = obs (measured) and ``targets`` = action (commanded).
They support both modes: kinematic replays obs directly without contact
response, while dynamic starts at the first obs pose and physically follows
the action track with MuJoCo contact enabled. Only a live SSE source is locked
to the authoritative kinematic Reality mode.

All MuJoCo/EGL objects must be created and used inside ONE thread (the sim
thread): the EGL context is thread-local.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
import ctypes
from dataclasses import dataclass
from dataclasses import field
import io
import os
from pathlib import Path
import time

import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
from PIL import Image
import polars as pl

from openarm_mission.config import ControllerConfig
from openarm_mission.config import FrictionMissionConfig
from openarm_mission.controller import BimanualCartesianController
from openarm_mission.dataset import CAMERAS
from openarm_mission.dataset import hide_collision_geomgroups
from openarm_mission.model import OpenArmMission


@dataclass(frozen=True)
class ViewerMissionConfig(FrictionMissionConfig):
    """Viewer scene: robot + table (with legs) + bottle + region markers.

    Matches the expert data-collection scene (``FrictionMissionConfig``) so the
    replayed arm trajectories interact with the same physical bottle/markers
    that produced them.
    """

    include_cup: bool = True
    include_region_markers: bool = True
    table_legs: bool = True


@dataclass(frozen=True)
class RealityViewerConfig(FrictionMissionConfig):
    """Reality mirror: robot plus a visual-only table, without task objects."""

    include_cup: bool = False
    include_region_markers: bool = False
    table_legs: bool = True


STATE_DIM = 16
DEFAULT_DATA_DIR = (
    Path(__file__).resolve().parents[1] / "artifacts" / "p10" / "openarm_paper_cup_relay"
)


def list_episodes(data_dir: Path) -> list[Path]:
    """Sorted ``episode_*.parquet`` files under a LeRobot data directory.

    Recurses into ``chunk-*`` so ``--data-dir`` may point at either the
    ``data/`` root or a single chunk directory.
    """
    return sorted(data_dir.rglob("episode_*.parquet"))


@dataclass
class EpisodeData:
    """One episode: joint targets + lazily loaded recorded images.

    ``image_factory`` (optional, e.g. real-robot JPEG streams) builds the
    per-frame byte list of one viewer camera; without it the recorded images
    are read from the LeRobot parquet image columns.
    """

    path: Path
    name: str
    states: np.ndarray  # (N, 16) float32 — measured/actual pose (real-data obs track)
    timestamps: np.ndarray  # (N,) float64
    fps: int
    has_images: bool
    targets: np.ndarray | None = None  # (N, 16) float32, optional commanded/action track
    reality: bool = False  # states are authoritative measured robot positions
    live: bool = False  # live Actual streams cannot be replayed dynamically
    _img_cache: dict[str, list[bytes]] = field(default_factory=dict)
    image_factory: Callable[[str], list[bytes]] | None = None

    @property
    def frame_count(self) -> int:
        return int(self.states.shape[0])

    def recorded_image_bytes(self, frame_index: int, camera: str) -> bytes | None:
        """JPEG/PNG bytes of the recorded view at ``frame_index`` (lazy per camera)."""
        if not self.has_images:
            return None
        if camera not in self._img_cache:
            if self.image_factory is not None:
                self._img_cache[camera] = self.image_factory(camera)
            else:
                column = f"observation.images.{camera}"
                rows = pl.read_parquet(self.path, columns=[column]).get_column(column).to_list()
                self._img_cache[camera] = [row["bytes"] for row in rows]
        return self._img_cache[camera][frame_index]


def _set_lookat(cam: mujoco.MjvCamera, x: float, y: float, z: float) -> None:
    """Write the free-camera lookat point into the C struct.

    MuJoCo 2.3.7 binding bug: the ``cam.lookat`` numpy view has strides=(0,),
    so any normal assignment aliases a single double. Write the three real
    contiguous doubles through ctypes instead.
    """
    buf = (ctypes.c_double * 3).from_address(cam.lookat.ctypes.data)
    buf[0], buf[1], buf[2] = float(x), float(y), float(z)


def _set_scene_geom_rgba(geom: mujoco.MjvGeom, rgba: tuple[float, float, float, float]) -> None:
    """Write RGBA around the MuJoCo 2.3.7 zero-stride numpy-view bug."""
    buf = (ctypes.c_float * 4).from_address(geom.rgba.ctypes.data)
    for index, value in enumerate(rgba):
        buf[index] = float(value)


def load_episode(path: Path, state_col: str = "observation.state", fps: int = 20) -> EpisodeData:
    """Load and validate the 16-dim joint target sequence of one parquet file."""
    path = Path(path)
    schema_names = list(pl.read_parquet_schema(path).keys())
    if state_col not in schema_names:
        raise ValueError(f"parquet {path.name} has no column {state_col!r}; available: {schema_names}")
    columns = [state_col] + (["timestamp"] if "timestamp" in schema_names else [])
    table = pl.read_parquet(path, columns=columns)
    states = np.asarray(table.get_column(state_col).to_list(), dtype=np.float32)
    if states.ndim != 2 or states.shape[1] != STATE_DIM:
        raise ValueError(f"column {state_col!r} must hold {STATE_DIM}-dim rows, got shape {states.shape}")
    if not np.all(np.isfinite(states)):
        raise ValueError(f"column {state_col!r} contains non-finite values")
    # Canonical viewer convention is physical opening in metres: 0 = closed,
    # 0.044 = fully open. LeRobot already stores model finger opening, so no
    # inversion is needed at the loader or at the MuJoCo boundary.
    if "timestamp" in columns:
        timestamps = np.asarray(table.get_column("timestamp").to_list(), dtype=np.float64)
    else:
        timestamps = np.arange(states.shape[0], dtype=np.float64) / fps
    has_images = all(f"observation.images.{cam}" in schema_names for cam in CAMERAS)
    return EpisodeData(
        path=path,
        name=path.stem,
        states=states,
        timestamps=timestamps,
        fps=fps,
        has_images=has_images,
    )


class SimPlayback:
    """Owns the mission, controller and renderer; call start() in the sim thread."""

    def __init__(
        self,
        *,
        render_width: int = 320,
        render_height: int = 240,
        free_width: int = 640,
        free_height: int = 480,
        jpeg_quality: int = 80,
        scene: str = "mission",
    ):
        self._render_width = render_width
        self._render_height = render_height
        self._free_width = free_width
        self._free_height = free_height
        self._jpeg_quality = jpeg_quality
        if scene not in ("mission", "reality"):
            raise ValueError(f"unknown scene {scene!r}")
        self._scene = scene
        self.mission: OpenArmMission | None = None
        self.controller: BimanualCartesianController | None = None
        self._renderer: mujoco.Renderer | None = None
        self._free_renderer: mujoco.Renderer | None = None
        self._target_data: mujoco.MjData | None = None
        self._target_perturb: mujoco.MjvPerturb | None = None
        self._target_ghost_enabled = True
        self._target_ghost_active = False
        self._free_cam: mujoco.MjvCamera | None = None
        self._follow = False
        self._scene_option: mujoco.MjvOption | None = None
        self._episode: EpisodeData | None = None
        self._mode = "dynamic"
        self._k = 0  # current frame index (targets[0..k] applied)
        self._jpeg_buf = io.BytesIO()
        self._joint_ranges: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self._gripper_limits = (0.0, 0.044)
        self._table_body_id: int = -1

    # -- lifecycle (sim thread only) -------------------------------------
    def start(self) -> None:
        config = RealityViewerConfig() if self._scene == "reality" else ViewerMissionConfig()
        self.mission = OpenArmMission(config)
        if self._scene == "reality":
            # The configured table is a spatial reference only. Disable contact
            # on both its top and legs even if model defaults change later.
            for geom_id in range(self.mission.model.ngeom):
                name = mujoco.mj_id2name(self.mission.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                if name == "mission_table_top" or name.startswith("mission_table_leg_"):
                    self.mission.model.geom_contype[geom_id] = 0
                    self.mission.model.geom_conaffinity[geom_id] = 0
        self.controller = BimanualCartesianController(self.mission, ControllerConfig())
        # A second data object carries commanded Target kinematics only. It
        # shares the compiled model but never participates in mj_step/contact;
        # its dynamic geoms are appended to the render scene as a translucent
        # overlay after the authoritative Actual scene has been built.
        self._target_data = mujoco.MjData(self.mission.model)
        self._target_perturb = mujoco.MjvPerturb()
        # Real-robot recordings may sit slightly outside the v1 model limits;
        # clamp targets so qpos writes and PD tracking always stay valid.
        self._joint_ranges = {
            side: (
                np.asarray(self.mission.model.jnt_range[arm.joint_ids, 0], dtype=np.float64),
                np.asarray(self.mission.model.jnt_range[arm.joint_ids, 1], dtype=np.float64),
            )
            for side, arm in self.mission.arms.items()
        }
        self._gripper_limits = (0.0, float(self.mission.config.open_finger_qpos))
        self._renderer = mujoco.Renderer(self.mission.model, height=self._render_height, width=self._render_width)
        self._free_renderer = mujoco.Renderer(self.mission.model, height=self._free_height, width=self._free_width)
        for renderer in (self._renderer, self._free_renderer):
            for flag in (
                mujoco.mjtRndFlag.mjRND_SHADOW,
                mujoco.mjtRndFlag.mjRND_REFLECTION,
                mujoco.mjtRndFlag.mjRND_SKYBOX,
                mujoco.mjtRndFlag.mjRND_FOG,
                mujoco.mjtRndFlag.mjRND_HAZE,
            ):
                renderer.scene.flags[flag] = 0
        self._scene_option = mujoco.MjvOption()
        # Groups 3+ are dense collision/debug meshes: not visual, and they make
        # EGL rendering much slower. ctypes-safe write (2.3.7 binding bug).
        hide_collision_geomgroups(self._scene_option)
        self._table_body_id = mujoco.mj_name2id(
            self.mission.model, mujoco.mjtObj.mjOBJ_BODY, "mission_table"
        )
        self.reset_camera()

    def close(self) -> None:
        for renderer in (self._renderer, self._free_renderer):
            if renderer is not None:
                # Same teardown fallback as dataset.py _close_media.
                try:
                    renderer.close()
                except AttributeError:
                    renderer._gl_context.free()  # noqa: SLF001
        self._renderer = None
        self._free_renderer = None
        self._target_data = None
        self._target_perturb = None
        self._target_ghost_active = False

    # -- interactive free camera -------------------------------------------
    def _table_top_z(self) -> float:
        """Actual table-top z from the current body position (not config)."""
        assert self.mission is not None
        return float(self.mission.model.body_pos[self._table_body_id][2] + self.mission.config.table_half_size[2])

    @property
    def table_top_z(self) -> float:
        return self._table_top_z()

    def reset_camera(self) -> None:
        """Restore the default free-camera orbit (table center lookat)."""
        assert self.mission is not None
        assert self._free_renderer is not None
        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        cam.azimuth = 90.0
        cam.elevation = -25.0
        cam.distance = 1.2
        _set_lookat(cam, self.mission.config.table_center[0], 0.0, self._table_top_z() + 0.05)
        self._free_cam = cam
        # One update so mjv_moveCamera has a valid scene to work against.
        self._free_renderer.update_scene(self.mission.data, camera=cam, scene_option=self._scene_option)

    def set_table_height(self, height_m: float) -> None:
        """Set table-top height in meters by moving the table body vertically."""
        assert self.mission is not None
        new_z = float(height_m) - self.mission.config.table_half_size[2]
        self.mission.model.body_pos[self._table_body_id][2] = new_z
        mujoco.mj_forward(self.mission.model, self.mission.data)
        self.reset_camera()

    def move_camera(self, action: str, dx: float, dy: float) -> None:
        """Orbit / pan / zoom the free camera with native viewer semantics."""
        assert self.mission is not None
        assert self._free_renderer is not None
        assert self._free_cam is not None
        model = self.mission.model
        scene = self._free_renderer.scene
        cam = self._free_cam
        move = mujoco.mjv_moveCamera
        if action == "rotate":
            move(model, mujoco.mjtMouse.mjMOUSE_ROTATE_H, dx, 0.0, scene, cam)
            move(model, mujoco.mjtMouse.mjMOUSE_ROTATE_V, 0.0, dy, scene, cam)
        elif action == "pan":
            move(model, mujoco.mjtMouse.mjMOUSE_MOVE_H, dx, 0.0, scene, cam)
            move(model, mujoco.mjtMouse.mjMOUSE_MOVE_V, 0.0, dy, scene, cam)
        elif action == "zoom":
            move(model, mujoco.mjtMouse.mjMOUSE_ZOOM, 0.0, dy, scene, cam)
        else:
            raise ValueError(f"unknown camera action {action!r}")

    def set_follow(self, *, enabled: bool) -> None:
        """Keep the free-camera lookat pinned to the cup when enabled."""
        self._follow = bool(enabled)

    # -- episode / mode / seek -------------------------------------------
    def load_episode(self, episode: EpisodeData) -> None:
        self._episode = episode
        if episode.live:
            self._mode = "kinematic"
        self._reset_to(0)

    def apply_live_frame(
        self,
        actual: np.ndarray,
        target: np.ndarray | None,
        timestamp_s: float,
    ) -> None:
        """Apply one authoritative live Actual frame without resetting MuJoCo."""
        actual = np.asarray(actual, dtype=np.float32)
        if actual.shape != (STATE_DIM,) or not np.all(np.isfinite(actual)):
            raise ValueError(f"live Actual must be finite ({STATE_DIM},), got {actual.shape}")
        targets = None
        if target is not None:
            target = np.asarray(target, dtype=np.float32)
            if target.shape != (STATE_DIM,) or not np.all(np.isfinite(target)):
                raise ValueError(f"live Target must be finite ({STATE_DIM},), got {target.shape}")
            targets = target.reshape(1, STATE_DIM)
        if self._episode is None or self._episode.name != "LIVE — OpenArm Panel":
            self._episode = EpisodeData(
                path=Path("LIVE"),
                name="LIVE — OpenArm Panel",
                states=actual.reshape(1, STATE_DIM),
                targets=targets,
                timestamps=np.asarray([timestamp_s], dtype=np.float64),
                fps=30,
                has_images=False,
                reality=True,
                live=True,
            )
        else:
            self._episode.states[0] = actual
            self._episode.targets = targets
            self._episode.timestamps[0] = float(timestamp_s)
        self._mode = "kinematic"
        self._k = 0
        self._apply_frame(0)

    @property
    def episode(self) -> EpisodeData | None:
        return self._episode

    @property
    def mode(self) -> str:
        return self._mode

    @property
    def frame_index(self) -> int:
        return self._k

    @property
    def finished(self) -> bool:
        return self._episode is not None and self._k >= self._episode.frame_count - 1

    def set_mode(self, mode: str) -> None:
        if mode not in ("dynamic", "kinematic"):
            raise ValueError(f"unknown mode {mode!r}")
        if self._episode is not None and self._episode.live and mode != "kinematic":
            raise ValueError("live Actual sources only support kinematic Reality mode")
        if mode == self._mode:
            return
        self._mode = mode
        self.seek(self._k)  # re-anchor the sim to the current frame

    def seek(self, frame_index: int) -> None:
        assert self._episode is not None
        k = int(np.clip(frame_index, 0, self._episode.frame_count - 1))
        self._reset_to(k)

    def _reset_to(self, k: int) -> None:
        assert self.mission is not None
        assert self.controller is not None
        self.mission.reset()
        self.controller.reset_targets()
        if self._mode == "kinematic":
            self._k = k
            self._apply_frame(k)
        elif self._episode.reality:
            # A real recording must begin from its measured first pose. From
            # frame 1 onward, action targets drive MuJoCo so table/robot/object
            # contacts can change the motion instead of starting from the
            # unrelated mission reset pose.
            self._place_state(self._episode.states[0], timestamp=float(self._episode.timestamps[0]))
            self._k = 0
            for i in range(1, k + 1):
                self._apply_frame(i)
            self._k = k
        else:
            # Re-simulate frames 0..k so the physical state is consistent.
            self._k = 0
            for i in range(k + 1):
                self._apply_frame(i)
            self._k = k

    # -- stepping ----------------------------------------------------------
    def step(self) -> bool:
        """Advance one control frame; False when the episode is over."""
        if self._episode is None or self.finished:
            return False
        self._k += 1
        self._apply_frame(self._k)
        return True

    def _apply_frame(self, k: int) -> None:
        assert self.mission is not None
        assert self.controller is not None
        assert self._episode is not None
        # Which 16-dim sequence drives this frame differs by mode:
        #   dynamic  → the commanded/action track when present, so the physics
        #              shows how the robot would follow the recorded commands;
        #              action-less episodes keep the old behavior (the recorded
        #              pose is itself the target).
        #   kinematic → the measured/obs track: qpos is placed exactly on the
        #              recorded real state, so `actual` is the measured robot
        #              and `target` (still the action track) overlays the
        #              commanded-vs-measured comparison.
        if self._mode == "kinematic":
            target = self._episode.states[k]
        elif self._episode.targets is not None:
            target = self._episode.targets[k]
        else:
            target = self._episode.states[k]
        # Gripper columns use physical opening metres, the same convention as
        # the v1 model finger qpos: 0 = closed, hi = fully open.
        lo_g, hi_g = self._gripper_limits
        if self._mode == "dynamic":
            data = self.mission.data
            for side, off in (("left", 0), ("right", 8)):
                arm = self.mission.arms[side]
                lo, hi = self._joint_ranges[side]
                self.controller.target_joint_qpos[side] = np.clip(
                    np.asarray(target[off : off + 7], dtype=np.float64), lo, hi
                )
                grip = float(np.clip(target[off + 7], lo_g, hi_g))
                self.controller.target_gripper[side] = grip
                # Drive the fingers kinematically even in dynamic mode. The v1
                # finger servo is underdamped and, when the recorded "closed"
                # pose has nothing between the pads (the real-data grasp pose
                # does not line up with the viewer-scene cup), overshoots past
                # the closed limit into the pad-pad contact. The two ellipsoid
                # pads then self-lock (curved surfaces near their apex give a
                # huge normal-force mechanical advantage) and can never reopen,
                # pinning "actual" at closed for the rest of the episode.
                # Placing the qpos directly (the finger servo is far faster
                # than the arm, so it tracks its target ~instantly at replay
                # fps) keeps the recorded gripper motion while the arms still
                # follow dynamically. The command is clamped to the model's
                # finger range, so targets never place the pads into contact.
                data.qpos[arm.finger_qpos_indices] = grip
                data.qvel[[self.mission.model.jnt_dofadr[j] for j in arm.finger_joint_ids]] = 0.0
            substeps = max(1, round(1.0 / (self._episode.fps * self.mission.config.timestep)))
            for _ in range(substeps):
                self.controller.compute_ctrl()
                mujoco.mj_step(self.mission.model, self.mission.data)
        else:
            self._place_state(target, timestamp=float(self._episode.timestamps[k]))

    def _place_state(self, state: np.ndarray, *, timestamp: float) -> None:
        """Place one recorded pose directly, without applying contact response."""
        assert self.mission is not None
        data = self.mission.data
        lo_g, hi_g = self._gripper_limits
        for side, off in (("left", 0), ("right", 8)):
            arm = self.mission.arms[side]
            lo, hi = self._joint_ranges[side]
            data.qpos[arm.qpos_indices] = np.clip(state[off : off + 7], lo, hi)
            grip = float(np.clip(state[off + 7], lo_g, hi_g))
            data.qpos[arm.finger_qpos_indices] = grip
            data.ctrl[arm.finger_actuator_ids] = grip
        data.qvel[:] = 0.0
        data.time = timestamp
        mujoco.mj_forward(self.mission.model, data)

    # -- observation ---------------------------------------------------------
    def target_state(self) -> np.ndarray | None:
        """16-dim commanded target (the action track), or the recorded state when
        no action track exists. In kinematic mode this is overlaid on the
        measured ``actual_state()`` so target/actual row shows the real robot's
        commanded-vs-measured pair, not a physics replay."""
        assert self._episode is not None
        if self._episode.targets is None and self._episode.reality:
            return None
        seq = self._episode.targets if self._episode.targets is not None else self._episode.states
        return seq[self._k]

    def actual_state(self) -> np.ndarray:
        """16-dim actual state, same order as the dataset (L7+grip, R7+grip).

        Kinematic Reality replay reports the original measured row so the table
        never hides model-limit clamping. Dynamic replay reports MuJoCo.
        """
        assert self.mission is not None
        assert self._episode is not None
        if self._episode.reality and self._mode == "kinematic":
            return np.asarray(self._episode.states[self._k], dtype=np.float32)
        hi_g = self._gripper_limits[1]
        values: list[float] = []
        for side in ("left", "right"):
            arm = self.mission.arms[side]
            values.extend(np.asarray(self.mission.data.qpos[arm.qpos_indices], dtype=np.float64).tolist())
            values.append(float(np.clip(np.mean(self.mission.data.qpos[arm.finger_qpos_indices]), 0.0, hi_g)))
        return np.asarray(values, dtype=np.float32)

    def render_clamped(self) -> bool:
        """Whether the authoritative Actual row exceeded MuJoCo pose limits."""
        if self._episode is None or not self._episode.reality or self._mode != "kinematic":
            return False
        raw = self._episode.states[self._k]
        for side, off in (("left", 0), ("right", 8)):
            lo, hi = self._joint_ranges[side]
            if np.any(raw[off : off + 7] < lo) or np.any(raw[off : off + 7] > hi):
                return True
        lo_g, hi_g = self._gripper_limits
        return bool(raw[7] < lo_g or raw[7] > hi_g or raw[15] < lo_g or raw[15] > hi_g)

    def frame_time(self) -> float:
        assert self._episode is not None
        return float(self._episode.timestamps[self._k])

    def _prepare_target_ghost(self) -> bool:
        """Forward Target in a render-only ``MjData`` without touching Actual."""
        self._target_ghost_active = False
        if not self._target_ghost_enabled or self._scene != "reality" or self._episode is None:
            return False
        target = self.target_state()
        if target is None:
            return False
        assert self.mission is not None
        assert self._target_data is not None
        data = self._target_data
        # Preserve every non-arm qpos from the Actual scene, then replace only
        # the 16 canonical arm/gripper coordinates with commanded Target.
        data.qpos[:] = self.mission.data.qpos
        data.qvel[:] = 0.0
        lo_g, hi_g = self._gripper_limits
        for side, off in (("left", 0), ("right", 8)):
            arm = self.mission.arms[side]
            lo, hi = self._joint_ranges[side]
            data.qpos[arm.qpos_indices] = np.clip(target[off : off + 7], lo, hi)
            data.qpos[arm.finger_qpos_indices] = float(np.clip(target[off + 7], lo_g, hi_g))
        data.time = self.mission.data.time
        mujoco.mj_forward(self.mission.model, data)
        self._target_ghost_active = True
        return True

    def _append_target_ghost(self, renderer: mujoco.Renderer) -> None:
        """Append cyan, non-physical Target geoms to an already-built scene."""
        if not self._target_ghost_active:
            return
        assert self.mission is not None
        assert self._target_data is not None
        assert self._target_perturb is not None
        start = renderer.scene.ngeom
        mujoco.mjv_addGeoms(
            self.mission.model,
            self._target_data,
            self._scene_option,
            self._target_perturb,
            mujoco.mjtCatBit.mjCAT_DYNAMIC,
            renderer.scene,
        )
        # Reality has no dynamic task object, so this category is exactly the
        # arms and mounted visual geometry. Styling MjvGeom copies cannot
        # affect model collision/contact properties.
        for index in range(start, renderer.scene.ngeom):
            geom = renderer.scene.geoms[index]
            _set_scene_geom_rgba(geom, (0.12, 0.82, 1.0, 0.04))
            geom.emission = 0.25
            geom.specular = 0.0
            geom.shininess = 0.0
            geom.reflectance = 0.0

    def target_ghost_visible(self) -> bool:
        """Whether the most recent render included a valid Target overlay."""
        return self._target_ghost_active

    @property
    def target_ghost_enabled(self) -> bool:
        return self._target_ghost_enabled

    def set_target_ghost(self, *, enabled: bool) -> None:
        """Enable or hide the render-only Target overlay."""
        self._target_ghost_enabled = bool(enabled)
        if not self._target_ghost_enabled:
            self._target_ghost_active = False

    def render(self) -> dict[str, np.ndarray]:
        """Render the three dataset cameras plus the interactive free view."""
        assert self.mission is not None
        assert self._renderer is not None
        assert self._free_renderer is not None
        assert self._free_cam is not None
        self._prepare_target_ghost()
        images = {}
        for key, camera in CAMERAS.items():
            self._renderer.update_scene(self.mission.data, camera=camera, scene_option=self._scene_option)
            self._append_target_ghost(self._renderer)
            images[key] = np.array(self._renderer.render(), dtype=np.uint8, copy=True)
        if self._follow and self.mission.has_cup:
            cup = self.mission.cup_position()
            _set_lookat(self._free_cam, cup[0], cup[1], float(cup[2]) + 0.05)
        self._free_renderer.update_scene(self.mission.data, camera=self._free_cam, scene_option=self._scene_option)
        self._append_target_ghost(self._free_renderer)
        images["free"] = np.array(self._free_renderer.render(), dtype=np.uint8, copy=True)
        return images

    def encode_jpeg(self, image: np.ndarray) -> bytes:
        self._jpeg_buf.seek(0)
        self._jpeg_buf.truncate()
        Image.fromarray(image).save(self._jpeg_buf, format="JPEG", quality=self._jpeg_quality)
        return self._jpeg_buf.getvalue()

    def recorded_jpeg(self, camera: str) -> bytes | None:
        """Re-encode the recorded parquet image of the current frame as JPEG."""
        assert self._episode is not None
        raw = self._episode.recorded_image_bytes(self._k, camera)
        if raw is None:
            return None
        image = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
        return self.encode_jpeg(image)


def _selftest(episode: EpisodeData) -> None:
    print(f"loaded {episode.name}: {episode.frame_count} frames @ {episode.fps} Hz, images={episode.has_images}")
    playback = SimPlayback()
    t0 = time.time()
    playback.start()
    print(f"sim startup {time.time() - t0:.2f}s")

    # Viewer scene matches the expert scene: bottle + region markers + legs.
    assert playback.mission.has_cup
    viewer_model = playback.mission.model
    assert mujoco.mj_name2id(viewer_model, mujoco.mjtObj.mjOBJ_GEOM, "mission_cup_body") >= 0
    assert mujoco.mj_name2id(viewer_model, mujoco.mjtObj.mjOBJ_GEOM, "mission_region_a") >= 0
    assert mujoco.mj_name2id(viewer_model, mujoco.mjtObj.mjOBJ_GEOM, "mission_region_b") >= 0
    assert mujoco.mj_name2id(viewer_model, mujoco.mjtObj.mjOBJ_GEOM, "mission_handoff_marker") >= 0
    assert mujoco.mj_name2id(viewer_model, mujoco.mjtObj.mjOBJ_GEOM, "mission_table_leg_pp") >= 0
    assert OpenArmMission(FrictionMissionConfig()).has_cup
    print("viewer scene ok: cup + region markers + table legs (matches expert scene)")
    playback.load_episode(episode)

    playback.set_mode("dynamic")
    t0 = time.time()
    for _ in range(5):
        assert playback.step()
    dt = (time.time() - t0) / 5
    actual = playback.actual_state()
    assert actual.shape == (16,)
    assert np.all(np.isfinite(actual))
    print(f"dynamic step {dt * 1000:.1f} ms/frame, frame={playback.frame_index}")

    t0 = time.time()
    images = playback.render()
    print(f"render 3 cams + free {(time.time() - t0) * 1000:.1f} ms")
    assert images["free"].shape[:2] == (playback._free_height, playback._free_width)  # noqa: SLF001

    free0 = images["free"].astype(np.int32)
    playback.move_camera("rotate", 0.3, 0.15)
    free1 = playback.render()["free"].astype(np.int32)
    rotated = float(np.abs(free1 - free0).mean())
    assert rotated > 1.0, rotated
    playback.move_camera("zoom", 0.0, -0.2)
    playback.move_camera("pan", 0.1, 0.0)
    playback.reset_camera()
    free2 = playback.render()["free"].astype(np.int32)
    reset_err = float(np.abs(free2 - free0).mean())
    assert reset_err < 1.0, reset_err
    print(f"free camera ok: rotate diff={rotated:.1f}, reset err={reset_err:.2f}")

    playback.set_mode("kinematic")
    playback.seek(100)
    playback.step()
    err = float(np.abs(playback.actual_state() - playback.target_state()).max())
    assert err < 1e-4, err
    print(f"kinematic exact: max|actual-target|={err:.2e} at frame {playback.frame_index}")

    playback.set_mode("dynamic")
    playback.seek(episode.frame_count - 1)
    assert playback.finished
    assert not playback.step()
    print("seek to end + finished flag ok")
    playback.close()
    print("selftest OK")


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline selftest for openarm_sim playback.")
    parser.add_argument("--parquet", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    args = parser.parse_args()
    if args.parquet is not None:
        episode = load_episode(args.parquet)
    else:
        parquets = list_episodes(args.data_dir)
        if parquets:
            episode = load_episode(parquets[0])
        else:
            # v0.3.0 dataset root (real_data tree or a converted sim set). Lazy
            # import to avoid a cycle (real_data imports EpisodeData from here).
            from openarm_mission.openarm_sim import real_data as _real

            real_dirs = _real.list_real_episodes(args.data_dir)
            if not real_dirs:
                raise SystemExit(
                    f"no episodes (episode_*.parquet / v0.3.0) found under {args.data_dir}"
                )
            episode = _real.load_real_episode(real_dirs[0])
    _selftest(episode)


if __name__ == "__main__":
    main()

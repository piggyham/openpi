"""Loader for OpenArm dataset-format v0.3.0 episodes.

Two layouts are discovered (``_episode_root``):

- legacy real-robot teleop tree: ``<data-dir>/real_data/episodes/<id>``
  (naming prefix ``real_``);
- a v0.3.0 dataset root itself: ``<data-dir>`` containing ``metadata.yaml``
  + ``episodes/<id>``, e.g. the converted P10 simulation set at
  ``artifacts/p10/openarm_paper_cup_relay`` (naming prefix ``sim_``).

Structure (OpenArm dataset format v0.3.0, matching the vendor
``openarm_dataset`` converter shipped next to the data):

    <episodes>/<id>/obs/arms/{left,right}/state.parquet
    <episodes>/<id>/cameras/{head,wrist_left,wrist_right}/<ns>.jpeg

Sampling uses a uniform grid over the common target/actual time range: Actual
is linearly interpolated, commanded Target is zero-order-held, and cameras use
nearest frames. The 8-dim
per-arm qpos (7 joints + gripper) is mapped into the viewer's 16-dim state
order (left arm first, see ``dataset.STATE_NAMES``):

- joints pass through unchanged (the real robot and the v1 sim model share the
  joint-zero convention; ``SimPlayback`` clamps to the model limits);
- new Panel datasets declare ``gripper_encoding: opening_m`` and pass through
  physical opening metres (0=closed, 0.044=open); legacy datasets without the
  declaration use normalized negative raw (0=closed, -1=open).

Recorded camera JPEGs are exposed through the same lazy per-camera interface
as the LeRobot parquet images (``EpisodeData.image_factory``), with the real
camera names mapped onto the viewer keys (head -> front, ...).
"""

from __future__ import annotations

from pathlib import Path
import warnings

import numpy as np
import polars as pl
import yaml

from openarm_mission.config import ControllerConfig
from openarm_mission.openarm_sim.playback import DEFAULT_DATA_DIR
from openarm_mission.openarm_sim.playback import EpisodeData

REAL_FPS = 30  # viewer rate; raw Panel parquet remains at its recorded 200 Hz
# Viewer camera key -> real-robot camera directory name.
CAMERA_DIRS = {"front": "head", "left_wrist": "wrist_left", "right_wrist": "wrist_right"}
_GRIPPER_OPEN_M = ControllerConfig().gripper_open_position


def _episode_root(data_dir: Path) -> Path | None:
    """The ``episodes/`` dir discoverable under ``data_dir``, or None.

    Accepts two layouts that share the OpenArm dataset format v0.3.0:

    - legacy real-robot tree: ``<data-dir>/real_data/episodes/``
    - a v0.3.0 dataset root itself (``metadata.yaml`` + ``episodes/``),
      e.g. the converted P10 simulation set at
      ``artifacts/p10/openarm_paper_cup_relay/``.
    """
    data_dir = Path(data_dir)
    legacy = data_dir / "real_data" / "episodes"
    if legacy.is_dir():
        return legacy
    if (data_dir / "metadata.yaml").is_file() and (data_dir / "episodes").is_dir():
        return data_dir / "episodes"
    return None


def _dataset_type(data_dir: Path) -> str:
    """``real`` (teleop) or ``sim`` (scripted) for a v0.3.0 data root.

    Reads ``operation_type`` from ``metadata.yaml`` when present, falling back
    to the directory layout. A legacy real-robot tree (``<data-dir>/real_data``
    or a bare ``real_data`` dir passed directly) is ``real``; a converted
    simulation root is ``sim``.
    """
    meta = Path(data_dir) / "metadata.yaml"
    if meta.is_file():
        with open(meta) as f:
            parsed = yaml.safe_load(f) or {}
        op = str(parsed.get("operation_type", "")).lower()
        if op:
            return "real" if op == "teleop" else "sim"
    # Layout fallback: a v0.3.0 dataset root has metadata.yaml next to episodes/.
    return "sim" if _episode_root(Path(data_dir)) == Path(data_dir) / "episodes" else "real"


def episode_prefix(data_dir: Path) -> str:
    """Dropdown prefix for episodes discovered under ``data_dir``.

    ``real`` for the legacy teleop tree, ``sim`` for a converted simulation
    root (distinguished by ``metadata.yaml``'s ``operation_type``).
    """
    return _dataset_type(data_dir)


def list_real_episodes(data_dir: Path) -> list[Path]:
    """Sorted episode dirs that contain both arm state parquets."""
    root = _episode_root(data_dir)
    if root is None:
        return []
    return [
        ep_dir
        for ep_dir in sorted(root.iterdir(), key=lambda p: (len(p.name), p.name))
        if ep_dir.is_dir()
        and not ep_dir.name.startswith(".")
        and all((ep_dir / f"obs/arms/{side}/state.parquet").is_file() for side in ("left", "right"))
    ]


def _read_arm_table(path: Path, value_col: str) -> tuple[np.ndarray, np.ndarray]:
    """Unix-second timestamps and (N, >=8) per-arm values from one parquet.

    ``value_col`` is ``"qpos"`` for ``obs/arms/*/state.parquet`` and ``"value"``
    for ``action/arms/*/qpos.parquet`` (the panel's recorder writes both).
    """
    if value_col not in pl.read_parquet_schema(path):
        raise ValueError(f"{path} has no column {value_col!r}")
    table = pl.read_parquet(path, columns=["timestamp", value_col])
    times = np.asarray([ts.timestamp() for ts in table.get_column("timestamp").to_list()], dtype=np.float64)
    values = np.asarray(table.get_column(value_col).to_list(), dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 8:
        raise ValueError(f"{path}: column {value_col!r} must hold >=8-dim rows, got {values.shape}")
    if times.shape[0] < 2 or not np.all(np.isfinite(times)) or not np.all(np.isfinite(values)):
        raise ValueError(f"real episode arm file {path} has invalid timestamps or values")
    return times, values[:, :8]


def _nearest_indices(times: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Index of the closest sample in ``times`` for each grid point."""
    idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    prev = np.clip(idx - 1, 0, len(times) - 1)
    return np.where(np.abs(times[prev] - grid) <= np.abs(times[idx] - grid), prev, idx)


def _episode_metadata(ep_dir: Path) -> dict:
    """Nearest metadata.yaml describing an episode, if present."""
    for parent in list(Path(ep_dir).parents)[:4]:
        path = parent / "metadata.yaml"
        if path.is_file():
            with open(path) as stream:
                return yaml.safe_load(stream) or {}
    return {}


def _gripper_opening(raw: np.ndarray, encoding: str) -> np.ndarray:
    """Normalize all dataset variants to opening metres (0 closed, 0.044 open)."""
    raw = np.asarray(raw, dtype=np.float64)
    if encoding == "opening_m":
        return np.clip(raw, 0.0, _GRIPPER_OPEN_M)
    # Historical OpenArm v0.3.0: 0=closed and increasingly negative means open.
    return _GRIPPER_OPEN_M * np.clip(-raw, 0.0, 1.0)


def _name_prefix(ep_dir: Path) -> str:
    """Dropdown prefix for one episode dir: ``sim`` under a v0.3.0 dataset root
    (``<root>/episodes/<id>`` with a ``metadata.yaml`` two levels up), else
    ``real`` for the legacy teleop tree (``<data-dir>/real_data/episodes/<id>``).
    """
    parents = list(Path(ep_dir).parents)
    for parent in (parents[1] if len(parents) >= 2 else None, parents[2] if len(parents) >= 3 else None):
        if parent is not None and (parent / "metadata.yaml").is_file():
            return _dataset_type(parent)
    return "real"


def _sample_values(times: np.ndarray, values: np.ndarray, grid: np.ndarray, method: str) -> np.ndarray:
    if method == "zoh":
        indices = np.clip(np.searchsorted(times, grid, side="right") - 1, 0, len(times) - 1)
        return values[indices]
    if method != "linear":
        raise ValueError(f"unknown resampling method {method!r}")
    return np.column_stack([np.interp(grid, times, values[:, i]) for i in range(values.shape[1])])


def _map16(
    left_t: np.ndarray, left_v: np.ndarray,
    right_t: np.ndarray, right_v: np.ndarray,
    grid: np.ndarray,
    *,
    method: str,
    gripper_encoding: str,
) -> np.ndarray:
    """Map left+right (N, 8) arm values onto the viewer's 16-dim state order.

    Order: left 7 joints + left opening + right 7 joints + right opening. Seven
    joints pass through; grippers normalize to metres (0 closed, 0.044 open).
    """
    left = _sample_values(left_t, left_v, grid, method)
    right = _sample_values(right_t, right_v, grid, method)
    states = np.empty((grid.shape[0], 16), dtype=np.float32)
    states[:, 0:7] = left[:, :7]
    states[:, 7] = _gripper_opening(left[:, 7], gripper_encoding)
    states[:, 8:15] = right[:, :7]
    states[:, 15] = _gripper_opening(right[:, 7], gripper_encoding)
    if not np.all(np.isfinite(states)):
        raise ValueError("mapped state contains non-finite values")
    return states


def load_real_episode(ep_dir: Path, fps: int = REAL_FPS) -> EpisodeData:
    """Resample one real-robot episode onto a uniform ``fps`` grid.

    Loads **two** 16-dim sequences when both are present:

    - ``targets`` = the **action** track (``action/arms/*/qpos.parquet``) — the
      position commands the follower was asked to track;
    - ``states``  = the **obs** track (``obs/arms/*/state.parquet``) — the
      jointly-measured follower state, i.e. the actual robot behaviour.

    A ``real`` episode with only ``obs`` keeps the legacy single-sequence
    behaviour (``targets`` is None and the viewer compares obs against its own
    physics). ``action`` is optional and degrades gracefully if missing or
    timing-truncated relative to ``obs``.
    """
    ep_dir = Path(ep_dir)
    metadata = _episode_metadata(ep_dir)
    gripper_encoding = str(metadata.get("gripper_encoding", "legacy_negative_normalized"))
    operation_type = str(metadata.get("operation_type", "teleop")).lower()
    is_reality = operation_type in ("teleop", "manual", "replay", "real")
    left_t, left_q = _read_arm_table(ep_dir / "obs/arms/left/state.parquet", "qpos")
    right_t, right_q = _read_arm_table(ep_dir / "obs/arms/right/state.parquet", "qpos")
    start = max(left_t[0], right_t[0])
    end = min(left_t[-1], right_t[-1])

    targets = None
    action_data = None
    action_paths = (
        ep_dir / "action" / "arms" / "left" / "qpos.parquet",
        ep_dir / "action" / "arms" / "right" / "qpos.parquet",
    )
    if all(p.is_file() for p in action_paths):
        try:
            la_t, la_v = _read_arm_table(action_paths[0], "value")
            ra_t, ra_v = _read_arm_table(action_paths[1], "value")
            start = max(start, la_t[0], ra_t[0])
            end = min(end, la_t[-1], ra_t[-1])
            action_data = (la_t, la_v, ra_t, ra_v)
        except (ValueError, OSError) as exc:
            warnings.warn(
                f"real episode {ep_dir.name}: could not load action track "
                f"({exc}) — replaying with targets=None (obs only)",
                stacklevel=2,
            )

    if end <= start:
        raise ValueError(f"real episode {ep_dir.name}: target/actual time ranges do not overlap")
    grid = np.arange(start, end, 1.0 / fps)
    if grid.size < 2:
        raise ValueError(f"real episode {ep_dir.name}: common time range is shorter than two frames")
    states = _map16(
        left_t, left_q, right_t, right_q, grid,
        method="linear", gripper_encoding=gripper_encoding,
    )
    if action_data is not None:
        la_t, la_v, ra_t, ra_v = action_data
        targets = _map16(
            la_t, la_v, ra_t, ra_v, grid,
            method="zoh", gripper_encoding=gripper_encoding,
        )

    camera_dirs = {}
    for key, cam in CAMERA_DIRS.items():
        cam_dir = ep_dir / "cameras" / cam
        if cam_dir.is_dir() and any(cam_dir.glob("*.jpeg")):
            camera_dirs[key] = cam_dir
    has_images = len(camera_dirs) == len(CAMERA_DIRS)

    def image_factory(camera: str) -> list[bytes]:
        """JPEG bytes of the temporally nearest real frame per state frame."""
        paths = sorted(camera_dirs[camera].glob("*.jpeg"), key=lambda p: int(p.stem))
        cam_t = np.asarray([int(p.stem) / 1e9 for p in paths], dtype=np.float64)
        return [paths[i].read_bytes() for i in _nearest_indices(cam_t, grid)]

    return EpisodeData(
        path=ep_dir,
        name=f"{_name_prefix(ep_dir)}_{ep_dir.name}",
        states=states,
        targets=targets,
        timestamps=grid - grid[0],
        fps=fps,
        has_images=has_images,
        image_factory=image_factory if has_images else None,
        reality=is_reality,
    )


def main() -> None:
    """Offline selftest: load every real episode and validate the mapping."""
    for ep_dir in list_real_episodes(DEFAULT_DATA_DIR):
        episode = load_real_episode(ep_dir)
        assert episode.states.shape == (episode.frame_count, 16)
        assert np.all(np.isfinite(episode.states))
        grip = episode.states[:, [7, 15]]
        assert grip.min() >= -1e-6
        assert grip.max() <= _GRIPPER_OPEN_M + 1e-6
        if episode.targets is not None:
            assert episode.targets.shape == episode.states.shape
            assert np.all(np.isfinite(episode.targets))
            tg = episode.targets[:, [7, 15]]
            assert tg.min() >= -1e-6
            assert tg.max() <= _GRIPPER_OPEN_M + 1e-6
            track = " (action/obs dual-track)"
        else:
            track = " (obs only)"
        print(
            f"{episode.name}: {episode.frame_count} frames @ {episode.fps} Hz, "
            f"duration={episode.timestamps[-1]:.2f}s, images={episode.has_images}, "
            f"gripper range=[{grip.min():.3f}, {grip.max():.3f}] m{track}"
        )
        if episode.has_images:
            for key in CAMERA_DIRS:
                blob = episode.recorded_image_bytes(0, key)
                assert blob is not None
                assert blob[:3] == b"\xff\xd8\xff"
            print("  recorded jpeg streams ok (front/left_wrist/right_wrist)")
    print("real_data selftest OK")


if __name__ == "__main__":
    main()

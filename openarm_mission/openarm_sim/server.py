"""Web server for the openarm_sim viewer.

One port serves both the single-file HTML frontend (GET /) and the WebSocket
stream (any other path upgrades). A dedicated sim thread owns all MuJoCo/EGL
objects (the EGL context is thread-local) and publishes encoded frames to the
asyncio side through a latest-frame-wins slot; control commands flow the other
way through a thread-safe queue. Multiple browser tabs share one session.

Run:
    MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server --port 8080
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import contextlib
from http import HTTPStatus
import json
import os
from pathlib import Path
import queue
import sys
import threading
import time
from urllib.parse import parse_qs
from urllib.parse import urlparse

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import websockets.asyncio.server as ws_server
from websockets.exceptions import ConnectionClosed

from openarm_mission.dataset import CAMERAS
from openarm_mission.dataset import STATE_NAMES
from openarm_mission.openarm_sim import playback as _playback
from openarm_mission.openarm_sim import real_data as _real
from openarm_mission.openarm_sim.live_source import LiveRealitySource

INDEX_HTML_PATH = Path(__file__).resolve().parent / "web" / "index.html"
SPEED_MIN, SPEED_MAX = 0.25, 4.0
MAX_MODES = ("dynamic", "kinematic")
LIVE_NAME = "LIVE — OpenArm Panel"
EpisodeSpec = tuple[str, str, Path | str]


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _round_list(values: np.ndarray | None, nd: int = 4) -> list[float] | None:
    if values is None:
        return None
    return [round(float(v), nd) for v in values]


class SimEngine:
    """Bridges the asyncio web side and the MuJoCo sim thread."""

    def __init__(self, args: argparse.Namespace, loop: asyncio.AbstractEventLoop):
        self.args = args
        self.loop = loop
        # Recorded specs can be replaced at runtime from the web UI.  The live
        # spec is kept separately so refreshing a shared recording directory
        # never drops the active Panel connection.
        self._data_dirs: list[Path] = []
        if args.parquet is not None:
            self._recorded_specs: list[EpisodeSpec] = [
                (Path(args.parquet).stem, "lerobot", Path(args.parquet))
            ]
            self.single_file = True
        else:
            self._data_dirs = _resolve_data_dirs(args.data_dir)
            self._recorded_specs = _collect_episodes(self._data_dirs)
            self.single_file = False
        self.live_url = str(args.live_url) if args.live_url else None
        self.episode_specs = self._combined_specs()
        if not self.episode_specs:
            raise RuntimeError(
                "no episodes (episode_*.parquet / real_data / v0.3.0 dataset) found in "
                + ", ".join(str(d) for d in self._data_dirs)
            )

        self.playback = _playback.SimPlayback(
            render_width=args.width,
            render_height=args.height,
            free_width=args.free_width,
            free_height=args.free_height,
            jpeg_quality=args.jpeg_quality,
            scene=args.scene,
        )
        self.live = (
            LiveRealitySource(self.live_url, token=os.environ.get("OPENARM_SIM_STREAM_TOKEN", ""))
            if self.live_url else None
        )
        self._active_kind = self.episode_specs[0][1]
        self._last_live_key: tuple[int, int, int] | None = None
        self._last_live_stale: bool | None = None
        self._last_episode_scan = 0.0
        self.playing = True
        self.speed = 1.0
        self.compare = self._active_kind == "live"

        self._commands: queue.Queue = queue.Queue()
        self._clients: dict = {}  # connection -> asyncio.Queue, asyncio side only
        self._frame_slot: dict = {"msg": None}
        self._frame_lock = threading.Lock()
        self._frame_event = asyncio.Event()
        self._publish_pending = True
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._ready_error: BaseException | None = None
        self._seq = 0

    # -- startup ---------------------------------------------------------
    def start(self) -> None:
        if self.live is not None:
            self.live.start()
        thread = threading.Thread(target=self._sim_loop, name="openarm-sim", daemon=True)
        thread.start()
        if not self._ready.wait(timeout=120):
            raise RuntimeError("sim thread failed to become ready in time")
        if self._ready_error is not None:
            raise RuntimeError(f"sim startup failed: {self._ready_error}") from self._ready_error

    def stop(self) -> None:
        self._stop.set()
        if self.live is not None:
            self.live.stop()

    # -- sim thread --------------------------------------------------------
    def _load_spec(self, spec: EpisodeSpec) -> _playback.EpisodeData:
        name, kind, path = spec
        if kind == "live":
            episode = _playback.EpisodeData(
                path=Path("LIVE"), name=name,
                states=np.zeros((1, 16), dtype=np.float32),
                targets=None,
                timestamps=np.zeros(1, dtype=np.float64),
                fps=self.args.live_fps,
                has_images=False,
                reality=True,
                live=True,
            )
        elif kind == "real":
            episode = _real.load_real_episode(Path(path), fps=self.args.real_fps)
        else:
            episode = _playback.load_episode(Path(path), self.args.state_col, self.args.fps)
        # Keep the episode name identical to the dropdown entry (the loader's
        # own default, e.g. ``real_<id>``, may disagree with the prefix chosen
        # for the data root, e.g. ``sim_<id>`` for a converted v0.3.0 set).
        episode.name = name
        return episode

    def _sim_loop(self) -> None:
        try:
            self.playback.start()
            episode = self._load_spec(self.episode_specs[0])
            self.playback.load_episode(episode)
            # Training episodes carry the three captured camera streams inside
            # their LeRobot parquet files.  Show them immediately next to the
            # freshly rendered SIM views instead of requiring the user to find
            # and enable the comparison checkbox first.
            self.compare = self._active_kind == "live" or episode.has_images
        except BaseException as exc:
            self._ready_error = exc
            self._ready.set()
            return
        self._ready.set()

        next_time = time.monotonic()
        while not self._stop.is_set():
            self._drain_commands()
            self._refresh_episode_specs()
            if self._active_kind == "live":
                self._live_tick()
                self._stop.wait(1.0 / self.args.live_fps)
                next_time = time.monotonic()
                continue
            # Per-iteration so both episode fps (20 Hz LeRobot vs 30 Hz real)
            # and speed changes take effect immediately.
            period = 1.0 / (self.playback.episode.fps * self.speed)
            publish = self._publish_pending
            self._publish_pending = False

            if self.playing:
                if self.playback.finished:
                    # e.g. after seeking to the last frame
                    self.playing = False
                    self._push_status()
                else:
                    now = time.monotonic()
                    # Catch-up without rendering if we fell behind schedule.
                    while next_time + period <= now and not self.playback.finished:
                        self.playback.step()
                        next_time += period
                        publish = True
                    if not self.playback.finished:
                        self.playback.step()
                        publish = True
                        next_time += period
                    if self.playback.finished:
                        self.playing = False
                        self._push_status()
            if not self.playing:
                next_time = time.monotonic() + period

            if publish:
                self._publish_frame()
            delay = next_time - time.monotonic()
            if delay > 0:
                self._stop.wait(delay)
        self.playback.close()

    def _live_tick(self) -> None:
        assert self.live is not None
        frame = self.live.latest()
        live_status = self.live.status()
        stale = bool(live_status["stale"])
        if frame is not None:
            key = (frame.generation, frame.seq, frame.sample_ts_ns)
            if key != self._last_live_key and frame.actual_valid:
                self.playback.apply_live_frame(
                    np.asarray(frame.actual, dtype=np.float32),
                    np.asarray(frame.target, dtype=np.float32) if frame.target_valid else None,
                    self.live.elapsed_s(frame),
                )
                self._last_live_key = key
                self._publish_frame()
        if stale != self._last_live_stale:
            self._last_live_stale = stale
            self._push_status()

    def _drain_commands(self) -> None:
        while True:
            try:
                command = self._commands.get_nowait()
            except queue.Empty:
                return
            self._handle_command(command)

    def _combined_specs(self) -> list[EpisodeSpec]:
        live_specs: list[EpisodeSpec] = []
        if self.live_url:
            live_specs.append((LIVE_NAME, "live", self.live_url))
        return live_specs + self._recorded_specs

    @staticmethod
    def _episode_payload(specs: list[EpisodeSpec]) -> list[dict]:
        return [
            {"name": name, "path": str(path), "source": "live" if kind == "live" else "recorded"}
            for name, kind, path in specs
        ]

    def _broadcast_episodes(self) -> None:
        self.loop.call_soon_threadsafe(
            self._enqueue_all,
            {"type": "episodes", "episodes": self._episode_payload(self.episode_specs)},
        )

    def _recorded_path(self) -> str | None:
        if self.single_file and self._recorded_specs:
            return str(self._recorded_specs[0][2])
        if self._data_dirs:
            return str(self._data_dirs[0])
        return None

    def _handle_command(self, command: dict) -> None:
        ctype = command.get("type")
        try:
            if ctype == "play":
                if self.playback.finished:
                    self.playback.seek(0)
                self.playing = True
            elif ctype == "pause":
                self.playing = False
            elif ctype == "set_speed":
                self.speed = float(np.clip(float(command["speed"]), SPEED_MIN, SPEED_MAX))
            elif ctype == "seek":
                self.playback.seek(int(command["frame"]))
                self._publish_pending = True
            elif ctype == "set_mode":
                self.playback.set_mode(str(command["mode"]))
                self._publish_pending = True
            elif ctype == "select_episode":
                self._select_episode(command)
                self._publish_pending = True
            elif ctype == "select_recorded_episode":
                self._select_recorded_episode(int(command["index"]))
                self._publish_pending = True
            elif ctype == "select_source":
                self._select_source(str(command["source"]))
                self._publish_pending = True
            elif ctype == "load_recorded_path":
                self._load_recorded_path(str(command["path"]))
                self._publish_pending = True
            elif ctype == "set_live_source":
                self._set_live_source(str(command["url"]))
                self._publish_pending = True
            elif ctype == "set_compare":
                enabled = bool(command["enabled"])
                if enabled and self._active_kind != "live" and not self.playback.episode.has_images:
                    self._push_error("this parquet has no recorded image columns")
                    enabled = False
                self.compare = enabled
                self._publish_pending = True
            elif ctype == "set_target_ghost":
                self.playback.set_target_ghost(enabled=bool(command["enabled"]))
                self._publish_pending = True
            elif ctype == "camera_move":
                self.playback.move_camera(str(command["action"]), float(command["dx"]), float(command["dy"]))
                self._publish_pending = True
            elif ctype == "camera_reset":
                self.playback.reset_camera()
                self._publish_pending = True
            elif ctype == "camera_follow":
                self.playback.set_follow(enabled=bool(command["enabled"]))
                self._publish_pending = True
            elif ctype == "set_table_height":
                self.playback.set_table_height(float(command["height"]))
                self._publish_pending = True
            else:
                self._push_error(f"unknown command {ctype!r}")
                return
        except (KeyError, TypeError, ValueError, OSError, RuntimeError) as exc:
            self._push_error(f"bad command {ctype!r}: {exc}")
            return
        self._push_status()

    def _refresh_episode_specs(self) -> None:
        """Discover atomically completed shared-directory episodes every 2 s."""
        if self.single_file or not self._data_dirs:
            return
        now = time.monotonic()
        if now - self._last_episode_scan < 2.0:
            return
        self._last_episode_scan = now
        disk = _collect_episodes(self._data_dirs)
        self._recorded_specs = disk
        specs = self._combined_specs()
        old = [(name, kind, str(path)) for name, kind, path in self.episode_specs]
        new = [(name, kind, str(path)) for name, kind, path in specs]
        if new == old:
            return
        self.episode_specs = specs
        self._broadcast_episodes()

    def _select_source(self, source: str) -> None:
        if source == "live":
            if self.live is None:
                raise ValueError("live source is not configured; enter a Panel SSE URL first")
            spec = next(spec for spec in self.episode_specs if spec[1] == "live")
        elif source == "recorded":
            if not self._recorded_specs:
                raise ValueError("no recorded episodes loaded; enter a file or directory path first")
            spec = self._recorded_specs[0]
        else:
            raise ValueError("source must be 'live' or 'recorded'")
        self._activate_spec(spec)

    def _select_recorded_episode(self, index: int) -> None:
        """Select a recorded episode by zero-based ordinal (ROS/API contract)."""
        if not 0 <= index < len(self._recorded_specs):
            raise ValueError(
                f"recorded episode index {index} out of range "
                f"[0, {max(0, len(self._recorded_specs) - 1)}]"
            )
        self._activate_spec(self._recorded_specs[index])

    def _load_recorded_path(self, raw_path: str) -> None:
        path, specs, single_file = _recorded_specs_for_path(raw_path)
        self._recorded_specs = specs
        self._data_dirs = [] if single_file else [path]
        self.single_file = single_file
        self.episode_specs = self._combined_specs()
        self._broadcast_episodes()
        self._activate_spec(self._recorded_specs[0])

    def _set_live_source(self, raw_url: str) -> None:
        url = _validate_live_url(raw_url)
        if self.live is not None:
            self.live.stop()
        self.live_url = url
        self.live = LiveRealitySource(
            url,
            token=os.environ.get("OPENARM_SIM_STREAM_TOKEN", ""),
        )
        self.live.start()
        self.episode_specs = self._combined_specs()
        self._broadcast_episodes()
        self._activate_spec(self.episode_specs[0])

    def _select_episode(self, command: dict) -> None:
        if "index" in command:
            index = int(command["index"])
            if not 0 <= index < len(self.episode_specs):
                raise ValueError(f"episode index {index} out of range")
            spec = self.episode_specs[index]
        else:
            name = str(command["episode"])
            matches = [s for s in self.episode_specs if s[0] == name]
            if not matches:
                raise ValueError(f"unknown episode {name!r}")
            spec = matches[0]
        self._activate_spec(spec)

    def _activate_spec(self, spec: EpisodeSpec) -> None:
        episode = self._load_spec(spec)
        self._active_kind = spec[1]
        self.playback.load_episode(episode)
        self.playing = self._active_kind != "live"
        # Each newly selected episode starts with its recorded camera row
        # visible when all three image columns are available.  The user can
        # still hide it with the comparison checkbox for the active episode.
        self.compare = self._active_kind == "live" or episode.has_images
        self._last_live_key = None
        self._last_live_stale = None

    def _publish_frame(self) -> None:
        images = self.playback.render()
        live_frame = self.live.latest() if self._active_kind == "live" and self.live is not None else None
        msg = {
            "type": "frame",
            "seq": self._seq,
            "frame_index": live_frame.seq if live_frame is not None else self.playback.frame_index,
            "t": round(self.playback.frame_time(), 3),
            "mode": "reality" if self._active_kind == "live" else self.playback.mode,
            "target": _round_list(self.playback.target_state()),
            "actual": _round_list(self.playback.actual_state()),
            "target_ghost": self.playback.target_ghost_visible(),
            "render_clamped": self.playback.render_clamped(),
            "images": {key: _b64(self.playback.encode_jpeg(img)) for key, img in images.items()},
            "recorded": None,
        }
        if self._active_kind == "live" and self.live is not None:
            camera_frames = self.live.camera_frames()
            if camera_frames:
                msg["recorded"] = {
                    key: _b64(camera_frames[key]) if key in camera_frames else None
                    for key in CAMERAS
                }
            msg["live"] = self.live.status()
            msg["live_generation"] = live_frame.generation if live_frame is not None else 0
        elif self.compare and self.playback.episode.has_images:
            msg["recorded"] = {key: _b64(self.playback.recorded_jpeg(key)) for key in CAMERAS}
        self._seq += 1
        with self._frame_lock:
            self._frame_slot["msg"] = msg
        self.loop.call_soon_threadsafe(self._frame_event.set)

    def _status_msg(self) -> dict:
        episode = self.playback.episode
        is_live = self._active_kind == "live"
        live_status = self.live.status() if is_live and self.live is not None else None
        recorded_index = next(
            (index for index, spec in enumerate(self._recorded_specs) if spec[0] == (episode.name if episode else None)),
            None,
        )
        return {
            "type": "status",
            "episode": episode.name if episode else None,
            "mode": "reality" if is_live else self.playback.mode,
            "playing": False if is_live else self.playing,
            "speed": self.speed,
            "frame_index": self.playback.frame_index,
            "frame_count": 0 if is_live else episode.frame_count if episode else 0,
            "ended": False if is_live else self.playback.finished,
            "compare": self.compare,
            "has_recorded_images": bool(
                any((live_status or {}).get("cameras", {}).values()) if is_live
                else episode and episode.has_images
            ),
            "has_action": bool(episode and episode.targets is not None),
            "target_ghost_available": bool(episode and episode.reality),
            "target_ghost_enabled": self.playback.target_ghost_enabled,
            "source": self._active_kind,
            "source_mode": "live" if is_live else "recorded",
            "live_url": self.live_url,
            "recorded_path": self._recorded_path(),
            "recorded_index": recorded_index,
            "available_modes": ["reality"] if is_live else list(MAX_MODES),
            "live": live_status,
        }

    def _push_status(self) -> None:
        msg = self._status_msg()
        self.loop.call_soon_threadsafe(self._enqueue_all, msg)

    def _push_error(self, message: str) -> None:
        self.loop.call_soon_threadsafe(self._enqueue_all, {"type": "error", "message": message})

    def _enqueue_all(self, msg: dict) -> None:
        for outbox in list(self._clients.values()):
            if outbox.full():
                with contextlib.suppress(asyncio.QueueEmpty):
                    outbox.get_nowait()
            outbox.put_nowait(msg)

    # -- asyncio side --------------------------------------------------------
    def submit(self, command: dict) -> None:
        self._commands.put(command)

    async def broadcast_frames(self) -> None:
        while True:
            await self._frame_event.wait()
            self._frame_event.clear()
            with self._frame_lock:
                msg = self._frame_slot["msg"]
            if msg is None:
                continue
            self._enqueue_all(msg)

    def http_hook(self, connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None  # let the handshake through
        parsed = urlparse(request.path)
        path = parsed.path
        if path in ("/", "/index.html"):
            response = connection.respond(HTTPStatus.OK, INDEX_HTML_PATH.read_text(encoding="utf-8"))
            # websockets Headers.append-on-assign: drop respond()'s text/plain first.
            del response.headers["Content-Type"]
            response.headers["Content-Type"] = "text/html; charset=utf-8"
            return response
        if path == "/healthz":
            return connection.respond(HTTPStatus.OK, "OK\n")
        if path == "/api/episode" or path.startswith("/api/episode/"):
            try:
                if path.startswith("/api/episode/"):
                    raw_index = path.removeprefix("/api/episode/")
                else:
                    values = parse_qs(parsed.query).get("index", [])
                    if len(values) != 1:
                        raise ValueError("provide exactly one integer index")
                    raw_index = values[0]
                index = int(raw_index)
                if not 0 <= index < len(self._recorded_specs):
                    raise ValueError(
                        f"index {index} out of range; loaded episodes={len(self._recorded_specs)}"
                    )
            except ValueError as exc:
                return self._json_response(
                    connection,
                    HTTPStatus.BAD_REQUEST,
                    {"ok": False, "error": str(exc)},
                )
            self.submit({"type": "select_recorded_episode", "index": index, "origin": "http"})
            return self._json_response(
                connection,
                HTTPStatus.ACCEPTED,
                {"ok": True, "index": index},
            )
        if path == "/favicon.ico":
            return connection.respond(HTTPStatus.OK, "")
        return None

    @staticmethod
    def _json_response(connection, status: HTTPStatus, payload: dict):
        response = connection.respond(status, json.dumps(payload, ensure_ascii=False) + "\n")
        del response.headers["Content-Type"]
        response.headers["Content-Type"] = "application/json; charset=utf-8"
        return response

    async def ws_handler(self, connection) -> None:
        outbox: asyncio.Queue = asyncio.Queue(maxsize=2)
        self._clients[connection] = outbox

        async def sender() -> None:
            while True:
                msg = await outbox.get()
                await asyncio.wait_for(connection.send(json.dumps(msg)), timeout=5.0)

        send_task = asyncio.create_task(sender())
        try:
            config = {
                "type": "config",
                "fps": self.args.fps,
                "width": self.args.width,
                "height": self.args.height,
                "episodes": [
                    *self._episode_payload(self.episode_specs)
                ],
                "default_episode": self.playback.episode.name if self.playback.episode else None,
                "cameras": [*CAMERAS, "free"],
                "free_size": [self.args.free_width, self.args.free_height],
                "has_cup": bool(self.playback.mission.has_cup),
                "state_names": list(STATE_NAMES),
                "modes": list(MAX_MODES),
                "scene": self.args.scene,
                "live_url": self.live_url,
                "recorded_path": self._recorded_path(),
                "single_file": self.single_file,
                "table_height": round(self.playback.table_top_z, 3),
            }
            await connection.send(json.dumps(config))
            await connection.send(json.dumps(self._status_msg()))
            async for raw in connection:
                try:
                    command = json.loads(raw)
                    if not isinstance(command, dict):
                        raise ValueError("command must be a JSON object")
                except ValueError as exc:
                    await connection.send(json.dumps({"type": "error", "message": str(exc)}))
                    continue
                self.submit(command)
        except ConnectionClosed:
            pass
        finally:
            self._clients.pop(connection, None)
            send_task.cancel()

    def fail_message(self, exc: BaseException) -> str:
        text = str(exc)
        if "EGL" in text or "OpenGL" in text or "GLX" in text:
            return (
                f"{text}\nFailed to create the MuJoCo renderer. Run with a working GL driver, "
                "e.g. `MUJOCO_GL=egl .venv/bin/python -m openarm_mission.openarm_sim.server`."
            )
        if "openarm" in text.lower() and ("missing" in text.lower() or "not" in text.lower()):
            return f"{text}\nFetch the official model first: bash openarm_mission/fetch_openarm_v1.sh"
        return text


def _resolve_data_dirs(data_dir_arg: list[Path] | None) -> list[Path]:
    """Normalize the (possibly repeated) ``--data-dir`` flag to a list."""
    if data_dir_arg:
        return [Path(d) for d in data_dir_arg]
    return [_playback.DEFAULT_DATA_DIR]


def _recorded_specs_for_path(raw_path: str) -> tuple[Path, list[EpisodeSpec], bool]:
    """Resolve a UI-supplied server path into one or more recorded episodes."""
    text = raw_path.strip()
    if not text:
        raise ValueError("recorded path is empty")
    path = Path(text).expanduser().resolve(strict=True)
    if path.is_file():
        if path.suffix.lower() != ".parquet":
            raise ValueError("recorded file must be an episode .parquet file")
        return path, [(path.stem, "lerobot", path)], True
    if not path.is_dir():
        raise ValueError(f"recorded path is neither a file nor a directory: {path}")
    specs: list[EpisodeSpec] = _collect_episodes([path])
    if not specs:
        raise ValueError(
            "no episodes found under path; expected LeRobot episode_*.parquet, "
            "real_data, or an OpenArm v0.3.0 dataset"
        )
    return path, specs, False


def _validate_live_url(raw_url: str) -> str:
    """Validate a runtime Panel SSE URL before starting network threads."""
    url = raw_url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ValueError("live URL must be an http(s) Panel SSE URL")
    return url


def _collect_episodes(data_dirs: list[Path]) -> list[tuple[str, str, Path]]:
    """Discover (name, kind, path) specs across several data dirs.

    Names are made unique so the frontend's name-keyed lookups
    (``applyStatus`` / ``_select_episode``) stay unambiguous. A single data
    dir keeps the historical naming (``sim_0``…), which the existing tests
    and README depend on; with several dirs each episode name is prefixed
    with a unique label for its source dir, e.g.
    ``p10_smoke_newlayout__sim_0``. Repeated leaf names get a numeric suffix
    (``p10__sim_0`` vs ``p10__2__sim_0``).
    """
    # Pick a distinct short label per dir: prefer the leaf name, but when that
    # collides (e.g. two dirs both named ``openarm_paper_cup_relay`` under
    # different parents) fall back to the parent's leaf name, then add a
    # numeric suffix if that still collides.
    labels: list[str] = []
    counts: dict[str, int] = {}
    for data_dir in data_dirs:
        base = data_dir.name
        counts[base] = counts.get(base, 0) + 1
        if counts[base] == 1:
            labels.append(base)
            continue
        parent = data_dir.parent.name
        key = f"{parent}/{base}"
        labels.append(key)
        for j in range(len(labels) - 1):
            if labels[j] == key:
                labels[j] = f"{key}__{j + 1}"
                break
    prefix_by_dir = [f"{label}__" for label in labels]

    specs: list[tuple[str, str, Path]] = []
    for data_dir, label_prefix in zip(data_dirs, prefix_by_dir, strict=True):
        prefix = _real.episode_prefix(data_dir)
        for path in _playback.list_episodes(data_dir):
            name = path.stem if len(data_dirs) == 1 else f"{label_prefix}{path.stem}"
            specs.append((name, "lerobot", path))
        for path in _real.list_real_episodes(data_dir):
            base = f"{prefix}_{path.name}"
            name = base if len(data_dirs) == 1 else f"{label_prefix}{base}"
            specs.append((name, "real", path))
    return specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OpenArm parquet pose-sim web viewer.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--data-dir",
        type=Path,
        action="append",
        default=None,
        help=(
            "episode source dir (LeRobot / real_data / v0.3.0 dataset root); "
            "may be given multiple times to serve several data sets at once"
        ),
    )
    parser.add_argument("--parquet", type=Path, default=None, help="Serve a single parquet file.")
    parser.add_argument("--state-col", default="observation.state")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--real-fps", type=int, default=_real.REAL_FPS)
    parser.add_argument(
        "--live-url", default=None,
        help="OpenArm Panel SSE endpoint, e.g. http://127.0.0.1:9000/sse/sim",
    )
    parser.add_argument("--live-fps", type=int, default=30, help="maximum live render rate")
    parser.add_argument(
        "--scene", choices=("mission", "reality"), default="mission",
        help="reality = robot + visual-only non-colliding table",
    )
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--free-width", type=int, default=640)
    parser.add_argument("--free-height", type=int, default=480)
    parser.add_argument("--jpeg-quality", type=int, default=80)
    return parser.parse_args()


async def _async_main(args: argparse.Namespace) -> None:
    loop = asyncio.get_running_loop()
    engine = SimEngine(args, loop)
    try:
        engine.start()
    except BaseException as exc:
        print(engine.fail_message(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    broadcaster = asyncio.create_task(engine.broadcast_frames())
    print(f"openarm_sim serving on http://{args.host}:{args.port}/", flush=True)
    try:
        async with ws_server.serve(
            engine.ws_handler,
            args.host,
            args.port,
            process_request=engine.http_hook,
            compression=None,
            max_size=2**20,
        ) as server:
            await server.serve_forever()
    finally:
        engine.stop()
        broadcaster.cancel()


def main() -> None:
    args = parse_args()
    with contextlib.suppress(KeyboardInterrupt):
        asyncio.run(_async_main(args))


if __name__ == "__main__":
    main()

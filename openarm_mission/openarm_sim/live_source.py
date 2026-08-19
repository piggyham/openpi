"""Network-only OpenArm Panel Reality source (SSE joints + MJPEG cameras).

This module never imports or touches MuJoCo. Network threads publish immutable
latest-frame snapshots; the server's dedicated simulation thread is the sole
consumer that applies them to qpos and renders EGL frames.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
import threading
import time
from urllib.parse import urljoin
from urllib.request import Request
from urllib.request import urlopen

SCHEMA = "openarm.panel.reality.v1"
CAMERAS = ("front", "left_wrist", "right_wrist")


@dataclass(frozen=True)
class RealityFrame:
    seq: int
    sample_ts_ns: int
    command_ts_ns: int
    actual_ts_ns: int
    target: tuple[float, ...]
    actual: tuple[float, ...]
    target_valid: bool
    actual_valid: bool
    status: dict
    generation: int
    received_monotonic: float


class LiveRealitySource:
    """Reconnectable latest-frame client for one Panel ``/sse/sim`` URL."""

    def __init__(self, url: str, *, token: str = "", stale_s: float = 0.5) -> None:
        self.url = str(url)
        self.token = token
        self.stale_s = float(stale_s)
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._frame: RealityFrame | None = None
        self._schema: dict | None = None
        self._camera_jpeg: dict[str, bytes] = {}
        self._camera_seen: dict[str, float] = {}
        self._camera_threads: dict[str, threading.Thread] = {}
        self._thread: threading.Thread | None = None
        self._generation = 0
        self._last_seq = -1
        self._last_ts_ns = -1
        self._generation_start_ns = 0
        self._error = "not connected"

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="openarm-panel-sse", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _request(self, url: str) -> Request:
        headers = {"Accept": "text/event-stream", "Cache-Control": "no-cache"}
        if self.token:
            headers["Authorization"] = f"Bearer {self.token}"
        return Request(url, headers=headers)

    def _run(self) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                with urlopen(self._request(self.url), timeout=10) as response:
                    backoff = 0.5
                    event = "message"
                    data_lines: list[str] = []
                    while not self._stop.is_set():
                        raw = response.readline()
                        if not raw:
                            raise ConnectionError("Panel SSE stream closed")
                        line = raw.decode("utf-8", errors="replace").rstrip("\r\n")
                        if not line:
                            if data_lines:
                                self._accept_event(event, "\n".join(data_lines))
                            event, data_lines = "message", []
                        elif line.startswith("event:"):
                            event = line[6:].strip()
                        elif line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
            except Exception as exc:  # network loop must survive Panel restarts
                with self._lock:
                    self._error = str(exc)
                self._stop.wait(backoff)
                backoff = min(8.0, backoff * 2.0)

    def _accept_event(self, event: str, payload: str) -> None:
        parsed = json.loads(payload)
        if parsed.get("schema") != SCHEMA:
            raise ValueError(f"unsupported Panel schema {parsed.get('schema')!r}")
        if event == "schema":
            order = parsed.get("joint_order")
            if not isinstance(order, list) or len(order) != 16:
                raise ValueError("Panel schema joint_order must contain 16 names")
            with self._lock:
                self._schema = parsed
                self._error = ""
            self._start_cameras(parsed.get("camera_urls") or {})
            return
        if event != "frame":
            return
        target = tuple(float(v) for v in parsed.get("target", ()))
        actual = tuple(float(v) for v in parsed.get("actual", ()))
        if len(target) != 16 or len(actual) != 16:
            raise ValueError("Panel frame target/actual must both be 16-D")
        if not all(math.isfinite(v) for v in (*target, *actual)):
            raise ValueError("Panel frame contains non-finite joint values")
        seq = int(parsed.get("seq", 0))
        sample_ts_ns = int(parsed.get("sample_ts_ns", 0))
        with self._lock:
            if seq < self._last_seq or sample_ts_ns < self._last_ts_ns:
                self._generation += 1
                self._generation_start_ns = sample_ts_ns
            elif self._generation_start_ns == 0:
                self._generation_start_ns = sample_ts_ns
            self._last_seq = seq
            self._last_ts_ns = sample_ts_ns
            valid = parsed.get("valid") or {}
            self._frame = RealityFrame(
                seq=seq,
                sample_ts_ns=sample_ts_ns,
                command_ts_ns=int(parsed.get("command_ts_ns", 0)),
                actual_ts_ns=int(parsed.get("actual_ts_ns", 0)),
                target=target,
                actual=actual,
                target_valid=bool(valid.get("command", False)),
                actual_valid=bool(valid.get("actual", False)),
                status=dict(parsed.get("status") or {}),
                generation=self._generation,
                received_monotonic=time.monotonic(),
            )
            self._error = ""

    def _start_cameras(self, camera_urls: dict) -> None:
        for camera in CAMERAS:
            rel = camera_urls.get(camera)
            if not rel or camera in self._camera_threads:
                continue
            url = urljoin(self.url, str(rel))
            thread = threading.Thread(
                target=self._camera_loop, args=(camera, url),
                name=f"openarm-panel-{camera}", daemon=True,
            )
            self._camera_threads[camera] = thread
            thread.start()

    def _camera_loop(self, camera: str, url: str) -> None:
        backoff = 0.5
        while not self._stop.is_set():
            try:
                with urlopen(self._request(url), timeout=10) as response:
                    backoff = 0.5
                    while not self._stop.is_set():
                        line = response.readline()
                        if not line:
                            raise ConnectionError(f"{camera} MJPEG stream closed")
                        if not line.startswith(b"--"):
                            continue
                        length = 0
                        while True:
                            header = response.readline()
                            if not header or header in (b"\r\n", b"\n"):
                                break
                            key, _, value = header.decode("latin1").partition(":")
                            if key.lower() == "content-length":
                                length = int(value.strip())
                        if length <= 0:
                            continue
                        jpeg = response.read(length)
                        response.readline()  # multipart trailing CRLF
                        if jpeg.startswith(b"\xff\xd8"):
                            with self._lock:
                                self._camera_jpeg[camera] = jpeg
                                self._camera_seen[camera] = time.monotonic()
            except Exception:
                self._stop.wait(backoff)
                backoff = min(8.0, backoff * 2.0)

    def latest(self) -> RealityFrame | None:
        with self._lock:
            return self._frame

    def elapsed_s(self, frame: RealityFrame) -> float:
        with self._lock:
            start = self._generation_start_ns or frame.sample_ts_ns
        return max(0.0, (frame.sample_ts_ns - start) / 1e9)

    def camera_frames(self) -> dict[str, bytes]:
        with self._lock:
            return dict(self._camera_jpeg)

    def status(self) -> dict:
        now = time.monotonic()
        with self._lock:
            frame = self._frame
            fresh = bool(
                frame and frame.actual_valid and now - frame.received_monotonic <= self.stale_s
            )
            return {
                "connected": fresh,
                "stale": not fresh,
                "age_s": round(now - frame.received_monotonic, 3) if frame else None,
                "error": self._error,
                "cameras": {
                    name: bool(now - self._camera_seen.get(name, 0.0) <= 1.0)
                    for name in CAMERAS
                },
            }

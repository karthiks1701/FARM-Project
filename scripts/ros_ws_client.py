#!/usr/bin/env python3
"""ROS-free WebSocket client for :mod:`scripts.ros_ws_bridge`.

Runs on the viser host (e.g. 192.168.1.212). Connects to the bridge on the
robot host (e.g. ``ws://192.168.1.192:8765``), decodes the odometry + JPEG
image frames it pushes, and exposes the latest of each — no ``rclpy`` and no
cross-host DDS discovery required here.

Frame alignment mirrors :class:`scripts.odom_ros_listener.OdomRosListener`:
``nav_msgs/Odometry`` carries ``T_odom_body``; pass the same 4x4 ``T_map_odom``
as ``world_transform`` (e.g. Spot ``seed_tform_body``) so the live trail lands
on a saved map.

    from ros_ws_client import RosWsClient

    c = RosWsClient("ws://192.168.1.192:8765", world_transform=T_map_odom)
    c.start()
    stamp, T = c.latest_odom()          # (float sec, (4,4) np.float64) or (None, None)
    istamp, rgb = c.latest_image()      # (float sec, HxWx3 uint8) or (None, None)
    c.stop()

Dependencies: ``websockets`` (already pulled in by ``viser``), ``numpy``,
``Pillow`` (or ``opencv-python``) to decode the compressed (JPEG/PNG) image.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import struct
import threading
import time
from typing import Optional, Tuple

import numpy as np

try:  # reuse the exact quaternion convention used elsewhere in scripts/
    from odom_ros_listener import quat_to_rotmat_xyzw
except Exception:  # pragma: no cover - standalone fallback
    def quat_to_rotmat_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
        n = x * x + y * y + z * z + w * w
        if n <= 1e-24:
            return np.eye(3)
        s = 2.0 / n
        return np.array(
            [
                [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
                [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
                [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
            ],
            dtype=np.float64,
        )


def _decode_image(data: bytes) -> Optional[np.ndarray]:
    """Compressed image bytes (JPEG or PNG) -> HxWx3 uint8 RGB, or None."""
    try:
        import cv2

        arr = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        return None if arr is None else np.ascontiguousarray(arr[..., ::-1])  # BGR -> RGB
    except Exception:
        pass
    try:
        from io import BytesIO

        from PIL import Image

        return np.asarray(Image.open(BytesIO(data)).convert("RGB"))
    except Exception:
        return None


class RosWsClient:
    """Background asyncio WebSocket reader with auto-reconnect."""

    def __init__(
        self,
        url: str,
        *,
        world_transform: Optional[np.ndarray] = None,
        want_image: bool = True,
        reconnect_max_s: float = 5.0,
        on_control=None,
    ) -> None:
        self._url = str(url)
        self._want_image = bool(want_image)
        self._reconnect_max_s = max(0.5, float(reconnect_max_s))
        # Called (in the asyncio thread) with the dict header of any inbound
        # 'goto' / 'cancel' control frame relayed by the bridge. Keep it quick
        # and thread-safe (e.g. push onto a queue.Queue).
        self._on_control = on_control
        self._T_map_odom = (
            np.eye(4, dtype=np.float64)
            if world_transform is None
            else np.asarray(world_transform, dtype=np.float64).reshape(4, 4)
        )

        self._lock = threading.Lock()
        self._odom: Tuple[Optional[float], Optional[np.ndarray]] = (None, None)
        self._image: Tuple[Optional[float], Optional[np.ndarray]] = (None, None)
        self._connected = False
        self._n_odom = 0
        self._n_image = 0

        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._ws = None  # current open connection (asyncio thread only)
        self._stop = threading.Event()

    # -- lifecycle -------------------------------------------------------
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="ros_ws_client", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        loop = self._loop
        if loop is not None:
            with contextlib.suppress(Exception):
                loop.call_soon_threadsafe(lambda: None)
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # -- accessors (thread-safe) --------------------------------------
    def latest_odom(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        with self._lock:
            s, T = self._odom
            return s, (None if T is None else T.copy())

    def latest_image(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        with self._lock:
            s, img = self._image
            return s, (None if img is None else img)

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def stats(self) -> Tuple[int, int]:
        with self._lock:
            return self._n_odom, self._n_image

    def send(self, payload: dict) -> bool:
        """Send a control message (e.g. a 'goto' goal) to the bridge, which
        relays it to the other clients. Thread-safe; returns False if not
        connected."""
        loop = self._loop
        if loop is None:
            return False
        try:
            hb = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            frame = struct.pack(">I", len(hb)) + hb
        except Exception:
            return False
        loop.call_soon_threadsafe(self._enqueue_send, frame)
        return True

    def _enqueue_send(self, frame: bytes) -> None:
        ws = self._ws
        if ws is None:
            return
        asyncio.create_task(self._safe_send(ws, frame))

    @staticmethod
    async def _safe_send(ws, frame: bytes) -> None:
        with contextlib.suppress(Exception):
            await ws.send(frame)

    # -- internals ----------------------------------------------------
    def _run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._reader())
        finally:
            with contextlib.suppress(Exception):
                self._loop.close()

    async def _reader(self) -> None:
        try:
            from websockets.asyncio.client import connect  # websockets >= 13
        except Exception:  # pragma: no cover - older websockets
            from websockets import connect  # type: ignore

        backoff = 0.5
        while not self._stop.is_set():
            try:
                async with connect(self._url, max_size=None, ping_interval=20, open_timeout=5) as ws:
                    self._ws = ws
                    with self._lock:
                        self._connected = True
                    print(f"[ros_ws_client] connected to {self._url}")
                    backoff = 0.5
                    async for message in ws:
                        if self._stop.is_set():
                            break
                        self._handle(message)
            except Exception as exc:  # noqa: BLE001 - reconnect on anything
                if not self._stop.is_set():
                    print(f"[ros_ws_client] disconnected ({exc}); retrying in {backoff:.1f}s")
            finally:
                self._ws = None
                with self._lock:
                    self._connected = False
            if self._stop.is_set():
                break
            await asyncio.sleep(backoff)
            backoff = min(self._reconnect_max_s, backoff * 2.0)

    def _handle(self, message) -> None:
        if isinstance(message, str):
            message = message.encode("utf-8")
        if len(message) < 4:
            return
        (hlen,) = struct.unpack(">I", message[:4])
        if 4 + hlen > len(message):
            return
        try:
            header = json.loads(message[4 : 4 + hlen].decode("utf-8"))
        except Exception:
            return
        payload = message[4 + hlen :]
        kind = header.get("type")
        if kind == "odom":
            self._handle_odom(header)
        elif kind == "image" and self._want_image:
            self._handle_image(header, payload)
        elif kind in ("goto", "goto_waypoint", "cancel") and self._on_control is not None:
            with contextlib.suppress(Exception):
                self._on_control(header)

    def _handle_odom(self, header: dict) -> None:
        try:
            px, py, pz = (float(v) for v in header["position"])
            ox, oy, oz, ow = (float(v) for v in header["orientation_xyzw"])
        except Exception:
            return
        T_odom_body = np.eye(4, dtype=np.float64)
        T_odom_body[:3, :3] = quat_to_rotmat_xyzw(ox, oy, oz, ow)
        T_odom_body[:3, 3] = (px, py, pz)
        T_map_body = self._T_map_odom @ T_odom_body
        if not np.all(np.isfinite(T_map_body)):
            return
        stamp = float(header.get("stamp") or time.time())
        with self._lock:
            self._odom = (stamp, T_map_body)
            self._n_odom += 1

    def _handle_image(self, header: dict, payload: bytes) -> None:
        if not payload:
            return
        rgb = _decode_image(payload)
        if rgb is None:
            return
        stamp = float(header.get("stamp") or time.time())
        with self._lock:
            self._image = (stamp, np.ascontiguousarray(rgb))
            self._n_image += 1


if __name__ == "__main__":  # quick manual check: python scripts/ros_ws_client.py ws://host:8765
    import sys

    url = sys.argv[1] if len(sys.argv) > 1 else "ws://127.0.0.1:8765"
    c = RosWsClient(url)
    c.start()
    try:
        while True:
            time.sleep(1.0)
            s, T = c.latest_odom()
            istamp, img = c.latest_image()
            no, ni = c.stats
            pos = None if T is None else np.round(T[:3, 3], 3).tolist()
            shape = None if img is None else img.shape
            print(f"connected={c.connected} odom#{no}@{pos} image#{ni}@{shape}")
    except KeyboardInterrupt:
        c.stop()

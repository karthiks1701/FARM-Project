#!/usr/bin/env python3
"""ROS 2 -> WebSocket bridge. Runs on the robot/sensor host (e.g. 192.168.1.192).

Subscribes to a ``nav_msgs/Odometry`` topic and a ``sensor_msgs/Image`` color
topic, and pushes both to every connected WebSocket client. The viser host
(e.g. 192.168.1.212) runs ``scripts/ros_ws_client.py`` / ``view_scene_state.py
--ws-url`` to receive them — no ROS install or cross-host DDS discovery needed
on that side.

Run on the host that HAS the topics::

    python scripts/ros_ws_bridge.py --host 0.0.0.0 --port 8765 \
        --odom-topic /odometry \
        --image-topic /camera/camera/color/image_raw \
        --image-max-fps 10 --image-max-side 640 --jpeg-quality 80

Then on the viser host::

    python scripts/view_scene_state.py --pt scene.pt --ws-url ws://192.168.1.192:8765

Dependencies here: ``rclpy`` (ROS 2), ``websockets`` (``pip install websockets``),
``numpy``; ``opencv-python`` is used for JPEG encoding if present, else Pillow.

Wire format — every message is one binary frame::

    [4 bytes big-endian uint32 = len(header)] [header: UTF-8 JSON] [payload bytes]

``header['type']`` is ``"odom"`` (payload empty; pose in the header) or
``"image"`` (payload = JPEG bytes).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import os
import struct
import threading
import time
from typing import Optional

import numpy as np

# ----------------------------------------------------------------------------
# image helpers (no cv_bridge dependency)
# ----------------------------------------------------------------------------
def _image_msg_to_rgb(msg) -> np.ndarray:
    """``sensor_msgs/Image`` -> contiguous HxWx3 uint8 RGB."""
    h, w, step = int(msg.height), int(msg.width), int(msg.step)
    enc = str(msg.encoding).lower()
    buf = np.frombuffer(bytes(msg.data), dtype=np.uint8)
    if enc in ("rgb8", "bgr8"):
        rows = buf.reshape(h, step)[:, : w * 3].reshape(h, w, 3)
        return np.ascontiguousarray(rows[..., ::-1] if enc == "bgr8" else rows)
    if enc in ("rgba8", "bgra8"):
        rows = buf.reshape(h, step)[:, : w * 4].reshape(h, w, 4)[..., :3]
        return np.ascontiguousarray(rows[..., ::-1] if enc == "bgra8" else rows)
    if enc in ("mono8", "8uc1"):
        rows = buf.reshape(h, step)[:, :w].reshape(h, w, 1)
        return np.ascontiguousarray(np.repeat(rows, 3, axis=2))
    raise ValueError(f"unsupported image encoding {msg.encoding!r} (want rgb8/bgr8/rgba8/bgra8/mono8)")


def _encode_jpeg(rgb: np.ndarray, max_side: int, quality: int) -> bytes:
    h, w = rgb.shape[:2]
    scale = float(max_side) / float(max(h, w)) if max_side > 0 and max(h, w) > max_side else 1.0
    try:
        import cv2

        img = rgb[..., ::-1]  # RGB -> BGR
        if scale < 1.0:
            img = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
        if not ok:
            raise RuntimeError("cv2.imencode failed")
        return enc.tobytes()
    except Exception:
        from io import BytesIO

        from PIL import Image

        im = Image.fromarray(rgb)
        if scale < 1.0:
            im = im.resize((max(1, int(w * scale)), max(1, int(h * scale))))
        bio = BytesIO()
        im.save(bio, format="JPEG", quality=int(quality))
        return bio.getvalue()


def _frame(header: dict, payload: bytes = b"") -> bytes:
    hb = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return struct.pack(">I", len(hb)) + hb + payload


# ----------------------------------------------------------------------------
# ROS side (background thread)
# ----------------------------------------------------------------------------
class _RosBridgeNode:
    def __init__(self, args, loop: asyncio.AbstractEventLoop, odom_q: asyncio.Queue, img_slot: "_LatestSlot") -> None:
        self._args = args
        self._loop = loop
        self._odom_q = odom_q
        self._img_slot = img_slot
        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._started = threading.Event()
        self._error: Optional[BaseException] = None
        self._last_image_s = 0.0

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="ros_ws_bridge_ros", daemon=True)
        self._thread.start()
        self._started.wait(timeout=8.0)
        if self._error is not None:
            raise RuntimeError(f"ROS bridge failed to start: {self._error}") from self._error

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)

    def _run(self) -> None:
        try:
            self._spin()
        except BaseException as exc:  # noqa: BLE001 - surface to start()
            self._error = exc
            self._started.set()

    def _spin(self) -> None:
        import rclpy
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.node import Node
        from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import Image

        a = self._args
        if a.ros_domain_id is not None:
            os.environ["ROS_DOMAIN_ID"] = str(int(a.ros_domain_id))
        if not rclpy.ok():
            rclpy.init(args=None)

        node = Node("ros_ws_bridge")

        def _qos(name: str) -> QoSProfile:
            rel = ReliabilityPolicy.BEST_EFFORT if name in ("best_effort", "sensor") else ReliabilityPolicy.RELIABLE
            return QoSProfile(depth=10, reliability=rel, durability=DurabilityPolicy.VOLATILE, history=HistoryPolicy.KEEP_LAST)

        node.create_subscription(Odometry, a.odom_topic, self._on_odom, _qos(a.odom_qos))
        node.create_subscription(Image, a.image_topic, self._on_image, _qos(a.image_qos))
        node.get_logger().info(
            f"[ros_ws_bridge] odom={a.odom_topic!r} image={a.image_topic!r} "
            f"domain={os.environ.get('ROS_DOMAIN_ID', '0')}"
        )

        ex = SingleThreadedExecutor()
        ex.add_node(node)
        self._started.set()
        try:
            while rclpy.ok() and not self._stop.is_set():
                ex.spin_once(timeout_sec=0.1)
        finally:
            with contextlib.suppress(Exception):
                ex.remove_node(node)
            with contextlib.suppress(Exception):
                node.destroy_node()
            with contextlib.suppress(Exception):
                rclpy.shutdown()

    # -- callbacks (ROS thread) --------------------------------------------
    def _on_odom(self, msg) -> None:
        st = msg.header.stamp
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        header = {
            "type": "odom",
            "stamp": float(st.sec) + float(st.nanosec) * 1e-9,
            "frame_id": str(msg.header.frame_id),
            "child_frame_id": str(msg.child_frame_id),
            "position": [float(p.x), float(p.y), float(p.z)],
            "orientation_xyzw": [float(o.x), float(o.y), float(o.z), float(o.w)],
        }
        frame = _frame(header)
        self._loop.call_soon_threadsafe(self._odom_q_put, frame)

    def _odom_q_put(self, frame: bytes) -> None:
        if self._odom_q.full():
            with contextlib.suppress(asyncio.QueueEmpty):
                self._odom_q.get_nowait()
        self._odom_q.put_nowait(frame)

    def _on_image(self, msg) -> None:
        a = self._args
        now = time.monotonic()
        if a.image_max_fps > 0 and (now - self._last_image_s) < (1.0 / a.image_max_fps):
            return
        try:
            rgb = _image_msg_to_rgb(msg)
            jpeg = _encode_jpeg(rgb, a.image_max_side, a.jpeg_quality)
        except Exception as exc:  # noqa: BLE001
            print(f"[ros_ws_bridge] image encode skipped: {exc}")
            return
        self._last_image_s = now
        st = msg.header.stamp
        header = {
            "type": "image",
            "stamp": float(st.sec) + float(st.nanosec) * 1e-9,
            "frame_id": str(msg.header.frame_id),
            "format": "jpeg",
            "width": int(rgb.shape[1]),
            "height": int(rgb.shape[0]),
        }
        frame = _frame(header, jpeg)
        self._loop.call_soon_threadsafe(self._img_slot.set, frame)


# ----------------------------------------------------------------------------
# WebSocket side (asyncio, main thread)
# ----------------------------------------------------------------------------
class _LatestSlot:
    """Single-value slot: newer frames overwrite an unsent one (image = latest wins)."""

    def __init__(self) -> None:
        self._value: Optional[bytes] = None
        self._event = asyncio.Event()

    def set(self, value: bytes) -> None:
        self._value = value
        self._event.set()

    async def get(self) -> bytes:
        await self._event.wait()
        self._event.clear()
        val = self._value
        self._value = None
        assert val is not None
        return val


async def _serve(args) -> None:
    try:
        from websockets.asyncio.server import serve  # websockets >= 13
    except Exception:  # pragma: no cover - older websockets
        from websockets import serve  # type: ignore

    loop = asyncio.get_running_loop()
    clients: set = set()
    odom_q: asyncio.Queue = asyncio.Queue(maxsize=64)
    img_slot = _LatestSlot()

    async def _handler(ws, *_compat) -> None:  # *_compat: legacy websockets passes a path arg
        clients.add(ws)
        peer = getattr(ws, "remote_address", None)
        print(f"[ros_ws_bridge] client connected: {peer} ({len(clients)} total)")
        try:
            await ws.wait_closed()
        finally:
            clients.discard(ws)
            print(f"[ros_ws_bridge] client gone: {peer} ({len(clients)} total)")

    async def _broadcast(frame: bytes) -> None:
        if not clients:
            return
        dead = []
        for ws in list(clients):
            try:
                await ws.send(frame)
            except Exception:
                dead.append(ws)
        for ws in dead:
            clients.discard(ws)

    async def _pump_odom() -> None:
        while True:
            await _broadcast(await odom_q.get())

    async def _pump_image() -> None:
        while True:
            await _broadcast(await img_slot.get())

    ros = _RosBridgeNode(args, loop, odom_q, img_slot)
    ros.start()

    async with serve(_handler, args.host, args.port, max_size=None, ping_interval=20):
        print(f"[ros_ws_bridge] serving ws://{args.host}:{args.port}  (Ctrl+C to stop)")
        try:
            await asyncio.gather(_pump_odom(), _pump_image())
        finally:
            ros.stop()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("--host", default="0.0.0.0", help="Interface to bind the WebSocket server on")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--odom-topic", default="/odometry")
    ap.add_argument("--image-topic", default="/camera/camera/color/image_raw")
    ap.add_argument("--odom-qos", choices=("reliable", "best_effort"), default="reliable")
    ap.add_argument("--image-qos", choices=("reliable", "best_effort"), default="best_effort",
                    help="RealSense image topics are usually best_effort")
    ap.add_argument("--image-max-fps", type=float, default=10.0, help="Cap forwarded image rate (0 = no cap)")
    ap.add_argument("--image-max-side", type=int, default=640, help="Downscale so the longest image side <= this (0 = keep)")
    ap.add_argument("--jpeg-quality", type=int, default=80)
    ap.add_argument("--ros-domain-id", type=int, default=None)
    args = ap.parse_args()
    try:
        asyncio.run(_serve(args))
    except KeyboardInterrupt:
        print("\n[ros_ws_bridge] bye.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

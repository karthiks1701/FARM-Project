#!/usr/bin/env python3
"""Background ROS 2 subscriber for a ``nav_msgs/Odometry`` stream (Spot's
``/odometry``), buffering poses as 4x4 matrices for the viser viewers.

Kept out of ``src/scene_graph/`` on purpose: that package is ROS-free. This
helper is imported only by ``scripts/view_scene_state.py`` (and other
scripts), which already run outside colcon.

Frame handling
--------------
``nav_msgs/Odometry`` carries ``T_odom_body`` (``pose.pose`` in ``header.frame_id``,
child ``child_frame_id``). The saved map may live in a different world frame
(e.g. Spot's ``seed`` frame, baked in by ``scripts/rgb_bag_frame.py
--world-transform``). Pass the *same* 4x4 ``T_map_odom`` here as
``world_transform`` so the live trail lands on the map:

    T_map_body(t) = world_transform @ T_odom_body(t)

Usage
-----
    from odom_ros_listener import OdomRosListener

    lis = OdomRosListener(topic="/odometry", ros_domain_id=42,
                          world_transform=T_map_odom)   # 4x4 or None
    lis.start()
    ...
    stamp, T = lis.latest()          # (float seconds, (4,4) np.float64) or (None, None)
    lis.stop()
"""

from __future__ import annotations

import contextlib
import os
import threading
from typing import List, Optional, Tuple

import numpy as np


def quat_to_rotmat_xyzw(x: float, y: float, z: float, w: float) -> np.ndarray:
    """Rotation matrix from a quaternion in (x, y, z, w) order. Mirrors the
    convention in ``scripts/rgb_bag_frame.py``."""
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


class OdomRosListener:
    """Spin ``rclpy`` in a daemon thread and keep a ring buffer of recent poses."""

    def __init__(
        self,
        *,
        topic: str = "/odometry",
        ros_domain_id: Optional[int] = None,
        qos: str = "reliable",
        world_transform: Optional[np.ndarray] = None,
        buffer_size: int = 4000,
        trail_seconds: float = 0.0,
        node_name: str = "farm_odom_listener",
    ) -> None:
        # ROS_DOMAIN_ID must be set before rclpy.init(); once the context is up
        # it cannot be changed for this process.
        if ros_domain_id is not None:
            os.environ["ROS_DOMAIN_ID"] = str(int(ros_domain_id))

        self._topic = topic
        self._qos_name = str(qos).lower()
        self._node_name = node_name
        self._buffer_size = max(1, int(buffer_size))
        self._trail_seconds = max(0.0, float(trail_seconds))

        if world_transform is None:
            self._T_map_odom = np.eye(4, dtype=np.float64)
        else:
            self._T_map_odom = np.asarray(world_transform, dtype=np.float64).reshape(4, 4)

        self._lock = threading.Lock()
        self._stamps: List[float] = []
        self._poses: List[np.ndarray] = []  # each (4,4) float64, already in map frame

        self._thread: Optional[threading.Thread] = None
        self._executor = None
        self._node = None
        self._rclpy_inited_here = False
        self._started = threading.Event()
        self._stop = threading.Event()
        self._error: Optional[BaseException] = None

    # ------------------------------------------------------------------ #
    # lifecycle
    # ------------------------------------------------------------------ #
    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name=self._node_name, daemon=True)
        self._thread.start()
        # Give the node a moment to come up so callers can log a clean status.
        self._started.wait(timeout=5.0)
        if self._error is not None:
            raise RuntimeError(f"odometry listener failed to start: {self._error}") from self._error

    def stop(self) -> None:
        self._stop.set()
        executor = self._executor
        if executor is not None:
            with contextlib.suppress(Exception):
                executor.wake()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    # ------------------------------------------------------------------ #
    # accessors (thread-safe)
    # ------------------------------------------------------------------ #
    def latest(self) -> Tuple[Optional[float], Optional[np.ndarray]]:
        with self._lock:
            if not self._poses:
                return None, None
            return self._stamps[-1], self._poses[-1].copy()

    def since(self, t: float) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return [(s, p.copy()) for s, p in zip(self._stamps, self._poses) if s >= t]

    def all(self) -> List[Tuple[float, np.ndarray]]:
        with self._lock:
            return [(s, p.copy()) for s, p in zip(self._stamps, self._poses)]

    def count(self) -> int:
        with self._lock:
            return len(self._poses)

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #
    def _run(self) -> None:
        try:
            self._spin()
        except BaseException as exc:  # noqa: BLE001 - surface to start()
            self._error = exc
            self._started.set()

    def _spin(self) -> None:
        import rclpy
        from rclpy.node import Node
        from rclpy.executors import SingleThreadedExecutor
        from rclpy.qos import (
            DurabilityPolicy,
            HistoryPolicy,
            QoSProfile,
            ReliabilityPolicy,
        )
        from nav_msgs.msg import Odometry

        if not rclpy.ok():
            rclpy.init(args=None)
            self._rclpy_inited_here = True

        node = Node(self._node_name)
        self._node = node

        reliability = (
            ReliabilityPolicy.BEST_EFFORT
            if self._qos_name in ("best_effort", "besteffort", "sensor")
            else ReliabilityPolicy.RELIABLE
        )
        qos = QoSProfile(
            depth=50,
            reliability=reliability,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
        )
        node.create_subscription(Odometry, self._topic, self._on_odom, qos)
        node.get_logger().info(
            f"[odom_listener] subscribed to {self._topic!r} "
            f"(domain={os.environ.get('ROS_DOMAIN_ID', '0')}, qos={self._qos_name})"
        )

        self._executor = SingleThreadedExecutor()
        self._executor.add_node(node)
        self._started.set()
        try:
            while rclpy.ok() and not self._stop.is_set():
                self._executor.spin_once(timeout_sec=0.1)
        finally:
            with contextlib.suppress(Exception):
                self._executor.remove_node(node)
            with contextlib.suppress(Exception):
                node.destroy_node()
            if self._rclpy_inited_here:
                with contextlib.suppress(Exception):
                    rclpy.shutdown()

    def _on_odom(self, msg) -> None:
        stamp = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        p = msg.pose.pose.position
        o = msg.pose.pose.orientation
        T_odom_body = np.eye(4, dtype=np.float64)
        T_odom_body[:3, :3] = quat_to_rotmat_xyzw(o.x, o.y, o.z, o.w)
        T_odom_body[:3, 3] = (p.x, p.y, p.z)
        T_map_body = self._T_map_odom @ T_odom_body
        if not np.all(np.isfinite(T_map_body)):
            return
        with self._lock:
            self._stamps.append(stamp)
            self._poses.append(T_map_body)
            if self._trail_seconds > 0.0:
                cutoff = stamp - self._trail_seconds
                keep = 0
                while keep < len(self._stamps) and self._stamps[keep] < cutoff:
                    keep += 1
                if keep:
                    del self._stamps[:keep]
                    del self._poses[:keep]
            if len(self._poses) > self._buffer_size:
                drop = len(self._poses) - self._buffer_size
                del self._stamps[:drop]
                del self._poses[:drop]


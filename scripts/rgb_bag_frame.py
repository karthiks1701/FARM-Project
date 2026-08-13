#!/usr/bin/env python3
"""Convert a ROS 2 bag with paired RGBD images + body-frame odometry into a
``frames.json`` scene consumable by ``scene_graph.offline.run --source frames-json``.

Unlike ``lidar_bag_to_frames.py`` (Odin1: single fisheye + LiDAR ``PointCloud2``,
depth *synthesized* by projecting the cloud), this assumes depth is already
produced upstream (stereo matching, RealSense/ZED, etc.) and published as a
standard ``sensor_msgs/Image`` depth topic aligned to the RGB stream. No LiDAR
projection, no fisheye rectification -- just pinhole RGB + metric depth + pose
per frame, written out as:

    <out>/frames.json
    <out>/rgb/<cam>/NNNNNN.jpeg
    <out>/depth/<cam>/NNNNNN.npy     # float32 metres, 0 = invalid

Then reconstruct:

    python -m scene_graph.offline.run --source frames-json \\
        --frames-json-dir <out> --save-path scene.pt --covisibility

Rate handling
-------------
RGB/depth (~15 Hz) and odometry (~10 Hz) do **not** need matching rates.
Odometry is buffered in full and SLERP-interpolated to each RGB frame's exact
timestamp (see ``interpolate_pose``) -- there is no synchronizer between the
two streams. RGB and depth *do* need pairing since they're separate topics:
each RGB message is matched to the nearest depth message within
``--sync-slop-sec`` (default 0.05s); pairs outside that tolerance are dropped.

Frame convention
-----------------
``T_world_cam`` must be in OpenCV/ROS-optical convention (X-right, Y-down,
Z-forward) -- see ``frame_sources/npz.py`` and the online path's TF lookups,
which always resolve to a camera's ``*_optical`` frame, never its physical
mounting/link frame. Odometry gives ``T_world_body`` (typically X-forward,
Z-up). ``T_CAM_BODY`` (hardcoded below) is the camera-*optical*-from-body
transform (mirrors the ``Tcl_0`` convention in ``lidar_bag_to_frames.py`` --
"maps body-frame points into the camera frame"). Composition:

    T_world_cam = T_world_body @ inv(T_cam_body)
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import sqlite3
import sys
from collections import deque
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def _prefer_repo_mapping_source() -> None:
    """Make ``mapping.lib.image_decoding`` importable from a bare checkout.

    Mirrors ``scene_graph.offline.run._prefer_repo_mapping_source``: prefer
    the checked-out ``ros/mapping`` source tree over a possibly-stale
    colcon-built copy.
    """
    import os

    repo_root = Path(__file__).resolve().parents[1]
    mapping_src = repo_root / "ros" / "mapping"
    if mapping_src.exists():
        src = str(mapping_src)
        if src not in sys.path:
            sys.path.insert(0, src)
        pythonpath = os.environ.get("PYTHONPATH", "")
        if src not in pythonpath.split(os.pathsep):
            os.environ["PYTHONPATH"] = src + (os.pathsep + pythonpath if pythonpath else "")


_prefer_repo_mapping_source()
from mapping.lib.image_decoding import decode_depth, decode_rgb  # noqa: E402


# --------------------------------------------------------------------------- #
# Extrinsic calibration
# --------------------------------------------------------------------------- #
# Camera-optical-from-body, 4x4. Camera optical center coincident with the
# body origin (zero translation); rotation is the fixed REP-103 body->optical
# axis convention (body: X-forward/Y-left/Z-up -> optical: X-right/Y-down/
# Z-forward). Replace with real calibration for anything beyond a smoke test.
T_CAM_BODY = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, -1.0, -0.3],
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)


# --------------------------------------------------------------------------- #
# Pose interpolation (slerp) -- same convention as lidar_bag_to_frames.py
# --------------------------------------------------------------------------- #
def quat_to_rotmat_xyzw(q) -> np.ndarray:
    x, y, z, w = (float(v) for v in q)
    n = x * x + y * y + z * z + w * w
    if n <= 1e-24:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - z * w), s * (x * z + y * w)],
        [s * (x * y + z * w), 1 - s * (x * x + z * z), s * (y * z - x * w)],
        [s * (x * z - y * w), s * (y * z + x * w), 1 - s * (x * x + y * y)],
    ], dtype=np.float64)


def slerp_xyzw(q0, q1, a):
    q0 = q0 / max(np.linalg.norm(q0), 1e-12)
    q1 = q1 / max(np.linalg.norm(q1), 1e-12)
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = max(-1.0, min(1.0, dot))
    if dot > 0.9995:
        q = q0 + a * (q1 - q0)
        return q / max(np.linalg.norm(q), 1e-12)
    t0 = math.acos(dot)
    st0 = math.sin(t0)
    t = t0 * a
    return (math.cos(t) - dot * math.sin(t) / st0) * q0 + (math.sin(t) / st0) * q1


def interpolate_pose(times, positions, quats, t) -> np.ndarray:
    if not times:
        raise ValueError("no odometry samples")
    if len(times) == 1 or t <= times[0]:
        pos, quat = np.asarray(positions[0]), np.asarray(quats[0])
    elif t >= times[-1]:
        pos, quat = np.asarray(positions[-1]), np.asarray(quats[-1])
    else:
        i1 = bisect.bisect_left(times, t)
        i0 = max(0, i1 - 1)
        t0, t1 = times[i0], times[i1]
        a = 0.0 if t1 <= t0 else (t - t0) / (t1 - t0)
        pos = (1 - a) * np.asarray(positions[i0]) + a * np.asarray(positions[i1])
        quat = slerp_xyzw(np.asarray(quats[i0]), np.asarray(quats[i1]), a)
    T = np.eye(4)
    T[:3, :3] = quat_to_rotmat_xyzw(quat)
    T[:3, 3] = pos
    return T


# --------------------------------------------------------------------------- #
# sqlite3 rosbag streaming
# --------------------------------------------------------------------------- #
def resolve_db(bag_dir: Path) -> Path:
    dbs = sorted(Path(bag_dir).glob("*.db3"))
    if not dbs:
        raise FileNotFoundError(f"no *.db3 under {bag_dir} (mcap bags not supported by this tool)")
    return dbs[0]


def stamp_s(header) -> float:
    return float(header.stamp.sec) + float(header.stamp.nanosec) * 1e-9


def get_first_message(conn, topic_id: int, msg_cls) -> Optional[object]:
    row = conn.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY id LIMIT 1", (topic_id,)
    ).fetchone()
    if row is None:
        return None
    return deserialize_message(row[0], msg_cls)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bag-dir", type=Path, required=True, help="rosbag2 dir containing *.db3 + metadata.yaml")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--camera", default="cam0")
    ap.add_argument("--rgb-topic", required=True)
    ap.add_argument("--depth-topic", required=True, help="raw sensor_msgs/Image (16UC1/MONO16/32FC1); CompressedImage depth not supported")
    ap.add_argument("--odometry-topic", required=True)
    ap.add_argument("--camera-info-topic", required=True, help="CameraInfo matching the DEPTH topic's resolution")
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth RGB message")
    ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--sync-slop-sec", type=float, default=0.05, help="max |t_rgb - t_depth| to pair a frame")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    args = ap.parse_args(argv)

    import cv2  # deferred: only needed for JPEG encode

    T_body_cam = np.linalg.inv(T_CAM_BODY)

    out = args.out_dir.expanduser().resolve()
    rgb_dir = out / "rgb" / args.camera
    depth_dir = out / "depth" / args.camera
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    db = resolve_db(args.bag_dir.expanduser().resolve())
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    topics = {name: (tid, typ) for tid, name, typ in conn.execute("SELECT id, name, type FROM topics")}
    required = (args.rgb_topic, args.depth_topic, args.odometry_topic, args.camera_info_topic)
    for t in required:
        if t not in topics:
            raise KeyError(f"topic {t!r} not in bag; available: {sorted(topics)}")

    # Intrinsics: must match the DEPTH topic's resolution (frames.json's K is
    # applied at depth resolution; RGB is resized to match at load time by
    # FramesJsonFrameSource, so no separate RGB intrinsics/resize needed here).
    info_id, info_type = topics[args.camera_info_topic]
    info_cls = get_message(info_type)
    camera_info = get_first_message(conn, info_id, info_cls)
    if camera_info is None:
        raise RuntimeError(f"no messages on {args.camera_info_topic!r}")
    K = np.array(
        [[float(camera_info.k[0]), 0.0, float(camera_info.k[2])],
         [0.0, float(camera_info.k[4]), float(camera_info.k[5])],
         [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    info_w, info_h = int(camera_info.width), int(camera_info.height)

    # Pass 1: odometry in full (small relative to image/depth payloads).
    odom_id, odom_type = topics[args.odometry_topic]
    odom_cls = get_message(odom_type)
    times: List[float] = []
    positions: List[Tuple[float, float, float]] = []
    quats: List[Tuple[float, float, float, float]] = []
    for (data,) in conn.execute(
        "SELECT data FROM messages WHERE topic_id = ? ORDER BY id", (odom_id,)
    ):
        msg = deserialize_message(data, odom_cls)
        t = stamp_s(msg.header)
        p, o = msg.pose.pose.position, msg.pose.pose.orientation
        times.append(t)
        positions.append((p.x, p.y, p.z))
        quats.append((o.x, o.y, o.z, o.w))
    if not times:
        raise RuntimeError(f"no odometry samples on {args.odometry_topic!r}")
    print(f"[rgbd2frames] loaded {len(times)} odometry samples "
          f"({times[0]:.2f}s .. {times[-1]:.2f}s)", flush=True)

    # Pass 2: stream rgb+depth in bag order, pair by nearest timestamp.
    rgb_id, rgb_type = topics[args.rgb_topic]
    depth_id, depth_type = topics[args.depth_topic]
    rgb_cls, depth_cls = get_message(rgb_type), get_message(depth_type)
    id2name = {rgb_id: "rgb", depth_id: "depth"}
    msg_cls = {rgb_id: rgb_cls, depth_id: depth_cls}

    pending_rgb: Deque[Tuple[object, float]] = deque()
    depth_buf: Deque[Tuple[float, object]] = deque()
    frames: List[dict] = []
    img_seen = 0
    n_dropped = 0
    last_t = 0.0
    depth_size_mismatch_warned = False

    def try_pair(force: bool):
        nonlocal n_dropped, depth_size_mismatch_warned
        while pending_rgb:
            rgb_msg, t_rgb = pending_rgb[0]
            newest_depth_t = depth_buf[-1][0] if depth_buf else float("-inf")
            if not force and newest_depth_t < t_rgb + args.sync_slop_sec:
                break
            pending_rgb.popleft()
            if args.max_frames is not None and len(frames) >= args.max_frames:
                continue

            best_dt, best = None, None
            for t_d, d_msg in depth_buf:
                dt = abs(t_d - t_rgb)
                if best_dt is None or dt < best_dt:
                    best_dt, best = dt, (t_d, d_msg)
            if best is None or best_dt > args.sync_slop_sec:
                n_dropped += 1
                continue
            t_depth, depth_msg = best

            rgb = decode_rgb(rgb_msg, logger=print)
            if rgb is None:
                n_dropped += 1
                continue
            depth = decode_depth(depth_msg)
            if depth is None:
                n_dropped += 1
                continue

            if (depth.shape[1] != info_w or depth.shape[0] != info_h) and not depth_size_mismatch_warned:
                print(
                    f"[rgbd2frames] WARNING: depth resolution {depth.shape[::-1]} != "
                    f"camera_info resolution ({info_w},{info_h}) -- K may not be valid for this "
                    f"depth topic; point --camera-info-topic at a CameraInfo matching --depth-topic.",
                    flush=True,
                )
                depth_size_mismatch_warned = True

            T_world_body = interpolate_pose(times, positions, quats, t_rgb)
            T_world_cam = T_world_body @ T_body_cam

            fid = f"{len(frames):06d}"
            rgb_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(rgb_dir / f"{fid}.jpeg"), rgb_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), args.jpeg_quality])
            np.save(depth_dir / f"{fid}.npy", depth.astype(np.float32))
            frames.append({
                "frame_id": fid, "camera": args.camera,
                "rgb_path": f"rgb/{args.camera}/{fid}.jpeg",
                "depth_path": f"depth/{args.camera}/{fid}.npy",
                "T_world_cam": T_world_cam.tolist(), "K": K.tolist(),
                "rgb_size": list(rgb.shape[:2]), "depth_size": list(depth.shape),
                "timestamp_ns": int(round(t_rgb * 1e9)),
                "depth_valid_pixels": int(np.count_nonzero(np.isfinite(depth) & (depth > 0))),
                "rgb_depth_skew_s": round(t_rgb - t_depth, 4),
            })
            if len(frames) % 50 == 0:
                print(f"[rgbd2frames] {len(frames)} frames (dropped={n_dropped}, "
                      f"last valid depth px={frames[-1]['depth_valid_pixels']})", flush=True)

        cutoff = (pending_rgb[0][1] - args.sync_slop_sec) if pending_rgb else (last_t - args.sync_slop_sec)
        while depth_buf and depth_buf[0][0] < cutoff:
            depth_buf.popleft()

    sql = f"SELECT topic_id, timestamp, data FROM messages WHERE topic_id IN ({','.join('?' * len(id2name))}) ORDER BY id"
    for tid, _ts, data in conn.execute(sql, tuple(id2name)):
        name = id2name[tid]
        msg = deserialize_message(data, msg_cls[tid])
        if name == "depth":
            t = stamp_s(msg.header)
            last_t = max(last_t, t)
            depth_buf.append((t, msg))
        else:  # rgb
            img_seen += 1
            if (img_seen - 1) % args.stride != 0:
                continue
            t = stamp_s(msg.header)
            last_t = max(last_t, t)
            pending_rgb.append((msg, t))
        try_pair(force=False)
        if args.max_frames is not None and len(frames) >= args.max_frames and not pending_rgb:
            break
    try_pair(force=True)

    index = {
        "schema_version": 1, "dataset": "custom_rgbd",
        "scene_id": args.out_dir.name, "cameras": [args.camera],
        "frames": frames,
        "notes": f"bag={db.parent.name};rgbd_direct",
    }
    (out / "frames.json").write_text(json.dumps(index, indent=1))
    med = int(np.median([f["depth_valid_pixels"] for f in frames])) if frames else 0
    print(f"[rgbd2frames] wrote {len(frames)} frames to {out} "
          f"(dropped={n_dropped}, median valid depth px/frame={med})", flush=True)
    print("[rgbd2frames] reconstruct:  python -m scene_graph.offline.run --source frames-json "
          f"--frames-json-dir {out} --save-path scene.pt --covisibility", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())


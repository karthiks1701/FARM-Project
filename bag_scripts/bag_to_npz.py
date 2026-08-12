#!/usr/bin/env python3
"""Convert the ria_rgb_d_mistlab_1 rosbag2 into FARM 'npz chunks' format.

Per DATA.md: each .npz holds images (N,H,W,3) uint8, depths (N,H,W) float32
metres (0/NaN = invalid), camtoworlds (N,4,4) float32, K (3,3) float32,
pose_convention, nominal_hz.

Pipeline for this bag:
  - RGB:   /oak/rgb/image_raw          (bgr8, ~19 Hz)
  - Depth: /oak/stereo/image_raw       (16UC1 mm, already registered to the
           RGB optical frame -- confirmed via matching header.frame_id)
  - Pose:  /laser_odometry (map -> os_lidar, ~3.16 Hz) interpolated to each
           RGB timestamp, then composed with the fixed static chain
           os_lidar -> oak_rgb_camera_frame -> oak_rgb_camera_optical_frame
           (camera mounted 30cm directly below the lidar; both /tf_static
           edges confirmed against user's physical description of the rig).

RGB and depth are synced by nearest timestamp within --sync-tol-ms.
"""
import argparse
import os
import bisect
import numpy as np
import rosbag2_py
from rclpy.serialization import deserialize_message
from rosidl_runtime_py.utilities import get_message


def quat_to_matrix(x, y, z, w):
    norm = x * x + y * y + z * z + w * w
    if norm <= 1e-12:
        return np.eye(3)
    s = 2.0 / norm
    xx, yy, zz = x * x * s, y * y * s, z * z * s
    xy, xz, yz = x * y * s, x * z * s, y * z * s
    wx, wy, wz = w * x * s, w * y * s, w * z * s
    return np.array([
        [1.0 - (yy + zz), xy - wz, xz + wy],
        [xy + wz, 1.0 - (xx + zz), yz - wx],
        [xz - wy, yz + wx, 1.0 - (xx + yy)],
    ])


def make_T(t, quat_xyzw):
    T = np.eye(4)
    T[:3, :3] = quat_to_matrix(*quat_xyzw)
    T[:3, 3] = t
    return T


def slerp(q0, q1, frac):
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    dot = np.dot(q0, q1)
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    dot = np.clip(dot, -1.0, 1.0)
    if dot > 0.9995:
        out = q0 + frac * (q1 - q0)
        return out / np.linalg.norm(out)
    theta0 = np.arccos(dot)
    theta = theta0 * frac
    q2 = q1 - q0 * dot
    q2 = q2 / np.linalg.norm(q2)
    return q0 * np.cos(theta) + q2 * np.sin(theta)


# --- fixed static chain: os_lidar -> oak_rgb_camera_frame -> optical -----
T_camFrame_to_lidar = make_T((0.0, 0.0, 0.3), (0.0, 0.0, 0.0, 1.0))
T_lidar_to_camFrame = np.linalg.inv(T_camFrame_to_lidar)
T_camFrame_to_optical = make_T((0.0, 0.0, 0.0), (0.5, -0.5, 0.5, -0.5))
T_lidar_to_optical = T_lidar_to_camFrame @ T_camFrame_to_optical


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("bag")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rgb-topic", default="/oak/rgb/image_raw")
    ap.add_argument("--depth-topic", default="/oak/stereo/image_raw")
    ap.add_argument("--rgb-info-topic", default="/oak/rgb/camera_info")
    ap.add_argument("--odom-topic", default="/laser_odometry")
    ap.add_argument("--sync-tol-ms", type=float, default=40.0)
    ap.add_argument("--chunk-size", type=int, default=300)
    ap.add_argument("--stride", type=int, default=1, help="keep every Nth synced frame")
    ap.add_argument("--max-frames", type=int, default=0, help="stop after N written frames (0 = no limit); for quick testing")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    def open_reader():
        r = rosbag2_py.SequentialReader()
        r.open(
            rosbag2_py.StorageOptions(uri=args.bag, storage_id="sqlite3"),
            rosbag2_py.ConverterOptions("", ""),
        )
        return r

    type_map = {t.name: t.type for t in open_reader().get_all_topics_and_types()}
    rgb_type = get_message(type_map[args.rgb_topic])
    depth_type = get_message(type_map[args.depth_topic])
    info_type = get_message(type_map[args.rgb_info_topic])
    odom_type = get_message(type_map[args.odom_topic])

    K = None
    W = H = None
    odom_t = []           # ns
    odom_pos = []
    odom_quat = []

    # --- pass 1: load all odometry (tiny) + camera_info up front, so pose
    # interpolation always has both past and future brackets available ---
    pass1 = open_reader()
    pass1.set_filter(rosbag2_py.StorageFilter(topics=[args.odom_topic, args.rgb_info_topic]))
    while pass1.has_next():
        topic, data, t = pass1.read_next()
        if topic == args.odom_topic:
            msg = deserialize_message(data, odom_type)
            p = msg.pose.pose.position
            q = msg.pose.pose.orientation
            odom_t.append(t)
            odom_pos.append(np.array([p.x, p.y, p.z]))
            odom_quat.append(np.array([q.x, q.y, q.z, q.w]))
        elif topic == args.rgb_info_topic and K is None:
            msg = deserialize_message(data, info_type)
            K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
            W, H = msg.width, msg.height
    print(f"loaded {len(odom_t)} odometry samples, K=\n{K}\nW,H=({W},{H})")

    # --- pass 2: stream rgb+depth in order, sync + interpolate pose ---
    reader = open_reader()
    reader.set_filter(rosbag2_py.StorageFilter(topics=[args.rgb_topic, args.depth_topic]))
    depth_buf = []       # (t_ns, msg)

    pending_rgb = []      # (t_ns, msg) waiting to be matched once depth/odom catch up
    chunk_images, chunk_depths, chunk_poses = [], [], []
    chunk_idx = 0
    n_written = 0
    n_seen_rgb = 0

    def flush_chunk():
        nonlocal chunk_images, chunk_depths, chunk_poses, chunk_idx
        if not chunk_images:
            return
        path = os.path.join(args.out_dir, f"frames_{chunk_idx:03d}.npz")
        np.savez(
            path,
            images=np.stack(chunk_images).astype(np.uint8),
            depths=np.stack(chunk_depths).astype(np.float32),
            camtoworlds=np.stack(chunk_poses).astype(np.float32),
            K=K.astype(np.float32),
            pose_convention="opencv",
            nominal_hz=np.float32(19.0),
        )
        print(f"wrote {path} ({len(chunk_images)} frames)")
        chunk_idx += 1
        chunk_images, chunk_depths, chunk_poses = [], [], []

    def interp_pose(t_ns):
        if len(odom_t) < 2:
            return None
        if t_ns < odom_t[0] or t_ns > odom_t[-1]:
            return None
        i = bisect.bisect_right(odom_t, t_ns) - 1
        i = max(0, min(i, len(odom_t) - 2))
        t0, t1 = odom_t[i], odom_t[i + 1]
        if t1 == t0:
            frac = 0.0
        else:
            frac = (t_ns - t0) / (t1 - t0)
        pos = odom_pos[i] + frac * (odom_pos[i + 1] - odom_pos[i])
        quat = slerp(odom_quat[i], odom_quat[i + 1], frac)
        T_map_lidar = make_T(pos, quat)
        return T_map_lidar @ T_lidar_to_optical

    def decode_rgb(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.step)[:, : msg.width * 3]
        arr = arr.reshape(msg.height, msg.width, 3)
        if msg.encoding == "bgr8":
            arr = arr[:, :, ::-1]
        elif msg.encoding != "rgb8":
            raise ValueError(f"unexpected rgb encoding {msg.encoding!r}")
        return arr

    def decode_depth(msg):
        arr = np.frombuffer(msg.data, dtype=np.uint16).reshape(msg.height, msg.step // 2)[:, : msg.width]
        depth_m = arr.astype(np.float32) / 1000.0
        depth_m[(arr == 0) | (arr == 65535)] = np.nan
        return depth_m

    tol_ns = int(args.sync_tol_ms * 1e6)

    def nearest_depth(t_ns):
        if not depth_buf:
            return None
        ts = [d[0] for d in depth_buf]
        i = bisect.bisect_left(ts, t_ns)
        best = None
        for j in (i - 1, i):
            if 0 <= j < len(ts):
                dt = abs(ts[j] - t_ns)
                if best is None or dt < best[0]:
                    best = (dt, j)
        if best is None or best[0] > tol_ns:
            return None
        return depth_buf[best[1]][1]

    def resolve(rgb_t, rgb_msg):
        nonlocal n_written
        if args.stride > 1 and (n_seen_rgb % args.stride) != 0:
            return
        depth_msg = nearest_depth(rgb_t)
        pose = interp_pose(rgb_t)
        if depth_msg is None or pose is None:
            return
        chunk_images.append(decode_rgb(rgb_msg))
        chunk_depths.append(decode_depth(depth_msg))
        chunk_poses.append(pose)
        n_written += 1
        if len(chunk_images) >= args.chunk_size:
            flush_chunk()

    # pending_rgb holds rgb frames waiting for enough *future* depth lookahead
    # (depth can arrive slightly after its matching rgb in bag order) before
    # we commit to a nearest match; resolved once the stream has advanced
    # past their tolerance window.
    pending_rgb = []  # list of (t_ns, msg, n_seen_rgb_at_arrival)

    while reader.has_next():
        if args.max_frames and n_written >= args.max_frames:
            break
        topic, data, t = reader.read_next()
        if topic == args.depth_topic:
            msg = deserialize_message(data, depth_type)
            depth_buf.append((t, msg))
        elif topic == args.rgb_topic:
            n_seen_rgb += 1
            msg = deserialize_message(data, rgb_type)
            pending_rgb.append((t, msg))

        while pending_rgb and (t - pending_rgb[0][0] > tol_ns):
            rgb_t, rgb_msg = pending_rgb.pop(0)
            resolve(rgb_t, rgb_msg)

        # trim depth_buf to what's still needed: newest pending rgb - tol
        cutoff = (pending_rgb[0][0] if pending_rgb else t) - tol_ns
        while depth_buf and depth_buf[0][0] < cutoff:
            depth_buf.pop(0)

    for rgb_t, rgb_msg in pending_rgb:
        resolve(rgb_t, rgb_msg)

    flush_chunk()
    print(f"\ndone: {n_written} frames written across {chunk_idx} chunk(s) "
          f"(of {n_seen_rgb} rgb messages seen)")


if __name__ == "__main__":
    main()

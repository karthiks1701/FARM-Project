#!/usr/bin/env python3
"""Build the ``cloud.npz`` background cloud for ``view_scene_state.py --cloud``.

Back-projects every frame's depth into the world frame and voxel-downsamples the
union — the same RGBD data that built your ``scene_state.pt``, so the cloud lines
up with the scene graph exactly. ROS-free: it reads a ``frames.json`` scene
directory (what ``scripts/rgbd_bag_to_frames.py`` / ``scripts/rgb_bag_frame.py``
produce from a rosbag), not the bag itself.

    # 1. rosbag -> frames.json  (already done if you built the .pt via --source frames-json)
    python scripts/rgbd_bag_to_frames.py --bag-dir /data/bags/run --out-dir /data/scenes/run \
        --camera cam0 --rgb-topic ... --depth-topic ... --odometry-topic ... --camera-info-topic ...

    # 2. frames.json -> cloud.npz
    python scripts/frames_to_cloud.py --frames-json-dir /data/scenes/run --voxel-size 0.03

    # 3. view
    python scripts/view_scene_state.py --pt /data/out/run.pt --cloud /data/scenes/run/cloud.npz

Output archive holds ``points`` (N,3 float32, world frame) and ``colors``
(N,3 uint8) — the key names ``view_scene_state.load_cloud`` expects.

If your bag instead carries an aggregated SLAM ``PointCloud2`` (LiDAR),
``scripts/lidar_bag_to_frames.py --cloud-topic <topic>`` already writes a
``cloud.npz`` (key ``xyz``) and you don't need this script.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def _voxel_dedup(pts: np.ndarray, cols: np.ndarray, voxel: float) -> tuple[np.ndarray, np.ndarray]:
    if voxel <= 0.0 or pts.shape[0] == 0:
        return pts, cols
    keys = np.floor(pts / voxel).astype(np.int64)
    _, idx = np.unique(keys, axis=0, return_index=True)
    return pts[idx], cols[idx]


def _backproject(depth: np.ndarray, rgb: np.ndarray, intr: dict, T_world_cam: np.ndarray,
                 depth_max: float) -> tuple[np.ndarray, np.ndarray]:
    """Pinhole back-projection (OpenCV/ROS-optical convention) -> world points + colors."""
    h, w = depth.shape
    fx, fy, cx, cy = intr["fx"], intr["fy"], intr["cx"], intr["cy"]
    valid = np.isfinite(depth) & (depth > 0.0)
    if depth_max > 0.0:
        valid &= depth <= depth_max
    if not np.any(valid):
        return np.zeros((0, 3), np.float32), np.zeros((0, 3), np.uint8)

    vs, us = np.nonzero(valid)
    z = depth[vs, us].astype(np.float32)
    x = (us.astype(np.float32) - cx) * z / fx
    y = (vs.astype(np.float32) - cy) * z / fy
    cam = np.stack([x, y, z], axis=1)
    world = cam @ np.asarray(T_world_cam[:3, :3], np.float32).T + np.asarray(T_world_cam[:3, 3], np.float32)

    finite = np.isfinite(world).all(axis=1)
    world = world[finite]
    if rgb is not None and rgb.shape[:2] == depth.shape:
        cols = rgb[vs, us][finite].astype(np.uint8)
    else:
        cols = np.full((world.shape[0], 3), 180, np.uint8)
    return world.astype(np.float32), cols


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-json-dir", type=Path, required=True,
                    help="Scene directory containing frames.json + rgb/ + depth/")
    ap.add_argument("--out", type=Path, default=None,
                    help="Output .npz (default: <frames-json-dir>/cloud.npz)")
    ap.add_argument("--cameras", nargs="+", default=None, help="Restrict to these cameras")
    ap.add_argument("--stride", type=int, default=1, help="Use every Nth frame")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=-1)
    ap.add_argument("--voxel-size", type=float, default=0.03, help="Voxel-downsample size in metres (0 = keep all)")
    ap.add_argument("--depth-max", type=float, default=0.0, help="Drop depth beyond this many metres (0 = no limit)")
    ap.add_argument("--pixel-stride", type=int, default=1,
                    help="Subsample pixels by this factor before back-projection (speed/memory)")
    ap.add_argument("--max-points", type=int, default=8_000_000, help="Random-subsample the final cloud to this many points")
    ap.add_argument("--no-colors", action="store_true", help="Store points only (smaller file)")
    ap.add_argument("--compact-every", type=int, default=20, help="Voxel-dedup the accumulator every N frames")
    args = ap.parse_args()

    from scene_graph.offline.frame_sources.frames_json import FramesJsonFrameSource

    out = args.out or (args.frames_json_dir.expanduser() / "cloud.npz")
    source = FramesJsonFrameSource(
        args.frames_json_dir.expanduser(),
        cameras=args.cameras,
        stride=max(1, args.stride),
        start=max(0, args.start),
        end=None if args.end < 0 else args.end,
        depth_clip_m=args.depth_max if args.depth_max > 0 else None,
    )

    ps = max(1, int(args.pixel_stride))
    acc_pts = np.zeros((0, 3), np.float32)
    acc_cols = np.zeros((0, 3), np.uint8)
    n_frames = 0
    t0 = time.time()
    for item in source:
        depth = np.asarray(item["depth_f32"], np.float32)
        rgb = np.asarray(item["rgb"], np.uint8) if not args.no_colors else None
        T = np.asarray(item["T_world_cam"], np.float32).reshape(4, 4)
        intr = item["depth_instrinsics"]
        if ps > 1:
            depth = depth[::ps, ::ps]
            if rgb is not None:
                rgb = rgb[::ps, ::ps]
            intr = {**intr, "fx": intr["fx"] / ps, "fy": intr["fy"] / ps,
                    "cx": intr["cx"] / ps, "cy": intr["cy"] / ps}
        if not np.all(np.isfinite(T)):
            continue
        pts, cols = _backproject(depth, rgb, intr, T, args.depth_max)
        if pts.shape[0] == 0:
            continue
        acc_pts = np.concatenate([acc_pts, pts], axis=0)
        acc_cols = np.concatenate([acc_cols, cols], axis=0)
        n_frames += 1
        if n_frames % max(1, args.compact_every) == 0:
            acc_pts, acc_cols = _voxel_dedup(acc_pts, acc_cols, args.voxel_size)
            print(f"  {n_frames} frames -> {acc_pts.shape[0]:,} voxels ({time.time() - t0:.1f}s)")

    acc_pts, acc_cols = _voxel_dedup(acc_pts, acc_cols, args.voxel_size)
    if acc_pts.shape[0] == 0:
        print("No points produced — check depth/pose in frames.json.")
        return 1

    if args.max_points > 0 and acc_pts.shape[0] > args.max_points:
        keep = np.random.default_rng(0).choice(acc_pts.shape[0], size=args.max_points, replace=False)
        acc_pts, acc_cols = acc_pts[keep], acc_cols[keep]

    out.parent.mkdir(parents=True, exist_ok=True)
    if args.no_colors:
        np.savez_compressed(out, points=acc_pts)
    else:
        np.savez_compressed(out, points=acc_pts, colors=acc_cols)
    print(
        f"Wrote {acc_pts.shape[0]:,} points from {n_frames} frames to {out} "
        f"(voxel {args.voxel_size} m, {time.time() - t0:.1f}s)"
    )
    print(f"  view: python scripts/view_scene_state.py --pt <scene>.pt --cloud {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""View a saved ``scene_state.pt`` in the browser (viser) — no re-mapping needed.

Loads a scene graph produced by ``python -m scene_graph.offline.run`` (or one
of the prebuilt graphs shipped with FARM-Scenes) and serves it interactively:
per-object voxel clouds and 3D boxes with captions, click-to-inspect, and the
**Query** panel, which runs the full relational retrieval pipeline
(``parse_query`` -> ``execute_spatial_query``) against the loaded scene.

Scene-state files use PyTorch pickle serialization. Only open files produced
by FARM or downloaded from a source you trust.

Examples (inside the container)::

    # Objects only
    python scripts/view_scene_state.py --pt /data/out/warehouse.pt

    # With the dataset's accumulated point cloud as background context
    python scripts/view_scene_state.py \
        --pt /data/scene_graphs/grandtour/2024-11-25_warehouse.pt \
        --cloud /data/scenes/grandtour/2024-11-25_warehouse/cloud.npz

    # Overlay a Spot Autowalk nav graph + a 1 m metric ground grid
    python scripts/view_scene_state.py --pt /data/out/site.pt \
        --walk /data/walks/site.walk --grid-cell-m 1.0

``--ws-url ws://<robot-host>:8765`` streams a remote robot's ``/odometry``
(live ``/robot_pose`` frame + trail) and color image (**Live RGB** panel) from a
``scripts/ros_ws_bridge.py`` running on that host — no ROS on this machine.
Live odometry is aligned to the map by ``DEFAULT_SEED_TFORM_BODY`` (a
``seed_tform_body`` baked into this script); override with ``--world-transform``
/ ``--world-transform-se3`` or disable with ``--no-world-transform``.

#   python scripts/view_scene_state.py --pt data/spot_scene_graph_caption_4_seed_1.pt --walk /data/walks/site.walk --grid-cell-m 1.0 --ws-url ws://192.168.1.192:8765

A metric ground grid is drawn by default (``--no-grid`` to suppress, or toggle
it in the **Metric grid** GUI panel). ``--walk`` overlays a Boston Dynamics
GraphNav / Autowalk ``.walk`` map: one coordinate frame per waypoint pose,
connecting edges, waypoint-name labels, and anchored fiducials, toggled in the
**Nav graph** panel. Needs ``bosdyn-api`` (``pip install bosdyn-api`` — protobuf
only). Waypoint poses come from the map's anchoring (``--walk-anchor seed``) or a
BFS over edge transforms (``--walk-anchor bfs``); ``auto`` picks anchoring when
the map has it.

Then open http://localhost:8080. For language queries, either click
"Start vLLM retrieval backend" in the Query panel (launches the servers on
this machine) or run ``./run.sh vllm`` first / point ``VLLM_BASE_URL`` +
``VLLM_EMBED_BASE_URL`` at running servers.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch


# Default map alignment for live odometry (``--ws-url`` / ``--odom-topic``):
# Spot ``seed_tform_body`` (SE3Pose — translation + quaternion in Boston
# Dynamics x,y,z,w order). Used when neither ``--world-transform`` nor
# ``--world-transform-se3`` is given, and the scene state carries none.
# Override on the CLI, or pass ``--no-world-transform`` to align 1:1.
DEFAULT_SEED_TFORM_BODY = {
    "position": (2.2729322207071028, 0.19774609072594404, -0.53609861630672606),
    "rotation_xyzw": (
        0.0047556974067231462,
        0.003794053310715498,
        0.99953135411525507,
        0.030001010685907416,
    ),
}


def se3_to_matrix(x: float, y: float, z: float, qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Row-major 4x4 from a translation + quaternion (x, y, z, w order)."""
    n = qx * qx + qy * qy + qz * qz + qw * qw
    s = 2.0 / n if n > 1e-12 else 0.0
    return np.array(
        [
            [1 - s * (qy * qy + qz * qz), s * (qx * qy - qz * qw), s * (qx * qz + qy * qw), x],
            [s * (qx * qy + qz * qw), 1 - s * (qx * qx + qz * qz), s * (qy * qz - qx * qw), y],
            [s * (qx * qz - qy * qw), s * (qy * qz + qx * qw), 1 - s * (qx * qx + qy * qy), z],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def load_state(pt_path: Path) -> dict:
    """Load a scene_state payload, normalising through the library loader."""
    payload = torch.load(pt_path, map_location="cpu", weights_only=False)
    feature_dim = None
    if isinstance(payload, dict):
        feature_dim = payload.get("feature_dim")
        if feature_dim is None and isinstance(payload.get("state"), dict):
            feats = payload["state"].get("features")
            if isinstance(feats, torch.Tensor) and feats.ndim == 2:
                feature_dim = int(feats.shape[1])
    if feature_dim is not None:
        from scene_graph.scene_state_io import load_scene_state

        return load_scene_state(pt_path, feature_dim=int(feature_dim), device="cpu")
    # Fallback: raw dict without the save_scene_state wrapper.
    return payload["state"] if isinstance(payload, dict) and "state" in payload else payload


def _record_field(rec: object, key: str) -> object:
    """Field access for image records, which are dicts in raw payloads and
    ``ImageRecord`` dataclasses after ``load_scene_state`` normalisation."""
    if isinstance(rec, dict):
        return rec.get(key)
    return getattr(rec, key, None)


def remap_image_refs(state: dict, frames_dir: Path) -> int:
    """Re-point saved image references at *frames_dir* (a scene directory with
    ``rgb/<camera>/`` frames) so click-to-inspect can show each object's anchor
    view. Saved graphs reference the reconstruction machine's paths; the same
    frames ship with the dataset."""
    remapped = 0
    for rec in state.get("images") or []:
        ref = str(_record_field(rec, "source_ref") or _record_field(rec, "storage_path") or "")
        base = Path(ref).name
        if not base:
            continue
        camera = str(_record_field(rec, "camera_id") or "")
        candidates = [frames_dir / "rgb" / camera / base] if camera else []
        candidates += [frames_dir / "rgb" / base, frames_dir / base]
        for cand in candidates:
            if cand.is_file():
                if isinstance(rec, dict):
                    rec["source_ref"] = str(cand)
                else:
                    rec.source_ref = str(cand)
                remapped += 1
                break
    return remapped


def load_cloud(cloud_path: Path, max_points: int) -> tuple[np.ndarray, np.ndarray | None]:
    """Read points (+ optional colors) from a ``cloud.npz``-style archive."""
    with np.load(cloud_path) as data:
        pts = None
        for key in ("points", "xyz", "cloud"):
            if key in data.files:
                pts = np.asarray(data[key], dtype=np.float32)
                break
        if pts is None:
            pts = np.asarray(data[data.files[0]], dtype=np.float32)
        pts = pts.reshape(-1, pts.shape[-1])[:, :3]
        cols = None
        for key in ("colors", "rgb", "color"):
            if key in data.files:
                cols = np.asarray(data[key], dtype=np.float32).reshape(-1, 3)
                break
    if max_points > 0 and pts.shape[0] > max_points:
        keep = np.random.default_rng(0).choice(pts.shape[0], size=max_points, replace=False)
        pts = pts[keep]
        if cols is not None and cols.shape[0] >= keep.max() + 1:
            cols = cols[keep]
    return pts, cols


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Serve a saved scene_state.pt in viser (objects, captions, retrieval).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--pt", type=Path, required=True, help="Path to scene_state.pt")
    parser.add_argument("--cloud", type=Path, default=None, help="Optional cloud.npz shown as static background")
    parser.add_argument("--frames-dir", type=Path, default=None, help="Scene directory with rgb/<camera>/ frames for click-to-inspect anchor views (default: the --cloud directory)")
    parser.add_argument("--host", default="127.0.0.1", help="Interface for the viser server")
    parser.add_argument("--port", type=int, default=8080, help="Port for the viser server")
    parser.add_argument("--point-size", type=float, default=0.02, help="Background cloud point size (m)")
    parser.add_argument("--max-cloud-points", type=int, default=2_000_000, help="Random-subsample the background cloud to this many points (0 = keep all)")
    parser.add_argument("--voxel-points-per-object", type=int, default=0, help="Per-object voxel evidence points to render (0 = clean default: boxes + cloud + trajectory only)")
    parser.add_argument("--walk", type=Path, default=None, help="Spot GraphNav / Autowalk '.walk' directory (or its 'graph' protobuf) to overlay: one coordinate frame per waypoint pose, connecting edges, name labels, and anchored fiducials. Toggle in the 'Nav graph' GUI panel.")
    parser.add_argument("--walk-anchor", choices=("auto", "seed", "bfs"), default="auto", help="How to lift waypoint poses into one frame: 'auto' uses the map's anchoring (seed frame) if present else BFS over edge transforms; 'seed' requires an anchored map; 'bfs' always composes edge transforms from the first waypoint")
    parser.add_argument("--walk-transform", type=float, nargs=16, default=None, metavar="M", help="Row-major 4x4 applied to every waypoint/anchor pose after resolving (residual nudge onto the scene frame)")
    parser.add_argument("--no-camera-trajectory", action="store_true", help="Don't draw the scene's capture path (one small coordinate frame per RGB image pose, the '/trajectory' loop). Useful when overlaying a '--walk' nav graph instead.")
    parser.add_argument("--show-all-boxes", action="store_true", help="Draw every object box on load. Default: boxes stay hidden until a query (or a Top-matches isolate click) focuses some; toggle live in Filters.")
    parser.add_argument("--top-down", action="store_true", help="Open with a fixed top-down (bird's-eye) camera looking straight down the up axis, and keep that orientation on every client connect / view reset.")
    parser.add_argument("--no-grid", action="store_true", help="Disable the metric ground grid overlay (on by default; also toggleable in the 'Metric grid' GUI panel)")
    parser.add_argument("--grid-cell-m", type=float, default=1.0, help="Metric grid cell size in meters (adjustable live in the GUI)")
    parser.add_argument("--query-examples", type=Path, default=None, help="Text file with one query per line for the Query panel's Examples dropdown (default: derive examples from the scene's own captioned objects)")
    parser.add_argument("--ws-url", default=None, help="WebSocket URL of a scripts/ros_ws_bridge.py running on the robot host (e.g. ws://192.168.1.192:8765). Streams /odometry (robot pose + trail) and the color image (Live RGB panel) without ROS on this machine. --world-transform aligns the odometry like --odom-topic.")
    parser.add_argument("--ws-no-image", action="store_true", help="With --ws-url, ignore the streamed camera image (odometry only)")
    parser.add_argument("--ws-odom-color", type=int, nargs=3, default=(236, 64, 200), metavar=("R", "G", "B"), help="RGB colour (0-255) for the WebSocket odometry trail + axes")
    parser.add_argument("--ws-odom-axes-spacing-m", type=float, default=0.5, help="Drop an orientation axes triad every this many metres along the WebSocket odometry trail")
    parser.add_argument("--ws-odom-axes-length", type=float, default=0.18, help="Length (m) of the WebSocket odometry axes triads")
    parser.add_argument("--odom-topic", default=None, help="Subscribe to this ROS 2 nav_msgs/Odometry topic (e.g. /odometry) and draw the robot's live pose + trail in the scene; each Query press also drops a marker where the robot was")
    parser.add_argument("--ros-domain-id", type=int, default=None, help="ROS_DOMAIN_ID for the --odom-topic subscription (default: current env)")
    parser.add_argument("--odom-qos", choices=("reliable", "best_effort"), default="reliable", help="QoS reliability for --odom-topic (Spot's driver often needs best_effort)")
    parser.add_argument("--odom-trail-seconds", type=float, default=0.0, help="Discard live odometry samples older than this many seconds (0 = keep the whole session)")
    parser.add_argument(
        "--world-transform",
        type=float,
        nargs=16,
        default=None,
        metavar="M",
        help="Row-major 4x4 T_map_odom applied to every live odometry pose before drawing "
        "(use the SAME matrix passed to scripts/rgb_bag_frame.py --world-transform, e.g. seed_tform_body)",
    )
    parser.add_argument(
        "--world-transform-se3",
        type=float,
        nargs=7,
        default=None,
        metavar=("X", "Y", "Z", "QX", "QY", "QZ", "QW"),
        help="Same as --world-transform but as an SE3Pose: translation x y z then quaternion "
        "qx qy qz qw (Boston Dynamics order). Handy for pasting a Spot seed_tform_body.",
    )
    parser.add_argument(
        "--no-world-transform",
        action="store_true",
        help="Ignore the built-in DEFAULT_SEED_TFORM_BODY (and any stored transform): draw live "
        "odometry 1:1 in its own frame.",
    )
    args = parser.parse_args()

    from scene_graph.visualization.viser_visualizer import PipelineViserVisualizer

    state = load_state(args.pt.expanduser())
    frames_dir = args.frames_dir
    if frames_dir is None and args.cloud is not None:
        frames_dir = args.cloud.expanduser().parent
    if frames_dir is not None:
        remapped = remap_image_refs(state, Path(frames_dir).expanduser())
        if remapped:
            print(f"Resolved {remapped} image references into {frames_dir}")
    means = state.get("means")
    n_objects = int(means.shape[0]) if isinstance(means, torch.Tensor) else 0
    captions = [c for c in (state.get("object_caption") or []) if isinstance(c, str) and c.strip()]
    print(f"Loaded {n_objects} objects ({len(captions)} captioned) from {args.pt}")

    query_examples = None
    if args.query_examples is not None:
        lines = args.query_examples.expanduser().read_text().splitlines()
        query_examples = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
        print(f"Loaded {len(query_examples)} query examples from {args.query_examples}")

    visualizer = PipelineViserVisualizer(
        enabled=True,
        host=args.host,
        port=args.port,
        query_examples=query_examples,
        live_rgb_enabled=bool(args.ws_url) and not args.ws_no_image,
        image_pose_axes_enabled=False,
        object_image_connections_enabled=False,
        covisibility_connections_enabled=False,
        regions_enabled=False,
        object_voxel_cloud_enabled=args.voxel_points_per_object > 0,
        object_voxel_max_points_per_object=max(0, args.voxel_points_per_object),
        object_box_from_voxels=True,
        object_boxes_start_hidden=not args.show_all_boxes,
        send_to_spot_enabled=bool(args.ws_url),
    )
    if not visualizer.enabled:
        print("viser is not available in this environment — aborting.")
        return 1

    frame_points = None
    if args.cloud is not None:
        try:
            points, colors = load_cloud(args.cloud.expanduser(), args.max_cloud_points)
            visualizer.add_background_point_cloud(points, colors, point_size=args.point_size)
            print(f"Background cloud: {points.shape[0]:,} points from {args.cloud}")
            frame_points = points
        except Exception as exc:  # noqa: BLE001 - cloud is optional context
            print(f"Skipping background cloud ({exc})")

    visualizer.update(colors=[], depths=[], intrinsics=[], poses=[], scene_state=state)
    poses = []
    if not args.no_camera_trajectory:
        for rec in state.get("images") or []:
            pose = _record_field(rec, "pose")
            if pose is None:
                continue
            try:
                arr = np.asarray(pose.cpu().numpy() if hasattr(pose, "cpu") else pose, dtype=np.float32)
            except Exception:
                continue
            if arr.shape == (4, 4) and np.isfinite(arr).all():
                poses.append(arr)
    if poses:
        try:
            visualizer.add_trajectory(np.stack(poses))
        except Exception as exc:  # noqa: BLE001 - trajectory is optional context
            print(f"Skipping trajectory ({exc})")
    if frame_points is None and isinstance(means, torch.Tensor) and means.numel():
        frame_points = means.detach().cpu().numpy().reshape(-1, 3)
    visualizer.set_home_view(frame_points, top_down=args.top_down)

    if not args.no_grid:
        try:
            visualizer.add_metric_grid(frame_points, cell_m=args.grid_cell_m)
            print(f"Metric grid: {args.grid_cell_m:g} m cells (toggle in the 'Metric grid' panel)")
        except Exception as exc:  # noqa: BLE001 - grid is optional context
            print(f"Skipping metric grid ({exc})")

    if args.walk is not None:
        try:
            from scene_graph.visualization.graphnav_walk import load_walk_graph

            walk_transform = None
            if args.walk_transform is not None:
                walk_transform = np.asarray(args.walk_transform, dtype=np.float64).reshape(4, 4)
            nav_graph = load_walk_graph(
                args.walk.expanduser(),
                anchor=args.walk_anchor,
                extra_transform=walk_transform,
            )
            visualizer.add_nav_graph(nav_graph)
            wp = nav_graph.waypoint_positions()
            print(
                f"Nav graph: {len(nav_graph.waypoints)} waypoints, "
                f"{len(nav_graph.edge_segments())} edges, "
                f"{len(nav_graph.anchored_objects)} anchored objects "
                f"(frame={nav_graph.frame}) from {args.walk}"
            )
            if wp.shape[0] and frame_points is not None and len(frame_points):
                fp = np.asarray(frame_points, dtype=np.float64).reshape(-1, 3)
                wp_c = wp.mean(axis=0)
                sc_c = fp.mean(axis=0)
                gap = float(np.linalg.norm(wp_c - sc_c))
                sc_span = float(np.linalg.norm(fp.max(0) - fp.min(0)))
                if gap > 0.5 * sc_span + 5.0:
                    print(
                        f"  ⚠ waypoints centre {np.round(wp_c, 1)} is {gap:.0f} m from the scene "
                        f"centre {np.round(sc_c, 1)} — the '.walk' seed frame and this scene are "
                        f"not aligned. They will render off to one side. Pass --walk-transform / "
                        f"--walk-anchor bfs, or check this is the right .walk for this scene."
                    )
        except Exception as exc:  # noqa: BLE001 - nav graph is optional context
            print(f"Skipping nav graph ({exc})")

    # Shared alignment for any live odometry source (--odom-topic or --ws-url).
    # Priority: --world-transform > --world-transform-se3 > stored in the scene
    # state > built-in DEFAULT_SEED_TFORM_BODY. --no-world-transform forces 1:1.
    world_transform = None
    if args.no_world_transform:
        print("Live odometry alignment: none (--no-world-transform)")
    elif args.world_transform is not None:
        world_transform = np.asarray(args.world_transform, dtype=np.float64).reshape(4, 4)
        print("Live odometry alignment: --world-transform (4x4)")
    elif args.world_transform_se3 is not None:
        world_transform = se3_to_matrix(*(float(v) for v in args.world_transform_se3))
        print("Live odometry alignment: --world-transform-se3")
    elif isinstance(state.get("world_transform"), (list, np.ndarray)):
        world_transform = np.asarray(state["world_transform"], dtype=np.float64).reshape(4, 4)
        print("Live odometry alignment: world_transform stored in the scene state")
    else:
        px, py, pz = DEFAULT_SEED_TFORM_BODY["position"]
        world_transform = se3_to_matrix(px, py, pz, *DEFAULT_SEED_TFORM_BODY["rotation_xyzw"])
        print(
            f"Live odometry alignment: built-in DEFAULT_SEED_TFORM_BODY "
            f"t=({px:.3f}, {py:.3f}, {pz:.3f}) (override with --world-transform*, "
            f"disable with --no-world-transform)"
        )

    ws_client = None
    if args.ws_url:
        try:
            from ros_ws_client import RosWsClient

            ws_client = RosWsClient(
                args.ws_url,
                world_transform=world_transform,
                want_image=not args.ws_no_image,
            )
            ws_client.start()

            def _send_goal(payload: dict) -> None:
                ok = ws_client.send(payload)
                kind = payload.get("type")
                if kind == "goto":
                    print(
                        f"[send-to-spot] {'sent' if ok else 'NOT SENT (offline)'} goal "
                        f"#{payload.get('object_id')} x={payload.get('x'):.2f} "
                        f"y={payload.get('y'):.2f} yaw={payload.get('yaw'):.3f} "
                        f"tol={payload.get('tol_m')}"
                    )
                else:
                    print(f"[send-to-spot] {'sent' if ok else 'NOT SENT'} {kind}")

            visualizer.set_send_goal_handler(_send_goal)
            print(
                f"Connecting to ROS bridge {args.ws_url} — robot pose + trail"
                + ("" if args.ws_no_image else " + Live RGB panel")
                + "; 'Send to Spot' panel active."
            )
        except Exception as exc:  # noqa: BLE001 - stream is optional context
            print(f"Could not start WebSocket client ({exc}); continuing without it.")
            ws_client = None

    odom_listener = None
    if args.odom_topic:
        try:
            from odom_ros_listener import OdomRosListener

            odom_listener = OdomRosListener(
                topic=args.odom_topic,
                ros_domain_id=args.ros_domain_id,
                qos=args.odom_qos,
                world_transform=world_transform,
                trail_seconds=args.odom_trail_seconds,
            )
            odom_listener.start()
            print(
                f"Listening to {args.odom_topic} on ROS_DOMAIN_ID="
                f"{args.ros_domain_id if args.ros_domain_id is not None else '(env)'} — "
                "robot pose + trail will appear in the scene."
            )
        except Exception as exc:  # noqa: BLE001 - odometry is optional context
            print(f"Could not start odometry listener ({exc}); continuing without it.")
            odom_listener = None

    print(f"Serving on http://localhost:{args.port} — Ctrl+C to stop.")
    last_image_stamp = None
    try:
        while True:
            active = False
            if odom_listener is not None:
                stamp, pose = odom_listener.latest()
                if pose is not None:
                    visualizer.set_robot_pose(pose, stamp=stamp)
                active = True
            if ws_client is not None:
                stamp, pose = ws_client.latest_odom()
                if pose is not None:
                    visualizer.add_live_odometry_pose(
                        pose,
                        stamp=stamp,
                        color=tuple(args.ws_odom_color),
                        axes_spacing_m=args.ws_odom_axes_spacing_m,
                        axes_length=args.ws_odom_axes_length,
                    )
                if not args.ws_no_image:
                    istamp, image = ws_client.latest_image()
                    if image is not None and istamp != last_image_stamp:
                        visualizer.set_live_rgb(image, caption=f"Live camera @ {istamp:.1f}s")
                        last_image_stamp = istamp
                active = True
            time.sleep(0.1 if active else 2.0)
    except KeyboardInterrupt:
        print("Bye.")
    finally:
        if odom_listener is not None:
            odom_listener.stop()
        if ws_client is not None:
            ws_client.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

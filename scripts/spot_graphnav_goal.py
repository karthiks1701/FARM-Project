#!/usr/bin/env python3
"""Drive Spot's GraphNav to navigation goals sent from the FARM viser
"Send to Spot" panel.

Runs on the robot host (e.g. 192.168.1.192) **inside the Spot SDK environment**
(the same env you run ``graph_nav_command_line`` from) — NOT the FARM Docker
container. It:

  * connects to Spot, checks the E-stop, takes the lease,
  * uploads the GraphNav map from ``--graph-path`` (your ``downloaded_graph/``),
  * localizes to the nearest fiducial (prompt, or ``--auto-localize``),
  * connects to the local ``ros_ws_bridge.py`` and waits for goals:
        viser "Send to Spot"  --WebSocket-->  ros_ws_bridge.py  -->  this script
  * for each goal, calls ``GraphNavClient.navigate_to_anchor`` with the
    seed-frame ``x, y, yaw`` + tolerance.

Every pose from the FARM scene graph is already in Spot's ``seed`` frame, so no
transform is applied here.

SAFETY
------
* A software E-stop must be running or Spot refuses to power its motors. Start
  it first, in its own terminal (spot-sdk ``examples/estop``)::

      python3 -m estop_gui  192.168.50.3        # big STOP button + keepalive

  then "Release" control on the tablet. This script only takes the *lease*.
* **Dry-run is the default** — goals are printed, motors are NOT powered, the
  robot does NOT move. Pass ``--execute`` to actually drive.
* Ctrl-C releases the lease -> Spot stops walking and stands (NOT an e-stop).
* Emergency: press STOP in estop_gui (or the tablet) -> motor power is cut
  immediately, regardless of this script.
* The viser panel is the per-goal confirmation (arm + send). This script
  executes goals as they arrive.

Credentials: copy ``scripts/spot_credentials.py.example`` to
``scripts/spot_credentials.py`` (git-ignored) and fill it in; falls back to
BOSDYN_CLIENT_USERNAME / BOSDYN_CLIENT_PASSWORD, then a prompt.

    python3 scripts/spot_graphnav_goal.py \
        --robot-hostname 192.168.50.3 \
        --graph-path ~/mist_ws_ros2/scene_graph/data_walk/nav_graph_spot_4/downloaded_graph/ \
        --ws-url ws://127.0.0.1:8765
        # add --execute when you are ready to move the robot
"""

from __future__ import annotations

import argparse
import contextlib
import math
import os
import queue
import sys
import time
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))


def _load_credentials() -> tuple[str, str]:
    try:
        import spot_credentials  # scripts/spot_credentials.py (git-ignored)

        u, p = str(spot_credentials.USERNAME), str(spot_credentials.PASSWORD)
        if u and p and p != "REPLACE_ME":
            return u, p
    except Exception:
        pass
    u = os.environ.get("BOSDYN_CLIENT_USERNAME", "")
    p = os.environ.get("BOSDYN_CLIENT_PASSWORD", "")
    if u and p:
        return u, p
    import getpass

    return input("Spot username: ").strip(), getpass.getpass("Spot password: ")


# ---------------------------------------------------------------------------
# GraphNav helpers (mirror bosdyn examples/graph_nav_command_line)
# ---------------------------------------------------------------------------
def _upload_graph(graph_nav_client, graph_path: Path):
    from bosdyn.api.graph_nav import map_pb2

    graph_path = graph_path.expanduser()
    with open(graph_path / "graph", "rb") as f:
        graph = map_pb2.Graph()
        graph.ParseFromString(f.read())
    print(
        f"[spot-nav] loaded graph: {len(graph.waypoints)} waypoints, "
        f"{len(graph.edges)} edges, {len(graph.anchoring.anchors)} anchors"
    )
    waypoint_snapshots, edge_snapshots = {}, {}
    for wp in graph.waypoints:
        if not wp.snapshot_id:
            continue
        sp = graph_path / "waypoint_snapshots" / wp.snapshot_id
        if not sp.is_file():
            continue
        with open(sp, "rb") as f:
            snap = map_pb2.WaypointSnapshot()
            snap.ParseFromString(f.read())
            waypoint_snapshots[snap.id] = snap
    for edge in graph.edges:
        if not edge.snapshot_id:
            continue
        sp = graph_path / "edge_snapshots" / edge.snapshot_id
        if not sp.is_file():
            continue
        with open(sp, "rb") as f:
            snap = map_pb2.EdgeSnapshot()
            snap.ParseFromString(f.read())
            edge_snapshots[snap.id] = snap

    graph_nav_client.clear_graph()
    generate_new_anchoring = len(graph.anchoring.anchors) == 0
    if generate_new_anchoring:
        print("[spot-nav] WARNING: graph has no anchoring; seed-frame goals will not match the scene graph.")
    resp = graph_nav_client.upload_graph(graph=graph, generate_new_anchoring=generate_new_anchoring)
    for snapshot_id in resp.unknown_waypoint_snapshot_ids:
        graph_nav_client.upload_waypoint_snapshot(waypoint_snapshots[snapshot_id])
    for snapshot_id in resp.unknown_edge_snapshot_ids:
        graph_nav_client.upload_edge_snapshot(edge_snapshots[snapshot_id])
    print("[spot-nav] graph + snapshots uploaded.")
    return graph


def _localize(graph_nav_client, auto: bool) -> bool:
    from bosdyn.api.graph_nav import graph_nav_pb2, nav_pb2

    state = graph_nav_client.get_localization_state()
    if state.localization.waypoint_id:
        print(f"[spot-nav] already localized to {state.localization.waypoint_id}")
        return True
    if not auto:
        ans = input("[spot-nav] Not localized. Localize to NEAREST fiducial now? [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("[spot-nav] skipped localization — navigation will fail until localized.")
            return False
    try:
        graph_nav_client.set_localization(
            initial_guess_localization=nav_pb2.Localization(),
            fiducial_init=graph_nav_pb2.SetLocalizationRequest.FIDUCIAL_INIT_NEAREST,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[spot-nav] set_localization failed: {exc}")
        return False
    state = graph_nav_client.get_localization_state()
    ok = bool(state.localization.waypoint_id)
    print(f"[spot-nav] localized to {state.localization.waypoint_id!r}" if ok else "[spot-nav] localization failed.")
    return ok


def _seed_tform_goal(x: float, y: float, yaw: float):
    from bosdyn.client.math_helpers import Quat, SE3Pose

    return SE3Pose(float(x), float(y), 0.0, Quat.from_yaw(float(yaw))).to_proto()


def _navigate_to_anchor(graph_nav_client, goal_pose, tol_m: float, yaw_tol_rad: float):
    """navigate_to_anchor with an optional goal tolerance (kwarg name / support
    varies by SDK version — fall back to the default tolerance)."""
    from bosdyn.api import geometry_pb2

    tol = geometry_pb2.Vec3(x=float(tol_m), y=float(tol_m), z=float(yaw_tol_rad))
    try:
        return graph_nav_client.navigate_to_anchor(
            goal_pose, 1.0, goal_waypoint_rt_seed_ewrt_seed_tolerance=tol
        )
    except TypeError:
        return graph_nav_client.navigate_to_anchor(goal_pose, 1.0)


def _run_goal(robot, graph_nav_client, command_client, payload: dict, args) -> None:
    x, y, yaw = float(payload["x"]), float(payload["y"]), float(payload["yaw"])
    tol = float(payload.get("tol_m", args.default_tolerance_m))
    label = payload.get("label") or f"#{payload.get('object_id')}"
    navigable = bool(payload.get("navigable", True))
    print(
        f"\n[spot-nav] GOAL {label}: seed x={x:.3f} y={y:.3f} yaw={math.degrees(yaw):.1f}° "
        f"tol={tol:.2f} m  navigable={navigable}"
    )
    if not navigable:
        print("[spot-nav] pose was flagged NOT navigable by the scene graph — proceeding anyway "
              "(GraphNav has its own obstacle avoidance).")

    if not args.execute:
        print(f"[spot-nav] [DRY RUN] would call navigate_to_anchor(x={x:.3f}, y={y:.3f}, "
              f"yaw={yaw:.4f}, tol={tol:.2f}). Pass --execute to move.")
        return

    if robot.is_estopped():
        print("[spot-nav] E-STOPPED — refusing to move. Clear the E-stop (estop_gui) and resend.")
        return
    if not robot.is_powered_on():
        print("[spot-nav] powering on motors…")
        robot.power_on(timeout_sec=20)
    from bosdyn.client.robot_command import blocking_stand

    blocking_stand(command_client, timeout_sec=10)

    goal_pose = _seed_tform_goal(x, y, yaw)
    yaw_tol = math.radians(args.yaw_tolerance_deg)
    _drive(graph_nav_client, lambda: _navigate_to_anchor(graph_nav_client, goal_pose, tol, yaw_tol),
           args.nav_timeout_s, "navigate_to_anchor")


def _resolve_waypoint(graph, token: str) -> str:
    """Map a token to a full GraphNav waypoint id. Accepts the id itself, a bare
    number ``38`` / ``#38`` (matched against a trailing number in each waypoint's
    name, else the list index), or empty -> ''."""
    token = str(token or "").strip().lstrip("#")
    if not token or graph is None:
        return token
    ids = [wp.id for wp in graph.waypoints]
    if token in ids:
        return token
    if token.isdigit():
        import re as _re

        n = int(token)
        for wp in graph.waypoints:
            m = _re.search(r"(\d+)\s*$", (wp.annotations.name or ""))
            if m and int(m.group(1)) == n:
                return wp.id
        if 0 <= n < len(ids):
            return ids[n]
    return token


def _run_waypoint_goal(robot, graph_nav_client, command_client, payload: dict, args, graph=None) -> None:
    wid = _resolve_waypoint(graph, payload.get("waypoint_id") or args.home_waypoint or "")
    label = payload.get("label") or wid
    if not wid:
        print("[spot-nav] goto_waypoint with no waypoint_id/number and no --home-waypoint — ignored.")
        return
    print(f"\n[spot-nav] WAYPOINT GOAL: {label}  (id {wid})")
    if not args.execute:
        print(f"[spot-nav] [DRY RUN] would call navigate_to({wid!r}). Pass --execute to move.")
        return
    if robot.is_estopped():
        print("[spot-nav] E-STOPPED — refusing to move. Clear the E-stop (estop_gui) and resend.")
        return
    if not robot.is_powered_on():
        print("[spot-nav] powering on motors…")
        robot.power_on(timeout_sec=20)
    from bosdyn.client.robot_command import blocking_stand

    blocking_stand(command_client, timeout_sec=10)
    _drive(graph_nav_client, lambda: graph_nav_client.navigate_to(wid, 1.0),
           args.nav_timeout_s, "navigate_to")


def _drive(graph_nav_client, reissue, timeout_s: float, what: str) -> None:
    """Re-issue the nav command on GraphNav's ~1 s watchdog and poll feedback
    until a terminal status or a client-side timeout."""
    from bosdyn.api.graph_nav import graph_nav_pb2

    terminal = {
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_REACHED_GOAL: "REACHED_GOAL",
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_NO_ROUTE: "NO_ROUTE",
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_NO_LOCALIZATION: "NO_LOCALIZATION",
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_LOST: "LOST",
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_STUCK: "STUCK",
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_ROBOT_IMPAIRED: "ROBOT_IMPAIRED",
        graph_nav_pb2.NavigationFeedbackResponse.STATUS_COMMAND_TIMED_OUT: "COMMAND_TIMED_OUT",
    }
    deadline = time.time() + float(timeout_s)
    while time.time() < deadline:
        nav_id = reissue()
        fb = graph_nav_client.navigation_feedback(nav_id)
        if fb.status in terminal:
            print(f"[spot-nav] {what} -> {terminal[fb.status]}")
            return
        time.sleep(0.5)
    print(f"[spot-nav] {what} timed out (client-side).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--robot-hostname", default="192.168.50.3")
    ap.add_argument("--graph-path", type=Path,
                    default=Path("~/mist_ws_ros2/scene_graph/data_walk/nav_graph_spot_4/downloaded_graph/"))
    ap.add_argument("--ws-url", default="ws://127.0.0.1:8765",
                    help="ros_ws_bridge.py running on this host")
    ap.add_argument("--default-tolerance-m", type=float, default=0.25,
                    help="Goal position tolerance if the viser panel sends none")
    ap.add_argument("--yaw-tolerance-deg", type=float, default=15.0)
    ap.add_argument("--nav-timeout-s", type=float, default=90.0)
    ap.add_argument("--home-waypoint", default="",
                    help="Waypoint id OR number (e.g. 38) used when a request carries none.")
    ap.add_argument("--auto-localize", action="store_true",
                    help="Localize to nearest fiducial without prompting")
    ap.add_argument("--execute", action="store_true",
                    help="ACTUALLY MOVE the robot. Without this it is a dry run (default).")
    ap.add_argument("--power-off-on-exit", action="store_true")
    args = ap.parse_args()

    if not (args.graph_path.expanduser() / "graph").is_file():
        print(f"[spot-nav] no 'graph' file under {args.graph_path}")
        return 2

    import bosdyn.client
    import bosdyn.client.util
    from bosdyn.client.graph_nav import GraphNavClient
    from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
    from bosdyn.client.robot_command import RobotCommandClient

    username, password = _load_credentials()
    sdk = bosdyn.client.create_standard_sdk("FARM-SpotGraphNavGoal")
    robot = sdk.create_robot(args.robot_hostname)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()

    if robot.is_estopped():
        print("[spot-nav] Robot is E-STOPPED. Start a software E-stop first, e.g.:\n"
              "    python3 -m estop_gui  {}\n"
              "then 'Release' control on the tablet, and rerun.".format(args.robot_hostname))
        return 1

    graph_nav_client = robot.ensure_client(GraphNavClient.default_service_name)
    command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    lease_client = robot.ensure_client(LeaseClient.default_service_name)

    ctrl_q: "queue.Queue[dict]" = queue.Queue()

    from ros_ws_client import RosWsClient

    ws = RosWsClient(args.ws_url, want_image=False, on_control=ctrl_q.put)
    ws.start()

    mode = "EXECUTE — the robot WILL move" if args.execute else "DRY RUN — no motion"
    print(f"[spot-nav] mode: {mode}")

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        graph = _upload_graph(graph_nav_client, args.graph_path)
        _localize(graph_nav_client, auto=args.auto_localize)
        print(f"[spot-nav] waiting for goals on {args.ws_url}  (Ctrl-C to stop)")
        try:
            while True:
                try:
                    payload = ctrl_q.get(timeout=1.0)
                except queue.Empty:
                    continue
                kind = payload.get("type")
                if kind == "cancel":
                    print("[spot-nav] STOP received.")
                    if args.execute:
                        from bosdyn.client.robot_command import RobotCommandBuilder

                        with contextlib.suppress(Exception):
                            command_client.robot_command(RobotCommandBuilder.stop_command())
                    continue
                try:
                    if kind == "goto":
                        _run_goal(robot, graph_nav_client, command_client, payload, args)
                    elif kind == "goto_waypoint":
                        _run_waypoint_goal(robot, graph_nav_client, command_client, payload, args, graph=graph)
                except Exception as exc:  # noqa: BLE001 - one bad goal must not kill the loop
                    print(f"[spot-nav] goal failed: {exc}")
        except KeyboardInterrupt:
            print("\n[spot-nav] Ctrl-C — releasing lease; Spot will stop and stand.")
        finally:
            ws.stop()
            if args.execute and args.power_off_on_exit and robot.is_powered_on():
                print("[spot-nav] powering off (safe).")
                robot.power_off(cut_immediately=False, timeout_sec=20)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
